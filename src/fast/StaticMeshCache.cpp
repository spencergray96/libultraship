#include "fast/StaticMeshCache.h"

#ifdef ENABLE_STATIC_BAKE

#include <cstring>
#include <unordered_map>
#include <vector>

#include "fast/interpreter.h"
#include "fast/PerfCounters.h"
#include "spdlog/spdlog.h"

namespace Fast {

bool gStaticBakeRecording = false;

namespace {

enum class BakeState : uint8_t {
    Unbaked,  // never recorded, or invalidated - the next frame records it
    Baked,    // has a GPU buffer and a draw table; replayed
    Rejected, // the recorder met something it cannot reproduce; interpreted forever
};

// One replayed draw call: a contiguous run of the persistent buffer that shares a program and a
// render state. The 256-triangle cap that breaks the interpreter's batches is a property of its
// staging buffer, not of the geometry, so consecutive captured flushes with identical state are
// merged here - that is the difference between ~9 draws per room and ~289.
struct BakedDraw {
    size_t byteOffset;
    size_t numTris;
    ShaderProgram* prg;
    uint8_t numFloats;
    uint8_t depthTestAndMask;
    uint8_t cull; // StaticBakeCull - what GfxSpTri1's CPU cull test would have done
    bool decal;
    bool alphaBlend;
    uint16_t primDepth;
};

struct Entry {
    BakeState state = BakeState::Unbaked;
    std::vector<float> staging; // record scratch; released once uploaded
    std::vector<BakedDraw> draws;
    uint32_t buffer = 0;
    size_t totalTris = 0;

    // The light state the vertex colours were lit under. Rooms are drawn with G_LIGHTING set and
    // the vertex normal in the colour slot, so a time-of-day change makes the baked colours
    // stale. Comparing what the interpreter is actually about to use against what it used at
    // record time is exact - no epsilon to tune, and no second copy of the light maths.
    uint8_t numLights = 0;
    F3DLight lights[MAX_LIGHTS + 1] = {};
};

std::unordered_map<const void*, Entry> sEntries;
bool sEnabled = false;

// The rendering backend, remembered the first time the interpreter reaches this file. The host
// calls StaticBakeReset() from the game thread's scene-load path, where no Interpreter is in
// hand, and freeing the GPU buffers is the one thing that needs the backend there.
GfxRenderingAPI* sRapi = nullptr;

// Recording state. Only meaningful while gStaticBakeRecording is true.
Entry* sRecording = nullptr;
const void* sRecordingKey = nullptr;
size_t sRecordDepth = 0;
float sSavedMpMatrix[4][4];
bool sRecordAborted = false;
const char* sAbortReason = nullptr;
// The cull mode the triangles now in the interpreter's staging buffer were emitted under. 0xFF
// until the first triangle of a recording, so that first triangle always opens a fresh batch.
uint8_t sRecordCull = 0xFF;

// Opcodes the recorder understands. Anything else - a matrix load, a segment write, a texture
// load, a branch_z - means the display list is doing something the replay could not reproduce,
// so the bake is abandoned for good rather than guessed at. Indexed by the raw opcode byte;
// values are F3DEX2's, which is what every compiled-in custom scene emits.
bool sOpcodeAllowed[256] = {};
bool sOpcodeTableBuilt = false;

void BuildOpcodeTable() {
    if (sOpcodeTableBuilt) {
        return;
    }
    sOpcodeTableBuilt = true;
    static const uint8_t kAllowed[] = {
        0x00, // G_NOOP
        0x01, // G_VTX
        0x03, // G_CULLDL      (an unimplemented stub in this interpreter; harmless either way)
        0x05, // G_TRI1
        0x06, // G_TRI2
        0x07, // G_QUAD
        0xd7, // G_TEXTURE     (scaling factors only - no texture is loaded)
        0xd9, // G_GEOMETRYMODE
        0xde, // G_DL          (a nested plain display list)
        0xdf, // G_ENDDL
        0xe2, // G_SETOTHERMODE_L
        0xe3, // G_SETOTHERMODE_H
        0xe6, // G_RDPLOADSYNC
        0xe7, // G_RDPPIPESYNC
        0xe8, // G_RDPTILESYNC
        0xe9, // G_RDPFULLSYNC
        0xfa, // G_SETPRIMCOLOR
        0xfb, // G_SETENVCOLOR
        0xfc, // G_SETCOMBINE
    };
    for (uint8_t op : kAllowed) {
        sOpcodeAllowed[op] = true;
    }
}

void ReleaseGpu(Entry& e) {
    if (e.buffer != 0 && sRapi != nullptr) {
        sRapi->DeleteStaticBuffer(e.buffer);
    }
    e.buffer = 0;
    e.draws.clear();
    e.draws.shrink_to_fit();
    e.staging.clear();
    e.staging.shrink_to_fit();
    e.totalTris = 0;
}

void AbortRecording(Interpreter* gfx, const char* reason) {
    if (sRecording == nullptr) {
        return;
    }
    sRecordAborted = true;
    if (sAbortReason == nullptr) {
        sAbortReason = reason;
    }
}

// The interpreter's own widescreen adjustment, as a plain multiplier. AdjXForAspectRatio is
// linear in x (it either returns x or scales it), so evaluating it at 1 gives the factor it would
// apply - including the framebuffer short-circuit, without having to reason about which branch is
// live. Recording divides it out; replay folds the current one back in, which is why a window
// resize does not need a rebake.
float AspectScale(Interpreter* gfx) {
    const float k = gfx->AdjXForAspectRatio(1.0f);
    return (k > 0.0001f || k < -0.0001f) ? k : 1.0f;
}

void BeginRecording(Interpreter* gfx, const void* key, Entry& e) {
    gfx->Flush(); // the live batch must not be swept into the recording

    memcpy(sSavedMpMatrix, gfx->mRsp->MP_matrix, sizeof(sSavedMpMatrix));

    // The record-pass matrix. Not quite identity: x is pre-divided by the widescreen factor that
    // GfxSpVertex is about to multiply back in, so the payload comes out in plain object space
    // without needing a branch on the per-vertex hot path.
    const float invAspect = 1.0f / AspectScale(gfx);
    memset(gfx->mRsp->MP_matrix, 0, sizeof(sSavedMpMatrix));
    gfx->mRsp->MP_matrix[0][0] = invAspect;
    gfx->mRsp->MP_matrix[1][1] = 1.0f;
    gfx->mRsp->MP_matrix[2][2] = 1.0f;
    gfx->mRsp->MP_matrix[3][3] = 1.0f;

    e.staging.clear();
    e.draws.clear();
    e.totalTris = 0;
    e.numLights = gfx->mRsp->current_num_lights;
    memcpy(e.lights, gfx->mRsp->current_lights, sizeof(e.lights));

    sRecording = &e;
    sRecordingKey = key;
    sRecordAborted = false;
    sAbortReason = nullptr;
    sRecordCull = 0xFF;
    // g_exec_stack.call() pushes exactly one frame for the display list we are about to enter.
    sRecordDepth = g_exec_stack.cmd_stack.size() + 1;
    gStaticBakeRecording = true;
}

void Replay(Interpreter* gfx, Entry& e) {
    gfx->Flush();

    // GfxSpTri1 applies a pending viewport/scissor change lazily, on the first triangle that needs
    // it. A replayed draw never reaches that code, so if a baked room happens to be the first
    // thing drawn after a change it has to apply it here or the room renders through the previous
    // frame's viewport.
    if (gfx->mRdp->viewport_or_scissor_changed) {
        if (memcmp(&gfx->mRdp->viewport, &gfx->mRenderingState.viewport, sizeof(gfx->mRdp->viewport)) != 0) {
            gfx->mRapi->SetViewport(gfx->mRdp->viewport.x, gfx->mRdp->viewport.y, gfx->mRdp->viewport.width,
                                    gfx->mRdp->viewport.height);
            gfx->mRenderingState.viewport = gfx->mRdp->viewport;
        }
        if (memcmp(&gfx->mRdp->scissor, &gfx->mRenderingState.scissor, sizeof(gfx->mRdp->scissor)) != 0) {
            gfx->mRapi->SetScissor(gfx->mRdp->scissor.x, gfx->mRdp->scissor.y, gfx->mRdp->scissor.width,
                                   gfx->mRdp->scissor.height);
            gfx->mRenderingState.scissor = gfx->mRdp->scissor;
        }
        gfx->mRdp->viewport_or_scissor_changed = false;
    }

    StaticBakeUniforms u = {};
    // Room display lists run under an identity modelview (gSPMatrix(&gMtxClear, ...) in z_room.c),
    // so MP_matrix at this point *is* the camera - already corrected for frame interpolation.
    memcpy(u.mvp, gfx->mRsp->MP_matrix, sizeof(u.mvp));
    const float aspect = AspectScale(gfx);
    for (int i = 0; i < 4; i++) {
        u.mvp[i][0] *= aspect;
    }
    u.fogColor[0] = gfx->mRdp->fog_color.r / 255.0f;
    u.fogColor[1] = gfx->mRdp->fog_color.g / 255.0f;
    u.fogColor[2] = gfx->mRdp->fog_color.b / 255.0f;
    u.fogColor[3] = 1.0f;
    u.fogMul = (float)gfx->mRsp->fog_mul;
    u.fogOffset = (float)gfx->mRsp->fog_offset;

    for (const BakedDraw& d : e.draws) {
        const bool depthTest = (d.depthTestAndMask & 1) != 0;
        const bool depthMask = (d.depthTestAndMask & 2) != 0;
        if (d.depthTestAndMask != gfx->mRenderingState.depth_test_and_mask) {
            gfx->mRapi->SetDepthTestAndMask(depthTest, depthMask);
            gfx->mRenderingState.depth_test_and_mask = d.depthTestAndMask;
        }
        if (d.decal != gfx->mRenderingState.decal_mode) {
            gfx->mRapi->SetZmodeDecal(d.decal);
            gfx->mRenderingState.decal_mode = d.decal;
        }
        if (d.alphaBlend != gfx->mRenderingState.alpha_blend) {
            gfx->mRapi->SetUseAlpha(d.alphaBlend);
            gfx->mRenderingState.alpha_blend = d.alphaBlend;
        }
        gfx->mRapi->SetCurrentPrimDepth((float)d.primDepth / 32767.0f);
        gfx->mRapi->DrawStaticTriangles(e.buffer, d.byteOffset, d.numTris, d.prg, u, d.cull, d.decal);

        gPerfCounters.draws++;
        gPerfCounters.drawsBaked++;
        gPerfCounters.trisBaked += d.numTris;
    }

    // The backend has just bound a shader and a vertex buffer the interpreter knows nothing
    // about. Clearing the memo makes the next interpreted triangle re-run its own binding path;
    // without it the corruption shows up in whatever draws *after* a baked room, not in the room.
    gfx->mRenderingState.mShaderProgram = nullptr;
}

// Have the lights the room is about to be drawn with moved since it was baked? The light block is
// a handful of bytes of u8 colour and s8 direction per light, so this is an exact comparison over
// (typically) two lights - no threshold, and it cannot drift the way a re-derived hash would.
bool LightsChanged(Interpreter* gfx, const Entry& e) {
    if (gfx->mRsp->current_num_lights != e.numLights) {
        return true;
    }
    const size_t n = (size_t)e.numLights;
    return memcmp(e.lights, gfx->mRsp->current_lights, n * sizeof(F3DLight)) != 0;
}

void FinishRecording(Interpreter* gfx) {
    gfx->Flush(); // capture the tail batch while still recording

    gStaticBakeRecording = false;
    memcpy(gfx->mRsp->MP_matrix, sSavedMpMatrix, sizeof(sSavedMpMatrix));

    Entry* e = sRecording;
    const void* key = sRecordingKey;
    if (e == nullptr) {
        sRecordingKey = nullptr;
        return;
    }

    bool rejected = sRecordAborted;
    const char* reason = sAbortReason;
    if (!rejected && (e->draws.empty() || e->staging.empty())) {
        rejected = true;
        reason = "nothing recordable came out of it";
    }

    // Every program the draw table names must have a transform-enabled twin, and the stride the
    // recording produced must be the one that twin's input layout reads. A mismatch here is the
    // difference between a wrong picture and a clean fallback, so it is checked before anything is
    // uploaded rather than diagnosed from a corrupt frame.
    if (!rejected) {
        for (const BakedDraw& d : e->draws) {
            if (!gfx->mRapi->PrepareStaticShader(d.prg)) {
                rejected = true;
                reason = "no transform-enabled shader variant";
                break;
            }
            const uint8_t expected = gfx->mRapi->GetShaderNumFloats(d.prg);
            if (expected != d.numFloats) {
                SPDLOG_ERROR("[staticbake] stride mismatch: recorded {} floats/vertex, shader expects {}", d.numFloats,
                             expected);
                rejected = true;
                reason = "recorded stride does not match the shader input layout";
                break;
            }
        }
    }

    if (!rejected) {
        e->buffer = gfx->mRapi->CreateStaticBuffer(e->staging.data(), e->staging.size() * sizeof(float));
        if (e->buffer == 0) {
            rejected = true;
            reason = "static vertex buffer creation failed";
        }
    }

    sRecording = nullptr;
    sRecordingKey = nullptr;

    if (rejected) {
        SPDLOG_WARN("[staticbake] display list {} rejected: {}", key, reason != nullptr ? reason : "unknown");
        e->state = BakeState::Rejected;
        ReleaseGpu(*e);
        return;
    }

    e->state = BakeState::Baked;
    SPDLOG_INFO("[staticbake] baked display list {}: {} draws, {} tris, {} KB", key, e->draws.size(), e->totalTris,
                (e->staging.size() * sizeof(float)) / 1024);
    e->staging.clear();
    e->staging.shrink_to_fit();

    // Draw it now rather than next frame: the record pass captured the geometry instead of
    // submitting it, so without this the room would be missing for exactly one frame.
    Replay(gfx, *e);
}

} // namespace

// ---------------------------------------------------------------------------
// Host-facing API
// ---------------------------------------------------------------------------

void StaticBakeSetEnabled(bool enabled) {
    sEnabled = enabled;
    BuildOpcodeTable();
}

bool StaticBakeIsEnabled() {
    return sEnabled;
}

void StaticBakeRegister(const void* displayList) {
    if (!sEnabled || displayList == nullptr) {
        return;
    }
    sEntries.emplace(displayList, Entry{});
}

void StaticBakeReset() {
    for (auto& kv : sEntries) {
        ReleaseGpu(kv.second);
    }
    sEntries.clear();
    gStaticBakeRecording = false;
    sRecording = nullptr;
    sRecordingKey = nullptr;
}

void StaticBakeInvalidateAll() {
    for (auto& kv : sEntries) {
        if (kv.second.state == BakeState::Baked) {
            // Drop the GPU buffer first: the re-record's FinishRecording overwrites e->buffer with a
            // fresh CreateStaticBuffer handle, so skipping this leaks one buffer per baked room per
            // ShaderCacheClear (which a graphics-settings change triggers).
            ReleaseGpu(kv.second);
            kv.second.state = BakeState::Unbaked;
        }
    }
}

void StaticBakeGetStats(uint32_t* registered, uint32_t* baked, uint32_t* rejected) {
    uint32_t r = 0, b = 0, j = 0;
    for (const auto& kv : sEntries) {
        r++;
        if (kv.second.state == BakeState::Baked) {
            b++;
        } else if (kv.second.state == BakeState::Rejected) {
            j++;
        }
    }
    if (registered != nullptr) {
        *registered = r;
    }
    if (baked != nullptr) {
        *baked = b;
    }
    if (rejected != nullptr) {
        *rejected = j;
    }
}

// ---------------------------------------------------------------------------
// Interpreter-facing API
// ---------------------------------------------------------------------------

bool StaticBakeIntercept(Interpreter* gfx, void* displayList) {
    if (!sEnabled || sEntries.empty() || displayList == nullptr || gStaticBakeRecording) {
        return false;
    }
    auto it = sEntries.find(displayList);
    if (it == sEntries.end()) {
        return false;
    }
    sRapi = gfx->mRapi;
    // Stepping through a frame in the GBI debugger has to see the real commands, so a debugging
    // session gets the interpreted path however the entry is marked.
    if (gfx->mGfxDebugger != nullptr && gfx->mGfxDebugger->IsDebugging()) {
        return false;
    }

    Entry& e = it->second;
    if (e.state == BakeState::Rejected) {
        return false;
    }
    if (e.state == BakeState::Baked) {
        if (LightsChanged(gfx, e)) {
            ReleaseGpu(e);
            e.state = BakeState::Unbaked;
        } else {
            Replay(gfx, e);
            return true;
        }
    }

    if (!gfx->mRapi->SupportsStaticBake()) {
        e.state = BakeState::Rejected;
        return false;
    }

    BeginRecording(gfx, it->first, e);
    return false; // let the walk run: this pass *is* the recording
}

void StaticBakeCaptureFlush(Interpreter* gfx) {
    Entry* e = sRecording;
    if (e == nullptr || gfx->mBufVboNumTris == 0) {
        return;
    }
    if (sRecordAborted) {
        return; // still recording (the display list has to finish) but nothing more is kept
    }

    const size_t floatsPerVertex = gfx->mBufVboLen / (gfx->mBufVboNumTris * 3);
    if (floatsPerVertex == 0 || floatsPerVertex * gfx->mBufVboNumTris * 3 != gfx->mBufVboLen || floatsPerVertex > 255) {
        AbortRecording(gfx, "flush did not divide into a whole number of floats per vertex");
        return;
    }

    // mRenderingState still describes the batch being flushed: GfxSpTri1 flushes *before* it
    // applies a state change, so these are the values the buffered triangles were drawn under.
    const uint8_t depthTestAndMask = gfx->mRenderingState.depth_test_and_mask;
    const bool decal = gfx->mRenderingState.decal_mode;
    const bool alphaBlend = gfx->mRenderingState.alpha_blend;
    ShaderProgram* prg = gfx->mRenderingState.mShaderProgram;
    const uint16_t primDepth = gfx->mRdp->prim_depth;
    // StaticBakeNoteMaterial closes the batch before the cull mode changes, so this is the mode
    // every triangle in the buffer was emitted under.
    const uint8_t cull = sRecordCull == 0xFF ? (uint8_t)STATIC_BAKE_CULL_NONE : sRecordCull;

    if (prg == nullptr) {
        AbortRecording(gfx, "a batch was flushed with no shader program bound");
        return;
    }

    const size_t byteOffset = e->staging.size() * sizeof(float);
    e->staging.insert(e->staging.end(), gfx->mBufVbo, gfx->mBufVbo + gfx->mBufVboLen);
    e->totalTris += gfx->mBufVboNumTris;

    // Merge into the previous draw when nothing that matters changed. Without this the
    // 256-triangle staging cap alone would give a 74k-triangle room ~289 draw calls.
    if (!e->draws.empty()) {
        BakedDraw& prev = e->draws.back();
        if (prev.prg == prg && prev.numFloats == floatsPerVertex && prev.depthTestAndMask == depthTestAndMask &&
            prev.cull == cull && prev.decal == decal && prev.alphaBlend == alphaBlend && prev.primDepth == primDepth &&
            prev.byteOffset + prev.numTris * 3 * floatsPerVertex * sizeof(float) == byteOffset) {
            prev.numTris += gfx->mBufVboNumTris;
            return;
        }
    }

    BakedDraw d = {};
    d.byteOffset = byteOffset;
    d.numTris = gfx->mBufVboNumTris;
    d.prg = prg;
    d.numFloats = (uint8_t)floatsPerVertex;
    d.depthTestAndMask = depthTestAndMask;
    d.cull = cull;
    d.decal = decal;
    d.alphaBlend = alphaBlend;
    d.primDepth = primDepth;
    e->draws.push_back(d);
}

void StaticBakeOnEndDl(Interpreter* gfx) {
    if (g_exec_stack.cmd_stack.size() < sRecordDepth) {
        FinishRecording(gfx);
    }
}

void StaticBakeOnOpcode(Interpreter* gfx, int8_t opcode) {
    if (!sOpcodeAllowed[(uint8_t)opcode]) {
        AbortRecording(gfx, "display list used an opcode the recorder does not understand");
        // Drop whatever object-space geometry is already buffered and hand the rest of the list
        // back to the normal path, correct matrix and all. One frame of this room draws short;
        // from the next frame on it is REJECTED and fully interpreted.
        gfx->mBufVboLen = 0;
        gfx->mBufVboNumTris = 0;
        gStaticBakeRecording = false;
        memcpy(gfx->mRsp->MP_matrix, sSavedMpMatrix, sizeof(sSavedMpMatrix));
        if (sRecording != nullptr) {
            SPDLOG_WARN("[staticbake] display list {} rejected at opcode {:#04x}", sRecordingKey, (uint8_t)opcode);
            sRecording->state = BakeState::Rejected;
            ReleaseGpu(*sRecording);
            sRecording = nullptr;
            sRecordingKey = nullptr;
        }
    }
}

void StaticBakeNoteMaterial(Interpreter* gfx, bool useFog, bool useBlendColor, bool useGrayscale, bool usedTexture0,
                            bool usedTexture1, uint8_t cullCode) {
    // Each of these would need its own replay-side handling that this prototype does not have:
    // textures need the sampler bindings restored, and grayscale and blend-colour fog carry
    // per-frame RDP colours in the vertex payload.
    if (usedTexture0 || usedTexture1) {
        AbortRecording(gfx, "textured material (Stage 1 is textureless-only)");
    } else if (useGrayscale) {
        AbortRecording(gfx, "grayscale material");
    } else if (useBlendColor) {
        AbortRecording(gfx, "blend-colour fog material");
    } else if ((gfx->mRsp->extra_geometry_mode & G_EX_INVERT_CULLING) != 0) {
        // MirroredWorld flips the winding test per frame; a recording would freeze whichever way
        // it was pointing on the frame it happened to be made.
        AbortRecording(gfx, "G_EX_INVERT_CULLING active");
    } else if (useFog != ((gfx->mRsp->geometry_mode & G_FOG) != 0)) {
        // GfxSpVertex stores the fog factor in the vertex's alpha channel, which is view-dependent
        // and therefore garbage in an object-space recording. That is fine when the material also
        // consumes it as fog, because the patched vertex shader recomputes it - but only then. A
        // fogged material with G_FOG clear would have the shader overwrite a real vertex alpha,
        // and an unfogged material with G_FOG set would feed the stale factor through as shade
        // alpha. Neither is reproducible, so neither is baked.
        AbortRecording(gfx, "material's fog usage disagrees with the G_FOG geometry mode");
    }

    // GfxSpTri1 drops every triangle under G_CULL_BOTH; rejecting is simpler than modelling it.
    if (cullCode == STATIC_BAKE_CULL_BOTH) {
        AbortRecording(gfx, "G_CULL_BOTH material");
    }

    // Carry the RSP's cull mode into the recording so the replay can ask the rasterizer for it.
    // A cull-mode change does not flush the interpreter's batch (it is a CPU-side decision there),
    // so close the batch here: one baked draw can only have one rasterizer state.
    const uint8_t cull = cullCode > STATIC_BAKE_CULL_BACK ? (uint8_t)STATIC_BAKE_CULL_NONE : cullCode;
    if (cull != sRecordCull) {
        gfx->Flush(); // captures what is buffered under the *previous* mode
        sRecordCull = cull;
    }
}

void StaticBakeEndFrame(Interpreter* gfx) {
    // A registered display list that never returned would otherwise leave the identity matrix and
    // the recording flag armed into the next frame. Bail out loudly instead.
    SPDLOG_WARN("[staticbake] display list {} never returned; recording abandoned", sRecordingKey);
    gStaticBakeRecording = false;
    memcpy(gfx->mRsp->MP_matrix, sSavedMpMatrix, sizeof(sSavedMpMatrix));
    if (sRecording != nullptr) {
        sRecording->state = BakeState::Rejected;
        ReleaseGpu(*sRecording);
        sRecording = nullptr;
        sRecordingKey = nullptr;
    }
}

} // namespace Fast

#else // !ENABLE_STATIC_BAKE

namespace Fast {

bool gStaticBakeRecording = false;

void StaticBakeSetEnabled(bool) {
}
bool StaticBakeIsEnabled() {
    return false;
}
void StaticBakeRegister(const void*) {
}
void StaticBakeReset() {
}
void StaticBakeInvalidateAll() {
}
void StaticBakeGetStats(uint32_t* registered, uint32_t* baked, uint32_t* rejected) {
    if (registered != nullptr) {
        *registered = 0;
    }
    if (baked != nullptr) {
        *baked = 0;
    }
    if (rejected != nullptr) {
        *rejected = 0;
    }
}

} // namespace Fast

#endif // ENABLE_STATIC_BAKE
