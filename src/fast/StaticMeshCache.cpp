#include "fast/StaticMeshCache.h"

#ifdef ENABLE_STATIC_BAKE

#include <algorithm>
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

// One replayed draw call: a contiguous run of the persistent buffer that shares a program, a render
// state and its textures. The 256-triangle cap that breaks the interpreter's batches is a property
// of its staging buffer, not of the geometry, so consecutive captured flushes with identical state
// are merged here - that is the difference between ~9 draws per room and ~289.
//
// `gpu` is what the backend draws with: the program, the cull mode GfxSpTri1's CPU test would have
// applied, the SHADE inputs the replay lights, the decal mode, and the draw's own held textures -
// one reference per slot the program samples, released in ReleaseGpu. Its bufferId is filled in
// once the room's buffer exists.
struct BakedDraw {
    StaticBakeDraw gpu;
    uint8_t numFloats;
    uint8_t depthTestAndMask;
    bool alphaBlend;
    uint16_t primDepth;
};

// Everything that has to match for a new capture to extend the previous draw instead of opening
// one. Textures compare by hold handle, which is equal exactly when the view and the sampler are
// the same objects - so two batches with one program but different textures, or one texture under
// different wrap or filter settings, stay separate draws.
bool SameDrawState(const BakedDraw& a, const BakedDraw& b) {
    return a.gpu.prg == b.gpu.prg && a.gpu.shadeMask == b.gpu.shadeMask && a.gpu.cullMode == b.gpu.cullMode &&
           a.gpu.zmodeDecal == b.gpu.zmodeDecal && a.gpu.textures[0] == b.gpu.textures[0] &&
           a.gpu.textures[1] == b.gpu.textures[1] && a.numFloats == b.numFloats &&
           a.depthTestAndMask == b.depthTestAndMask && a.alphaBlend == b.alphaBlend && a.primDepth == b.primDepth;
}

struct Entry {
    BakeState state = BakeState::Unbaked;
    std::vector<float> staging; // record scratch; released once uploaded
    std::vector<BakedDraw> draws;
    uint32_t buffer = 0;
    size_t totalTris = 0;
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
// The cull mode and shade-input mask the triangles now in the interpreter's staging buffer were
// emitted under. 0xFF until the first triangle of a recording, so that first triangle always opens
// a fresh batch. (0xFF is never a real mask: a combiner has at most 7 inputs.)
uint8_t sRecordCull = 0xFF;
uint8_t sRecordShadeMask = 0xFF;

// Opcodes the recorder understands. Anything else - a matrix load, a segment write, a branch_z -
// means the display list is doing something the replay could not reproduce, so the bake is
// abandoned for good rather than guessed at. Indexed by the raw opcode byte; values are F3DEX2's,
// which is what every compiled-in custom scene emits.
//
// The texture loads run as normal while recording: they only stage RDP state, and the triangles
// after them import and bind the texture exactly as a live frame would, which is what the capture
// then holds. They are exactly what our scenes emit (sturdy-bassoon#142 step 0): gsDPLoadTextureBlock
// is FD F5 E6 F3 E7 F5 F2, and Fast64's CI4 material adds F0.
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
        0xd7, // G_TEXTURE     (scaling factors and the tile to use)
        0xd9, // G_GEOMETRYMODE
        0xde, // G_DL          (a nested plain display list; not through a segment - see the guard)
        0xdf, // G_ENDDL
        0xe2, // G_SETOTHERMODE_L
        0xe3, // G_SETOTHERMODE_H
        0xe6, // G_RDPLOADSYNC
        0xe7, // G_RDPPIPESYNC
        0xe8, // G_RDPTILESYNC
        0xe9, // G_RDPFULLSYNC
        0xf0, // G_LOADTLUT    (a palette; CI textures are expanded to RGBA at upload, so only here)
        0xf2, // G_SETTILESIZE (a scroll changes it per frame, but reaches the list through a segment)
        0xf3, // G_LOADBLOCK
        0xf5, // G_SETTILE
        0xfa, // G_SETPRIMCOLOR
        0xfb, // G_SETENVCOLOR
        0xfc, // G_SETCOMBINE
        0xfd, // G_SETTIMG     (not through a segment - see the guard)
    };
    for (uint8_t op : kAllowed) {
        sOpcodeAllowed[op] = true;
    }
}

void ReleaseTextures(const StaticBakeDraw& d) {
    if (sRapi == nullptr) {
        return;
    }
    for (uint32_t handle : d.textures) {
        if (handle != 0) {
            sRapi->ReleaseStaticTexture(handle);
        }
    }
}

void ReleaseGpu(Entry& e) {
    if (e.buffer != 0 && sRapi != nullptr) {
        sRapi->DeleteStaticBuffer(e.buffer);
    }
    for (const BakedDraw& d : e.draws) {
        ReleaseTextures(d.gpu);
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

    ReleaseGpu(e); // an UNBAKED entry holds nothing, but if it ever did its textures would leak here

    sRecording = &e;
    sRecordingKey = key;
    sRecordAborted = false;
    sAbortReason = nullptr;
    sRecordCull = 0xFF;
    sRecordShadeMask = 0xFF;
    // g_exec_stack.call() pushes exactly one frame for the display list we are about to enter.
    sRecordDepth = g_exec_stack.cmd_stack.size() + 1;
    gStaticBakeRecording = true;
}

// The RSP counts the ambient light among its lights, last.
int NumDirLights(Interpreter* gfx) {
    return (int)gfx->mRsp->current_num_lights - 1;
}

// Can the replay shader light this frame? It carries a fixed number of directional lights, and
// there has to be an ambient one. A frame that fails this is interpreted, not replayed: the bake
// itself holds no lighting, so it is still good for the next frame that passes.
bool LightsFitReplay(Interpreter* gfx) {
    const int numDir = NumDirLights(gfx);
    return numDir >= 0 && numDir <= STATIC_BAKE_MAX_DIR_LIGHTS;
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

    // The lights, as GfxSpVertex would read them for this display list: the last one is the
    // ambient, and each directional light's direction goes through CalculateNormalDir under the
    // current modelview - the same call, so the replay shader lights from the same numbers.
    // StaticBakeIntercept has already checked the count fits.
    const int numDir = NumDirLights(gfx);
    const F3DLight_t& ambient = gfx->mRsp->current_lights[numDir].l;
    for (int c = 0; c < 3; c++) {
        u.ambient[c] = (float)ambient.col[c];
    }
    u.numDirLights = (uint32_t)numDir;
    for (int i = 0; i < numDir; i++) {
        const F3DLight_t& light = gfx->mRsp->current_lights[i].l;
        gfx->CalculateNormalDir(&light, u.lightDir[i]);
        for (int c = 0; c < 3; c++) {
            u.lightColor[i][c] = (float)light.col[c];
        }
    }

    for (const BakedDraw& d : e.draws) {
        const bool depthTest = (d.depthTestAndMask & 1) != 0;
        const bool depthMask = (d.depthTestAndMask & 2) != 0;
        if (d.depthTestAndMask != gfx->mRenderingState.depth_test_and_mask) {
            gfx->mRapi->SetDepthTestAndMask(depthTest, depthMask);
            gfx->mRenderingState.depth_test_and_mask = d.depthTestAndMask;
        }
        if (d.gpu.zmodeDecal != gfx->mRenderingState.decal_mode) {
            gfx->mRapi->SetZmodeDecal(d.gpu.zmodeDecal);
            gfx->mRenderingState.decal_mode = d.gpu.zmodeDecal;
        }
        if (d.alphaBlend != gfx->mRenderingState.alpha_blend) {
            gfx->mRapi->SetUseAlpha(d.alphaBlend);
            gfx->mRenderingState.alpha_blend = d.alphaBlend;
        }
        gfx->mRapi->SetCurrentPrimDepth((float)d.primDepth / 32767.0f);
        gfx->mRapi->DrawStaticTriangles(d.gpu, u);

        gPerfCounters.draws++;
        gPerfCounters.drawsBaked++;
        gPerfCounters.trisBaked += d.gpu.numTris;
    }

    // The backend has just bound a shader and a vertex buffer the interpreter knows nothing
    // about. Clearing the memo makes the next interpreted triangle re-run its own binding path;
    // without it the corruption shows up in whatever draws *after* a baked room, not in the room.
    //
    // Textures need nothing here. The draws bound their held textures straight to the GPU and left
    // the backend's own memo saying so, so the next interpreted draw rebinds any slot that differs.
    // The interpreter's side - mRenderingState.mTextures and the cache id each slot names - was
    // never touched, and still describes the texture the interpreter will ask for.
    gfx->mRenderingState.mShaderProgram = nullptr;
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
            if (!gfx->mRapi->PrepareStaticShader(d.gpu.prg, d.gpu.shadeMask)) {
                rejected = true;
                reason = "no transform-enabled shader variant";
                break;
            }
            const uint8_t expected = gfx->mRapi->GetShaderNumFloats(d.gpu.prg);
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

    // The textured draws and the distinct textures they hold, for the log: a room that loads textures
    // and reports 0 here is not drawing them.
    size_t texturedDraws = 0;
    std::vector<uint32_t> held;
    for (BakedDraw& d : e->draws) {
        d.gpu.bufferId = e->buffer;
        bool textured = false;
        for (uint32_t handle : d.gpu.textures) {
            if (handle != 0) {
                textured = true;
                if (std::find(held.begin(), held.end(), handle) == held.end()) {
                    held.push_back(handle);
                }
            }
        }
        texturedDraws += textured ? 1 : 0;
    }

    e->state = BakeState::Baked;
    SPDLOG_INFO("[staticbake] baked display list {}: {} draws, {} tris, {} KB, {} textured draws, {} textures", key,
                e->draws.size(), e->totalTris, (e->staging.size() * sizeof(float)) / 1024, texturedDraws, held.size());
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
    if (displayList == nullptr) {
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
    // Checked before recording as well as before replaying, because a recording ends by replaying.
    if (!LightsFitReplay(gfx)) {
        return false;
    }
    if (e.state == BakeState::Baked) {
        Replay(gfx, e);
        return true;
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
    // StaticBakeNoteMaterial closes the batch before the cull mode or the shade mask changes, so
    // these are what every triangle in the buffer was emitted under.
    const uint8_t cull = sRecordCull == 0xFF ? (uint8_t)STATIC_BAKE_CULL_NONE : sRecordCull;
    const uint8_t shadeMask = sRecordShadeMask == 0xFF ? 0 : sRecordShadeMask;

    if (prg == nullptr) {
        AbortRecording(gfx, "a batch was flushed with no shader program bound");
        return;
    }

    const size_t byteOffset = e->staging.size() * sizeof(float);
    e->staging.insert(e->staging.end(), gfx->mBufVbo, gfx->mBufVbo + gfx->mBufVboLen);
    e->totalTris += gfx->mBufVboNumTris;

    BakedDraw d = {};
    d.gpu.byteOffset = byteOffset;
    d.gpu.numTris = gfx->mBufVboNumTris;
    d.gpu.prg = prg;
    d.gpu.shadeMask = shadeMask;
    d.gpu.cullMode = cull;
    d.gpu.zmodeDecal = decal;
    d.numFloats = (uint8_t)floatsPerVertex;
    d.depthTestAndMask = depthTestAndMask;
    d.alphaBlend = alphaBlend;
    d.primDepth = primDepth;

    // Hold the textures the batch was drawn with, for every slot the program samples. What the
    // backend has bound right now is this batch's: GfxSpTri1 flushes before it imports a texture
    // and before it changes a sampler. The hold copies what it needs rather than pointing into the
    // cache or at mRdp, both of which have moved on to the next batch or will. A slot the program
    // samples but the combiner does not read (2-cycle marks TEXEL1 used whenever TEXEL0 is) is held
    // as bound, even if that is nothing, which is what the interpreter would draw with; the slots the
    // combiner does read were checked in StaticBakeNoteMaterial.
    uint8_t numInputs = 0;
    bool usedTextures[STATIC_BAKE_TEXTURE_SLOTS] = {};
    gfx->mRapi->ShaderGetInfo(prg, &numInputs, usedTextures);
    for (int i = 0; i < STATIC_BAKE_TEXTURE_SLOTS; i++) {
        d.gpu.textures[i] = usedTextures[i] ? gfx->mRapi->HoldStaticTexture(i) : 0;
    }

    // Merge into the previous draw when nothing that matters changed. Without this the
    // 256-triangle staging cap alone would give a 74k-triangle room ~289 draw calls.
    if (!e->draws.empty()) {
        BakedDraw& prev = e->draws.back();
        if (SameDrawState(prev, d) &&
            prev.gpu.byteOffset + prev.gpu.numTris * 3 * floatsPerVertex * sizeof(float) == byteOffset) {
            prev.gpu.numTris += d.gpu.numTris;
            ReleaseTextures(d.gpu); // the same handles prev already holds, one reference too many
            return;
        }
    }
    e->draws.push_back(d);
}

void StaticBakeOnEndDl(Interpreter* gfx) {
    if (g_exec_stack.cmd_stack.size() < sRecordDepth) {
        FinishRecording(gfx);
    }
}

void StaticBakeOnOpcode(Interpreter* gfx, int8_t opcode, uintptr_t w1) {
    // A segmented operand is resolved through mSegmentPointers at record time, and a segment is how
    // OoT feeds a display list something that changes every frame: vanilla's texture scrolls
    // (Gfx_TexScroll, a G_SETTILESIZE behind a segment - which the whitelist now lets through) and
    // its animated materials. A recording would freeze whichever frame it happened on. Odd = segmented
    // is SoH's convention (Interpreter::SegAddr). No compiled-in custom scene does this today; the
    // guard is there so that nothing that does can bake silently.
    constexpr uint8_t kOpVtx = 0x01;     // G_VTX
    constexpr uint8_t kOpDl = 0xde;      // G_DL
    constexpr uint8_t kOpSetTImg = 0xfd; // G_SETTIMG
    const uint8_t op = (uint8_t)opcode;
    if ((op == kOpDl || op == kOpSetTImg || op == kOpVtx) && (w1 & 1) != 0) {
        AbortRecording(gfx, op == kOpDl        ? "nested display list reached through a segment"
                            : op == kOpSetTImg ? "texture image reached through a segment"
                                               : "vertices reached through a segment");
    }
    if (!sOpcodeAllowed[op]) {
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

void StaticBakeNoteMaterial(Interpreter* gfx, const StaticBakeMaterial& m) {
    static_assert(sizeof(m.combTextures) / sizeof(m.combTextures[0]) == STATIC_BAKE_TEXTURE_SLOTS,
                  "StaticBakeMaterial names one flag per texture slot a baked draw can bind");

    // A texture slot the combiner reads has to be one the replay can hold: an entry the import
    // really uploaded (a null entry or a non-upload would bind whatever the reused id held before).
    // GfxSpTri1 has just imported these, so this is checked against the live state, per triangle -
    // which is per batch, since the interpreter flushes before any import.
    bool textureUnholdable = false;
    for (int i = 0; i < STATIC_BAKE_TEXTURE_SLOTS; i++) {
        if (m.combTextures[i]) {
            const TextureCacheNode* node = gfx->mRenderingState.mTextures[i];
            if (node == nullptr || !node->second.uploaded) {
                textureUnholdable = true;
            }
        }
    }

    // Each of these would need its own replay-side handling that the bake does not have: HD mask and
    // blend textures bind four more slots, and grayscale and blend-colour fog carry per-frame RDP
    // colours in the vertex payload.
    if (m.maskedOrBlended) {
        AbortRecording(gfx, "masked or blended (HD) texture");
    } else if (textureUnholdable) {
        AbortRecording(gfx, "a texture slot the combiner reads uploaded nothing");
    } else if (m.useGrayscale) {
        AbortRecording(gfx, "grayscale material");
    } else if (m.useBlendColor) {
        AbortRecording(gfx, "blend-colour fog material");
    } else if ((gfx->mRsp->extra_geometry_mode & G_EX_INVERT_CULLING) != 0) {
        // MirroredWorld flips the winding test per frame; a recording would freeze whichever way
        // it was pointing on the frame it happened to be made.
        AbortRecording(gfx, "G_EX_INVERT_CULLING active");
    } else if (m.useFog != ((gfx->mRsp->geometry_mode & G_FOG) != 0)) {
        // GfxSpVertex stores the fog factor in the vertex's alpha channel, which is view-dependent
        // and therefore garbage in an object-space recording. That is fine when the material also
        // consumes it as fog, because the patched vertex shader recomputes it - but only then. A
        // fogged material with G_FOG clear would have the shader overwrite a real vertex alpha,
        // and an unfogged material with G_FOG set would feed the stale factor through as shade
        // alpha. Neither is reproducible, so neither is baked.
        AbortRecording(gfx, "material's fog usage disagrees with the G_FOG geometry mode");
    }

    // GfxSpTri1 drops every triangle under G_CULL_BOTH; rejecting is simpler than modelling it.
    if (m.cullCode == STATIC_BAKE_CULL_BOTH) {
        AbortRecording(gfx, "G_CULL_BOTH material");
    }

    // Carry the RSP's cull mode into the recording so the replay can ask the rasterizer for it.
    // A cull-mode change does not flush the interpreter's batch (it is a CPU-side decision there),
    // so close the batch here: one baked draw can only have one rasterizer state.
    //
    // The shade mask is the same story. Two combiners can share one shader program with SHADE on
    // different inputs, and the interpreter does not flush between them (the program did not
    // change; only which input is fed the vertex colour did). The replay shader lights a fixed
    // input, so one baked draw can only have one mask.
    const uint8_t cull = m.cullCode > STATIC_BAKE_CULL_BACK ? (uint8_t)STATIC_BAKE_CULL_NONE : m.cullCode;
    if (cull != sRecordCull || m.shadeMask != sRecordShadeMask) {
        gfx->Flush(); // captures what is buffered under the *previous* mode and mask
        sRecordCull = cull;
        sRecordShadeMask = m.shadeMask;
    }
}

void StaticBakeAbort(Interpreter* gfx, const char* reason) {
    AbortRecording(gfx, reason);
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
