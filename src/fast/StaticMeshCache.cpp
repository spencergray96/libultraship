#include "fast/StaticMeshCache.h"

#ifdef ENABLE_STATIC_BAKE

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "fast/interpreter.h"
#include "fast/PerfCounters.h"
#include "spdlog/spdlog.h"

namespace Fast {

bool gStaticBakeRecording = false;
bool gStaticBakeScrollsBound = false;
bool gStaticBakeWindOn = StaticBakeWind{}.amplitude != 0.0f;
uint32_t gStaticBakeWindGeneration = 0;

namespace {

enum class BakeState : uint8_t {
    Unbaked,  // never recorded, or invalidated - the next frame records it
    Baked,    // has a GPU buffer and a draw table; replayed
    Rejected, // the recorder met something it cannot reproduce; interpreted forever
};

// One replayed draw call: a contiguous run of the persistent buffer that shares a program, a render
// state and its textures. The 256-triangle cap that breaks the interpreter's batches is a property
// of its staging buffer, not of the geometry, so consecutive captured flushes with identical state
// are merged here - that is the difference between ~9 draws per room and ~289. SortByMaterial then
// merges the ones that were not consecutive, before the upload.
//
// `gpu` is what the backend draws with: the program, the cull mode GfxSpTri1's CPU test would have
// applied, the SHADE inputs the replay lights, the decal mode, and the draw's own held textures -
// one reference per slot the program samples, released in ReleaseGpu. Its bufferId is filled in
// once the room's buffer exists.
//
// `scroll` is the scroll rate the registry gave each slot's texture when the batch was recorded
// (sturdy-bassoon#187 A1), in texture widths and heights a second: xy TEXEL0's, zw TEXEL1's. The
// replay turns TEXEL0's into the draw's offset. TEXEL1's splits batches but is not applied: a baked
// prop samples one texture today, and several textures a prop (#201) is what makes it move.
struct BakedDraw {
    StaticBakeDraw gpu;
    uint8_t numFloats;
    uint8_t depthTestAndMask;
    bool alphaBlend;
    uint16_t primDepth;
    float scroll[4];
};

// TEXEL0 has a rate: the one the replay applies, so the draw moves. (TEXEL1's splits, but is not applied.)
bool ScrollsTexel0(const BakedDraw& d) {
    return d.scroll[0] != 0.0f || d.scroll[1] != 0.0f;
}

// Everything that has to match for a new capture to extend the previous draw instead of opening
// one. Textures compare by hold handle, which is equal exactly when the view and the sampler are
// the same objects - so two batches with one program but different textures, or one texture under
// different wrap or filter settings, stay separate draws. The scroll rates compare too: a draw has one
// offset. (Today a rate belongs to a texture, so two batches with equal hold handles have equal rates
// and the term never decides on its own; it keeps the draw's state complete if that ever changes.)
bool SameDrawState(const BakedDraw& a, const BakedDraw& b) {
    return a.gpu.prg == b.gpu.prg && a.gpu.shadeMask == b.gpu.shadeMask && a.gpu.cullMode == b.gpu.cullMode &&
           a.gpu.zmodeDecal == b.gpu.zmodeDecal && a.gpu.textures[0] == b.gpu.textures[0] &&
           a.gpu.textures[1] == b.gpu.textures[1] && a.numFloats == b.numFloats &&
           a.depthTestAndMask == b.depthTestAndMask && a.alphaBlend == b.alphaBlend && a.primDepth == b.primDepth &&
           memcmp(a.scroll, b.scroll, sizeof(a.scroll)) == 0;
}

struct Entry {
    BakeState state = BakeState::Unbaked;
    std::vector<float> staging; // record scratch; released once uploaded
    std::vector<BakedDraw> draws;
    uint32_t buffer = 0;
    size_t totalTris = 0;
    // Wind (#209 W1): the weighted vertices the recording loaded, and the triangles with a weighted
    // corner. A list with none replays with the wind registers at 0, as it did before wind existed.
    uint32_t windVertices = 0;
    uint32_t windTris = 0;
    const char* rejectReason = nullptr; // a literal, set wherever state becomes Rejected
};

void Reject(Entry& e, const char* reason) {
    e.state = BakeState::Rejected;
    e.rejectReason = reason;
}

std::unordered_map<const void*, Entry> sEntries;
bool sEnabled = false;
bool sSortByMaterial = true;

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
// The scroll rates (BakedDraw::scroll) the buffered triangles were emitted under; closed the same way.
float sRecordScroll[4] = {};

// The texture scroll registry (sturdy-bassoon#187 A1), for the life of the process: see the header's
// "Texture scroll". Keyed by the path as an archive list names it, without "__OTR__". A path is bound
// when it resolves: `image` is the image data, which is what a texture-cache key names, and `owner`
// the texture resource, held so that address stays this texture's for as long as it is bound.
struct ScrollReg {
    float rate[2] = {};
    const void* image = nullptr;
    std::shared_ptr<const void> owner;
};
std::unordered_map<std::string, ScrollReg> sScrollByPath;
// The bound addresses and their rates: the lookup the recorder and the interpreter make per triangle.
// Exactly one per bound path; gStaticBakeScrollsBound says whether it is empty.
std::unordered_map<const void*, std::pair<float, float>> sScrollByImage;

// The clock (StaticBakeClockSeconds): real time since the process started, or pinned.
const std::chrono::steady_clock::time_point sClockEpoch = std::chrono::steady_clock::now();
double sClockPinned = -1.0; // < 0: running
double sClockNow = 0.0;     // this frame's value, sampled by StaticBakeBeginFrame

double RealClockSeconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - sClockEpoch).count();
}

const char* StripOtr(const char* path) {
    return strncmp(path, "__OTR__", 7) == 0 ? path + 7 : path;
}

void Unbind(ScrollReg& reg) {
    if (reg.image != nullptr) {
        sScrollByImage.erase(reg.image);
    }
    reg.image = nullptr;
    reg.owner.reset();
    gStaticBakeScrollsBound = !sScrollByImage.empty();
}

// The rate of the texture whose image data is at `image`, into out[0..1]; false (out untouched) when
// it does not scroll.
bool ScrollRateOf(const void* image, float* out) {
    if (!gStaticBakeScrollsBound || image == nullptr) {
        return false;
    }
    auto it = sScrollByImage.find(image);
    if (it == sScrollByImage.end()) {
        return false;
    }
    out[0] = it->second.first;
    out[1] = it->second.second;
    return true;
}

// frac(rate x clock) per axis: only the fraction, so the number stays small and exact however long the
// session runs, and a wrapping texture shows the same picture either way.
void ScrollOffsetNow(const float* rate, float* out) {
    for (int i = 0; i < 2; i++) {
        const double x = (double)rate[i] * sClockNow;
        out[i] = (float)(x - std::floor(x));
    }
}

// The frame's wind (sturdy-bassoon#209 W1; the header's "Wind in the replay"), and what it did: the
// frame being drawn counts into sWindNow, which StaticBakeBeginFrame hands to sWindLast.
StaticBakeWind sWind;
StaticBakeWindStats sWindNow;
StaticBakeWindStats sWindLast;

// A recorded vertex's w is (1 or 2) + 4 x its wind code, so a code is there when w reaches 4.
bool RecordedWeighted(float w) {
    return w >= STATIC_BAKE_WIND_W_SCALE;
}

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
        // Archive-loaded display lists (sturdy-bassoon#171, proven in #160): Fast64's hm64 XML emits
        // these. Each names a resource by path, resolved through the resource manager as it runs, so
        // the recording captures what the path resolved to then - as it does a texture's pixels. A
        // path that resolves to nothing is refused in its handler (StaticBakeAbort), never frozen
        // into a bake with a piece missing. The by-hash forms (0x20 G_VTX_OTR_HASH, 0x31
        // G_DL_OTR_HASH, 0x32 G_SETTIMG_OTR_HASH) stay refused until something needs them.
        0x24, // G_VTX_OTR_FILEPATH
        0x25, // G_SETTIMG_OTR_FILEPATH
        0x26, // G_TRI1_OTR    (G_TRI1 with 16-bit indices)
        0x27, // G_DL_OTR_FILEPATH
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
    e.windVertices = 0;
    e.windTris = 0;
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
    memset(sRecordScroll, 0, sizeof(sRecordScroll));
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

    // What moves (#187 A1): per list entry the wind registers; per draw TEXEL0's scroll offset, 0 for a
    // still draw. The backend re-sends the block only when it changes, so a list with nothing
    // scrolling sends it at most once.
    StaticBakeAnimUniforms anim = {};
    // Wind (#209 W1), worked out once for the entry from its modelview: room lists run under the
    // identity, an actor's list under the actor's matrix. Only a list that recorded a weighted vertex
    // gets them; every other list keeps them at 0, so its buffer is exactly what it was without wind.
    if (e.windTris != 0 && gStaticBakeWindOn && gfx->mRsp->modelview_matrix_stack_size > 0) {
        StaticBakeWindVectors(gfx->mRsp->modelview_matrix_stack[gfx->mRsp->modelview_matrix_stack_size - 1],
                              anim.windK, anim.windB, false);
        sWindNow.replayEntries++;
    }

    for (const BakedDraw& d : e.draws) {
        if (ScrollsTexel0(d)) {
            ScrollOffsetNow(d.scroll, anim.uvOffset);
        } else {
            anim.uvOffset[0] = anim.uvOffset[1] = 0.0f;
        }
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
        gfx->mRapi->DrawStaticTriangles(d.gpu, u, anim);

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

// How far a batch may move when a recording is ordered by material (sturdy-bassoon#158).
enum class BatchOrder : uint8_t {
    // Opaque, depth-tested and depth-written, not a decal: the depth test decides what shows, so
    // these draw the same picture in any order - bar exactly coplanar surfaces, where the order
    // picks the winner. Grouped by material.
    Free,
    // A decal or an alpha-blended batch under a depth test. Each has to draw after what it sits on
    // or shows through to, so it goes after the Free batches around it, in its original order.
    AfterOpaque,
    // Anything else - no depth test, or an opaque batch that does not write depth - covers or is
    // covered by exactly what came before and after it. Stays where it is; nothing moves past it.
    Fixed,
};

BatchOrder OrderOf(const BakedDraw& d) {
    const bool depthTest = (d.depthTestAndMask & 1) != 0;
    const bool depthMask = (d.depthTestAndMask & 2) != 0;
    if (!d.alphaBlend && !d.gpu.zmodeDecal && depthTest && depthMask) {
        return BatchOrder::Free;
    }
    if (depthTest && (d.alphaBlend || d.gpu.zmodeDecal)) {
        return BatchOrder::AfterOpaque;
    }
    return BatchOrder::Fixed;
}

// Reorder a finished recording so batches of one material sit together, then merge them into one
// draw each. A draw call is one material, and the capture merges a batch only into the one right
// before it, so a list that alternates materials - props of different kinds, or one prop drawn in
// two materials - was a draw per batch. Grouped, it is about one draw per material per run of Free
// batches. Materials keep the order they first appear in, so the result is the same every session.
//
// Called before the upload, on the staging buffer: rebuilds it in the new order (one copy), and
// releases the texture references the merges make redundant, as the capture's own merge does.
// Returns the time it took, for the log.
double SortByMaterial(Entry& e) {
    const auto start = std::chrono::steady_clock::now();
    const std::vector<BakedDraw>& in = e.draws;

    std::vector<size_t> order;
    order.reserve(in.size());
    std::vector<std::vector<size_t>> groups; // this run's Free batches, one group per material
    std::vector<size_t> afterOpaque;         // this run's AfterOpaque batches, in list order
    auto closeRun = [&]() {
        for (const std::vector<size_t>& g : groups) {
            order.insert(order.end(), g.begin(), g.end());
        }
        order.insert(order.end(), afterOpaque.begin(), afterOpaque.end());
        groups.clear();
        afterOpaque.clear();
    };
    for (size_t i = 0; i < in.size(); i++) {
        switch (OrderOf(in[i])) {
            case BatchOrder::Free: {
                auto g = std::find_if(groups.begin(), groups.end(),
                                      [&](const std::vector<size_t>& grp) { return SameDrawState(in[grp[0]], in[i]); });
                if (g != groups.end()) {
                    g->push_back(i);
                } else {
                    groups.push_back({ i });
                }
                break;
            }
            case BatchOrder::AfterOpaque:
                afterOpaque.push_back(i);
                break;
            case BatchOrder::Fixed:
                closeRun();
                order.push_back(i);
                break;
        }
    }
    closeRun();

    bool moved = false;
    for (size_t i = 0; i < order.size() && !moved; i++) {
        moved = order[i] != i;
    }
    if (moved) {
        std::vector<float> staging;
        staging.reserve(e.staging.size());
        std::vector<BakedDraw> draws;
        for (size_t idx : order) {
            BakedDraw d = in[idx];
            const size_t first = d.gpu.byteOffset / sizeof(float);
            const size_t count = (size_t)d.gpu.numTris * 3 * d.numFloats;
            d.gpu.byteOffset = staging.size() * sizeof(float);
            staging.insert(staging.end(), e.staging.begin() + first, e.staging.begin() + first + count);
            // Appended in order, so a draw of the same state is always contiguous with the last one.
            if (!draws.empty() && SameDrawState(draws.back(), d)) {
                draws.back().gpu.numTris += d.gpu.numTris;
                ReleaseTextures(d.gpu); // the same handles the merged-into draw already holds
            } else {
                draws.push_back(d);
            }
        }
        e.staging.swap(staging);
        e.draws.swap(draws);
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
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

    const size_t listOrderDraws = e->draws.size();
    double sortMs = 0.0;
    if (!rejected && sSortByMaterial) {
        sortMs = SortByMaterial(*e);
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
        Reject(*e, reason != nullptr ? reason : "unknown");
        ReleaseGpu(*e);
        return;
    }

    // The textured draws and the distinct textures they hold, for the log: a room that loads textures
    // and reports 0 here is not drawing them.
    size_t texturedDraws = 0;
    size_t scrollingDraws = 0;
    std::vector<uint32_t> held;
    for (BakedDraw& d : e->draws) {
        d.gpu.bufferId = e->buffer;
        scrollingDraws += ScrollsTexel0(d) ? 1 : 0;
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
    // The draws the list would have in list order, and what ordering it by material cost, after the
    // fields older run scripts parse; then the scrolling draws (#187 A1), and the wind (#209 W1).
    SPDLOG_INFO("[staticbake] baked display list {}: {} draws, {} tris, {} KB, {} textured draws, {} textures; "
                "{} draws in list order, {}; {} scrolling draws; {} wind vertices in {} tris",
                key, e->draws.size(), e->totalTris, (e->staging.size() * sizeof(float)) / 1024, texturedDraws,
                held.size(), listOrderDraws,
                sSortByMaterial ? fmt::format("sorted by material in {:.2f} ms", sortMs) : std::string("not sorted"),
                scrollingDraws, e->windVertices, e->windTris);
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

void StaticBakeSetSortByMaterial(bool sort) {
    if (sort != sSortByMaterial) {
        sSortByMaterial = sort;
        StaticBakeInvalidateAll(); // the order is decided at record time
    }
}

bool StaticBakeSortsByMaterial() {
    return sSortByMaterial;
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

namespace {

StaticBakeEntryInfo InfoOf(const Entry& e) {
    StaticBakeEntryInfo info;
    switch (e.state) {
        case BakeState::Unbaked:
            info.state = StaticBakeEntryState::Unbaked;
            break;
        case BakeState::Baked:
            info.state = StaticBakeEntryState::Baked;
            info.draws = (uint32_t)e.draws.size();
            info.tris = (uint32_t)e.totalTris;
            for (const BakedDraw& d : e.draws) {
                if (ScrollsTexel0(d)) {
                    info.scrollingDraws++;
                    info.scrollingTris += (uint32_t)d.gpu.numTris;
                }
            }
            info.windVertices = e.windVertices;
            info.windTris = e.windTris;
            break;
        case BakeState::Rejected:
            info.state = StaticBakeEntryState::Rejected;
            info.rejectReason = e.rejectReason;
            break;
    }
    return info;
}

} // namespace

StaticBakeEntryInfo StaticBakeGetEntry(const void* displayList) {
    auto it = sEntries.find(displayList);
    if (it == sEntries.end()) {
        return {};
    }
    return InfoOf(it->second);
}

std::vector<StaticBakeScrollingEntry> StaticBakeGetScrollingEntries() {
    std::vector<StaticBakeScrollingEntry> out;
    for (const auto& kv : sEntries) {
        StaticBakeEntryInfo info = InfoOf(kv.second);
        if (info.scrollingDraws != 0) {
            out.push_back({ kv.first, info });
        }
    }
    return out;
}

std::vector<StaticBakeScrollingEntry> StaticBakeGetWindEntries() {
    std::vector<StaticBakeScrollingEntry> out;
    for (const auto& kv : sEntries) {
        StaticBakeEntryInfo info = InfoOf(kv.second);
        if (info.windVertices != 0 || info.windTris != 0) {
            out.push_back({ kv.first, info });
        }
    }
    return out;
}

bool StaticBakeSetTextureScroll(const char* path, float du, float dv) {
    if (path == nullptr || *path == '\0' || !std::isfinite(du) || !std::isfinite(dv)) {
        return false;
    }
    // -0 and +0 are one rate: a draw state compares rates bit for bit.
    du = du == 0.0f ? 0.0f : du;
    dv = dv == 0.0f ? 0.0f : dv;
    const std::string key = StripOtr(path);
    auto it = sScrollByPath.find(key);
    if (du == 0.0f && dv == 0.0f) {
        if (it == sScrollByPath.end()) {
            return false;
        }
        Unbind(it->second);
        sScrollByPath.erase(it);
        return true;
    }
    if (it == sScrollByPath.end()) {
        ScrollReg reg;
        reg.rate[0] = du;
        reg.rate[1] = dv;
        sScrollByPath.emplace(key, std::move(reg)); // bound when an archive list next names it
        return true;
    }
    ScrollReg& reg = it->second;
    if (reg.rate[0] == du && reg.rate[1] == dv) {
        return false; // the idempotent case: a host re-registering at every list load
    }
    reg.rate[0] = du;
    reg.rate[1] = dv;
    if (reg.image != nullptr) {
        sScrollByImage[reg.image] = { du, dv };
    }
    return true;
}

void StaticBakeClearTextureScrolls() {
    sScrollByPath.clear(); // releases every held texture resource
    sScrollByImage.clear();
    gStaticBakeScrollsBound = false;
}

std::vector<StaticBakeTextureScroll> StaticBakeGetTextureScrolls() {
    std::vector<StaticBakeTextureScroll> out;
    out.reserve(sScrollByPath.size());
    for (const auto& kv : sScrollByPath) {
        StaticBakeTextureScroll s;
        s.path = kv.first;
        s.du = kv.second.rate[0];
        s.dv = kv.second.rate[1];
        s.bound = kv.second.image != nullptr;
        out.push_back(std::move(s));
    }
    return out;
}

void StaticBakePinClock(double seconds) {
    sClockPinned = seconds >= 0.0 ? seconds : -1.0;
    sClockNow = sClockPinned >= 0.0 ? sClockPinned : RealClockSeconds(); // from the next draw, not frame
    gStaticBakeWindGeneration++; // the wind's phase follows the clock
}

bool StaticBakeClockIsPinned() {
    return sClockPinned >= 0.0;
}

double StaticBakeClockSeconds() {
    return sClockNow;
}

bool StaticBakeSetWind(const StaticBakeWind& wind) {
    if (!std::isfinite(wind.amplitude) || !std::isfinite(wind.frequency) || !std::isfinite(wind.wavelength) ||
        !std::isfinite(wind.yawDeg) || !std::isfinite(wind.ripple) || wind.amplitude < 0.0f ||
        wind.wavelength < 0.0f) {
        return false;
    }
    sWind = wind;
    gStaticBakeWindOn = sWind.amplitude != 0.0f;
    gStaticBakeWindGeneration++;
    return true;
}

StaticBakeWind StaticBakeGetWind() {
    return sWind;
}

StaticBakeWindStats StaticBakeGetWindStats() {
    return sWindLast;
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
        Reject(e, "this rendering backend cannot bake");
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
    // The triangles that sway (#209 W1): a corner's w (its fourth float) carries its wind code.
    for (size_t t = 0; t < gfx->mBufVboNumTris; t++) {
        const float* tri = gfx->mBufVbo + t * 3 * floatsPerVertex;
        if (RecordedWeighted(tri[3]) || RecordedWeighted(tri[floatsPerVertex + 3]) ||
            RecordedWeighted(tri[2 * floatsPerVertex + 3])) {
            e->windTris++;
        }
    }

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
    // Closed before it changes, like the cull mode: what every buffered triangle was emitted under.
    memcpy(d.scroll, sRecordScroll, sizeof(d.scroll));

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
            Reject(*sRecording, "display list used an opcode the recorder does not understand");
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
    //
    // And the scroll rates (#187 A1): the replay gives a draw one offset, so a texture that scrolls at
    // another rate is another draw. Each slot's rate is its texture's registration, if the combiner
    // reads it - the same test GfxSpTri1's interpreted mirror makes (comb->usedTextures[0]), and the
    // shader patch's (the program samples TEXEL0), so the three agree on which draws move. In practice
    // a texture change has already flushed (GfxSpTri1 flushes before it imports one), so the flush here
    // closes nothing new today; what carries the split is that the rate is TRACKED here, and the flush
    // is what keeps the stamp right if that order ever changes (#187 A1's run, "The batch split,
    // planted").
    const uint8_t cull = m.cullCode > STATIC_BAKE_CULL_BACK ? (uint8_t)STATIC_BAKE_CULL_NONE : m.cullCode;
    float scroll[4] = {};
    for (int i = 0; i < STATIC_BAKE_TEXTURE_SLOTS; i++) {
        const TextureCacheNode* node = gfx->mRenderingState.mTextures[i];
        if (m.combTextures[i] && node != nullptr) {
            ScrollRateOf(node->first.texture_addr, &scroll[2 * i]);
        }
    }
    if (cull != sRecordCull || m.shadeMask != sRecordShadeMask || memcmp(scroll, sRecordScroll, sizeof(scroll)) != 0) {
        gfx->Flush(); // captures what is buffered under the *previous* mode, mask and rates
        sRecordCull = cull;
        sRecordShadeMask = m.shadeMask;
        memcpy(sRecordScroll, scroll, sizeof(scroll));
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
        Reject(*sRecording, "display list never returned");
        ReleaseGpu(*sRecording);
        sRecording = nullptr;
        sRecordingKey = nullptr;
    }
}

void StaticBakeBeginFrame() {
    sClockNow = sClockPinned >= 0.0 ? sClockPinned : RealClockSeconds();
    gStaticBakeWindGeneration++; // a new clock value: every list's wind phase moves on
    sWindLast = sWindNow;
    sWindNow = {};
}

void StaticBakeWindVectors(const float mv[4][4], float k[4], float b[4], bool fromInterpreter) {
    for (int i = 0; i < 4; i++) {
        k[i] = b[i] = 0.0f;
    }
    if (fromInterpreter) {
        sWindNow.interpVectors++;
    }
    if (!gStaticBakeWindOn) {
        return;
    }
    constexpr double kTau = 6.283185307179586;
    const double yaw = (double)sWind.yawDeg * kTau / 360.0;
    const double dir[3] = { std::sin(yaw), 0.0, std::cos(yaw) };
    const double waveNumber = sWind.wavelength > 0.0f ? kTau / (double)sWind.wavelength : 0.0;

    // The bend along the wind: the object-space vector whose image under M is amplitude x dir, so
    // b = (amplitude x dir) M^-1. A matrix that cannot be inverted (a list squashed flat) bends nothing.
    const double m00 = mv[0][0], m01 = mv[0][1], m02 = mv[0][2];
    const double m10 = mv[1][0], m11 = mv[1][1], m12 = mv[1][2];
    const double m20 = mv[2][0], m21 = mv[2][1], m22 = mv[2][2];
    const double det = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20);
    if (!(std::fabs(det) > 1e-12)) {
        return;
    }
    const double inv[3][3] = {
        { (m11 * m22 - m12 * m21) / det, (m02 * m21 - m01 * m22) / det, (m01 * m12 - m02 * m11) / det },
        { (m12 * m20 - m10 * m22) / det, (m00 * m22 - m02 * m20) / det, (m02 * m10 - m00 * m12) / det },
        { (m10 * m21 - m11 * m20) / det, (m01 * m20 - m00 * m21) / det, (m00 * m11 - m01 * m10) / det },
    };
    for (int j = 0; j < 3; j++) {
        double s = 0.0;
        for (int i = 0; i < 3; i++) {
            s += (double)sWind.amplitude * dir[i] * inv[i][j];
        }
        b[j] = (float)s;
    }
    b[3] = sWind.ripple;

    // The wave: world = p M + T (row vectors, as GfxSpVertex multiplies), so K . world = (M K) . p + K . T.
    // It travels along the wind, sin(2 pi f t - K . world): the object-space vector is -(M K), and the
    // phase takes K . T. Both reduced in double, so the phase stays small however long the session runs
    // and wherever the list stands.
    double kDotT = 0.0;
    for (int i = 0; i < 3; i++) {
        double s = 0.0;
        for (int j = 0; j < 3; j++) {
            s += (double)mv[i][j] * dir[j] * waveNumber;
        }
        k[i] = (float)-s;
        kDotT += dir[i] * waveNumber * (double)mv[3][i];
    }
    const double cycles = (double)sWind.frequency * sClockNow;
    const double phase = kTau * (cycles - std::floor(cycles)) - std::fmod(kDotT, kTau);
    k[3] = (float)phase;
}

void StaticBakeWindDirection(uint32_t directionCode, float out[3]) {
    // (code - 1) x pi / 127, in float, as the replay shader computes it.
    const float a = (float)(directionCode - 1) * (3.14159265f / (float)STATIC_BAKE_WIND_DIRECTIONS);
    out[0] = sinf(a);
    out[1] = 0.0f;
    out[2] = cosf(a);
}

void StaticBakeNoteWindVertices(uint32_t n) {
    if (gStaticBakeRecording) {
        if (sRecording != nullptr && !sRecordAborted) {
            sRecording->windVertices += n;
        }
    } else {
        sWindNow.interpVertices += n;
    }
}

void StaticBakeNoteTexture(const char* path, const void* imageData, const std::shared_ptr<Texture>& owner) {
    if (sScrollByPath.empty() || path == nullptr || imageData == nullptr) {
        return;
    }
    auto it = sScrollByPath.find(StripOtr(path));
    if (it == sScrollByPath.end()) {
        return;
    }
    ScrollReg& reg = it->second;
    if (reg.image == imageData) {
        return; // the usual case: bound on an earlier frame, to the resource still held
    }
    // First resolve, or the path now resolves to a new resource (the alt-assets toggle reloads it).
    // The old address leaves the lookup as the old resource is let go, so it can never stand for
    // whatever is allocated there next.
    Unbind(reg);
    reg.image = imageData;
    reg.owner = owner;
    sScrollByImage[imageData] = { reg.rate[0], reg.rate[1] };
    gStaticBakeScrollsBound = true;
}

bool StaticBakeTextureScrollOffset(const void* imageData, float out[2]) {
    float rate[2];
    if (!ScrollRateOf(imageData, rate)) {
        return false;
    }
    ScrollOffsetNow(rate, out);
    return true;
}

} // namespace Fast

#else // !ENABLE_STATIC_BAKE

namespace Fast {

bool gStaticBakeRecording = false;
bool gStaticBakeScrollsBound = false;
bool gStaticBakeWindOn = false;
uint32_t gStaticBakeWindGeneration = 0;

void StaticBakeSetEnabled(bool) {
}
bool StaticBakeIsEnabled() {
    return false;
}
void StaticBakeSetSortByMaterial(bool) {
}
bool StaticBakeSortsByMaterial() {
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

StaticBakeEntryInfo StaticBakeGetEntry(const void*) {
    return {};
}

std::vector<StaticBakeScrollingEntry> StaticBakeGetScrollingEntries() {
    return {};
}
std::vector<StaticBakeScrollingEntry> StaticBakeGetWindEntries() {
    return {};
}
bool StaticBakeSetWind(const StaticBakeWind&) {
    return false;
}
StaticBakeWind StaticBakeGetWind() {
    return {};
}
StaticBakeWindStats StaticBakeGetWindStats() {
    return {};
}
bool StaticBakeSetTextureScroll(const char*, float, float) {
    return false;
}
void StaticBakeClearTextureScrolls() {
}
std::vector<StaticBakeTextureScroll> StaticBakeGetTextureScrolls() {
    return {};
}
void StaticBakePinClock(double) {
}
bool StaticBakeClockIsPinned() {
    return false;
}
double StaticBakeClockSeconds() {
    return 0.0;
}

} // namespace Fast

#endif // ENABLE_STATIC_BAKE
