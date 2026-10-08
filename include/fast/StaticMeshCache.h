#pragma once

#include <stdint.h>
#include <stddef.h>

// Static-geometry bake - record a room's display list once, replay it from a persistent GPU
// buffer with the camera, fog and lights as uniforms (sturdy-bassoon#40 Stage 1; lighting moved
// into the replay, and textured geometry admitted, by sturdy-bassoon#142).
//
// Why this exists: the Fast3D interpreter re-walks every room display list once per *rendered*
// frame, transforming and lighting every vertex on the CPU and streaming the result to a dynamic
// vertex buffer in 256-triangle batches. Measurement (docs/test-runs/2026-08-29-render-counters)
// put ~63% of Release interpreter time in that walk, over all resident geometry, whether or not
// any of it survives culling. Nothing about a static room changes between frames except the
// camera, the fog band and the light state.
//
// The trick that makes it cheap to build: the emitted vertex payload is normally clip-space,
// because GfxSpVertex multiplies by mRsp->MP_matrix. Room display lists run under an identity
// modelview, so at their entry MP_matrix *is* the camera. Force it to (near-)identity for one
// pass and the interpreter's own emission becomes an object-space payload - already interleaved
// in exactly the layout the material's generated shader expects. Layout mismatch, the riskiest
// failure mode, is designed out rather than debugged.
//
// Replay binds the same shader with a transform-enabled vertex stage, supplies the current
// camera/fog/lights as uniforms, and skips the walk. Lighting stays per vertex, as the interpreter
// does it: a lit vertex is recorded with its normal where its lit colour would have gone, and the
// vertex stage runs GfxSpVertex's directional-light sum on it. So a light change - time of day,
// the Sun's Song - costs nothing, where a recording of lit colours would have to be redone.
//
// Textures: the record pass lets the texture-load commands run, so the interpreter imports and
// binds each texture exactly as it would live, and the UVs land in the payload already normalised
// by the tile. Each baked draw then holds its OWN reference to the GPU texture and sampler it was
// recorded with (GfxRenderingAPI::HoldStaticTexture) - not a texture-cache id, which an eviction can
// hand to another texture, and not a cache pin, which every ocarina textbox's full cache clear
// would have to be taught to skip. So nothing the cache does reaches a bake; only a real content
// change (the alt-assets toggle, a texture-filter change) invalidates one.
//
// Safety model: a display list is only ever considered if the host explicitly registered it
// (compiled-in custom scenes only - vanilla display lists are never registered, so they can never
// take this path), and any command or material feature the recorder does not understand aborts
// that display list's bake *permanently* and falls back to interpretation. Safe by construction,
// not by analysis.

namespace Fast {

class Interpreter;
struct StaticBakeUniforms;

// ---------------------------------------------------------------------------
// Host-facing API
//
// Everything here is inert until StaticBakeSetEnabled(true): with the gate off, every registered
// display list is interpreted, and the interpreter's hooks early-out on the gate before looking
// anything up.
// ---------------------------------------------------------------------------

// The runtime switch. Off: every display list is interpreted, and registrations and existing bakes
// are kept, so switching back on replays them without re-recording. That is what makes a baked /
// interpreted A/B possible within one session, at one camera. Safe to flip between frames.
void StaticBakeSetEnabled(bool enabled);
bool StaticBakeIsEnabled();

// Order each recording by material before it is uploaded (sturdy-bassoon#158), so one material is
// one draw per list instead of one per stretch of the list that uses it. On by default. Opaque,
// depth-tested batches regroup; decals and alpha-blended batches go after them in their original
// order; a batch with no depth test, or opaque without depth writes, stays where it is. A change
// sends every bake back to be recorded (StaticBakeInvalidateAll). Exists to compare the two orders
// in one session, and as the way back if some content turns out to depend on list order (two
// exactly coplanar opaque surfaces are the one known case).
void StaticBakeSetSortByMaterial(bool sort);
bool StaticBakeSortsByMaterial();

// Offer one display list for baking. Keyed by pointer, which is only sound for display lists
// whose address is a stable C symbol for the life of the process - i.e. compiled-in scene data.
// Registering the same pointer twice is a no-op, so this is safe to call on every room init.
// Registration does not depend on the gate, so a room loaded while the bake is off still bakes
// when it is switched on.
void StaticBakeRegister(const void* displayList);

// Drop every registration and release every GPU buffer. Call on a scene change: the next scene's
// display lists are different symbols, and nothing else would ever free the old buffers.
void StaticBakeReset();

// Send every baked entry back to UNBAKED so the next frame re-records it, releasing its buffer and
// held textures. Recording costs one interpreted pass - what every frame costs today - so this is
// cheap enough to be naive. Call it when what a bake holds is no longer what the interpreter would
// draw: a shader-cache clear, the alt-assets toggle, a texture-filter change. NOT on a plain
// texture-cache clear (an ocarina textbox does one every time): bakes hold their own textures.
void StaticBakeInvalidateAll();

// How many display lists are registered, and how many of those are currently baked / rejected.
// For host-side logging only.
void StaticBakeGetStats(uint32_t* registered, uint32_t* baked, uint32_t* rejected);

// One registered display list, for a host that reports on the lists it offered by name
// (sturdy-bassoon#171: the archive prop lists). `draws` and `tris` are the baked entry's replay draws
// and triangles, 0 unless Baked. `rejectReason` is a string literal naming why the recorder refused
// it, nullptr unless Rejected.
enum class StaticBakeEntryState : int8_t { NotRegistered = -1, Unbaked = 0, Baked = 1, Rejected = 2 };
struct StaticBakeEntryInfo {
    StaticBakeEntryState state = StaticBakeEntryState::NotRegistered;
    uint32_t draws = 0;
    uint32_t tris = 0;
    const char* rejectReason = nullptr;
};
StaticBakeEntryInfo StaticBakeGetEntry(const void* displayList);

// THROWAWAY (sturdy-bassoon#187 route A): a texture that scrolls, as RS's do. A texture registered
// by archive path scrolls at (du, dv) texture widths a second wherever it is drawn: a baked draw
// carries the rate and the replay adds the offset in its vertex stage; an interpreted draw adds it
// to the UVs it writes. Rates of 0 remove it. Registering re-records nothing: the rate is read at
// record time, so register before the list first draws.
void StaticBakeSetTextureScroll(const char* path, float du, float dv);
void StaticBakeClearTextureScrolls();
// Pin the scroll clock at t seconds (t >= 0, for same-picture comparisons), or let it run (t < 0).
void StaticBakeSetScrollClock(float t);
// The scroll rate of the texture whose image data is at addr; false when it does not scroll.
bool StaticBakeScrollRate(const void* texAddr, float rate[2]);
// The offset to add now for a rate: frac(rate * clock).
void StaticBakeScrollOffset(const float rate[2], float out[2]);
// From the G_SETTIMG_OTR_FILEPATH handler: binds a registered path to the image data it resolved to.
void StaticBakeNoteTexturePath(const char* path, const void* imageData);
// Record passes and their total wall time since the last call, for the host's report.
void StaticBakeTakeRecordTime(uint32_t* passes, double* ms);

// THROWAWAY (sturdy-bassoon#208): wind in the replay. A vertex whose F3DVtx_t.flag is
// STATIC_BAKE_WIND_MARK | q (q 1..255) bends by weight w = q/255:
//   p' = p + bend * w * sin(2 pi f t - K . world(p) + ripple * w)
// where K is the wind direction times 2 pi / wavelength. The bend is along the wind (axisMode 0, for
// trees and wheat) or along a fixed axis of the display list's own space (axisMode 1, for a flag
// whose cloth faces one way in its kind). amplitude is in world units. A recording keeps the raw
// position and carries q in position.w; the replay bends in its vertex stage, and the interpreter
// bends the same way in GfxSpVertex. Nothing is read at record time but q, so no rebake is needed
// when these change. amplitude 0 turns it off.
constexpr uint16_t STATIC_BAKE_WIND_MARK = 0x5700;
struct StaticBakeWindParams {
    float amplitude = 0.0f;  // world units at weight 1
    float frequency = 1.0f;  // Hz
    float wavelength = 0.0f; // world units; 0 = every vertex in phase
    float yawDeg = 0.0f;     // world wind direction: x = sin(yaw), z = cos(yaw), as OoT's yaw
    float ripple = 0.0f;     // radians of phase lag from weight 0 to weight 1 (a wave down the cloth)
    int axisMode = 0;        // 0 = bend along the wind; 1 = along localAxis
    float localAxis[3] = { 0.0f, 0.0f, 1.0f };
};
void StaticBakeSetWind(const StaticBakeWindParams& p);
StaticBakeWindParams StaticBakeGetWind();
// The per-draw wind vectors for a display list drawn under modelview mv (row-vector convention:
// world = p * mv). k.xyz: the wave vector in the list's own space, k.w: the phase now (mod 2 pi);
// b.xyz: the bend at weight 1 in the list's own space, b.w: ripple. False (and zeros) when wind is off.
bool StaticBakeWindVectors(const float mv[4][4], float k[4], float b[4]);
// The bake's clock, in seconds: the pinned time (StaticBakeSetScrollClock), or the running one.
double StaticBakeClockSeconds();

// THROWAWAY (sturdy-bassoon#208): a flipbook picked at render time. `head` is a display-list pointer
// unique to one copy (it is never walked); a G_DL to it is redirected to poses[k], with
// k = floor(clock * posesPerSecond + phase) mod n, on every rendered frame - including the frames
// frame interpolation makes between game ticks, which a pose chosen in an actor's Draw cannot reach.
// The poses are ordinary lists: bake-registered ones replay, others are interpreted.
void StaticBakeRegisterFlip(const void* head, const void* const* poses, int n, float posesPerSecond, float phase);
void StaticBakeUnregisterFlip(const void* head);
// The pose a G_DL to displayList draws now: itself unless it is a registered flip head.
const void* StaticBakeResolveFlip(const void* displayList);
// Pose changes the resolver has handed out since the last call (counted per head), for rate checks.
uint32_t StaticBakeTakeFlipSwitches();

// ---------------------------------------------------------------------------
// Interpreter-facing API (libultraship internal)
// ---------------------------------------------------------------------------

// True only while a bake is being recorded. Read on the interpreter's hot path, so it is a plain
// global rather than anything that needs a lock or a lookup: there is exactly one Interpreter,
// and it is only ever written between commands on that same thread.
extern bool gStaticBakeRecording;

// Called at every G_DL that would call into a display list. Returns true when the list was
// replayed from its baked buffer and the caller must skip the walk entirely.
bool StaticBakeIntercept(Interpreter* gfx, void* displayList);

// Called from Interpreter::Flush() instead of drawing, while recording.
void StaticBakeCaptureFlush(Interpreter* gfx);

// Called from the G_ENDDL handler after the return, while recording: ends the bake when the
// display list that opened it has returned.
void StaticBakeOnEndDl(Interpreter* gfx);

// Opcode whitelist, plus a guard on the operand: anything not on the list, or a display list,
// vertex or texture reached through a segment, aborts the bake in progress. w1 is the command's
// second word.
void StaticBakeOnOpcode(Interpreter* gfx, int8_t opcode, uintptr_t w1);

// What GfxSpTri1 knows about the material a triangle is drawn with, for the recorder's
// material-level whitelist.
struct StaticBakeMaterial {
    bool useFog;
    bool useBlendColor;
    bool useGrayscale;
    // Texture slots the combiner reads, i.e. the ones the interpreter imported for this triangle.
    bool combTextures[2];
    // An HD mask or blend texture rides behind a slot (SHADER_FIRST_MASK_TEXTURE on).
    bool maskedOrBlended;
    // A StaticBakeCull value - the CPU cull decision the recording is skipping, which the replay
    // hands to the rasterizer instead.
    uint8_t cullCode;
    // Bit j set when colour input j is SHADE, the input a lit vertex records its normal in.
    uint8_t shadeMask;
};

// Material-level whitelist, from GfxSpTri1: the recorder can only reproduce non-grayscale
// materials whose fog (if any) it can recompute in the vertex shader, and whose textures really
// were imported and uploaded.
void StaticBakeNoteMaterial(Interpreter* gfx, const StaticBakeMaterial& material);

// Refuse the bake in progress, for something met outside the opcode and material checks - from
// GfxSpVertex, a lit vertex using lighting the replay shader does not model; from ImportTexture, a
// texture that cannot be held. The display list finishes its walk and is then interpreted for
// good. No-op when nothing is recording.
void StaticBakeAbort(Interpreter* gfx, const char* reason);

// Safety net: a display list that never returns would otherwise leave recording armed across
// frames. Called once at the end of Interpreter::Run.
void StaticBakeEndFrame(Interpreter* gfx);

} // namespace Fast
