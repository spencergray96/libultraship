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
//
// Scrolling textures (sturdy-bassoon#187 A1): a texture can slide across a baked surface without a
// re-record, the way lighting changes without one. The recording stays as it is; the replay's vertex
// stage adds frac(rate x clock) to TEXEL0's coordinate, and the interpreter adds the same offset to
// the coordinates it writes, so a frame drawn without the bake (the bake off, a frame that falls back,
// a backend that cannot bake) shows the same picture. See "Texture scroll" below.

#include <memory>
#include <string>
#include <vector>

namespace Fast {

class Interpreter;
class Texture;
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
// and triangles, 0 unless Baked; `scrollingDraws` and `scrollingTris` the part of them recorded with a
// scroll rate (#187 A1). `rejectReason` is a string literal naming why the recorder refused it,
// nullptr unless Rejected.
enum class StaticBakeEntryState : int8_t { NotRegistered = -1, Unbaked = 0, Baked = 1, Rejected = 2 };
struct StaticBakeEntryInfo {
    StaticBakeEntryState state = StaticBakeEntryState::NotRegistered;
    uint32_t draws = 0;
    uint32_t tris = 0;
    uint32_t scrollingDraws = 0;
    uint32_t scrollingTris = 0;
    const char* rejectReason = nullptr;
};
StaticBakeEntryInfo StaticBakeGetEntry(const void* displayList);

// Every baked entry with at least one scrolling draw, keyed as registered, for a report that cannot
// enumerate the lists itself (`staticbake props`, which joins these to its own lines by key).
struct StaticBakeScrollingEntry {
    const void* key = nullptr;
    StaticBakeEntryInfo info;
};
std::vector<StaticBakeScrollingEntry> StaticBakeGetScrollingEntries();

// ---------------------------------------------------------------------------
// Texture scroll (sturdy-bassoon#187 A1)
//
// A texture registered by archive path scrolls at (du, dv) texture widths and heights a second
// wherever it is drawn as TEXEL0, baked or interpreted: RS's texture animation, which belongs to the
// texture, not to the model that uses it. Texture rectangles never scroll.
//
// LIFETIME: the registry belongs to the process, not to a scene, a bake group or a recording. A
// registration survives StaticBakeReset (a scene or bake-group change, `staticbake reset`),
// StaticBakeInvalidateAll (`staticbake rebake`, the alt-assets toggle, a filter change) and every
// texture-cache clear, and lasts until it is changed, removed or cleared. Nothing else ever drops it.
//
// How a path reaches the pixels: the path is bound to the image data it resolves to each time an
// archive list's G_SETTIMG_OTR_FILEPATH names it, which is the address a texture-cache key names. The
// registry holds that texture resource while bound, so its address can never be reused by another
// texture under it; a path that later resolves to a new resource (the alt-assets toggle) is re-bound
// to it. A path never drawn by an archive list never binds, and never scrolls.
//
// RECORD TIME: a baked draw carries the rate the registry gave its texture WHEN IT WAS RECORDED. So
// register before the list first draws; a change to a path an existing bake already recorded shows
// on baked draws only after StaticBakeInvalidateAll (`staticbake rebake`), while interpreted draws
// follow at once. The recorder closes a batch where TEXEL0's rate changes, so a scrolling material is
// a draw of its own; a list with no scrolling texture records exactly as before.
// ---------------------------------------------------------------------------

// Register, change or remove one texture's scroll: `path` as an archive list names it (with or without
// "__OTR__"), (du, dv) in texture widths and heights a second; (0, 0) removes it. IDEMPOTENT: the same
// rate again changes nothing and costs a lookup, so a host can call it every time it loads a list that
// draws the texture (#187 A2 does, at list load, before the list's first draw). Returns true when the
// registry changed: a rate added, changed or removed - the case where a bake recorded under the old
// rate needs StaticBakeInvalidateAll. Rates must be finite; a non-finite rate is refused (false).
bool StaticBakeSetTextureScroll(const char* path, float du, float dv);
// Remove every registration (the console's `staticbake scroll clear`). Same record-time rule.
void StaticBakeClearTextureScrolls();
// The registrations, in no particular order. `bound`: the path has resolved to image data since it
// was registered, so draws of it scroll.
struct StaticBakeTextureScroll {
    std::string path;
    float du = 0.0f;
    float dv = 0.0f;
    bool bound = false;
};
std::vector<StaticBakeTextureScroll> StaticBakeGetTextureScrolls();

// The clock every moving thing in the replay reads (the scroll; wind, #209): seconds since the process
// started, sampled once at the start of each rendered frame so every draw of a frame - baked or
// interpreted, interpolated frames included - sees one value. Pin it for same-picture comparisons:
// StaticBakePinClock(t) holds it at t seconds (t >= 0) until StaticBakePinClock with t < 0 lets it
// run again, from real time.
void StaticBakePinClock(double seconds);
bool StaticBakeClockIsPinned();
double StaticBakeClockSeconds();

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

// Texture scroll, the interpreter's side (#187 A1).
//
// True while at least one registered path is bound to image data: the interpreter's per-triangle gate,
// read on its hot path like gStaticBakeRecording, so a session that registers nothing pays one test of
// a bool per triangle.
extern bool gStaticBakeScrollsBound;
// From the start of Interpreter::Run: samples the clock for this frame (StaticBakeClockSeconds).
void StaticBakeBeginFrame();
// From the G_SETTIMG_OTR_FILEPATH handler: `path` resolved to `imageData`, owned by `owner` (the
// texture resource). Binds a registered path, holding `owner` so the address stays its own, and moves
// the binding (letting the old resource go) when the path now resolves somewhere else. No-op for an
// unregistered path, and while nothing is registered.
void StaticBakeNoteTexture(const char* path, const void* imageData, const std::shared_ptr<Texture>& owner);
// The offset to add now, in texture widths and heights, to the coordinates of a triangle whose TEXEL0
// is the image data at `imageData`; false (and `out` untouched) when that texture does not scroll.
bool StaticBakeTextureScrollOffset(const void* imageData, float out[2]);

} // namespace Fast
