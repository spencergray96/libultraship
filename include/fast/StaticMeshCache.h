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
// and triangles, 0 unless Baked; `scrollingDraws` and `scrollingTris` the part of them whose TEXEL0
// was recorded with a scroll rate, the draws that move (#187 A1; a rate on TEXEL1 alone is stamped but
// not applied, so not counted). `windVertices` and `windTris` (#209 W1): the weighted vertices its
// recording loaded, and the triangles with at least one weighted corner, the part that sways.
// `rejectReason` is a string literal naming why the recorder refused it, nullptr unless Rejected.
enum class StaticBakeEntryState : int8_t { NotRegistered = -1, Unbaked = 0, Baked = 1, Rejected = 2 };
struct StaticBakeEntryInfo {
    StaticBakeEntryState state = StaticBakeEntryState::NotRegistered;
    uint32_t draws = 0;
    uint32_t tris = 0;
    uint32_t scrollingDraws = 0;
    uint32_t scrollingTris = 0;
    uint32_t windVertices = 0;
    uint32_t windTris = 0;
    const char* rejectReason = nullptr;
};
StaticBakeEntryInfo StaticBakeGetEntry(const void* displayList);

// A baked entry keyed as registered, for a report that cannot enumerate the lists itself (`staticbake
// props`, which joins these to its own lines by key): what the listings below return.
struct StaticBakeKeyedEntry {
    const void* key = nullptr;
    StaticBakeEntryInfo info;
};
// Every baked entry with at least one scrolling draw.
std::vector<StaticBakeKeyedEntry> StaticBakeGetScrollingEntries();

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
// archive list's G_SETTIMG_OTR_FILEPATH names it, which is the address a texture-cache key names when
// the texture is loaded whole (G_LOADBLOCK, what every archive prop emits). The registry holds that
// texture resource while bound, so its address can never be reused by another texture under it; a
// path that later resolves to a new resource (the alt-assets toggle) is re-bound to it. A path never
// drawn by an archive list never binds, and never scrolls; nor does a texture loaded from an offset
// into its image (G_LOADTILE), whose key is another address - which the recorder refuses anyway, so
// baked and interpreted still agree. One image is one path: two paths that resolved to the same image
// data would share one binding.
//
// NOT GATED: unlike the rest of this section, a scroll does not wait for StaticBakeSetEnabled(true).
// With the bake off - or on a backend that cannot bake - the interpreter still scrolls a registered
// texture, which is what keeps a baked frame and an interpreted one the same picture.
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
// Wind in the replay (sturdy-bassoon#209 W1)
//
// A vertex that carries a wind weight bends by a sine of time and world position, the scroll's trick
// applied to positions: the recording stays as it is and the replay's vertex stage moves the vertex,
// so a swaying banner or field of wheat stays in a map's baked list, with no actor and no animation
// data. With weight w (0-1) and bend vector B,
//
//     p' = p + B x w x sin(2 pi f t - K . world(p) + ripple x w)
//
// where K is the wind's direction times 2 pi over the wavelength, so the wave travels across the world
// and neighbouring placements sway out of step, and the ripple lags the hem behind the top.
//
// THE CARRIER is Vtx_t.flag, which the RSP ignores (see the STATIC_BAKE_WIND_FLAG_* constants): a
// marker bit, a bend direction and the weight. An archive's XML vertex reads it from its `Flag`
// attribute; nothing else is ever decoded (THE GUARD, below). A recording keeps the raw position and
// carries the code in position.w as (1 or 2) + STATIC_BAKE_WIND_W_SCALE x code, so no vertex attribute
// is added and vertex alpha is untouched.
//
// THE GUARD: only vertices loaded from an XML vertex resource (an archive's, G_VTX_OTR_FILEPATH) can
// bend. Vanilla vertices (binary resources, whose flag is passed through from the ROM data) and
// compiled-in ones (raw G_VTX; Fast64 writes packed normals into the flag) are never decoded, whatever
// their flag holds. And the XML reader keeps a Flag only in the marker's form, so an archive written
// before this existed, or by another tool, loads with every flag 0, as it always did.
//
// THE PARAMETERS belong to the frame, not to a recording: amplitude, frequency, wavelength, direction
// and ripple (StaticBakeWind), and the clock (StaticBakeClockSeconds, the scroll's, pinnable). Nothing
// but the weight is read at record time, so a change needs no rebake. A host sets them from its
// settings, and code can change them at any time (a scripted gust; the weather).
//
// THE BEND LINE is the vertex's (the owner's choice, sturdy-bassoon ADR 2026-10-07-animated-rs-props,
// decisions 19-22): along the frame's wind direction (line 0, wheat and trees), or along a fixed
// horizontal line of the list's own space (lines 1-127, the cloth's own normal for a flag or banner).
// In a map's list every placement is pre-transformed, so that line is written per vertex, turned with
// its placement. Either way the bend moves `amplitude` world units at weight 1 - under a list matrix
// that only turns, moves and scales evenly, as a map list's identity and an actor's matrix do.
//
// NOT GATED: like the scroll, wind does not wait for StaticBakeSetEnabled(true). With the bake off, or
// on a backend that cannot bake, the interpreter bends the same vertices by the same formula, so a
// baked frame and an interpreted one are the same picture.
// ---------------------------------------------------------------------------

// Vtx_t.flag, as an archive's XML `Flag` attribute writes it:
//   bit 15      STATIC_BAKE_WIND_FLAG_MARK: the vertex carries wind
//   bits 8-14   the bend line: 0 = along the wind; d = 1-127 = the horizontal line at angle
//               (d - 1) x 180 / 127 degrees from the list's +z towards +x (OoT's yaw: x = sin, z = cos).
//               A line, not an arrow: a bend is a sine, so the opposite direction is the same sway
//               half a period apart.
//   bits 0-7    the weight q, 1-255 (w = q / 255): 0 at what holds the vertex still, 255 at the hem
// A flag without the marker, or with q = 0, is no wind at all.
constexpr uint16_t STATIC_BAKE_WIND_FLAG_MARK = 0x8000;
constexpr int STATIC_BAKE_WIND_FLAG_LINE_SHIFT = 8;
constexpr uint16_t STATIC_BAKE_WIND_FLAG_WEIGHT_MASK = 0x00FF;
constexpr int STATIC_BAKE_WIND_LINES = 127; // lines 1-127 split 180 degrees

// The code a flag carries into a recording, the flag without its marker: q + 256 x line, 1-32767; 0 when
// it carries no wind. Its weight is code & STATIC_BAKE_WIND_FLAG_WEIGHT_MASK, its line
// code >> STATIC_BAKE_WIND_FLAG_LINE_SHIFT.
constexpr uint16_t StaticBakeWindCode(uint16_t flag) {
    return ((flag & STATIC_BAKE_WIND_FLAG_MARK) != 0 && (flag & STATIC_BAKE_WIND_FLAG_WEIGHT_MASK) != 0)
               ? (uint16_t)(flag & ~STATIC_BAKE_WIND_FLAG_MARK)
               : (uint16_t)0;
}

// The frame's wind. The defaults are the owner's pick after sturdy-bassoon#208: 6 units of swing at
// the hem (RS's own size of motion; 9 and more show the gaps between a banner's cloth strips), RS's
// 0.94 s loop, a 400-unit wave, and a ripple of 1.5 radians down the cloth.
struct StaticBakeWind {
    float amplitude = 6.0f;    // world units of swing at weight 1; 0 bends nothing
    float frequency = 1.0638f; // Hz
    float wavelength = 400.0f; // world units; 0 = every placement in step
    float yawDeg = 0.0f;       // where it blows to, as OoT's yaw (degrees): x = sin, z = cos
    float ripple = 1.5f;       // radians of phase the hem (weight 1) lags behind weight 0

    bool operator==(const StaticBakeWind& o) const {
        return amplitude == o.amplitude && frequency == o.frequency && wavelength == o.wavelength &&
               yawDeg == o.yawDeg && ripple == o.ripple;
    }
};
// Set the frame's wind, from the next vertex drawn. Refused (false, nothing changed) when a value is
// not finite, or the amplitude or wavelength is negative. Safe at any time on the game thread: a host's
// settings at startup, a console, a scripted gust.
bool StaticBakeSetWind(const StaticBakeWind& wind);
StaticBakeWind StaticBakeGetWind();

// Every baked entry that recorded a weighted vertex (windTris != 0: the same test the replay makes).
std::vector<StaticBakeKeyedEntry> StaticBakeGetWindEntries();

// What the wind did in the last whole frame drawn, for a host's report and for checks:
//   replayEntries   baked list entries replayed with wind vectors (a list holding weighted vertices
//                   while the amplitude is non-zero)
//   interpVertices  weighted vertices the interpreter bent (none of them baked)
//   interpVectors   times the interpreter worked out the wind vectors: once per list modelview it met
//                   with a weighted vertex in it, not once per vertex load
struct StaticBakeWindStats {
    uint32_t replayEntries = 0;
    uint32_t interpVertices = 0;
    uint32_t interpVectors = 0;
};
StaticBakeWindStats StaticBakeGetWindStats();

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

// Wind, the interpreter's side (#209 W1).
//
// True while the amplitude is non-zero: the gate on the interpreter's bend. A record pass notes the
// weights whatever it is, because the amplitude is read at replay.
extern bool gStaticBakeWindOn;
// Bumped whenever the vectors below would come out different for the same modelview: each frame (the
// clock) and each StaticBakeSetWind. The interpreter keeps the vectors it last worked out, with the
// generation and the modelview they were for, and works them out again only when either changes.
extern uint32_t gStaticBakeWindGeneration;
// The two vectors the replay shader and the interpreter bend by, for a list drawn under modelview
// `mv` (row vectors, world = p M + T, as GfxSpVertex multiplies) at this frame's clock:
//   k: xyz the wave vector in the list's own space, -(M K); w the phase now, 2 pi f t - K . T,
//      reduced into (-2 pi, 2 pi)
//   b: xyz the bend at weight 1 along the wind, in the list's own space, (amplitude x dir) M^-1, so it
//      moves `amplitude` world units; w the ripple. A vertex with its own direction bends by |b.xyz|
//      along that direction instead: the same world length under a turned and uniformly scaled matrix.
// All zero while the amplitude is 0, or under a matrix that cannot be inverted. `fromInterpreter`
// counts it in StaticBakeWindStats::interpVectors.
void StaticBakeWindVectors(const float mv[4][4], float k[4], float b[4], bool fromInterpreter);
// The unit bend direction of a vertex's line (1-127), as the replay shader computes it:
// (sin a, 0, cos a) with a = (line - 1) x pi / 127.
void StaticBakeWindLine(uint32_t line, float out[3]);
// From GfxSpVertex: in a record pass, `n` weighted vertices were loaded into the list being recorded;
// otherwise, `n` weighted vertices were bent on the CPU.
void StaticBakeNoteWindVertices(uint32_t n);

} // namespace Fast
