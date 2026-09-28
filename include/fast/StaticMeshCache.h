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
