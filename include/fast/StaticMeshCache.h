#pragma once

#include <stdint.h>
#include <stddef.h>

// Static-geometry bake - record a room's display list once, replay it from a persistent GPU
// buffer with the camera as a uniform (sturdy-bassoon#40 Stage 1, prototype).
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
// camera/fog as uniforms, and skips the walk.
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
// Everything here is inert until StaticBakeSetEnabled(true). With the gate off there are no
// registrations, so the interpreter hooks all early-out on an empty registry.
// ---------------------------------------------------------------------------

void StaticBakeSetEnabled(bool enabled);
bool StaticBakeIsEnabled();

// Offer one display list for baking. Keyed by pointer, which is only sound for display lists
// whose address is a stable C symbol for the life of the process - i.e. compiled-in scene data.
// Registering the same pointer twice is a no-op, so this is safe to call on every room init.
void StaticBakeRegister(const void* displayList);

// Drop every registration and release every GPU buffer. Call on a scene change: the next scene's
// display lists are different symbols, and nothing else would ever free the old buffers.
void StaticBakeReset();

// Send every baked entry back to UNBAKED so the next frame re-records it. Recording costs one
// interpreted pass - what every frame costs today - so this is cheap enough to be naive.
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

// Opcode whitelist. Anything not on it aborts the bake in progress.
void StaticBakeOnOpcode(Interpreter* gfx, int8_t opcode);

// Material-level whitelist, from GfxSpTri1: the recorder can only reproduce untextured,
// non-grayscale materials whose fog (if any) it is able to recompute in the vertex shader.
// cullCode is a StaticBakeCull value - the CPU cull decision the recording is skipping, which the
// replay hands to the rasterizer instead.
void StaticBakeNoteMaterial(Interpreter* gfx, bool useFog, bool useBlendColor, bool useGrayscale,
                            bool usedTexture0, bool usedTexture1, uint8_t cullCode);

// Safety net: a display list that never returns would otherwise leave recording armed across
// frames. Called once at the end of Interpreter::Run.
void StaticBakeEndFrame(Interpreter* gfx);

} // namespace Fast
