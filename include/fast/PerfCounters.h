#pragma once

#include <stdint.h>

// Render-side instrumentation for the Fast3D interpreter.
//
// Nothing outside libultraship can see how much of a frame the display list actually costs: a
// host's frame timer measures the whole present loop and is pinned by any fps cap, and a host's
// game-tick timer stops before submission begins. These counters answer "how long did walking
// the display list take, and how much geometry came out of it" - the numbers any geometry-caching
// or batching work has to beat, and the only way to tell per-vertex cost apart from draw-call
// count.
//
// Cost budget: one steady_clock pair per rendered frame, and three integer adds per flush. No
// allocation, no logging, no locking - instrumentation that moved the figures it measures would
// be worthless.
//
// Threading: written only from the thread that runs Interpreter::Run, and meant to be read from
// that same thread. Values are plain rather than atomic because making them atomic would cost
// more than the work being measured; a cross-thread reader could observe a torn 64-bit field.

namespace Fast {

struct PerfCounters {
    // Monotonic totals since process start.
    uint64_t frames;   // Interpreter::Run calls, i.e. rendered frames - NOT host game ticks, of
                       // which there are typically several rendered frames each.
    double interpMs;   // summed wall time spent inside Interpreter::Run
    uint64_t flushes;  // Interpreter::Flush calls, including those with nothing buffered
    uint64_t draws;    // flushes that issued geometry: one DrawTriangles, i.e. one GPU draw call
    uint64_t tris;     // triangles submitted through those draws

    // The most recently completed rendered frame on its own, published by PerfCountersEndFrame.
    // A fixed viewpoint makes these stable frame to frame, so lastTris is directly comparable to
    // a scene's known triangle count in a way a windowed mean would not be.
    double lastFrameMs;
    uint64_t lastFlushes;
    uint64_t lastDraws;
    uint64_t lastTris;
};

// The live block. The hot path increments it directly so a flush costs an add rather than a
// call; everyone else should go through PerfCountersGet().
extern PerfCounters gPerfCounters;

// Wall-clock bracket around one Interpreter::Run. Prefer PerfCountersFrameScope.
void PerfCountersBeginFrame();
void PerfCountersEndFrame();

struct PerfCountersFrameScope {
    PerfCountersFrameScope() {
        PerfCountersBeginFrame();
    }
    ~PerfCountersFrameScope() {
        PerfCountersEndFrame();
    }
};

// A copy of the counters, for a host that wants to report them. Cheap enough to call every tick.
// Totals are cumulative, so a caller wanting per-window figures should keep its own previous
// snapshot and subtract.
PerfCounters PerfCountersGet();

} // namespace Fast
