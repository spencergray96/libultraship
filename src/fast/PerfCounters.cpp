#include "fast/PerfCounters.h"

#include <chrono>

namespace Fast {

PerfCounters gPerfCounters = {};

namespace {
// The totals as they stood when the frame in flight opened. Keeping them here rather than in the
// public block means the hot path only ever touches the totals, and the frame's own figures fall
// out as a subtraction once per frame.
//
// On MSVC and libstdc++ alike steady_clock is the platform's high-resolution monotonic timer
// (QueryPerformanceCounter on Windows), which is what an optimized frame of a few milliseconds
// needs - a millisecond-resolution clock would quantize it to nothing.
std::chrono::steady_clock::time_point sFrameStart;
uint64_t sFrameStartFlushes = 0;
uint64_t sFrameStartDraws = 0;
uint64_t sFrameStartTris = 0;
uint64_t sFrameStartDrawsBaked = 0;
uint64_t sFrameStartTrisBaked = 0;
bool sFrameOpen = false;
} // namespace

void PerfCountersBeginFrame() {
    sFrameStartFlushes = gPerfCounters.flushes;
    sFrameStartDraws = gPerfCounters.draws;
    sFrameStartTris = gPerfCounters.tris;
    sFrameStartDrawsBaked = gPerfCounters.drawsBaked;
    sFrameStartTrisBaked = gPerfCounters.trisBaked;
    sFrameOpen = true;
    sFrameStart = std::chrono::steady_clock::now();
}

void PerfCountersEndFrame() {
    if (!sFrameOpen) {
        return;
    }
    sFrameOpen = false;

    const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - sFrameStart;
    const double ms = elapsed.count();

    gPerfCounters.lastFrameMs = ms;
    gPerfCounters.lastFlushes = gPerfCounters.flushes - sFrameStartFlushes;
    gPerfCounters.lastDraws = gPerfCounters.draws - sFrameStartDraws;
    gPerfCounters.lastTris = gPerfCounters.tris - sFrameStartTris;
    gPerfCounters.lastDrawsBaked = gPerfCounters.drawsBaked - sFrameStartDrawsBaked;
    gPerfCounters.lastTrisBaked = gPerfCounters.trisBaked - sFrameStartTrisBaked;

    gPerfCounters.interpMs += ms;
    gPerfCounters.frames++;
}

PerfCounters PerfCountersGet() {
    return gPerfCounters;
}

} // namespace Fast
