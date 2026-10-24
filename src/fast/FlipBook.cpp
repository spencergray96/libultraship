#include "fast/FlipBook.h"

#include <cmath>
#include <unordered_map>

#include "fast/StaticMeshCache.h"

// Flip heads (sturdy-bassoon#204 C4b): fast/FlipBook.h has the design. Deliberately outside every static-bake
// guard: a flip picks its pose whether or not the bake is on, or the backend can bake.

namespace Fast {

bool gFlipBookTracing = false;

namespace {

struct Flip {
    FlipBookDesc desc;
    uint16_t serial = 0;
    int32_t last = -1;
    uint64_t resolves = 0;
    uint64_t changes = 0;
};

std::unordered_map<const void*, Flip> sFlips;
uint16_t sNextSerial = 1;
int32_t sForced = FLIPBOOK_FORCE_CLOCK;
FlipBookStats sStats;

struct Trace {
    const void* handle = nullptr;
    uint32_t left = 0; // rendered frames still to trace
    uint64_t lastFrame = UINT64_MAX;
    std::vector<FlipBookTraceRow> rows;
} sTrace;

bool Valid(const FlipBookDesc& d) {
    if (d.poseCount < 1 || d.poseCount > FLIPBOOK_POSES_MAX) {
        return false;
    }
    for (int32_t k = 0; k < d.poseCount; k++) {
        if (d.cycles[k] < 1) {
            return false;
        }
    }
    return std::isfinite(d.origin);
}

bool Loops(const FlipBookDesc& d) {
    return d.frameStep >= 1 && d.frameStep <= d.poseCount;
}

void TraceNote(int32_t pose, bool head) {
    if (sTrace.left == 0 || sTrace.lastFrame == sStats.frames) {
        return; // one row a rendered frame
    }
    sTrace.lastFrame = sStats.frames;
    sTrace.rows.push_back({ sStats.frames, StaticBakeClockSeconds(), pose, head });
}

} // namespace

uint16_t FlipBookRegister(const void* handle, const FlipBookDesc& desc) {
    if (handle == nullptr || !Valid(desc)) {
        return 0;
    }
    Flip& f = sFlips[handle];
    f = Flip{};
    f.desc = desc;
    f.serial = sNextSerial;
    sNextSerial = sNextSerial == UINT16_MAX ? 1 : (uint16_t)(sNextSerial + 1);
    return f.serial;
}

void FlipBookSetOrigin(const void* handle, double origin) {
    auto it = sFlips.find(handle);
    if (it != sFlips.end() && std::isfinite(origin)) {
        it->second.desc.origin = origin;
    }
}

void FlipBookUnregister(const void* handle) {
    sFlips.erase(handle);
    if (sTrace.handle == handle) {
        sTrace.left = 0;
        gFlipBookTracing = false;
    }
}

void FlipBookClear() {
    sFlips.clear();
    sTrace.left = 0;
    gFlipBookTracing = false;
}

// RS's DynamicObject, as Rs_AnimProp's RsAnimProp_PoseAt (#204 C4a) - `animprop selftest` holds the two equal.
int32_t FlipBookPoseAt(const FlipBookDesc& d, int64_t cycle) {
    int64_t total = 0;
    int64_t loop = 0;
    int32_t from = 0;

    if (cycle < 0) {
        cycle = 0;
    }
    for (int32_t k = 0; k < d.poseCount; k++) {
        total += d.cycles[k];
    }
    if (cycle >= total) {
        if (!Loops(d)) {
            return d.poseCount;
        }
        from = d.poseCount - d.frameStep;
        for (int32_t k = from; k < d.poseCount; k++) {
            loop += d.cycles[k];
        }
        cycle = (cycle - total) % loop;
    }
    for (int32_t k = from; k < d.poseCount; k++) {
        if (cycle < d.cycles[k]) {
            return k;
        }
        cycle -= d.cycles[k];
    }
    return d.poseCount - 1; // not reached: the cycles add up
}

int64_t FlipBookCycleAt(const FlipBookDesc& d, double clockSeconds) {
    return (int64_t)std::floor((clockSeconds - d.origin) * FLIPBOOK_CYCLES_PER_SECOND) + d.cycleOffset;
}

int32_t FlipBookPoseAtClock(const void* handle, double clockSeconds) {
    auto it = sFlips.find(handle);
    if (it == sFlips.end()) {
        return -1;
    }
    return FlipBookPoseAt(it->second.desc, FlipBookCycleAt(it->second.desc, clockSeconds));
}

void FlipBookForce(int32_t pose) {
    sForced = pose < 0 && pose != FLIPBOOK_FORCE_REST ? FLIPBOOK_FORCE_CLOCK : pose;
}

FlipBookStats FlipBookGetStats() {
    FlipBookStats s = sStats;
    s.registered = (uint32_t)sFlips.size();
    return s;
}

bool FlipBookGetEntry(const void* handle, FlipBookEntryInfo* out) {
    auto it = sFlips.find(handle);
    if (it == sFlips.end()) {
        return false;
    }
    out->last = it->second.last;
    out->resolves = it->second.resolves;
    out->changes = it->second.changes;
    return true;
}

bool FlipBookTraceStart(const void* handle, uint32_t frames) {
    sTrace = Trace{};
    if (sFlips.find(handle) == sFlips.end() || frames == 0) {
        gFlipBookTracing = false;
        return false;
    }
    sTrace.handle = handle;
    sTrace.left = frames;
    sTrace.rows.reserve(frames);
    gFlipBookTracing = true;
    return true;
}

const std::vector<FlipBookTraceRow>& FlipBookTraceRows(bool* running) {
    if (running != nullptr) {
        *running = sTrace.left > 0;
    }
    return sTrace.rows;
}

void FlipBookBeginFrame() {
    // The frame that just ended counts against the trace when the flip was drawn in it.
    if (sTrace.left > 0 && sTrace.lastFrame == sStats.frames) {
        if (--sTrace.left == 0) {
            gFlipBookTracing = false;
        }
    }
    sStats.frames++;
}

const void* FlipBookResolve(const void* handle, uint16_t serial) {
    auto it = sFlips.find(handle);
    if (it == sFlips.end() || it->second.serial != serial) {
        sStats.stale++;
        return nullptr;
    }
    Flip& f = it->second;
    int32_t pose;
    if (sForced == FLIPBOOK_FORCE_REST) {
        pose = f.desc.poseCount;
    } else if (sForced >= 0) {
        pose = sForced < f.desc.poseCount ? sForced : f.desc.poseCount - 1;
    } else {
        pose = FlipBookPoseAt(f.desc, FlipBookCycleAt(f.desc, StaticBakeClockSeconds()));
    }
    if (f.last >= 0 && pose != f.last) {
        f.changes++;
        sStats.changes++;
    }
    f.last = pose;
    f.resolves++;
    sStats.resolves++;
#if defined(ENABLE_STATIC_BAKE) && !defined(NDEBUG)
    // The Debug lifetime check (fast/FlipBook.h, LIFETIME).
    if (f.desc.poses[pose] != nullptr &&
        StaticBakeGetEntry(f.desc.poses[pose]).state == StaticBakeEntryState::NotRegistered) {
        sStats.unheld++;
    }
#endif
    if (gFlipBookTracing && handle == sTrace.handle) {
        TraceNote(pose, true);
    }
    return f.desc.poses[pose];
}

void FlipBookTraceCall(const void* displayList) {
    auto it = sFlips.find(sTrace.handle);
    if (displayList == nullptr || it == sFlips.end()) {
        return;
    }
    const FlipBookDesc& d = it->second.desc;
    for (int32_t k = 0; k <= d.poseCount; k++) {
        if (d.poses[k] == displayList) {
            TraceNote(k, false);
            return;
        }
    }
}

} // namespace Fast
