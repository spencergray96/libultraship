#pragma once

#include <stdint.h>
#include <vector>

// Flip heads: a flipbook's pose picked at render time (sturdy-bassoon#204 C4b; the prototype was #208's).
//
// Why this exists: SoH builds one display list a 20 Hz game tick and replays it for every rendered frame,
// interpolated ones included. A pose an actor picks in its Draw therefore changes at most once a tick: a
// RuneScape flipbook whose poses last 100 and 80 ms holds 50 or 100 instead. A flip head is one command,
// G_FLIPBOOK (gbi.h gSPFlipBook), that the host writes where it would have written a G_DL to the pose: the
// interpreter picks the pose from the clock on every rendered frame and calls it exactly as a pushed G_DL
// would (the bake's intercept, the mips scope, the call). So a baked pose list replays, an unbaked one is
// interpreted, and nothing else in the frame pays anything: ordinary G_DLs never look a flip up.
//
// THE COMMAND: w0 = G_FLIPBOOK << 24 | serial, w1 = handle. The handle is any address the host keeps stable
// for as long as the flip is registered (ArchiveProps uses the copy's own pose-pointer row); the serial is the
// one FlipBookRegister returned. A handle that is not registered, or whose serial is not the command's (it was
// released and registered again since the command was built), draws NOTHING and counts `stale`: a command
// built before a release is never followed to a freed list.
//
// THE CLOCK is RuneScape's, on StaticBakeClockSeconds() (sampled once a rendered frame, so `staticbake clock`
// pins poses as it pins the scroll): cycle = floor((clock - origin) x 50) + cycleOffset, in 20 ms client
// cycles; each pose lasts its own cycles; past the last pose the walk steps back frameStep poses and goes
// round again, and a frameStep outside 1..poseCount plays once and ends on rest (poses[poseCount]). A cycle
// before the play's start (a pinned clock earlier than a play-once origin) is the first pose, never garbage.
// A looping flip uses origin 0 and its random start as cycleOffset; a play-once one its own start as origin.
//
// RECORDING: a flip head inside a list the bake is recording would freeze whichever pose was current then.
// The recorder's whitelist does not admit G_FLIPBOOK, so such a list is rejected for good ("a flip head
// inside a recorded list") and interpreted, where the flip resolves per frame as it should.
//
// NOT GATED on the bake: with the bake off, or on a backend that cannot bake, a flip head still picks its pose
// and the pose list is interpreted.
//
// LIFETIME: the registry holds the host's pose pointers, not their resources. The host must unregister a flip
// before the lists it names can go (ArchiveProps does it in ReleaseHeld, before it drops them).

namespace Fast {

constexpr int FLIPBOOK_POSES_MAX = 16;
constexpr double FLIPBOOK_CYCLES_PER_SECOND = 50.0; // RS's client cycle, 20 ms

struct FlipBookDesc {
    // poses[k] for k < poseCount, then rest at poses[poseCount]. NULL draws nothing for that pose.
    const void* poses[FLIPBOOK_POSES_MAX + 1] = {};
    int32_t cycles[FLIPBOOK_POSES_MAX] = {}; // each pose's length in client cycles, >= 1
    int32_t poseCount = 0;                   // 1..FLIPBOOK_POSES_MAX
    int32_t frameStep = 0;                   // RS's: 1..poseCount loops; anything else plays once
    int64_t cycleOffset = 0;                 // added to the clock's cycles: a loop's random start
    double origin = 0.0;                     // clock seconds the play counts from
};

// Register (or replace) the flip at `handle`; returns its serial, never 0. 0 when the description does not
// read (no handle, a pose count out of range, a length below 1): nothing is registered.
uint16_t FlipBookRegister(const void* handle, const FlipBookDesc& desc);
// A play-once flip's start, once its owner knows it. No-op for an unknown handle.
void FlipBookSetOrigin(const void* handle, double origin);
void FlipBookUnregister(const void* handle);
void FlipBookClear();

// The walk itself, pure: which pose `cycle` client cycles into the play shows (poseCount: rest).
int32_t FlipBookPoseAt(const FlipBookDesc& desc, int64_t cycle);
// The cycle the clock puts the flip at.
int64_t FlipBookCycleAt(const FlipBookDesc& desc, double clockSeconds);
// The pose the registered flip at `handle` shows at `clockSeconds`, forced pose ignored; -1 for an unknown
// handle. Pure: no counter moves. For the host's self-test against its own pick.
int32_t FlipBookPoseAtClock(const void* handle, double clockSeconds);

// A forced pose for every flip: FLIPBOOK_FORCE_CLOCK (the default), FLIPBOOK_FORCE_REST, or a pose index
// (clamped to each flip's last pose). For same-picture checks; the host's `animprop pose`.
constexpr int32_t FLIPBOOK_FORCE_CLOCK = -1;
constexpr int32_t FLIPBOOK_FORCE_REST = -2;
void FlipBookForce(int32_t pose);

struct FlipBookStats {
    uint32_t registered = 0;
    uint64_t resolves = 0; // flip heads resolved to a pose
    uint64_t changes = 0;  // resolves that picked another pose than the flip's last
    uint64_t stale = 0;    // commands whose handle or serial was not registered: drawn as nothing
    uint64_t frames = 0;   // rendered frames since boot
};
FlipBookStats FlipBookGetStats();
// One flip's: false for an unknown handle.
struct FlipBookEntryInfo {
    uint16_t serial = 0;
    int32_t last = -1; // the pose last resolved
    uint64_t resolves = 0;
    uint64_t changes = 0;
};
bool FlipBookGetEntry(const void* handle, FlipBookEntryInfo* out);

// THE TRACE: which of one flip's pose lists the interpreter called, once a rendered frame, through a flip head
// or through a plain G_DL to the same list (so a pose a Draw picked shows the same way). For the holds and the
// rate: a frame's clock is real time unless pinned. Starting a trace replaces any before it; it stops itself
// after `frames` rendered frames, and the rows stay until the next start.
struct FlipBookTraceRow {
    uint64_t frame = 0; // FlipBookStats::frames when it was drawn
    double clock = 0.0;
    int32_t pose = -1; // poseCount: rest
    bool head = false; // through a flip head (else a plain G_DL)
};
bool FlipBookTraceStart(const void* handle, uint32_t frames);
const std::vector<FlipBookTraceRow>& FlipBookTraceRows(bool* running);

// ---------------------------------------------------------------------------
// Interpreter-facing (libultraship internal)
// ---------------------------------------------------------------------------

// True while a trace runs: the G_DL handler's one extra branch.
extern bool gFlipBookTracing;
void FlipBookBeginFrame();
// The pose list a G_FLIPBOOK draws now, or nullptr (stale, or the pose has no list).
const void* FlipBookResolve(const void* handle, uint16_t serial);
// A plain G_DL to `displayList`, while tracing.
void FlipBookTraceCall(const void* displayList);

} // namespace Fast
