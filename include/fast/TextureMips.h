#pragma once

#include <stdint.h>

// Mipmaps for the host's own compiled-in scene textures (sturdy-bassoon#146).
//
// Fast3D uploads every texture with one level, so a texture seen from far enough away that many
// texels land in one pixel is sampled one texel per pixel, and the texel it picks changes as the
// camera moves: distant textures crawl. This builds the smaller copies at upload - on the CPU, the
// textures are tiny - and the three-point shader picks and blends two of them by distance.
//
// Scoped to textures the HOST names, not to every texture: vanilla assets and texture packs are
// untouched until the look has been judged. The host names them indirectly. It hands over the
// display lists it knows are its own (for SoH, the rooms of a scene defined in the fork's C), and
// every raw texture address one of those lists loads with G_SETTIMG while the interpreter walks it
// joins the set - an archive texture ("__OTR__" path) never does. Membership is decided before the
// texture's first upload, because the set grows at the G_SETTIMG that precedes the import.
//
// A texture gets mips only when both sides are powers of two and larger than 1x1 (the hardware wraps
// a smaller level cleanly only then), and only on a backend that builds them: DX11 today. On any
// other backend this is all inert.
//
// Two ways one of the host's textures can miss out, neither reached by today's scenes:
//   - a G_LOADTILE from a nonzero corner imports from base + offset, which is not the address the
//     G_SETTIMG named, so the lookup misses. The grid tool and Fast64 load whole textures (LoadBlock).
//   - a texture uploaded before any scoped list loaded it keeps its single level until the cache
//     drops it (an eviction, a clear, a toggle of this switch). The host's arrays are drawn only by
//     its own lists, so the first upload is always from inside one.
// The shader side lives in soh.o2r (shaders/directx/default.shader.hlsl). An exe with this change
// and an old archive samples a mipped texture with the old three-point filter, which lets the point
// sampler pick the level from snapped coordinates: distant textures come out noisy. Regenerate the
// archive and copy it next to the exe.

namespace Fast {

// Offer a display list whose textures should be mipmapped. Keyed by pointer, so only for lists whose
// address is a stable C symbol for the life of the process; registering one twice is a no-op, so it
// is safe on every room init.
void TextureMipsRegisterDisplayList(const void* displayList);

// The runtime switch. A change clears the texture cache and sends every bake back to be recorded, so
// both paths redraw with or without mips on the next frame. On by default. Safe between frames.
void TextureMipsSetEnabled(bool enabled);
bool TextureMipsIsEnabled();

// For the host's status line: display lists offered; raw addresses those lists have named with
// G_SETTIMG (a palette's too - a TLUT load names its palette the same way - so this is an upper bound
// on textures); and uploads that really built a chain of more than one level since boot (a
// re-upload after a cache clear counts again).
void TextureMipsGetStats(uint32_t* lists, uint32_t* addresses, uint64_t* mippedUploads);

// The texture filter the renderer is running (FilteringMode: 0 three-point, 1 linear, 2 none), or -1
// before there is one. The mips reach three-point (the shader picks the level) and linear (the
// sampler does); under none they are sampled from the nearest level. Here so a host status line can
// say which, without including the interpreter.
int TextureMipsFilterMode();

} // namespace Fast
