#pragma once

#include <stdint.h>

#include <unordered_map>
#include <set>
#include "imconfig.h"

namespace Fast {
struct ShaderProgram;

struct GfxClipParameters {
    bool z_is_from_0_to_1;
    bool invertY;
};

enum FilteringMode { FILTER_THREE_POINT, FILTER_LINEAR, FILTER_NONE };

// Face culling for a baked draw. The interpreter culls per triangle on the CPU, in clip space,
// which a recording cannot do: at record time the vertices are in object space and there is no
// camera to be facing away from. The recorded draw therefore carries the RSP's cull mode and the
// backend asks the rasterizer for it instead - the same decision, one stage later.
enum StaticBakeCull : uint8_t {
    STATIC_BAKE_CULL_NONE = 0,
    STATIC_BAKE_CULL_FRONT,
    STATIC_BAKE_CULL_BACK,
    STATIC_BAKE_CULL_BOTH, // recording-side only: GfxSpTri1 drops these, so the bake is refused
};

// Everything a baked (pre-recorded, object-space) draw needs that used to be folded into the
// vertex payload by the CPU. See fast/StaticMeshCache.h. Laid out to be memcpy'd straight into a
// 16-byte-aligned constant buffer.
struct StaticBakeUniforms {
    // Camera * projection as the interpreter would have applied it, with the widescreen X
    // adjustment folded in. Row-major, i.e. the same memory order as RSP::MP_matrix, so
    // clip = mul(float4(objectPos, 1), mvp) in HLSL.
    float mvp[4][4];
    float fogColor[4];
    float fogMul;    // RSP fog_mul, as G_MW_FOG left it
    float fogOffset; // RSP fog_offset
    float pad[2];    // constant buffers are multiples of 16 bytes
};

// A hash function used to hash a: pair<float, float>
struct hash_pair_ff {
    size_t operator()(const std::pair<float, float>& p) const {
        const auto hash1 = std::hash<float>{}(p.first);
        const auto hash2 = std::hash<float>{}(p.second);

        // If hash1 == hash2, their XOR is zero.
        return (hash1 != hash2) ? hash1 ^ hash2 : hash1;
    }
};

class GfxRenderingAPI {
  public:
    virtual ~GfxRenderingAPI() = default;
    virtual const char* GetName() = 0;
    virtual int GetMaxTextureSize() = 0;
    virtual GfxClipParameters GetClipParameters() = 0;
    virtual void UnloadShader(ShaderProgram* oldPrg) = 0;
    virtual void LoadShader(ShaderProgram* newPrg) = 0;
    virtual void ClearShaderCache() = 0;
    virtual ShaderProgram* CreateAndLoadNewShader(uint64_t shaderId0, uint64_t shaderId1) = 0;
    virtual ShaderProgram* LookupShader(uint64_t shaderId0, uint64_t shaderId1) = 0;
    virtual void ShaderGetInfo(ShaderProgram* prg, uint8_t* numInputs, bool usedTextures[2]) = 0;
    virtual uint32_t NewTexture() = 0;
    virtual void SelectTexture(int tile, uint32_t textureId) = 0;
    virtual void UploadTexture(const uint8_t* rgba32Buf, uint32_t width, uint32_t height) = 0;
    virtual void SetSamplerParameters(int sampler, bool linear_filter, uint32_t cms, uint32_t cmt) = 0;
    virtual void SetDepthTestAndMask(bool depth_test, bool z_upd) = 0;
    virtual void SetZmodeDecal(bool decal) = 0;
    virtual void SetViewport(int x, int y, int width, int height) = 0;
    virtual void SetScissor(int x, int y, int width, int height) = 0;
    virtual void SetUseAlpha(bool useAlpha) = 0;
    virtual void DrawTriangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) = 0;
    virtual void Init() = 0;
    virtual void OnResize() = 0;
    virtual void StartFrame() = 0;
    virtual void EndFrame() = 0;
    virtual void FinishRender() = 0;
    virtual int CreateFramebuffer() = 0;
    virtual void UpdateFramebufferParameters(int fb_id, uint32_t width, uint32_t height, uint32_t msaa_level,
                                             bool opengl_invertY, bool render_target, bool has_depth_buffer,
                                             bool can_extract_depth) = 0;
    virtual void StartDrawToFramebuffer(int fbId, float noiseScale) = 0;
    virtual void CopyFramebuffer(int fbDstId, int fbSrcId, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0,
                                 int dstY0, int dstX1, int dstY1) = 0;
    virtual void ClearFramebuffer(bool color, bool depth) = 0;
    virtual void ClearDepthRegion(int x, int y, int w, int h) {
        // Default: full depth clear. Backends that support scissored depth clears
        // (e.g. OpenGL) should override for a more precise partial clear.
        ClearFramebuffer(false, true);
    }
    virtual void ReadFramebufferToCPU(int fbId, uint32_t width, uint32_t height, uint16_t* rgba16Buf) = 0;
    virtual void ResolveMSAAColorBuffer(int fbIdTarger, int fbIdSrc) = 0;
    virtual std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
    GetPixelDepth(int fb_id, const std::set<std::pair<float, float>>& coordinates) = 0;
    virtual void* GetFramebufferTextureId(int fbId) = 0;
    virtual void SelectTextureFb(int fbId) = 0;
    virtual void DeleteTexture(uint32_t texId) = 0;
    virtual void SetTextureFilter(FilteringMode mode) = 0;
    virtual FilteringMode GetTextureFilter() = 0;
    virtual void SetSrgbMode() = 0;
    virtual ImTextureID GetTextureById(int id) = 0;
    virtual void SetCurrentPrimDepth(float depth) = 0;

    // ---- Static-geometry bake (sturdy-bassoon#40). ----
    // Defaults are no-ops so a backend that has not implemented the path still compiles and
    // simply never bakes: StaticBakeIntercept refuses to record when SupportsStaticBake() is
    // false, and every display list stays interpreted. DX11 is the only implementation today.
    virtual bool SupportsStaticBake() {
        return false;
    }
    // Upload an immutable vertex buffer. Returns 0 on failure; ids are otherwise opaque.
    virtual uint32_t CreateStaticBuffer(const void* data, size_t sizeBytes) {
        return 0;
    }
    virtual void DeleteStaticBuffer(uint32_t bufferId) {
    }
    // Build (or find) the transform-enabled twin of an already-created shader program. Returning
    // false rejects the bake rather than drawing something wrong.
    virtual bool PrepareStaticShader(struct ShaderProgram* prg) {
        return false;
    }
    // Floats per vertex the given program's input layout expects - used to cross-check the stride
    // the recording actually produced before anything is drawn with it.
    virtual uint8_t GetShaderNumFloats(struct ShaderProgram* prg) {
        return 0;
    }
    // One replayed draw. The caller has already applied depth/decal state through the normal
    // setters; this binds the persistent buffer, the transform-enabled shader and the uniforms,
    // draws, and leaves the backend's "currently bound" memo invalidated so the next interpreted
    // draw rebinds from scratch.
    virtual void DrawStaticTriangles(uint32_t bufferId, size_t byteOffset, size_t numTris, struct ShaderProgram* prg,
                                     const StaticBakeUniforms& uniforms, uint8_t cullMode, bool zmodeDecal) {
    }

  protected:
    int8_t mCurrentDepthTest = 0;
    int8_t mCurrentDepthMask = 0;
    int8_t mCurrentZmodeDecal = 0;
    int8_t mLastDepthTest = -1;
    int8_t mLastDepthMask = -1;
    int8_t mLastZmodeDecal = -1;
    bool mSrgbMode = false;
    float mCurrentPrimDepth = 0.0f;
    bool mPrimDepthDirty = true;
};
} // namespace Fast
