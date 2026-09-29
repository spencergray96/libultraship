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

// Directional lights the replay shader can light a baked vertex with. OoT binds at most 7
// (Lights_FindSlot), and rooms get two - the environment's dirLight1/2. A frame that has more is
// interpreted instead of replayed (StaticBakeIntercept), so this is a limit on the bake, never on
// the picture.
constexpr int STATIC_BAKE_MAX_DIR_LIGHTS = 7;

// A recorded vertex that was lit under G_LIGHTING carries its raw object-space normal where its
// lit colour would have gone, and this value in position.w to say so. Recorded w is otherwise
// always exactly 1 (the record pass runs under a near-identity matrix), and the replay shader
// rebuilds w from the camera, so the slot is free.
constexpr float STATIC_BAKE_LIT_W = 2.0f;

// Everything a baked (pre-recorded, object-space) draw needs that used to be folded into the
// vertex payload by the CPU. See fast/StaticMeshCache.h. Laid out to be memcpy'd straight into a
// 16-byte-aligned constant buffer: the HLSL cbuffer in StaticBakePatchSource mirrors it member for
// member, and HLSL packs each float4 (and each array element) on a 16-byte register.
struct StaticBakeUniforms {
    // Camera * projection as the interpreter would have applied it, with the widescreen X
    // adjustment folded in. Row-major, i.e. the same memory order as RSP::MP_matrix, so
    // clip = mul(float4(objectPos, 1), mvp) in HLSL.
    float mvp[4][4];
    float fogColor[4];
    float fogMul;          // RSP fog_mul, as G_MW_FOG left it
    float fogOffset;       // RSP fog_offset
    uint32_t numDirLights; // RSP current_num_lights - 1 (the last light is the ambient one)
    float pad;
    // The light state GfxSpVertex would have lit a vertex with, in the same units it uses:
    // colours are the lights' 0..255 bytes, and each direction is CalculateNormalDir's output - the
    // light's direction in the display list's object space, normalised. Only xyz / rgb are read.
    float ambient[4];
    float lightDir[STATIC_BAKE_MAX_DIR_LIGHTS][4];
    float lightColor[STATIC_BAKE_MAX_DIR_LIGHTS][4];
};

// Texture slots a baked draw can bind: TEXEL0 and TEXEL1. The HD mask and blend slots behind them
// (SHADER_FIRST_MASK_TEXTURE on) are refused at record time, so a baked draw never needs them.
constexpr int STATIC_BAKE_TEXTURE_SLOTS = 2;

// One replayed draw, as the backend needs it. Everything that is per draw rather than per room
// travels in here; the per-room camera, fog and lights are StaticBakeUniforms.
struct StaticBakeDraw {
    uint32_t bufferId; // CreateStaticBuffer's handle for the room
    size_t byteOffset; // where this draw's vertices start in that buffer
    size_t numTris;
    ShaderProgram* prg; // the interpreted program; the backend binds its transform-enabled twin
    uint8_t shadeMask;  // colour inputs that are SHADE, which the twin lights
    uint8_t cullMode;   // StaticBakeCull
    bool zmodeDecal;
    // HoldStaticTexture handles, one per slot the program samples, taken when the draw was recorded.
    // 0 binds no texture to that slot.
    uint32_t textures[STATIC_BAKE_TEXTURE_SLOTS];
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
    // false rejects the bake rather than drawing something wrong. shadeMask has bit j set when the
    // program's colour input j is SHADE: the twin lights those inputs from the recorded normal.
    // The program alone cannot say which input that is (inputs are numbered by first use in the
    // combiner), so the same program can need more than one twin.
    virtual bool PrepareStaticShader(struct ShaderProgram* prg, uint8_t shadeMask) {
        return false;
    }
    // Floats per vertex the given program's input layout expects - used to cross-check the stride
    // the recording actually produced before anything is drawn with it.
    virtual uint8_t GetShaderNumFloats(struct ShaderProgram* prg) {
        return 0;
    }
    // Take a reference to the texture and sampler bound to `slot` right now, for a baked draw to
    // bind on replay. The reference is to the GPU objects themselves, not to a texture-cache id, so
    // nothing the cache does afterwards - an eviction, an id handed to another texture, a full clear -
    // can change what the baked draw samples. Holding the same objects twice returns the same handle
    // with one more reference, so two handles compare equal exactly when they bind the same texture
    // with the same sampler and the same filter flag. 0 means nothing is bound.
    virtual uint32_t HoldStaticTexture(int slot) {
        return 0;
    }
    virtual void ReleaseStaticTexture(uint32_t handle) {
    }
    // One replayed draw. The caller has already applied depth/decal/prim-depth state through the
    // normal setters; this binds the persistent buffer, the transform-enabled shader, the draw's
    // held textures and the uniforms, draws, and leaves the backend's "currently bound" memo true
    // (or invalidated) so the next interpreted draw rebinds whatever it needs.
    virtual void DrawStaticTriangles(const StaticBakeDraw& draw, const StaticBakeUniforms& uniforms) {
    }

    // ---- Mipmaps for the host's own textures (sturdy-bassoon#146, fast/TextureMips.h). ----
    // Set by the interpreter around one texture import: the next UploadTexture should build a mip
    // chain. A backend that ignores it (every one but DX11 today) uploads one level, as before.
    virtual void SetNextUploadMipmaps(bool mipmaps) {
    }
    // How the shader picks a mipmapped texture's level: fast/TextureMips.h TextureMipsLodMode, and a
    // bias added to it. Takes effect from the next draw.
    virtual void SetMipLod(int mode, float bias) {
    }
    // Uploads that really built more than one level: a texture that is not a power of two gets one
    // level even when asked, and so does every texture on a backend that ignores the request.
    uint64_t MippedUploads() const {
        return mMippedUploads;
    }

    // Textures the backend has actually created through UploadTexture. The static bake compares it
    // across a texture-cache miss to tell an import that uploaded from one that returned early and
    // left the id holding its previous occupant's picture. Only backends that bake count.
    uint64_t TexturesUploaded() const {
        return mTexturesUploaded;
    }

  protected:
    uint64_t mTexturesUploaded = 0;
    uint64_t mMippedUploads = 0;
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
