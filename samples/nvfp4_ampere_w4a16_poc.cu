// SPDX-License-Identifier: Apache-2.0
//
// Experimental Ampere NVFP4 W4A16 GEMM proof of concept.
//
// The NVFP4 weight stays packed as E2M1 in HBM. Each CTA dequantizes only its
// live 128x64 weight tile to BF16 in shared memory, then feeds Ampere's
// m16n8k16 BF16 tensor-core MMA. Activations remain BF16 (W4A16).
//
// This is intentionally standalone: samples/nvfp4_ampere_w4a16_poc.py builds
// this file with torch.utils.cpp_extension. Once the path is validated on SM8x,
// the kernel can be integrated into comfy-kitchen's normal CUDA backend.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

namespace {

__device__ __forceinline__ uint32_t cvta_smem_u32(const void* ptr) {
    uint32_t s;
    asm("{ .reg .u64 ll; cvta.to.shared.u64 ll, %1; cvt.u32.u64 %0, ll; }"
        : "=r"(s) : "l"(ptr));
    return s;
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&dst)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
                 : "r"(addr));
#endif
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
}

// NVIDIA/PyTorch float8_e4m3fn: E4M3, bias 7, finite exponent 0xf values,
// with only exp=0xf/mantissa=7 reserved for NaN. NVFP4 scales are clamped to
// 448, so the NaN encoding is not expected here.
__device__ __forceinline__ float decode_e4m3fn(uint8_t x) {
    const uint32_t sign = x >> 7;
    const uint32_t exp = (x >> 3) & 0x0f;
    const uint32_t mant = x & 0x07;

    float mag;
    if (exp == 0) {
        // (mant / 8) * 2^(1-bias) = mant * 2^-9.
        mag = ldexpf(static_cast<float>(mant), -9);
    } else if (exp == 0x0f && mant == 0x07) {
        mag = nanf("");
    } else {
        mag = ldexpf(1.0f + static_cast<float>(mant) * 0.125f,
                     static_cast<int>(exp) - 7);
    }
    return sign ? -mag : mag;
}

// E2M1: sign(1), exponent(2, bias 1), mantissa(1). Magnitudes are
// {0, .5, 1, 1.5, 2, 3, 4, 6}.
__device__ __forceinline__ float decode_e2m1(uint8_t x) {
    const uint32_t sign = x >> 3;
    const uint32_t exp = (x >> 1) & 0x03;
    const uint32_t mant = x & 0x01;

    float mag;
    if (exp == 0) {
        mag = mant ? 0.5f : 0.0f;
    } else {
        mag = ldexpf(1.0f + static_cast<float>(mant) * 0.5f,
                     static_cast<int>(exp) - 1);
    }
    return sign ? -mag : mag;
}

// Same cuBLAS/NVFP4 scale-factor swizzle used by comfy-kitchen's native
// quantize/dequantize kernels. col_length is the number of 16-value blocks in
// a logical row (K / 16), not its padded storage width.
__device__ __forceinline__ size_t scale_factor_swizzled_offset(
    size_t row_idx, size_t col_idx, uint32_t col_length) {
    constexpr uint32_t kTotalRowsPerBaseBlock = 128;
    constexpr uint32_t kRowsPerBaseBlockCol = 32;
    constexpr uint32_t kColsPerBaseBlockCol = 4;

    const size_t rb = row_idx / kTotalRowsPerBaseBlock;
    const size_t rem = row_idx % kTotalRowsPerBaseBlock;
    const size_t d4 = rem / kRowsPerBaseBlockCol;
    const size_t d3 = rem % kRowsPerBaseBlockCol;
    const size_t cbg = col_idx / kColsPerBaseBlockCol;
    const size_t d5 = col_idx % kColsPerBaseBlockCol;
    const size_t cbg_cnt =
        (col_length + kColsPerBaseBlockCol - 1) / kColsPerBaseBlockCol;

    return ((rb * cbg_cnt + cbg) * kRowsPerBaseBlockCol + d3) * 16
        + d4 * kColsPerBaseBlockCol + d5;
}

__global__ void nvfp4_w4a16_mma_kernel(
    const __nv_bfloat16* __restrict__ x,       // (M, K)
    const uint8_t* __restrict__ qweight,       // (N_storage, K/2), hi-first
    const uint8_t* __restrict__ block_scales,  // swizzled E4M3FN
    const float* __restrict__ global_scale,    // scalar decode scale
    __nv_bfloat16* __restrict__ out,           // (M, N)
    int M,
    int N,
    int K,
    int qweight_rows) {
    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 64;
    constexpr int BLOCK_KH = BLOCK_K / 2;
    constexpr int NUM_WARPS = 4;
    constexpr int CTA_THREADS = NUM_WARPS * 32;
    constexpr int WARP_N = BLOCK_N / NUM_WARPS;  // 32
    constexpr int N_MMA = WARP_N / 8;            // 4
    constexpr int K_MMA = BLOCK_K / 16;          // 4
    constexpr int SMEM_STRIDE_K = BLOCK_K + 8;   // avoid ldmatrix bank conflicts

    __shared__ alignas(16) __nv_bfloat16 x_sh[BLOCK_M * SMEM_STRIDE_K];
    __shared__ alignas(16) __nv_bfloat16 w_sh[BLOCK_N * SMEM_STRIDE_K];

    const int cta_n = blockIdx.x * BLOCK_N;
    const int cta_m = blockIdx.y * BLOCK_M;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_n_base = warp_id * WARP_N;

    const int k_half = K / 2;
    const int k_blocks16 = K / 16;
    const float tensor_scale = global_scale[0];

    float acc[N_MMA][4];
#pragma unroll
    for (int i = 0; i < N_MMA; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k_base = 0; k_base < K; k_base += BLOCK_K) {
        const int kh_base = k_base / 2;
        const int n_global = cta_n + tid;

        // One thread owns one output-feature row for the dequant pass.
        if (n_global < N && n_global < qweight_rows) {
            float scales[4];
#pragma unroll
            for (int sb = 0; sb < 4; ++sb) {
                const int block_col = (k_base / 16) + sb;
                const size_t off = scale_factor_swizzled_offset(
                    static_cast<size_t>(n_global),
                    static_cast<size_t>(block_col),
                    static_cast<uint32_t>(k_blocks16));
                scales[sb] = decode_e4m3fn(block_scales[off]) * tensor_scale;
            }

            const uint8_t* qw_row = qweight +
                static_cast<size_t>(n_global) * k_half + kh_base;
#pragma unroll
            for (int kh = 0; kh < BLOCK_KH; ++kh) {
                const uint8_t packed = qw_row[kh];
                // comfy-kitchen NVFP4 default/cublas convention is hi_first:
                // even K element in high nibble, odd K element in low nibble.
                const uint8_t q_even = packed >> 4;
                const uint8_t q_odd = packed & 0x0f;
                const float s = scales[kh >> 3];  // 8 bytes == 16 values
                w_sh[tid * SMEM_STRIDE_K + 2 * kh] =
                    __float2bfloat16_rn(decode_e2m1(q_even) * s);
                w_sh[tid * SMEM_STRIDE_K + 2 * kh + 1] =
                    __float2bfloat16_rn(decode_e2m1(q_odd) * s);
            }
        } else {
#pragma unroll
            for (int kk = 0; kk < BLOCK_K; ++kk) {
                w_sh[tid * SMEM_STRIDE_K + kk] = __float2bfloat16_rn(0.0f);
            }
        }

        // 1024 BF16 values / 128 threads = one 16-byte vector load each.
        {
            const int v = tid;
            const int mm = (v * 8) / BLOCK_K;
            const int kk = (v * 8) % BLOCK_K;
            const int m_global = cta_m + mm;
            const int k_global = k_base + kk;
            uint4* dst = reinterpret_cast<uint4*>(
                &x_sh[mm * SMEM_STRIDE_K + kk]);
            if (m_global < M) {
                *dst = *reinterpret_cast<const uint4*>(
                    &x[static_cast<size_t>(m_global) * K + k_global]);
            } else {
                *dst = make_uint4(0, 0, 0, 0);
            }
        }

        __syncthreads();

#pragma unroll
        for (int k_mma = 0; k_mma < K_MMA; ++k_mma) {
            const int k_off = k_mma * 16;

            uint32_t a_frag[4];
            const __nv_bfloat16* a_addr =
                &x_sh[(lane % 16) * SMEM_STRIDE_K + (lane / 16) * 8 + k_off];
            ldmatrix_x4(a_frag, cvta_smem_u32(a_addr));

#pragma unroll
            for (int b_pair = 0; b_pair < N_MMA / 2; ++b_pair) {
                const int n_off = warp_n_base + b_pair * 16;
                uint32_t b_frag4[4];
                const __nv_bfloat16* b_addr =
                    &w_sh[(n_off + (lane % 16)) * SMEM_STRIDE_K
                          + (lane / 16) * 8 + k_off];
                ldmatrix_x4(b_frag4, cvta_smem_u32(b_addr));

                const uint32_t b0[2] = {b_frag4[0], b_frag4[2]};
                mma_m16n8k16_bf16(acc[b_pair * 2], a_frag, b0);

                const uint32_t b1[2] = {b_frag4[1], b_frag4[3]};
                mma_m16n8k16_bf16(acc[b_pair * 2 + 1], a_frag, b1);
            }
        }

        __syncthreads();
    }

    const int row_lo = lane / 4;
    const int col_lo = (lane % 4) * 2;
#pragma unroll
    for (int n_mma = 0; n_mma < N_MMA; ++n_mma) {
        const int n_global_base = cta_n + warp_n_base + n_mma * 8;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int m_global = cta_m + row_lo + half * 8;
            if (m_global < M) {
                const int n0 = n_global_base + col_lo;
                const int n1 = n0 + 1;
                if (n0 < N) {
                    out[static_cast<size_t>(m_global) * N + n0] =
                        __float2bfloat16_rn(acc[n_mma][half * 2]);
                }
                if (n1 < N) {
                    out[static_cast<size_t>(m_global) * N + n1] =
                        __float2bfloat16_rn(acc[n_mma][half * 2 + 1]);
                }
            }
        }
    }
}

torch::Tensor nvfp4_w4a16(torch::Tensor x,
                           torch::Tensor qweight,
                           torch::Tensor block_scales,
                           torch::Tensor global_scale,
                           int64_t logical_n) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(qweight.is_cuda(), "qweight must be CUDA");
    TORCH_CHECK(block_scales.is_cuda(), "block_scales must be CUDA");
    TORCH_CHECK(global_scale.is_cuda(), "global_scale must be CUDA");
    TORCH_CHECK(x.device() == qweight.device() &&
                x.device() == block_scales.device() &&
                x.device() == global_scale.device(),
                "all tensors must be on the same CUDA device");

    TORCH_CHECK(x.scalar_type() == at::kBFloat16,
                "POC supports BF16 activations only");
    TORCH_CHECK(qweight.scalar_type() == at::kByte,
                "qweight must be torch.uint8 packed E2M1");
    TORCH_CHECK(global_scale.scalar_type() == at::kFloat,
                "global_scale must be float32");
    TORCH_CHECK(block_scales.element_size() == 1,
                "block_scales must use one-byte E4M3FN storage");

    TORCH_CHECK(x.dim() == 2, "x must be 2D (M, K)");
    TORCH_CHECK(qweight.dim() == 2, "qweight must be 2D (N_storage, K/2)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(qweight.is_contiguous(), "qweight must be contiguous");
    TORCH_CHECK(block_scales.is_contiguous(), "block_scales must be contiguous");
    TORCH_CHECK(global_scale.numel() == 1, "global_scale must be scalar");

    const int64_t M64 = x.size(0);
    const int64_t K64 = x.size(1);
    const int64_t N_storage64 = qweight.size(0);
    TORCH_CHECK(qweight.size(1) * 2 == K64,
                "qweight inner dimension must be K/2");
    TORCH_CHECK(K64 > 0 && (K64 % 64) == 0,
                "K must be a positive multiple of 64 for this POC");
    TORCH_CHECK(logical_n > 0 && logical_n <= N_storage64,
                "logical_n must be in (0, qweight.size(0)]");
    TORCH_CHECK(M64 <= INT32_MAX && logical_n <= INT32_MAX && K64 <= INT32_MAX,
                "dimensions exceed int32 kernel limits");

    int device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    TORCH_CHECK(prop.major >= 8,
                "Ampere BF16 MMA requires SM80 or newer, got sm",
                prop.major, prop.minor);

    auto out = torch::empty({M64, logical_n}, x.options());
    if (M64 == 0) {
        return out;
    }

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 128;
    constexpr int THREADS = 128;
    const dim3 block(THREADS);
    const dim3 grid(
        static_cast<unsigned>((logical_n + BLOCK_N - 1) / BLOCK_N),
        static_cast<unsigned>((M64 + BLOCK_M - 1) / BLOCK_M));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
    nvfp4_w4a16_mma_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        qweight.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(block_scales.data_ptr()),
        global_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        static_cast<int>(M64),
        static_cast<int>(logical_n),
        static_cast<int>(K64),
        static_cast<int>(N_storage64));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_w4a16", &nvfp4_w4a16,
          "Experimental NVFP4 weight-only BF16 matmul for SM80+");
}
