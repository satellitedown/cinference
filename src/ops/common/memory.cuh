// Modified by satellitedown for Cinference: add evict-first L2 fill policies.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

namespace ninfer::ops {

enum class Cache { ca, cg };

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return *reinterpret_cast<const V*>(ptr);
}

template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return __ldg(reinterpret_cast<const V*>(ptr));
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    *reinterpret_cast<V*>(ptr) = value;
}

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes));
    }
}

// L2 priority for weight fills. A weight tile that a single CTA reads (one token tile) is streamed
// once per call, and the weights of a layer are far larger than L2: evict-first fills leave the
// small activation and record working set resident across layers instead of writing it back and
// refetching it every layer. Tiles that several token-tile CTAs reuse keep the normal priority.
__device__ __forceinline__ unsigned long long l2_weight_policy(bool read_once) {
    unsigned long long policy;
    if (read_once) {
        asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    } else {
        asm("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(policy));
    }
    return policy;
}

// 16-byte cp.async.cg whose L2 fill carries `policy` (see l2_weight_policy).
__device__ __forceinline__ void cp_async_cg_policy(void* smem_dst, const void* gmem_src,
                                                   unsigned long long policy) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                 :
                 : "r"(smem_addr(smem_dst)), "l"(gmem_src), "l"(policy));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
}

__device__ __forceinline__ void pipe_commit() { __pipeline_commit(); }

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
    __pipeline_wait_prior(Groups);
}

} // namespace ninfer::ops
