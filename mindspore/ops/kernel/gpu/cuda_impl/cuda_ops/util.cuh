/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UTIL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UTIL_CUH_
#include <cuda_fp16.h>
#include <cuComplex.h>
#include <algorithm>
#include <limits>
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

#define kThreadsPerBlock (256)
#define kBlocksPerGrid(n) ((n + kThreadsPerBlock - 1) / kThreadsPerBlock)

namespace atomic {
constexpr size_t OneByte = 1;
constexpr size_t TwoByte = 2;
constexpr size_t FourByte = 4;
constexpr size_t EightByte = 8;

template <typename Func, typename T, size_t Bytes = sizeof(T)>
struct MsAtomicBinaryOpImpl;

template <typename Func, typename T>
struct MsAtomicBinaryOpImpl<Func, T, OneByte> {
  __device__ __forceinline__ T operator()(T *address, T val) {
    // We use cuda's atomicCAS(unsigned int*, unsigned int, unsigned int) to
    // implement MsAtomicAdd. An unsigned char may not be 4 byte aligned, but
    // unsigned int* must be 4 byte aligned. This variable contains the offset,
    // in bytes, of the beginning of address, within the 4 byte aligned space that
    // contains it.
    size_t address_offset = reinterpret_cast<size_t>(address) & 3;

    // Address of the 4 byte aligned space that contains address.
    unsigned int *aligned =
      reinterpret_cast<unsigned int *>(reinterpret_cast<unsigned char *>(address) - address_offset);

    // Constants which will be used later with __byte_perm. __byte_perm is a cuda
    // function which takes 3 unsigned int's (x, y, selector) as parameters and
    // returns an int. __byte_perm returns an integer by selecting bytes from x
    // and y based on the given selector. The selector 0x3210 in will select all
    // four bytes from x, preserving their original order. The position of the
    // "4" in the selector indicates the position in the output where the first
    // byte of y will end up.
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};

    // Gets the selector that will select the bytes at address from aligned
    unsigned int selector = selectors[address_offset];

    unsigned int old = *aligned;
    unsigned int assumed = 0;

    do {
      assumed = old;
      // Selects the byte associated with address and put it as the first byte of
      // this variable, so that we can add val to the value at address.
      uint8_t old_byte = __byte_perm(old, 0, address_offset);
      T old_value = *(reinterpret_cast<T *>(&old_byte));

      T new_value = Func()(old_value, val);

      unsigned int new_byte = *(reinterpret_cast<uint8_t *>(&new_value));

      // Takes old and replaces the byte corresponding to address with the sum.
      unsigned int replacement = __byte_perm(old, new_byte, selector);

      // Try to replace the old value with the new value
      old = atomicCAS(aligned, assumed, replacement);
    } while (old != assumed);
    // Select the single byte corredsponding to address and return it.
    return __byte_perm(old, 0, address_offset);
  }
};

template <typename Func, typename T>
struct MsAtomicBinaryOpImpl<Func, T, TwoByte> {
  __device__ __forceinline__ T operator()(T *address, T val) {
    bool is_4_byte_aligned = (reinterpret_cast<size_t>(address) & 2) == 0;
    unsigned int *aligned = reinterpret_cast<unsigned int *>(reinterpret_cast<size_t>(address) & ~2);
    unsigned int old = *aligned;
    unsigned int assumed;

    do {
      assumed = old;
      uint16_t old_byte = is_4_byte_aligned ? (old & 0xffff) : (old >> 16);
      T old_value = *(reinterpret_cast<T *>(&old_byte));
      // Do the binary operation.
      T new_value = Func()(old_value, val);

      unsigned int new_byte = *(reinterpret_cast<uint16_t *>(&new_value));
      if (is_4_byte_aligned) {
        new_byte = (old & 0xffff0000) | new_byte;
      } else {
        new_byte = (old & 0xffff) | (new_byte << 16);
      }
      // Try to replace the old value with the new value.
      // If failed, the current value of *address would be used to update the old value.
      old = atomicCAS(aligned, assumed, new_byte);
    } while (assumed != old);

    if (is_4_byte_aligned) {
      return T(old & 0xffff);  // NOLINT
    } else {
      return T(old >> 16);  // NOLINT
    }
  }
};

template <typename Func, typename T>
struct MsAtomicBinaryOpImpl<Func, T, FourByte> {
  __device__ __forceinline__ T operator()(T *address, T val) {
    unsigned int *address_as_uint32 = reinterpret_cast<unsigned int *>(address);
    unsigned int old = *address_as_uint32;
    unsigned int assumed;
    do {
      assumed = old;
      T old_value = *(reinterpret_cast<T *>(&old));
      // Do the binary operation.
      T new_value = Func()(old_value, val);
      unsigned int new_byte = *(reinterpret_cast<unsigned int *>(&new_value));
      // Try to replace the old value with the new value.
      // If failed, the current value of *address would be used to update the old value.
      old = atomicCAS(address_as_uint32, assumed, new_byte);
    } while (assumed != old);
    return T(old);
  }
};

template <typename Func, typename T>
struct MsAtomicBinaryOpImpl<Func, T, EightByte> {
  __device__ __forceinline__ T operator()(T *address, T val) {
    unsigned long long int *address_as_uint64 = reinterpret_cast<unsigned long long int *>(address);  // NOLINT
    unsigned long long int old = *address_as_uint64;                                                  // NOLINT
    unsigned long long int assumed;                                                                   // NOLINT
    do {
      assumed = old;
      T old_value = *(reinterpret_cast<T *>(&old));
      // Do the binary operation.
      T new_value = Func()(old_value, val);
      unsigned long long int new_byte = *(reinterpret_cast<unsigned long long int *>(&new_value));  // NOLINT
      // Try to replace the old value with the new value.
      // If failed, the current value of *address would be used to update the old value.
      old = atomicCAS(address_as_uint64, assumed, new_byte);
    } while (assumed != old);
    return T(old);
  }
};

struct Add {
  template <typename T>
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return lhs + rhs;
  }
};

struct Sub {
  template <typename T>
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return lhs - rhs;
  }
};

struct Mul {
  template <typename T>
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return lhs * rhs;
  }
};

struct Div {
  template <typename T>
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return lhs / rhs;
  }
};

struct Min {
  template <typename T>
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return lhs < rhs ? lhs : rhs;
  }
};

struct Max {
  template <typename T>
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return lhs > rhs ? lhs : rhs;
  }
};
}  // namespace atomic

// atomic add
template <typename T>
__device__ __forceinline__ T MsAtomicAdd(T *address, T val) {
  return atomic::MsAtomicBinaryOpImpl<atomic::Add, T>()(address, val);
}

// For following types, call CUDA API directly
template <>
__device__ __forceinline__ int MsAtomicAdd(int *address, int val) {
  return atomicAdd(address, val);
}

template <>
__device__ __forceinline__ unsigned int MsAtomicAdd(unsigned int *address, unsigned int val) {
  return atomicAdd(address, val);
}

template <>
__device__ __forceinline__ unsigned long long int MsAtomicAdd(unsigned long long int *address,  // NOLINT
                                                              unsigned long long int val) {     // NOLINT
  return atomicAdd(address, val);
}

template <>
__device__ __forceinline__ float MsAtomicAdd(float *address, float val) {
  return atomicAdd(address, val);
}

template <>
__device__ __forceinline__ bool MsAtomicAdd(bool *address, bool val) {
  *address = address && val;
  return address[0];
}

template <>
__device__ __forceinline__ Complex<float> MsAtomicAdd(Complex<float> *address, Complex<float> val) {
  float *realAddr = reinterpret_cast<float *>(address);
  return Complex<float>(MsAtomicAdd(realAddr, val.real()), MsAtomicAdd(realAddr + 1, val.imag()));
}

template <>
__device__ __forceinline__ Complex<double> MsAtomicAdd(Complex<double> *address, Complex<double> val) {
  double *realAddr = reinterpret_cast<double *>(address);
  return Complex<double>(MsAtomicAdd(realAddr, val.real()), MsAtomicAdd(realAddr + 1, val.imag()));
}

// atomic sub
template <typename T>
__device__ __forceinline__ T MsAtomicSub(T *address, T val) {
  return atomic::MsAtomicBinaryOpImpl<atomic::Sub, T>()(address, val);
}

// For following types, call CUDA API directly
template <>
__device__ __forceinline__ unsigned int MsAtomicSub(unsigned int *address, unsigned int val) {
  return atomicSub(address, val);
}

// atomic min
template <typename T>
__device__ __forceinline__ T MsAtomicMin(T *address, T val) {
  return atomic::MsAtomicBinaryOpImpl<atomic::Min, T>()(address, val);
}

// For following types, call CUDA API directly
template <>
__device__ __forceinline__ int MsAtomicMin(int *address, int val) {
  return atomicMin(address, val);
}

template <>
__device__ __forceinline__ unsigned int MsAtomicMin(unsigned int *address, unsigned int val) {
  return atomicMin(address, val);
}

template <>
__device__ __forceinline__ unsigned long long int MsAtomicMin(unsigned long long int *address,  // NOLINT
                                                              unsigned long long int val) {     // NOLINT
  return atomicMin(address, val);
}

template <>
__device__ __forceinline__ long long int MsAtomicMin(long long int *address, long long int val) {  // NOLINT
  return atomicMin(address, val);
}

// atomic max
template <typename T>
__device__ __forceinline__ T MsAtomicMax(T *address, T val) {
  return atomic::MsAtomicBinaryOpImpl<atomic::Max, T>()(address, val);
}

// For following types, call CUDA API directly
template <>
__device__ __forceinline__ int MsAtomicMax(int *address, int val) {
  return atomicMax(address, val);
}

template <>
__device__ __forceinline__ unsigned int MsAtomicMax(unsigned int *address, unsigned int val) {
  return atomicMax(address, val);
}

template <>
__device__ __forceinline__ unsigned long long int MsAtomicMax(unsigned long long int *address,  // NOLINT
                                                              unsigned long long int val) {     // NOLINT
  return atomicMax(address, val);
}

template <>
__device__ __forceinline__ long long int MsAtomicMax(long long int *address, long long int val) {  // NOLINT
  return atomicMax(address, val);
}

// atomic mul
template <typename T>
__device__ __forceinline__ T MsAtomicMul(T *address, T val) {
  return atomic::MsAtomicBinaryOpImpl<atomic::Mul, T>()(address, val);
}

template <>
__device__ __forceinline__ bool MsAtomicMul(bool *address, bool val) {
  *address = address && val;
  return address[0];
}

// atomic div
template <typename T>
__device__ __forceinline__ T MsAtomicDiv(T *address, T val) {
  return atomic::MsAtomicBinaryOpImpl<atomic::Div, T>()(address, val);
}

__device__ __forceinline__ unsigned BallotSync(int predicate, unsigned mask = 0xffffffff) {
  return __ballot_sync(mask, predicate);
}

struct MsAtomicAddFunctor {
  template <typename T>
  __device__ __forceinline__ T operator()(T *address, T val) {
    return MsAtomicAdd(address, val);
  }
};

struct MsAtomicSubFunctor {
  template <typename T>
  __device__ __forceinline__ T operator()(T *address, T val) {
    return MsAtomicSub(address, val);
  }
};

struct MsAtomicMulFunctor {
  template <typename T>
  __device__ __forceinline__ T operator()(T *address, T val) {
    return MsAtomicMul(address, val);
  }
};

struct MsAtomicDivFunctor {
  template <typename T>
  __device__ __forceinline__ T operator()(T *address, T val) {
    return MsAtomicDiv(address, val);
  }
};

struct MsAtomicMinFunctor {
  template <typename T>
  __device__ __forceinline__ T operator()(T *address, T val) {
    return MsAtomicMin(address, val);
  }
};

struct MsAtomicMaxFunctor {
  template <typename T>
  __device__ __forceinline__ T operator()(T *address, T val) {
    return MsAtomicMax(address, val);
  }
};

enum : unsigned { warp_size = 32, log_wap_size = 5 };
__device__ __forceinline__ unsigned LaneId() { return threadIdx.x & (warp_size - 1); }
__device__ __forceinline__ unsigned WarpId(const unsigned &tid) { return tid >> log_wap_size; }

template <typename T>
struct Epsilon {
  static constexpr float value = std::numeric_limits<T>::epsilon();
};

template <>
struct Epsilon<half> {
  static constexpr float value = 0.000977;
};

// Some bit-related function
inline int Log2Floor(uint32_t n) {
  if (n == 0) return -1;
  int log = 0;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    uint32_t x = n >> shift;
    if (x) {
      n = x;
      log += shift;
    }
  }
  return log;
}

inline int Log2Ceil(uint32_t n) {
  int floor = Log2Floor(n);
  if (n == (n & ~(n - 1)))
    return floor;
  else
    return floor + 1;
}

inline int Log2Floor64(uint64_t n) {
  // Scan n first high 32 then low 32 bits.
  const uint32_t high_32_bit = static_cast<uint32_t>(n >> 32);
  if (high_32_bit == 0) {
    return Log2Floor(static_cast<uint32_t>(n));
  } else {
    return 32 + Log2Floor(high_32_bit);
  }
}

inline int Log2Ceil64(uint64_t n) {
  int floor = Log2Floor64(n);
  if (n == (n & ~(n - 1)))
    return floor;
  else
    return floor + 1;
}

template <typename T>
__device__ __forceinline__ T ZeroImpl() {
  return 0;
}

template <>
__device__ __forceinline__ cuComplex ZeroImpl() {
  return make_cuComplex(0., 0.);
}

template <>
__device__ __forceinline__ cuDoubleComplex ZeroImpl() {
  return make_cuDoubleComplex(0., 0.);
}

template <typename T>
__device__ __forceinline__ T shfl_xor_sync(unsigned mask, T var, int lane_mask) {
  return __shfl_xor_sync(mask, var, lane_mask);
}

template <>
__device__ __forceinline__ Complex<float> shfl_xor_sync(unsigned mask, Complex<float> var, int lane_mask) {
  return Complex<float>(__shfl_xor_sync(mask, var.real(), lane_mask), __shfl_xor_sync(mask, var.imag(), lane_mask));
}

template <>
__device__ __forceinline__ Complex<double> shfl_xor_sync(unsigned mask, Complex<double> var, int lane_mask) {
  return Complex<double>(__shfl_xor_sync(mask, var.real(), lane_mask), __shfl_xor_sync(mask, var.imag(), lane_mask));
}

template <typename T>
__device__ __forceinline__ T shfl_down_sync(unsigned mask, T var, int lane_mask) {
  return __shfl_down_sync(mask, var, lane_mask);
}

template <>
__device__ __forceinline__ Complex<float> shfl_down_sync(unsigned mask, Complex<float> var, int lane_mask) {
  return Complex<float>(__shfl_down_sync(mask, var.real(), lane_mask), __shfl_down_sync(mask, var.imag(), lane_mask));
}

template <>
__device__ __forceinline__ Complex<double> shfl_down_sync(unsigned mask, Complex<double> var, int lane_mask) {
  return Complex<double>(__shfl_down_sync(mask, var.real(), lane_mask), __shfl_down_sync(mask, var.imag(), lane_mask));
}

template <typename T>
__device__ __forceinline__ void FastAtomicAdd(T *base, size_t offset, const size_t length, T value) {
  MsAtomicAdd(base + offset, value);
}

template <>
__device__ __forceinline__ void FastAtomicAdd(half *base, size_t offset, const size_t length, half value) {
#if ((defined(CUDA_VERSION) && (CUDA_VERSION < 10000)) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  MsAtomicAdd(reinterpret_cast<half *>(base + offset), static_cast<half>(value));
#else
  // Accounts for the chance base falls on an odd 16 bit alignment (ie, not 32 bit aligned)
  __half *target_addr = reinterpret_cast<__half *>(base + offset);
  bool low_byte = (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__half2) == 0);

  if (low_byte && offset < (length - 1)) {
    __half2 value2;
    value2.x = value;
    value2.y = __float2half_rz(0);
    atomicAdd(reinterpret_cast<__half2 *>(target_addr), value2);
  } else if (!low_byte && offset > 0) {
    __half2 value2;
    value2.x = __float2half_rz(0);
    value2.y = value;
    atomicAdd(reinterpret_cast<__half2 *>(target_addr - 1), value2);
  } else {
    MsAtomicAdd(reinterpret_cast<__half *>(base) + offset, static_cast<__half>(value));
  }
#endif
}

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UTIL_CUH_
