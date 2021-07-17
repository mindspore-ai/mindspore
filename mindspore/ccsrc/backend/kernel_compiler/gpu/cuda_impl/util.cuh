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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_UTIL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_UTIL_H_

#include <cuda_fp16.h>

#include <algorithm>

#include "runtime/device/gpu/cuda_common.h"

#define kThreadsPerBlock (256)
#define kBlocksPerGrid(n) ((n + kThreadsPerBlock - 1) / kThreadsPerBlock)

// atomic add

__device__ static inline double MsAtomicAdd(double *address, const double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;  // NOLINT
  unsigned long long int old = *address_as_ull;                                // NOLINT
  unsigned long long int assumed;                                              // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);  // NOLINT
  return __longlong_as_double(old);
}

__device__ static inline float MsAtomicAdd(float *address, const float val) { return atomicAdd(address, val); }

__device__ static inline int MsAtomicAdd(int *address, int val) { return atomicAdd(address, val); }

__device__ static inline unsigned int MsAtomicAdd(unsigned int *address, unsigned int val) {
  return atomicAdd(address, val);
}

__device__ static inline int8_t MsAtomicAdd(int8_t *address, int8_t val) {
  size_t offset = (size_t)address & 3;
  uint32_t *address_as_ui = (uint32_t *)((char *)address - offset);  // NOLINT
  uint32_t old = *address_as_ui;
  uint32_t shift = offset * 8;
  uint32_t old_byte;
  uint32_t newval;
  uint32_t assumed;

  do {
    assumed = old;
    old_byte = (old >> shift) & 0xff;
    newval = static_cast<uint8_t>(val + old_byte);
    newval = (old & ~(0x000000ff << shift)) | (newval << shift);
    old = atomicCAS(address_as_ui, assumed, newval);
  } while (assumed != old);
  return __byte_perm(old, 0, offset);
}

__device__ static inline int64_t MsAtomicAdd(int64_t *address, int64_t val) {
  unsigned long long *address_as_ui = (unsigned long long *)(address);  // NOLINT
  unsigned long long old = *address_as_ui;                              // NOLINT
  unsigned long long newval;                                            // NOLINT
  unsigned long long assumed;                                           // NOLINT

  do {
    assumed = old;
    newval = val + (int64_t)old;
    old = atomicCAS(address_as_ui, assumed, newval);
  } while (assumed != old);
  return (int64_t)old;
}

__device__ static inline bool MsAtomicAdd(bool *address, bool val) {
  *address = address && val;
  return address[0];
}

__device__ static inline unsigned char MsAtomicAdd(short *address, short val) {  // NOLINT
  bool is_4_byte_aligned = ((size_t)address & 2) == 0;
  unsigned int *aligned = (unsigned int *)((size_t)address & ~2);
  unsigned int old = *aligned;
  unsigned int assumed;

  do {
    assumed = old;
    unsigned int replacement;

    if (is_4_byte_aligned) {
      replacement = (old & 0xffff0000) | (((old & 0xffff) + val) & 0xffff);
    } else {
      replacement = old + ((unsigned int)val << 16);
    }

    old = atomicCAS(aligned, assumed, replacement);
  } while (assumed != old);

  if (is_4_byte_aligned) {
    return (short)(old & 0xffff);  // NOLINT
  } else {
    return (short)(old >> 16);  // NOLINT
  }
}

__device__ static inline half MsAtomicAdd(half *address, half val) {
  unsigned int *aligned =
    reinterpret_cast<unsigned int *>(reinterpret_cast<size_t>(address) - (reinterpret_cast<size_t>(address) & 2));
  unsigned int old = *aligned;
  unsigned int assumed;
  unsigned short old_as_us;  // NOLINT
  do {
    assumed = old;
    old_as_us =
      static_cast<unsigned short>(reinterpret_cast<size_t>(address) & 2 ? old >> 16 : old & 0xffff);  // NOLINT
    half sum = __float2half_rn(__half2float(__ushort_as_half(old_as_us)) + static_cast<float>(val));
    unsigned short sum_as_us = __half_as_ushort(sum);  // NOLINT
    unsigned int sum_as_ui =
      reinterpret_cast<size_t>(address) & 2 ? (sum_as_us << 16) | (old & 0xffff) : (old & 0xffff0000) | sum_as_us;
    old = atomicCAS(aligned, assumed, sum_as_ui);
  } while (assumed != old);
  __half_raw raw = {old_as_us};
  return half(raw);
}

__device__ static inline unsigned char MsAtomicAdd(unsigned char *address, unsigned char val) {
  // We use cuda's atomicCAS(unsigned int*, unsigned int, unsigned int) to
  // implement MsAtomicAdd. An unsigned char may not be 4 byte aligned, but
  // unsigned int* must be 4 byte aligned. This variable contains the offset,
  // in bytes, of the beginning of address, within the 4 byte aligned space that
  // contains it.
  size_t address_offset = (size_t)address & 3;

  // Address of the 4 byte aligned space that contains address.
  unsigned int *aligned = (unsigned int *)((unsigned char *)address - address_offset);

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
    unsigned int sum = val + __byte_perm(old, 0, address_offset);

    // Takes old and replaces the byte corresponding to address with the sum.
    unsigned int replacement = __byte_perm(old, sum, selector);

    // Try to replace the old value with the new value
    old = atomicCAS(aligned, assumed, replacement);
  } while (old != assumed);
  // Select the single byte corredsponding to address and return it.
  return __byte_perm(old, 0, address_offset);
}

__device__ static inline char MsAtomicAdd(char *address, char val) {
  size_t address_offset = (size_t)address & 3;
  unsigned int *aligned = reinterpret_cast<unsigned int *>(reinterpret_cast<char *>(address) - address_offset);
  unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
  unsigned int selector = selectors[address_offset];
  unsigned int old = *aligned;
  unsigned int assumed = 0;

  do {
    assumed = old;

    unsigned int sum = val + __byte_perm(old, 0, address_offset);
    unsigned int replacement = __byte_perm(old, sum, selector);

    old = atomicCAS(aligned, assumed, replacement);
  } while (old != assumed);
  return __byte_perm(old, 0, address_offset);
}

// atomic sub

template <typename T>
__device__ static inline T MsAtomicSub(T *address, const T val) {
  return MsAtomicAdd(address, -val);
}

template <>
__device__ inline unsigned char MsAtomicSub(unsigned char* address, unsigned char val) {
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int selector = selectors[(size_t)address & 3];
    unsigned int old, assumed, difference, new_value;
    old = *base_address;

    do {
        assumed = old;
        difference = (unsigned char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440) - val;
        new_value = __byte_perm(old, difference, selector);
        if (new_value == old)
            break;
        old = atomicCAS(base_address, assumed, new_value);
    } while (assumed != old);
    return old;
}

// atomic max

__device__ static inline float MsAtomicMax(int *address, const int val) {
  unsigned int *address_as_ui = (unsigned int *)address;  // NOLINT
  unsigned int old = *address_as_ui;                       // NOLINT
  unsigned int assumed;                                    // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address_as_ui, assumed, max(val, (int)assumed)); // NOLINT
  } while (assumed != old);  // NOLINT
  return __longlong_as_double(old);
}

__device__ static inline char MsAtomicMax(char* address, char val) {
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int selector = selectors[(size_t)address & 3];
    unsigned int old, assumed, max_value, new_value;
    old = *base_address;

    do {
        assumed = old;
        max_value = max(val, (char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440)); // NOLINT
        new_value = __byte_perm(old, max_value, selector);
        if (new_value == old)
            break;
        old = atomicCAS(base_address, assumed, new_value);
    } while (assumed != old);
    return old;
}

__device__ static inline unsigned char MsAtomicMax(unsigned char* address, unsigned char val) {
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int selector = selectors[(size_t)address & 3];
    unsigned int old, assumed, max_value, new_value;
    old = *base_address;

    do {
        assumed = old;
        max_value = max(val, (unsigned char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440));
        new_value = __byte_perm(old, max_value, selector);
        if (new_value == old)
            break;
        old = atomicCAS(base_address, assumed, new_value);
    } while (assumed != old);
    return old;
}

__device__ static inline float MsAtomicMax(float *address, const float val) {
  unsigned int *address_as_ui = (unsigned int *)address;  // NOLINT
  unsigned int old = *address_as_ui;                       // NOLINT
  unsigned int assumed;                                    // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address_as_ui, assumed, __float_as_uint(max(val, __uint_as_float(assumed))));
  } while (assumed != old);  // NOLINT
  return __longlong_as_double(old);
}

__device__ static inline half MsAtomicMax(half *address, half val) {
  unsigned int *aligned =
    reinterpret_cast<unsigned int *>(reinterpret_cast<size_t>(address) - (reinterpret_cast<size_t>(address) & 2));
  unsigned int old = *aligned;
  unsigned int assumed;
  unsigned short old_as_us;  // NOLINT
  do {
    assumed = old;
    old_as_us =
      static_cast<unsigned short>(reinterpret_cast<size_t>(address) & 2 ? old >> 16 : old & 0xffff);  // NOLINT
    half max_value = __float2half_rn(max(__half2float(__ushort_as_half(old_as_us)), static_cast<float>(val)));
    unsigned short max_as_us = __half_as_ushort(max_value);  // NOLINT
    unsigned int max_as_ui =
      reinterpret_cast<size_t>(address) & 2 ? (max_as_us << 16) | (old & 0xffff) : (old & 0xffff0000) | max_as_us;
    old = atomicCAS(aligned, assumed, max_as_ui);
  } while (assumed != old);
  __half_raw raw = {old_as_us};
  return half(raw);
}

// atomic min

__device__ static inline float MsAtomicMin(int *address, const int val) {
  unsigned int *address_as_ui = (unsigned int *)address;  // NOLINT
  unsigned int old = *address_as_ui;                       // NOLINT
  unsigned int assumed;                                    // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address_as_ui, assumed, min(val, (int)assumed)); // NOLINT
  } while (assumed != old);  // NOLINT
  return __longlong_as_double(old);
}

__device__ static inline char MsAtomicMin(char* address, char val) {
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int selector = selectors[(size_t)address & 3];
    unsigned int old, assumed, min_value, new_value;
    old = *base_address;

    do {
        assumed = old;
        min_value = min(val, (char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440)); // NOLINT
        new_value = __byte_perm(old, min_value, selector);
        if (new_value == old)
            break;
        old = atomicCAS(base_address, assumed, new_value);
    } while (assumed != old);
    return old;
}

__device__ static inline unsigned char MsAtomicMin(unsigned char* address, unsigned char val) {
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int selector = selectors[(size_t)address & 3];
    unsigned int old, assumed, min_value, new_value;
    old = *base_address;

    do {
        assumed = old;
        min_value = min(val, (unsigned char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440));
        new_value = __byte_perm(old, min_value, selector);
        if (new_value == old)
            break;
        old = atomicCAS(base_address, assumed, new_value);
    } while (assumed != old);
    return old;
}

__device__ static inline float MsAtomicMin(float *address, const float val) {
  unsigned int *address_as_ui = (unsigned int *)address;  // NOLINT
  unsigned int old = *address_as_ui;                       // NOLINT
  unsigned int assumed;                                    // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address_as_ui, assumed, __float_as_uint(min(val, __uint_as_float(assumed))));
  } while (assumed != old);  // NOLINT
  return __longlong_as_double(old);
}

__device__ static inline half MsAtomicMin(half *address, half val) {
  unsigned int *aligned =
    reinterpret_cast<unsigned int *>(reinterpret_cast<size_t>(address) - (reinterpret_cast<size_t>(address) & 2));
  unsigned int old = *aligned;
  unsigned int assumed;
  unsigned short old_as_us;  // NOLINT
  do {
    assumed = old;
    old_as_us =
      static_cast<unsigned short>(reinterpret_cast<size_t>(address) & 2 ? old >> 16 : old & 0xffff);  // NOLINT
    half min_value = __float2half_rn(min(__half2float(__ushort_as_half(old_as_us)), static_cast<float>(val)));
    unsigned short min_as_us = __half_as_ushort(min_value);  // NOLINT
    unsigned int min_as_ui =
      reinterpret_cast<size_t>(address) & 2 ? (min_as_us << 16) | (old & 0xffff) : (old & 0xffff0000) | min_as_us;
    old = atomicCAS(aligned, assumed, min_as_ui);
  } while (assumed != old);
  __half_raw raw = {old_as_us};
  return half(raw);
}

// atomic mul

__device__ static inline float MsAtomicMul(int *address, const int val) {
  unsigned int *address_as_ui = (unsigned int *)address;  // NOLINT
  unsigned int old = *address_as_ui;                       // NOLINT
  unsigned int assumed;                                    // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address_as_ui, assumed, val * (int)assumed); // NOLINT
  } while (assumed != old);  // NOLINT
  return __longlong_as_double(old);
}

__device__ static inline char MsAtomicMul(char* address, char val) {
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int selector = selectors[(size_t)address & 3];
    unsigned int old, assumed, product, new_value;
    old = *base_address;

    do {
        assumed = old;
        product = val * (char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440); // NOLINT
        new_value = __byte_perm(old, product, selector);
        if (new_value == old)
            break;
        old = atomicCAS(base_address, assumed, new_value);
    } while (assumed != old);
    return old;
}

__device__ static inline unsigned char MsAtomicMul(unsigned char* address, unsigned char val) {
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int selector = selectors[(size_t)address & 3];
    unsigned int old, assumed, product, new_value;
    old = *base_address;

    do {
        assumed = old;
        product = val * (unsigned char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440);
        new_value = __byte_perm(old, product, selector);
        if (new_value == old)
            break;
        old = atomicCAS(base_address, assumed, new_value);
    } while (assumed != old);
    return old;
}

__device__ static inline float MsAtomicMul(float *address, const float val) {
  unsigned int *address_as_ui = (unsigned int *)address;  // NOLINT
  unsigned int old = *address_as_ui;                       // NOLINT
  unsigned int assumed;                                    // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address_as_ui, assumed, __float_as_uint(val * uint_as_float(assumed)));
  } while (assumed != old);  // NOLINT
  return __longlong_as_double(old);
}

__device__ static inline half MsAtomicMul(half *address, half val) {
  unsigned int *aligned =
    reinterpret_cast<unsigned int *>(reinterpret_cast<size_t>(address) - (reinterpret_cast<size_t>(address) & 2));
  unsigned int old = *aligned;
  unsigned int assumed;
  unsigned short old_as_us;  // NOLINT
  do {
    assumed = old;
    old_as_us =
      static_cast<unsigned short>(reinterpret_cast<size_t>(address) & 2 ? old >> 16 : old & 0xffff);  // NOLINT
    // we cast val to float here, otherwise we get a compile error saying there is
    // more than one * operator that matches these operands.
    half product = __float2half_rn(__half2float(__ushort_as_half(old_as_us)) * (float)val); // NOLINT
    unsigned short product_as_us = __half_as_ushort(product);  // NOLINT
    unsigned int product_as_ui = reinterpret_cast<size_t>(address) & 2 ? // NOLINT
        (product_as_us << 16) | (old & 0xffff) :
        (old & 0xffff0000) | product_as_us;
    old = atomicCAS(aligned, assumed, product_as_ui);
  } while (assumed != old);
  __half_raw raw = {old_as_us};
  return half(raw);
}

// atomic div

__device__ static inline float MsAtomicDiv(int *address, const int val) {
  unsigned int *address_as_ui = (unsigned int *)address;  // NOLINT
  unsigned int old = *address_as_ui;                       // NOLINT
  unsigned int assumed;                                    // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address_as_ui, assumed, ((int)assumed) / val); // NOLINT
  } while (assumed != old);  // NOLINT
  return __longlong_as_double(old);
}

__device__ static inline char MsAtomicDiv(char* address, char val) {
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int selector = selectors[(size_t)address & 3];
    unsigned int old, assumed, quotient, new_value;
    old = *base_address;

    do {
        assumed = old;
        quotient = ((char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440)) / val; // NOLINT
        new_value = __byte_perm(old, quotient, selector);
        if (new_value == old)
            break;
        old = atomicCAS(base_address, assumed, new_value);
    } while (assumed != old);
    return old;
}

__device__ static inline unsigned char MsAtomicDiv(unsigned char* address, unsigned char val) {
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int selector = selectors[(size_t)address & 3];
    unsigned int old, assumed, quotient, new_value;
    old = *base_address;

    do {
        assumed = old;
        quotient = ((unsigned char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440)) / val;
        new_value = __byte_perm(old, quotient, selector);
        if (new_value == old)
            break;
        old = atomicCAS(base_address, assumed, new_value);
    } while (assumed != old);
    return old;
}

__device__ static inline float MsAtomicDiv(float *address, const float val) {
  unsigned int *address_as_ui = (unsigned int *)address;  // NOLINT
  unsigned int old = *address_as_ui;                       // NOLINT
  unsigned int assumed;                                    // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address_as_ui, assumed, __float_as_uint(uint_as_float(assumed) / val));
  } while (assumed != old);  // NOLINT
  return __longlong_as_double(old);
}

__device__ static inline half MsAtomicDiv(half *address, half val) {
  unsigned int *aligned =
    reinterpret_cast<unsigned int *>(reinterpret_cast<size_t>(address) - (reinterpret_cast<size_t>(address) & 2));
  unsigned int old = *aligned;
  unsigned int assumed;
  unsigned short old_as_us;  // NOLINT
  do {
    assumed = old;
    old_as_us =
      static_cast<unsigned short>(reinterpret_cast<size_t>(address) & 2 ? old >> 16 : old & 0xffff);  // NOLINT
    // we cast val to float here, otherwise we get a compile error saying there is
    // more than one * operator that matches these operands.
    half product = __float2half_rn(__half2float(__ushort_as_half(old_as_us)) / (float)val); // NOLINT
    unsigned short product_as_us = __half_as_ushort(product);  // NOLINT
    unsigned int product_as_ui = reinterpret_cast<size_t>(address) & 2 ? // NOLINT
        (product_as_us << 16) | (old & 0xffff) :
        (old & 0xffff0000) | product_as_us;
    old = atomicCAS(aligned, assumed, product_as_ui);
  } while (assumed != old);
  __half_raw raw = {old_as_us};
  return half(raw);
}

__device__ __forceinline__ unsigned BallotSync(int predicate, unsigned mask = 0xffffffff) {
  return __ballot_sync(mask, predicate);
}

enum : unsigned { warp_size = 32, log_wap_size = 5 };
__device__ __forceinline__ unsigned LaneId() { return threadIdx.x & (warp_size - 1); }
__device__ __forceinline__ unsigned WarpId(const unsigned &tid) { return tid >> log_wap_size; }

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_UTIL_H_
