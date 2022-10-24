/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <iostream>
constexpr int kWarpSize = 32;
constexpr int64_t MAX_GRID_SIZE = 65535LL;

template <typename T>
__host__ __device__ __forceinline__ T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}
template <typename T>
__host__ __device__ __forceinline__ T RoundUp(T a, T b) {
  return CeilDiv(a, b) * b;
}

static dim3 getGrid(int64_t slices_number) {
  int64_t gridX = slices_number > MAX_GRID_SIZE ? MAX_GRID_SIZE : slices_number;
  int64_t gridY = 1;
  int64_t gridZ = 1;

  if (slices_number > MAX_GRID_SIZE) {
    slices_number = CeilDiv(slices_number, MAX_GRID_SIZE);
    gridY = slices_number > MAX_GRID_SIZE ? MAX_GRID_SIZE : slices_number;

    if (slices_number > MAX_GRID_SIZE) {
      slices_number = CeilDiv(slices_number, MAX_GRID_SIZE);
      gridZ = slices_number > MAX_GRID_SIZE ? MAX_GRID_SIZE : slices_number;
    }
  }

  return dim3(gridX, gridY, gridZ);
}

__device__ __forceinline__ int64_t getLinearBlockId() {
  return blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
}

__device__ __forceinline__ int getLaneId() {
  int laneId;
  asm("mov.s32 %0, %%laneid;" : "=r"(laneId));
  return laneId;
}

template <typename T>
struct Bitvalue {};

template <>
struct Bitvalue<uint32_t> {
  static __device__ __forceinline__ uint32_t getBitvalue(uint32_t val, int pos, int len) {
    uint32_t ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
    return ret;
  }

  static __device__ __forceinline__ uint32_t setBitvalue(uint32_t val, uint32_t toInsert, int pos, int len) {
    uint32_t ret;
    asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
    return ret;
  }
};

template <>
struct Bitvalue<uint64_t> {
  static __device__ __forceinline__ uint64_t getBitvalue(uint64_t val, int pos, int len) {
    uint64_t ret;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
    return ret;
  }

  static __device__ __forceinline__ uint64_t setBitvalue(uint64_t val, uint64_t toInsert, int pos, int len) {
    uint64_t ret;
    asm("bfi.b64 %0, %1, %2, %3, %4;" : "=l"(ret) : "l"(toInsert), "l"(val), "r"(pos), "r"(len));
    return ret;
  }
};

template <typename scalar_t>
struct NthElementType {};

template <>
struct NthElementType<float> {
  typedef uint32_t TargetType;

  static inline __device__ TargetType change(float v) {
    TargetType x = __float_as_int(v);
    TargetType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

    return (v == v) ? (x ^ mask) : 0xffffffff;
  }

  static inline __device__ float recover(TargetType v) {
    TargetType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

    return __int_as_float(v ^ mask);
  }
};
template <>
struct NthElementType<uint8_t> {
  typedef uint32_t TargetType;

  static inline __device__ TargetType change(uint8_t v) { return v; }

  static inline __device__ uint8_t recover(TargetType v) { return v; }
};

template <>
struct NthElementType<int8_t> {
  typedef uint32_t TargetType;

  static inline __device__ TargetType change(int8_t v) { return 128u + v; }

  static inline __device__ int8_t recover(TargetType v) { return v - 128; }
};

template <>
struct NthElementType<int16_t> {
  typedef uint32_t TargetType;

  static inline __device__ TargetType change(int16_t v) {
    static_assert(sizeof(int16_t) == 2, "");
    return 32768u + v;
  }

  static inline __device__ int16_t recover(TargetType v) { return v - 32768; }
};

template <>
struct NthElementType<int32_t> {
  typedef uint32_t TargetType;

  static inline __device__ TargetType change(int32_t v) {
    static_assert(sizeof(int) == 4, "");
    return 2147483648u + v;
  }

  static inline __device__ int32_t recover(TargetType v) { return v - 2147483648u; }
};

template <>
struct NthElementType<int64_t> {
  typedef uint64_t TargetType;

  static inline __device__ TargetType change(int64_t v) {
    static_assert(sizeof(int64_t) == 8, "");
    return 9223372036854775808ull + v;
  }

  static inline __device__ int64_t recover(TargetType v) { return v - 9223372036854775808ull; }
};

template <>
struct NthElementType<double> {
  typedef uint64_t TargetType;

  static inline __device__ TargetType change(double v) {
    TargetType x = __double_as_longlong(v);
    TargetType mask = -((x >> 63)) | 0x8000000000000000;
    return (v == v) ? (x ^ mask) : 0xffffffffffffffff;
  }

  static inline __device__ double recover(TargetType v) {
    TargetType mask = ((v >> 63) - 1) | 0x8000000000000000;
    return __longlong_as_double(v ^ mask);
  }
};

template <>
struct NthElementType<half> {
  typedef uint32_t TargetType;

  static inline __device__ TargetType change(half v) {
    TargetType x = __half_as_ushort(v);
    TargetType mask = (x & 0x00008000) ? 0x0000ffff : 0x00008000;
    return (v == v) ? (x ^ mask) : 0xffff;
  }

  static inline __device__ half recover(TargetType v) {
    TargetType mask = (v & 0x00008000) ? 0x00008000 : 0x0000ffff;
    return __ushort_as_half(v ^ mask);
  }
};

template <>
struct NthElementType<uint16_t> {
  typedef uint32_t TargetType;

  static inline __device__ TargetType change(uint16_t v) {
    TargetType x = v;
    TargetType mask = (x & 0x00008000) ? 0x0000ffff : 0x00008000;
    return (v == v) ? (x ^ mask) : 0xffff;
  }

  static inline __device__ uint16_t recover(TargetType v) {
    TargetType mask = (v & 0x00008000) ? 0x00008000 : 0x0000ffff;
    uint16_t r;
    r = (v ^ mask);
    return r;
  }
};
