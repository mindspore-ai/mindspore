/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_UTILS_CPU_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_UTILS_CPU_UTILS_H_

#include <cmath>
#include <utility>
#include <algorithm>
#include <complex>

#include "kernel/cpu/cpu_kernel.h"
#include "base/float16.h"
#include "kernel/cpu/utils/visible.h"

namespace mindspore {
namespace kernel {
OPS_HOST_API void ForceLinkOpsHost();
template <typename S, typename T>
void Cast(const S *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if constexpr ((std::is_same_v<S, std::complex<float>>) || (std::is_same_v<S, std::complex<double>>)) {
        if constexpr ((std::is_same_v<T, std::complex<float>>) || (std::is_same_v<T, std::complex<double>>)) {
          out[i] = static_cast<T>(in[i]);
        } else {
          out[i] = static_cast<T>(std::real(in[i]));
        }
      } else if constexpr ((std::is_same_v<T, std::complex<float>>) || (std::is_same_v<T, std::complex<double>>)) {
        double realValue = static_cast<double>(in[i]);
        std::complex<double> complexValue(realValue, 0.0);
        out[i] = (std::is_same_v<T, std::complex<float>>) ? static_cast<T>(complexValue) : complexValue;
      } else {
        out[i] = static_cast<T>(in[i]);
      }
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
}

template <typename S, typename T>
void CastKernelTensor(KernelTensor *source, KernelTensor *target) {
  MS_EXCEPTION_IF_NULL(source);
  MS_EXCEPTION_IF_NULL(source->device_ptr());
  S *source_addr = reinterpret_cast<S *>(source->device_ptr());
  MS_EXCEPTION_IF_NULL(target);
  MS_EXCEPTION_IF_NULL(target->device_ptr());
  T *target_addr = reinterpret_cast<T *>(target->device_ptr());
  Cast(source_addr, target_addr, source->size() / sizeof(S));
}

template <typename T>
inline T offset_to_index_init(T offset) {
  return offset;
}

template <typename T, typename... Args>
inline T offset_to_index_init(T offset, T *x, const T &X, Args &&...args) {
  offset = offset_to_index_init(offset, std::forward<Args>(args)...);
  *x = offset % X;
  return offset / X;
}

inline bool offset_to_index_step() { return true; }

template <typename T, typename... Args>
inline bool offset_to_index_step(T *x, const T &X, Args &&...args) {
  if (offset_to_index_step(std::forward<Args>(args)...)) {
    *x = ((*x + 1) == X) ? 0 : (*x + 1);
    return *x == 0;
  }
  return false;
}

// compatible with MSVC
template <typename T>
inline bool IsNan(T x) {
  return std::isnan(x);
}

template <>
inline bool IsNan<float16>(float16 x) {
  return isnan(x);
}

#ifdef _MSC_VER
template <>
inline bool IsNan<int8_t>(int8_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<uint8_t>(uint8_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<int16_t>(int16_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<uint16_t>(uint16_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<int32_t>(int32_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<uint32_t>(uint32_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<int64_t>(int64_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<uint64_t>(uint64_t x) {
  return isnan(static_cast<double>(x));
}
#endif

inline bool DataIndexStep() { return true; }

template <typename T, typename... Args>
inline bool DataIndexStep(T *x, const T *X, Args &&...args) {
  if (DataIndexStep(std::forward<Args>(args)...)) {
    *x = ((*x + 1) == *X) ? 0 : (*x + 1);
    return *x == 0;
  }
  return false;
}

template <typename T>
inline T DataIndexInit(const T *offset) {
  return *offset;
}

template <typename T, typename... Args>
inline T DataIndexInit(T *offset, T *x, const T *X, Args &&...args) {
  auto off = DataIndexInit(offset, std::forward<Args>(args)...);
  *x = off % *X;
  return off / *X;
}

template <typename T>
inline T ComputeLerp(T top_left, T top_right, T bottom_left, T bottom_right, T x_lerp, T y_lerp) {
  T top = top_left + (top_right - top_left) * x_lerp;
  T bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

template <typename T>
struct HalfPixelFunc {
  T operator()(const T &new_x, const int &old_length, const int &new_length) const {
    constexpr auto half_pixel = 0.5;
    return new_length > 1 ? (new_x + half_pixel) * old_length / new_length - half_pixel : 0;
  }
};

template <typename T>
struct AlignCornersFunc {
  T operator()(const T &new_x, const int &old_length, const int &new_length) const {
    return new_length != 1 ? new_x * (old_length - 1) / (new_length - 1) : 0;
  }
};

struct CachedInterpolation {
  size_t lower;
  size_t upper;
  float lerp;
};

template <typename T>
inline T AreaPixelComputeSourceIndex(T scale, int64_t dst_index, bool align_corners) {
  if (align_corners) {
    return scale * static_cast<T>(dst_index);
  } else {
    constexpr T zero = 0.;
    T src_idx = scale * (LongToDouble(dst_index) + 0.5) - 0.5;
    return src_idx < zero ? zero : src_idx;
  }
}

template <typename T>
inline void ComputeSourceIndexAndLambda(int64_t *const input_index0, int64_t *const input_index1, T *const lambda0,
                                        T *const lambda1, T ratio, int64_t output_index, int64_t input_size,
                                        int64_t output_size, bool align_corners) {
  if (output_size == input_size) {
    // scale_factor = 1
    *input_index0 = output_index;
    *input_index1 = output_index;
    *lambda0 = static_cast<T>(1);
    *lambda1 = static_cast<T>(0);
  } else {
    const T real_input_index = AreaPixelComputeSourceIndex<T>(ratio, output_index, align_corners);
    *input_index0 = static_cast<int64_t>(real_input_index);
    int64_t offset = (*input_index0 < input_size - 1) ? 1 : 0;
    *input_index1 = *input_index0 + offset;
    *lambda1 = real_input_index - static_cast<T>(*input_index0);
    constexpr T one = 1.0;
    *lambda0 = one - *lambda1;
  }
}
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_UTILS_CPU_UTILS_H_
