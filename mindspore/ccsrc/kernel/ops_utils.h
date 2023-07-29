/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_KERNEL_OPS_UTILS_H_
#define MINDSPORE_CCSRC_KERNEL_OPS_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include "kernel/kernel.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
BACKEND_EXPORT float Scaling(size_t in_size, size_t out_size, bool align_corners);
float ScaleGrid(const int x, const float scale);
inline float Scaler(const size_t x, const float scale, bool half_pixel_centers) {
  if (half_pixel_centers) {
    /**
     * function with a std::floor(), so instead of subtracting the 0.5 as we
     * do in HalfPixelScale, we leave it as is, as the std::floor does the
     * correct thing.
     * */
    return (static_cast<float>(x) + 0.5f) * scale;
  } else {
    /**
     * Older incorrect scaling method that causes all resizes to have a slight
     * translation leading to inconsistent results. For example, a flip then a
     * resize gives different results then a resize then a flip.
     * */
    return static_cast<float>(x) * scale;
  }
}

struct CachedInterpolation {
  size_t lower;
  size_t upper;
  float lerp;
};

template <typename T>
struct AlignCornersFunc {
  T operator()(const T &new_x, const int &old_length, const int &new_length) const {
    return new_length != 1 ? new_x * (old_length - 1) / (new_length - 1) : 0;
  }
};

template <typename T>
struct HalfPixelFunc {
  T operator()(const T &new_x, const int &old_length, const int &new_length) const {
    constexpr auto half_pixel = 0.5;
    return new_length > 1 ? (new_x + half_pixel) * old_length / new_length - half_pixel : 0;
  }
};

void ComputeInterpolationWeights(const size_t out_size, const size_t in_size, const float scale,
                                 CachedInterpolation *interpolation, bool half_pixel_centers);

template <typename T>
inline T ComputeLerp(T top_left, T top_right, T bottom_left, T bottom_right, T x_lerp, T y_lerp) {
  T top = top_left + (top_right - top_left) * x_lerp;
  T bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

BACKEND_EXPORT std::vector<bool> Dec2Bin(const int64_t &mask);
BACKEND_EXPORT void FillEmptyDims(const BaseOperatorPtr &base_operator, std::vector<int64_t> *begin,
                                  std::vector<int64_t> *end, std::vector<int64_t> *stride, ShapeVector *input_shape,
                                  bool is_gpu_strided = false);
BACKEND_EXPORT void ParseStrideSliceMasks(const BaseOperatorPtr &base_operator, std::vector<int64_t> *begin,
                                          std::vector<int64_t> *end, std::vector<int64_t> *stride,
                                          const ShapeVector &input_shape);

template <typename T>
inline T ComputeScales(const double &scale, const size_t &input_size, const size_t &output_size) {
  if (scale > 0.) {
    return static_cast<T>(1.0 / scale);
  } else if (output_size > 0) {
    return (static_cast<T>(input_size) / output_size);
  }
  return 0;
}

template <typename T>
inline T ComputeScalesBackward(const double scale, const int64_t src_size, const int64_t dst_size) {
  if (scale > 0.) {
    return static_cast<T>(scale);
  } else if (dst_size > 0) {
    return static_cast<T>(src_size) / dst_size;
  }
  return 0;
}

inline size_t NearestNeighborSourceIndex(const float &scale, const size_t &dst_index, const size_t &input_size) {
  size_t src_index = std::min(static_cast<size_t>(floorf(SizeToFloat(dst_index) * scale)), input_size - 1);
  return src_index;
}

inline size_t NearestIndex(const size_t &output_index, const size_t &input_size, const size_t &output_size,
                           const double &scales) {
  constexpr size_t kNumberTwo = 2;
  if (output_size == input_size) {
    // scale_factor = 1
    return output_index;
  } else if (output_size == kNumberTwo * input_size) {
    // scale_factor = 2, shift input index
    return output_index >> 1;
  } else {
    auto scale = ComputeScales<float>(scales, input_size, output_size);
    return NearestNeighborSourceIndex(scale, output_index, input_size);
  }
}

template <typename T>
inline T AreaPixelComputeScale(int64_t input_size, int64_t output_size, bool align_corners, double scale) {
  if (align_corners) {
    if (output_size > 1) {
      return static_cast<T>(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<T>(0);
    }
  } else {
    return ComputeScales<T>(scale, input_size, output_size);
  }
}

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
inline T DataIndexInit(const T *offset) {
  return *offset;
}

template <typename T, typename... Args>
inline T DataIndexInit(T *offset, T *x, const T *X, Args &&... args) {
  auto off = DataIndexInit(offset, std::forward<Args>(args)...);
  *x = off % *X;
  return off / *X;
}

inline bool DataIndexStep() { return true; }

template <typename T, typename... Args>
inline bool DataIndexStep(T *x, const T *X, Args &&... args) {
  if (DataIndexStep(std::forward<Args>(args)...)) {
    *x = ((*x + 1) == *X) ? 0 : (*x + 1);
    return *x == 0;
  }
  return false;
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

BACKEND_EXPORT void CheckSliceValid(const std::vector<int64_t> &start, const std::vector<int64_t> &stop,
                                    const std::vector<int64_t> &step, const std::vector<int64_t> &input_shape);
BACKEND_EXPORT size_t CalOffset(const std::vector<int64_t> &start, const std::vector<int64_t> &stop,
                                const std::vector<int64_t> &dim_offset);
BACKEND_EXPORT std::vector<int64_t> CalDimOffset(const std::vector<int64_t> &input_shape);
BACKEND_EXPORT size_t GetCopySize(const std::vector<int64_t> &dim_offset, const std::vector<int64_t> &start,
                                  const std::vector<int64_t> &stop);

BACKEND_EXPORT std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment> GetAlignments(const std::string &alignment);

namespace broadcast_utils {
BACKEND_EXPORT bool AlignedBroadCastShape(size_t align_rank, std::vector<size_t> *broadcast, std::vector<size_t> *lhs,
                                          std::vector<size_t> *rhs);
}  // namespace broadcast_utils

#define CHECK_KERNEL_WORKSPACE_SIZE(actual_size, expect_size, kernel_name)                                           \
  do {                                                                                                               \
    if ((actual_size) != (expect_size)) {                                                                            \
      MS_LOG(EXCEPTION) << (kernel_name) << " requires " << (expect_size) << " workspace, but got " << (actual_size) \
                        << ".";                                                                                      \
    }                                                                                                                \
  } while (0)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_OPS_UTILS_H_
