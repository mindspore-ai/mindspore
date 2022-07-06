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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_GRID_SAMPLER_GRAD_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_GRID_SAMPLER_GRAD_IMPL_CUH_
#include <algorithm>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/grid_sampler_impl.cuh"

template <typename T>
CUDA_LIB_EXPORT void GridSampler2DGrad(const size_t size, const size_t dinput_size,
                                       const size_t dgrid_size, T *grad_addr, T *input_addr,
                                       T *grid_addr, T *dinput_addr, T *dgrid_addr,
                                       const std::vector<size_t> &grad_shape,
                                       const std::vector<size_t> &input_shape,
                                       const std::vector<size_t> &grid_shape,
                                       const std::vector<size_t> &dinput_shape,
                                       const std::vector<size_t> &dgrid_shape,
                                       const std::vector<size_t> &grad_stride,
                                       const std::vector<size_t> &input_stride,
                                       const std::vector<size_t> &grid_stride,
                                       const std::vector<size_t> &dinput_stride,
                                       const std::vector<size_t> &dgrid_stride,
                                       const GridSamplerInterpolationMode interpolation_mode,
                                       const GridSamplerPaddingMode padding_mode,
                                       const bool align_corners,
                                       cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT void GridSampler3DGrad(const size_t size, const size_t dinput_size,
                                       const size_t dgrid_size, T *grad_addr, T *input_addr,
                                       T *grid_addr, T *dinput_addr, T *dgrid_addr,
                                       const std::vector<size_t> &grad_shape,
                                       const std::vector<size_t> &input_shape,
                                       const std::vector<size_t> &grid_shape,
                                       const std::vector<size_t> &dinput_shape,
                                       const std::vector<size_t> &dgrid_shape,
                                       const std::vector<size_t> &grad_stride,
                                       const std::vector<size_t> &input_stride,
                                       const std::vector<size_t> &grid_stride,
                                       const std::vector<size_t> &dinput_stride,
                                       const std::vector<size_t> &dgrid_stride,
                                       const GridSamplerInterpolationMode interpolation_mode,
                                       const GridSamplerPaddingMode padding_mode,
                                       const bool align_corners,
                                       cudaStream_t cuda_stream);

template <typename T>
static __forceinline__ __device__
T grid_sampler_unnormalize_set_grad(T coord, int size,
                                    bool align_corners, T *din) {
  if (!align_corners) {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    *din = static_cast<T>(size) / 2;
    return ((coord + 1.f) * size - 1) / 2;
  } else {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    *din = static_cast<T>(size - 1) / 2;
    return ((coord + 1.f) / 2) * (size - 1);
  }
}

template <typename T>
static __forceinline__ __device__
T clip_coordinates_set_grad(T in, int clip_limit, T *din) {
  if (in > static_cast<T>(0)) {
    T max = static_cast<T>(clip_limit - 1);
    if (in >= max) {
      *din = static_cast<T>(0);
      return max;
    } else {
      *din = static_cast<T>(1);
      return in;
    }
  } else {
    *din = static_cast<T>(0);
    return static_cast<T>(0);
  }
}

template <typename T>
static __forceinline__ __device__
T reflect_coordinates_set_grad(T in, int twice_low, int twice_high,
                               T *din) {
  if (twice_low != twice_high) {
    int din_mult_;
    T min = static_cast<T>(twice_low) / 2;
    T span = static_cast<T>(twice_high - twice_low) / 2;
    in = in - min;
    if (in >= static_cast<T>(0)) {
      din_mult_ = 1;
    } else {
      din_mult_ = -1;
      in = -in;
    }
    // `fmod` returns same sign as `in`, which is positive after the `if` above.
    T extra = ::fmod(in, span);
    int flips = static_cast<int>(::floor(in / span));
    if (flips % 2 != 0) {
      *din = static_cast<T>(-din_mult_);
      return span - extra + min;
    } else {
      *din = static_cast<T>(din_mult_);
      return extra + min;
    }
  } else {
    *din = static_cast<T>(0);
    return static_cast<T>(0);
  }
}

template <typename T>
static __forceinline__ __device__
T reflect_coordinates_set_grad(half in, int twice_low, int twice_high,
                               half *din) {
  if (twice_low != twice_high) {
    int din_mult_;
    float min = static_cast<float>(twice_low) / 2;
    float span = static_cast<float>(twice_high - twice_low) / 2;
    float new_in = __half2float(in) - min;
    if (in >= static_cast<T>(0)) {
      din_mult_ = 1;
    } else {
      din_mult_ = -1;
      new_in = -new_in;
    }
    // `fmod` returns same sign as `in`, which is positive after the `if` above.
    float extra = ::fmod(new_in, span);
    int flips = static_cast<int>(::floor(new_in / span));
    if (flips % 2 != 0) {
      *din = static_cast<T>(-din_mult_);
      return __float2half(span - extra + min);
    } else {
      *din = static_cast<T>(din_mult_);
      return __float2half(extra + min);
    }
  } else {
    *din = static_cast<T>(0);
    return static_cast<T>(0);
  }
}

// Calculate the differential of the cubic convolution, i.e. `d coeff / d x`
template<typename T>
static __forceinline__ __device__
void get_cubic_coefficients_grad(
    T coeffs[4],
    T t) {
  const T A = -0.75;

  T x;
  x = -1 - t;  // 1 < x = |-1 - tx| < 2
  coeffs[0] = (-3 * A * x - 10 * A) * x - 8 * A;
  x = -t;     // x = |0 - tx| <= 1
  coeffs[1] = (-3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 1 - t;  // x = |1 - tx| <= 1
  coeffs[2] = (3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 2 - t;  // 1 < x = |2 - tx| < 2
  coeffs[3] = (3 * A * x - 10 * A) * x + 8 * A;
}

template <typename T>
static __forceinline__ __device__
T grid_sampler_compute_source_index_set_grad(
    T coord,
    int size,
    GridSamplerPaddingMode padding_mode,
    bool align_corners,
    T *din) {
  T dclip, drefl;
  coord = grid_sampler_unnormalize_set_grad(coord, size, align_corners, din);
  if (padding_mode == GridSamplerPaddingMode::REFLECTION) {
    // reflect coordinates by image borders
    if (!align_corners) {
      coord = reflect_coordinates_set_grad<T>(coord, -1, 2*size - 1, &drefl);
    } else {
      coord = reflect_coordinates_set_grad<T>(coord, 0, 2*(size - 1), &drefl);
    }
    // clip coordinates to image borders
    coord = clip_coordinates_set_grad(coord, size, &dclip);
    *din = (*din) * drefl * dclip;
  } else if (padding_mode == GridSamplerPaddingMode::BORDER) {
    // clip coordinates to image borders
    coord = clip_coordinates_set_grad(coord, size, &dclip);
    *din = (*din) * dclip;
  }

  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_GRID_SAMPLER_GRAD_IMPL_CUH_
