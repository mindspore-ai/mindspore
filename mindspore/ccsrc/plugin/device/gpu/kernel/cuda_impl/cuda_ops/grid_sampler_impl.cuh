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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_GRID_SAMPLER_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_GRID_SAMPLER_CUH_
#include <vector>
#include <algorithm>
#include <map>
#include <string>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

enum GridSamplerInterpolationMode { BILINEAR = 0, NEAREST, BICUBIC };

enum GridSamplerPaddingMode { ZEROS = 0, BORDER, REFLECTION };

static std::map<std::string, GridSamplerInterpolationMode> kGridSamplerInterpolationMap{
  {"bilinear", GridSamplerInterpolationMode::BILINEAR},
  {"nearest", GridSamplerInterpolationMode::NEAREST},
  {"bicubic", GridSamplerInterpolationMode::BICUBIC}};

static std::map<std::string, GridSamplerPaddingMode> kGridSamplerPaddingMap{
  {"zeros", GridSamplerPaddingMode::ZEROS},
  {"border", GridSamplerPaddingMode::BORDER},
  {"reflection", GridSamplerPaddingMode::REFLECTION}};

template <typename T>
CUDA_LIB_EXPORT void GridSampler2D(const size_t size, const T *input_addr, const T *grid_addr, T *output_addr,
                                   const std::vector<size_t> &input_shape, const std::vector<size_t> &grid_shape,
                                   const std::vector<size_t> &output_shape, const std::vector<size_t> &input_stride,
                                   const std::vector<size_t> &grid_stride, const std::vector<size_t> &output_stride,
                                   const GridSamplerInterpolationMode interpolation_mode,
                                   const GridSamplerPaddingMode padding_mode, const bool align_corners,
                                   cudaStream_t stream);

template <typename T>
CUDA_LIB_EXPORT void GridSampler3D(const size_t size, const T *input_addr, const T *grid_addr, T *output_addr,
                                   const std::vector<size_t> &input_shape, const std::vector<size_t> &grid_shape,
                                   const std::vector<size_t> &output_shape, const std::vector<size_t> &input_stride,
                                   const std::vector<size_t> &grid_stride, const std::vector<size_t> &output_stride,
                                   const GridSamplerInterpolationMode interpolation_mode,
                                   const GridSamplerPaddingMode padding_mode, const bool align_corners,
                                   cudaStream_t stream);

template <typename T>
static __forceinline__ __device__ T clip_coordinates(T in, int clip_limit) {
  in = in > static_cast<T>(0) ? in : static_cast<T>(0);
  return static_cast<T>(clip_limit - 1) < in ? static_cast<T>(clip_limit - 1) : in;
}

template <typename T>
static __forceinline__ __device__ T reflect_coordinates(T in, int twice_low, int twice_high) {
  if (twice_low != twice_high) {
    T min = static_cast<T>(twice_low) / 2;
    T span = static_cast<T>(twice_high - twice_low) / 2;
    in = ::fabs(in - min);
    // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
    T extra = ::fmod(in, span);
    int flips = static_cast<int>(::floor(in / span));
    if (flips % 2 != 0) {
      return span - extra + min;
    } else {
      return extra + min;
    }
  } else {
    return static_cast<T>(0);
  }
}

template <typename T>
static __forceinline__ __device__ half reflect_coordinates(half in, int twice_low, int twice_high) {
  if (twice_low != twice_high) {
    float min = static_cast<float>(twice_low) / 2;
    float span = static_cast<float>(twice_high - twice_low) / 2;
    float new_in = ::fabs(__half2float(in) - min);
    // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
    float extra = ::fmod(in, span);
    int flips = static_cast<int>(::floor(new_in / span));
    if (flips % 2 != 0) {
      return __float2half(span - extra + min);
    } else {
      return __float2half(extra + min);
    }
  } else {
    return static_cast<half>(0.0);
  }
}

template <typename T>
static __forceinline__ __device__ T safe_downgrade_to_int_range(T x) {
  if (x > static_cast<T>(INT_MAX - 1) || x < static_cast<T>(INT_MIN) || !::isfinite(static_cast<double>(x))) {
    return static_cast<T>(-100.0);
  } else {
    return x;
  }
}

template <typename T>
__device__ __forceinline__ static T cubic_convolution_one(T x, const T A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename T>
__device__ __forceinline__ static T cubic_convolution_two(T x, const T A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

static __forceinline__ __device__ bool within_bounds_2d(int h, int w, int H, int W) {
  return w >= 0 && w < W && h >= 0 && h < H;
}

static __forceinline__ __device__ bool within_bounds_3d(int d, int h, int w, int D, int H, int W) {
  return w >= 0 && w < W && h >= 0 && h < H && d >= 0 && d < D;
}

template <typename T>
static __forceinline__ __device__ T grid_sampler_unnormalize(T coord, const int size, bool align_corners) {
  if (!align_corners) {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1.f) * size - 1) / 2;
  } else {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    return ((coord + 1.f) / 2) * (size - 1);
  }
}

template <typename T>
static __forceinline__ __device__ T compute_coordinates(T coord, const size_t size,
                                                        GridSamplerPaddingMode padding_mode,
                                                        bool align_corners) {
  if (padding_mode == GridSamplerPaddingMode::REFLECTION) {
    if (!align_corners) {
      coord = reflect_coordinates<T>(coord, -1, 2 * size - 1);
    } else {
      coord = reflect_coordinates<T>(coord, 0, 2 * (size - 1));
    }
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == GridSamplerPaddingMode::BORDER) {
    coord = clip_coordinates(coord, size);
  }

  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

template <typename T>
static __forceinline__ __device__ T grid_sampler_compute_source_index(T coord, size_t size,
                                                                      GridSamplerPaddingMode padding_mode,
                                                                      bool align_corners) {
  coord = compute_coordinates(grid_sampler_unnormalize(coord, size, align_corners),
                              size, padding_mode, align_corners);
  return coord;
}

template <typename T>
__device__ __forceinline__ static void get_cubic_upsampling_coefficients(T coeffs[4], T t) {
  const T A = -0.75;

  // opposite coefficients
  T op_x = 1.0 - t;
  coeffs[2] = cubic_convolution_one<T>(op_x, A);
  coeffs[3] = cubic_convolution_two<T>(op_x + 1.0, A);

  T x = t;
  coeffs[0] = cubic_convolution_two<T>(x + 1.0, A);
  coeffs[1] = cubic_convolution_one<T>(x, A);
}

template <typename T, typename S>
__device__ __forceinline__ static T cubic_interp1d(T x0, T x1, T x2, T x3, S t) {
  S coeffs[4];
  get_cubic_upsampling_coefficients<S>(coeffs, t);

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

template <typename S>
__device__ __forceinline__ static half cubic_interp1d(half x0, half x1, half x2, half x3, S t) {
  S coeffs[4];
  get_cubic_upsampling_coefficients<S>(coeffs, t);

  return __float2half(__half2float(x0) * coeffs[0] + __half2float(x1) * coeffs[1] + __half2float(x2) * coeffs[2] +
         __half2float(x3) * coeffs[3]);
}

template <typename T>
static __forceinline__ __device__ T get_value_bounded(const T *data, T x, T y, const size_t W, const size_t H,
                                                      const size_t sW, const size_t sH,
                                                      GridSamplerPaddingMode padding_mode, bool align_corners) {
  int ix = static_cast<int>(compute_coordinates(x, W, padding_mode, align_corners));
  int iy = static_cast<int>(compute_coordinates(y, H, padding_mode, align_corners));

  if (within_bounds_2d(iy, ix, H, W)) {
    return data[iy * sH + ix * sW];
  }
  return static_cast<T>(0);
}

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_GRID_SAMPLER_CUH_
