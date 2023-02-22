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

#include "plugin/device/cpu/kernel/resize_v2_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include <unordered_map>
#include "mindspore/core/ops/resize_v2.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "kernel/common_utils.h"

namespace mindspore::kernel {
namespace {
constexpr auto kResizeV2 = "ResizeV2";
constexpr float half = 0.5;
constexpr float kCubicCoeffA = -0.75;
constexpr size_t kCubicOne = 1;
constexpr size_t kCubicTwo = 2;
constexpr size_t kCubicThree = 3;
constexpr size_t kCubicFour = 4;
constexpr size_t kCubicFive = 5;
constexpr size_t kCubicEight = 8;
constexpr float kCubicScopeTwo = 2.0;
constexpr size_t kResizeV2InputsSizesSize = 4;
}  // namespace

struct CachedInterpolationCubic {
  int64_t lowest;  // Lowest source index used in the interpolation
  int64_t lower;   // Lower source index used in the interpolation
  int64_t upper;   // Upper source index used in the interpolation
  int64_t uppest;  // Uppest source index used in the interpolation
  float lerp;
};

float ResizeV2CpuKernelMod::ComputeScale(size_t in_size, size_t out_size, bool align_corners) {
  if (align_corners) {
    if (out_size > 1) {
      return static_cast<float>(in_size - 1) / (out_size - 1);
    } else {
      return static_cast<float>(0);
    }
  } else {
    return static_cast<float>(in_size / static_cast<float>(out_size));
  }
}

static float ComputeSourceIndex(const int x, const float scale, bool align_corners) {
  if (align_corners) {
    return static_cast<float>(x) * scale;
  } else {
    return (static_cast<float>(x) + half) * scale - half;
  }
}

static float ComputeSourceIndexByNearest(const int x, const float scale) { return static_cast<float>(x) * scale; }

// Nearest ComputeInterpolationWeights
static void ComputeInterpolationWeightsByNearest(const size_t out_size, const size_t in_size, const float scale,
                                                 CachedInterpolation *interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (size_t i = 0; i <= out_size - 1; ++i) {
    const float in = ComputeSourceIndexByNearest(i, scale);
    const float in_f = std::floor(in);
    interpolation[i].lower = (in_f > 0) ? static_cast<size_t>(in_f) : 0;
    interpolation[i].upper = std::min(static_cast<size_t>(std::ceil(in)), in_size - 1);
    interpolation[i].lerp = in - in_f;
  }
}

// Linear ComputeInterpolationWeights
static void ComputeInterpolationWeightsByLinear(const size_t out_size, const size_t in_size, const float scale,
                                                bool align_corners_, CachedInterpolation *interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (size_t i = 0; i <= out_size - 1; ++i) {
    const float in = ComputeSourceIndex(i, scale, align_corners_);
    const float in_f = std::floor(in);
    interpolation[i].lower = (in_f > 0) ? static_cast<size_t>(in_f) : 0;
    interpolation[i].upper = std::min(static_cast<size_t>(std::ceil(in)), in_size - 1);
    interpolation[i].lerp = in - in_f;
  }
}

// Cubic ComputeInterpolationWeights
static void ComputeInterpolationWeightsByCubic(const size_t out_size, const size_t in_size, const float scale,
                                               bool align_corners, CachedInterpolationCubic *interpolation) {
  interpolation[out_size].lowest = 0;
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  interpolation[out_size].uppest = 0;
  for (size_t i = 0; i <= out_size - 1; ++i) {
    const float in = ComputeSourceIndex(i, scale, align_corners);
    const float in_f = std::floor(in);
    const float in_c = std::ceil(in);
    interpolation[i].lowest = (in_f - 1 > 0) ? static_cast<size_t>(in_f - 1) : 0;
    interpolation[i].lower = (in_f > 0) ? static_cast<size_t>(in_f) : 0;
    interpolation[i].upper = (in_c < in_size - 1) ? static_cast<size_t>(in_c) : in_size - 1;
    interpolation[i].uppest = (in_c + 1 < in_size - 1) ? static_cast<size_t>(in_c + 1) : in_size - 1;
    interpolation[i].lerp = in - in_f;
  }
}

// data[4] = {top_left, top_right, bottom_left, bottom_right}
template <typename T>
static inline T ComputeLerpByNearest(T *data, float x_lerp, float y_lerp, const std::string &nearest_mode) {
  T top_left = data[0];
  if (nearest_mode != "floor") {
    MS_LOG(ERROR) << "For 'ResizeV2', nearest_mode must be 'floor'.";
  }
  return top_left;
}

// data[4] = {top_left, top_right, bottom_left, bottom_right}
template <typename T>
static inline T ComputeLerpByBiLinear(T *data, float x_lerp, float y_lerp) {
  float top_left = static_cast<float>(data[0]);
  float top_right = static_cast<float>(data[1]);
  float bottom_left = static_cast<float>(data[2]);
  float bottom_right = static_cast<float>(data[3]);
  float top = top_left + (top_right - top_left) * x_lerp;
  float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return static_cast<T>(top + (bottom - top) * y_lerp);
}

static float GetBiCubicWeight(float lerp, float a) {
  if (a != kCubicCoeffA) {
    MS_LOG(ERROR) << "For 'ResizeV2', cubic_coeff_a must be -0.75.";
  }
  if (lerp <= 1.0) {
    return ((a + kCubicTwo) * lerp - (a + kCubicThree)) * lerp * lerp + kCubicOne;
  } else if (lerp < kCubicScopeTwo) {
    return ((a * lerp - kCubicFive * a) * lerp + kCubicEight * a) * lerp - kCubicFour * a;
  } else {
    return 0.0;
  }
}

// data[16]
template <typename T>
static inline T ComputeLerpByBiCubic(T *data, float x_lerp, float y_lerp, float a) {
  float w_x_0 = GetBiCubicWeight(1 + x_lerp, a);
  float w_x_1 = GetBiCubicWeight(x_lerp, a);
  float w_x_2 = GetBiCubicWeight(1 - x_lerp, a);
  float w_x_3 = GetBiCubicWeight(2 - x_lerp, a);

  float w_y_0 = GetBiCubicWeight(1 + y_lerp, a);
  float w_y_1 = GetBiCubicWeight(y_lerp, a);
  float w_y_2 = GetBiCubicWeight(1 - y_lerp, a);
  float w_y_3 = GetBiCubicWeight(2 - y_lerp, a);

  float result = static_cast<float>(data[0]) * w_y_0 * w_x_0 + static_cast<float>(data[1]) * w_y_0 * w_x_1 +
                 static_cast<float>(data[2]) * w_y_0 * w_x_2 + static_cast<float>(data[3]) * w_y_0 * w_x_3 +
                 static_cast<float>(data[4]) * w_y_1 * w_x_0 + static_cast<float>(data[5]) * w_y_1 * w_x_1 +
                 static_cast<float>(data[6]) * w_y_1 * w_x_2 + static_cast<float>(data[7]) * w_y_1 * w_x_3 +
                 static_cast<float>(data[8]) * w_y_2 * w_x_0 + static_cast<float>(data[9]) * w_y_2 * w_x_1 +
                 static_cast<float>(data[10]) * w_y_2 * w_x_2 + static_cast<float>(data[11]) * w_y_2 * w_x_3 +
                 static_cast<float>(data[12]) * w_y_3 * w_x_0 + static_cast<float>(data[13]) * w_y_3 * w_x_1 +
                 static_cast<float>(data[14]) * w_y_3 * w_x_2 + static_cast<float>(data[15]) * w_y_3 * w_x_3;
  return static_cast<T>(result);
}

bool ResizeV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ResizeV2>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();

  std::string coordinate_transformation_mode = kernel_ptr->get_coordinate_transformation_mode();
  if (coordinate_transformation_mode == "align_corners") {
    align_corners_ = true;
  } else if (coordinate_transformation_mode == "pytorch_half_pixel") {
    align_corners_ = false;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', coordinate_transformation_mode: " << coordinate_transformation_mode
                  << "is not support.";
    return false;
  }

  std::string mode = kernel_ptr->get_mode();
  if (mode == "nearest") {
    mode_ = "nearest";
    align_corners_ = false;
  } else if (mode == "linear") {
    mode_ = "linear";
  } else if (mode == "cubic") {
    mode_ = "cubic";
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', mode: " << mode << "is not support.";
    return false;
  }

  TypeId input_dtype = inputs[0]->GetDtype();
  if (mode_ != "nearest") {
    if (input_dtype != kNumberTypeFloat16 && input_dtype != kNumberTypeFloat32 && input_dtype != kNumberTypeFloat64) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' linear and cubic mode only support float16, float32, float64.";
      return false;
    }
  }
  sizes_dtype_ = inputs[kIndex3]->GetDtype();
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int ResizeV2CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != 0) {
    return ret;
  }
  std::vector<int64_t> shape = inputs[kIndex0]->GetShapeVector();
  batch_size_ = LongToSize(shape[kIndex0]);
  channel_ = LongToSize(shape[kIndex1]);
  in_height_ = LongToSize(shape[kIndex2]);
  in_width_ = LongToSize(shape[kIndex3]);

  return KRET_OK;
}

template <typename T>
bool ResizeV2CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  T *input_addr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  T *output_addr = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  MS_ERROR_IF_NULL_W_RET_VAL(input_addr, false);
  MS_ERROR_IF_NULL_W_RET_VAL(output_addr, false);

  auto sizes = inputs[kIndex3];
  if (sizes_dtype_ == kNumberTypeInt64) {
    int64_t *sizes_data = reinterpret_cast<int64_t *>(sizes->addr);
    MS_ERROR_IF_NULL_W_RET_VAL(sizes_data, false);
    out_height_ = LongToSize(sizes_data[kIndex2]);
    out_width_ = LongToSize(sizes_data[kIndex3]);
  } else {
    int32_t *sizes_data = reinterpret_cast<int32_t *>(sizes->addr);
    MS_ERROR_IF_NULL_W_RET_VAL(sizes_data, false);
    std::vector<int64_t> sizes_v;
    sizes_v.push_back(static_cast<int64_t>(sizes_data[kIndex2]));
    sizes_v.push_back(static_cast<int64_t>(sizes_data[kIndex3]));
    out_height_ = LongToSize(sizes_v[kIndex0]);
    out_width_ = LongToSize(sizes_v[kIndex1]);
  }

  bc_ = batch_size_ * channel_;
  out_hw_size_ = out_height_ * out_width_;
  in_hw_size_ = in_height_ * in_width_;
  bhwc_size_ = in_hw_size_ * channel_ * batch_size_;

  if (out_height_ == in_height_ && out_width_ == in_width_) {
    for (size_t i = 0; i < bhwc_size_; ++i) {
      output_addr[i] = input_addr[i];
    }
    return true;
  }

  if (mode_ == "nearest") {
    align_corners_ = false;
  }

  height_scale_ = ComputeScale(in_height_, out_height_, align_corners_);
  width_scale_ = ComputeScale(in_width_, out_width_, align_corners_);

  bool result = false;
  if (mode_ == "nearest") {
    result = LaunchKernelByNearest<T>(inputs, outputs);
  } else if (mode_ == "linear") {
    result = LaunchKernelByLinear<T>(inputs, outputs);
  } else {
    result = LaunchKernelByCubic<T>(inputs, outputs);
  }

  return result;
}

template <typename T>
bool ResizeV2CpuKernelMod::LaunchKernelByNearest(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  T *input_addr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  T *output_addr = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  std::vector<CachedInterpolation> ys(out_height_ + 1);
  std::vector<CachedInterpolation> xs(out_width_ + 1);

  ComputeInterpolationWeightsByNearest(out_height_, in_height_, height_scale_, ys.data());
  ComputeInterpolationWeightsByNearest(out_width_, in_width_, width_scale_, xs.data());

  auto task = [input_addr, output_addr, ys, xs, this](size_t start, size_t end) {
    for (size_t c = start; c < end; ++c) {
      auto c_input_addr = input_addr + c * in_hw_size_;
      auto c_output_addr = output_addr + c * out_hw_size_;
      for (size_t h = 0; h < out_height_; ++h) {
        const T *ys_input_lower_ptr = c_input_addr + ys[h].lower * in_width_;
        const T *ys_input_upper_ptr = c_input_addr + ys[h].upper * in_width_;
        const float ys_lerp = static_cast<float>(ys[h].lerp);
        for (size_t w = 0; w < out_width_; ++w) {
          const size_t xs_lower = xs[w].lower;
          const size_t xs_upper = xs[w].upper;
          const float xs_lerp = static_cast<float>(xs[w].lerp);
          const T top_left(ys_input_lower_ptr[xs_lower]);
          const T top_right(ys_input_lower_ptr[xs_upper]);
          const T bottom_left(ys_input_upper_ptr[xs_lower]);
          const T bottom_right(ys_input_upper_ptr[xs_upper]);
          int64_t output_index = h * out_width_ + w;
          T data[4] = {top_left, top_right, bottom_left, bottom_right};
          c_output_addr[output_index] = ComputeLerpByNearest(data, xs_lerp, ys_lerp, "floor");
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, bc_, this, &parallel_search_info_, pool_);
  return true;
}

template <typename T>
bool ResizeV2CpuKernelMod::LaunchKernelByLinear(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  T *input_addr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  T *output_addr = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  std::vector<CachedInterpolation> ys(out_height_ + 1);
  std::vector<CachedInterpolation> xs(out_width_ + 1);

  ComputeInterpolationWeightsByLinear(out_height_, in_height_, height_scale_, align_corners_, ys.data());
  ComputeInterpolationWeightsByLinear(out_width_, in_width_, width_scale_, align_corners_, xs.data());

  auto task = [input_addr, output_addr, ys, xs, this](size_t start, size_t end) {
    for (size_t c = start; c < end; ++c) {
      auto c_input_addr = input_addr + c * in_hw_size_;
      auto c_output_addr = output_addr + c * out_hw_size_;
      for (size_t h = 0; h < out_height_; ++h) {
        const T *ys_input_lower_ptr = c_input_addr + ys[h].lower * in_width_;
        const T *ys_input_upper_ptr = c_input_addr + ys[h].upper * in_width_;
        const float ys_lerp = static_cast<float>(ys[h].lerp);
        for (size_t w = 0; w < out_width_; ++w) {
          const size_t xs_lower = xs[w].lower;
          const size_t xs_upper = xs[w].upper;
          const float xs_lerp = static_cast<float>(xs[w].lerp);
          const T top_left(ys_input_lower_ptr[xs_lower]);
          const T top_right(ys_input_lower_ptr[xs_upper]);
          const T bottom_left(ys_input_upper_ptr[xs_lower]);
          const T bottom_right(ys_input_upper_ptr[xs_upper]);
          int64_t output_index = h * out_width_ + w;
          T data[4] = {top_left, top_right, bottom_left, bottom_right};
          c_output_addr[output_index] = ComputeLerpByBiLinear(data, xs_lerp, ys_lerp);
        }
      }
    }
  };

  ParallelLaunchAutoSearch(task, bc_, this, &parallel_search_info_, pool_);
  return true;
}

template <typename T>
bool ResizeV2CpuKernelMod::LaunchKernelByCubic(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  T *input_addr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  T *output_addr = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  std::vector<CachedInterpolationCubic> ys(out_height_ + 1);
  std::vector<CachedInterpolationCubic> xs(out_width_ + 1);

  ComputeInterpolationWeightsByCubic(out_height_, in_height_, height_scale_, align_corners_, ys.data());
  ComputeInterpolationWeightsByCubic(out_width_, in_width_, width_scale_, align_corners_, xs.data());

  auto task = [input_addr, output_addr, ys, xs, this](size_t start, size_t end) {
    for (size_t c = start; c < end; ++c) {
      auto c_input_addr = input_addr + c * in_hw_size_;
      auto c_output_addr = output_addr + c * out_hw_size_;
      for (size_t h = 0; h < out_height_; ++h) {
        const T *ys_input_lowest_ptr = c_input_addr + ys[h].lowest * in_width_;
        const T *ys_input_lower_ptr = c_input_addr + ys[h].lower * in_width_;
        const T *ys_input_upper_ptr = c_input_addr + ys[h].upper * in_width_;
        const T *ys_input_uppest_ptr = c_input_addr + ys[h].uppest * in_width_;
        const float ys_lerp = static_cast<float>(ys[h].lerp);
        for (size_t w = 0; w < out_width_; ++w) {
          const size_t xs_lowest = xs[w].lowest;
          const size_t xs_lower = xs[w].lower;
          const size_t xs_upper = xs[w].upper;
          const size_t xs_uppest = xs[w].uppest;
          const float xs_lerp = static_cast<float>(xs[w].lerp);
          const T i_0_0(ys_input_lowest_ptr[xs_lowest]);
          const T i_0_1(ys_input_lowest_ptr[xs_lower]);
          const T i_0_2(ys_input_lowest_ptr[xs_upper]);
          const T i_0_3(ys_input_lowest_ptr[xs_uppest]);
          const T i_1_0(ys_input_lower_ptr[xs_lowest]);
          const T i_1_1(ys_input_lower_ptr[xs_lower]);
          const T i_1_2(ys_input_lower_ptr[xs_upper]);
          const T i_1_3(ys_input_lower_ptr[xs_uppest]);
          const T i_2_0(ys_input_upper_ptr[xs_lowest]);
          const T i_2_1(ys_input_upper_ptr[xs_lower]);
          const T i_2_2(ys_input_upper_ptr[xs_upper]);
          const T i_2_3(ys_input_upper_ptr[xs_uppest]);
          const T i_3_0(ys_input_uppest_ptr[xs_lowest]);
          const T i_3_1(ys_input_uppest_ptr[xs_lower]);
          const T i_3_2(ys_input_uppest_ptr[xs_upper]);
          const T i_3_3(ys_input_uppest_ptr[xs_uppest]);
          int64_t output_index = h * out_width_ + w;
          T data[16] = {i_0_0, i_0_1, i_0_2, i_0_3, i_1_0, i_1_1, i_1_2, i_1_3,
                        i_2_0, i_2_1, i_2_2, i_2_3, i_3_0, i_3_1, i_3_2, i_3_3};
          c_output_addr[output_index] = ComputeLerpByBiCubic(data, xs_lerp, ys_lerp, -0.75);
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, bc_, this, &parallel_search_info_, pool_);
  return true;
}

#define RESIZE_CPU_REG(MS_T, T, SIZES_T) \
  KernelAttr()                           \
    .AddInputAttr(MS_T)                  \
    .AddInputAttr(kNumberTypeFloat32)    \
    .AddInputAttr(kNumberTypeFloat32)    \
    .AddInputAttr(SIZES_T)               \
    .AddOutputAttr(MS_T),                \
    &ResizeV2CpuKernelMod::LaunchKernel<T>

const std::vector<std::pair<KernelAttr, ResizeV2CpuKernelMod::KernelRunFunc>> &ResizeV2CpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, ResizeV2CpuKernelMod::KernelRunFunc>> func_list = {
    {RESIZE_CPU_REG(kNumberTypeFloat16, float16, kNumberTypeInt64)},
    {RESIZE_CPU_REG(kNumberTypeFloat32, float, kNumberTypeInt64)},
    {RESIZE_CPU_REG(kNumberTypeFloat64, double, kNumberTypeInt64)},
    {RESIZE_CPU_REG(kNumberTypeInt8, int8_t, kNumberTypeInt64)},
    {RESIZE_CPU_REG(kNumberTypeUInt8, uint8_t, kNumberTypeInt64)},
    {RESIZE_CPU_REG(kNumberTypeInt16, int16_t, kNumberTypeInt64)},
    {RESIZE_CPU_REG(kNumberTypeInt32, int32_t, kNumberTypeInt64)},
    {RESIZE_CPU_REG(kNumberTypeInt64, int64_t, kNumberTypeInt64)},
    {RESIZE_CPU_REG(kNumberTypeFloat16, float16, kNumberTypeInt32)},
    {RESIZE_CPU_REG(kNumberTypeFloat32, float, kNumberTypeInt32)},
    {RESIZE_CPU_REG(kNumberTypeFloat64, double, kNumberTypeInt32)},
    {RESIZE_CPU_REG(kNumberTypeInt8, int8_t, kNumberTypeInt32)},
    {RESIZE_CPU_REG(kNumberTypeUInt8, uint8_t, kNumberTypeInt32)},
    {RESIZE_CPU_REG(kNumberTypeInt16, int16_t, kNumberTypeInt32)},
    {RESIZE_CPU_REG(kNumberTypeInt32, int32_t, kNumberTypeInt32)},
    {RESIZE_CPU_REG(kNumberTypeInt64, int64_t, kNumberTypeInt32)},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ResizeV2, ResizeV2CpuKernelMod);
}  // namespace mindspore::kernel
