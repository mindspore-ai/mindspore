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

#include "plugin/device/cpu/kernel/resize_linear_1d_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include <unordered_map>
#include "kernel/ops_utils.h"
#include "mindspore/core/ops/ops_func_impl/resize_linear_1d.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore::kernel {
constexpr auto kResizeLinear1D = "ResizeLinear1D";
constexpr const size_t kResizeLinear1DInputsNum = 3;
constexpr const size_t kResizeLinear1DOutputsNum = 1;
constexpr const size_t kResizeDims = 3;

template <typename T>
void ResizeLinear1DCpuKernelMod::ComputeInterpolationCaches(const size_t out_size, const size_t in_size,
                                                            const CoordinateTransformationFunc<T> &func,
                                                            size_t *interp_lower, size_t *interp_upper,
                                                            T *interp_lerp) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      const T in = func(i, in_size, out_size);
      const T in_floor = std::floor(in);
      const T in_ceil = std::ceil(in);
      interp_lower[i] = static_cast<size_t>(in_floor > 0 ? in_floor : 0);
      interp_upper[i] = static_cast<size_t>(in_ceil < static_cast<T>(in_size - 1) ? in_ceil : in_size - 1);
      interp_lerp[i] = in - in_floor;
    }
  };
  ParallelLaunchAutoSearch(task, out_size, this, &parallel_search_info_, pool_);
  return;
}

template <typename T>
bool ResizeLinear1DCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &workspace,
                                              const std::vector<kernel::KernelTensor *> &outputs) {
  auto input = GetDeviceAddress<T>(inputs, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);
  auto output = GetDeviceAddress<T>(outputs, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(output, false);

  if (out_width_ == in_width_) {
    auto task = [input, output](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        output[i] = input[i];
      }
    };
    ParallelLaunchAutoSearch(task, inputs[kIndex0]->size() / sizeof(T), this, &parallel_search_info_, pool_);
    return true;
  }

  auto interp_lower = GetDeviceAddress<size_t>(workspace, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(interp_lower, false);
  auto interp_upper = GetDeviceAddress<size_t>(workspace, kIndex1);
  MS_ERROR_IF_NULL_W_RET_VAL(interp_upper, false);
  auto interp_lerp = GetDeviceAddress<T>(workspace, kIndex2);
  MS_ERROR_IF_NULL_W_RET_VAL(interp_lerp, false);

  auto coordinate_transformation_func = ChooseCoordinateTransformationFunc<T>(coordinate_transformation_mode_);

  ComputeInterpolationCaches(out_width_, in_width_, coordinate_transformation_func, interp_lower, interp_upper,
                             interp_lerp);

  auto task = [input, output, interp_lower, interp_upper, interp_lerp, this](size_t start, size_t end) {
    for (size_t index = start; index < end; ++index) {
      for (size_t w = 0; w < out_width_; ++w) {
        const T left(static_cast<T>(*(input + index * in_width_ + interp_lower[w])));
        const T right(static_cast<T>(*(input + index * in_width_ + interp_upper[w])));
        *(output + index * out_width_ + w) = left + (right - left) * interp_lerp[w];
      }
    }
  };

  ParallelLaunchAutoSearch(task, batch_ * channel_, this, &parallel_search_info_, pool_);
  return true;
}

#define RESIZE_LINEAR_1D_CPU_REG(MS_T, MS_S, T)        \
  KernelAttr()                                         \
    .AddInputAttr(MS_T)                                \
    .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)  \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
    .AddOutputAttr(MS_T),                              \
    &ResizeLinear1DCpuKernelMod::LaunchKernel<T>

const std::vector<std::pair<KernelAttr, ResizeLinear1DCpuKernelMod::KernelRunFunc>>
  &ResizeLinear1DCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ResizeLinear1DCpuKernelMod::KernelRunFunc>> func_list = {
    {RESIZE_LINEAR_1D_CPU_REG(kNumberTypeFloat32, kNumberTypeInt32, float)},
    {RESIZE_LINEAR_1D_CPU_REG(kNumberTypeFloat64, kNumberTypeInt32, double)},
    {RESIZE_LINEAR_1D_CPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float)},
    {RESIZE_LINEAR_1D_CPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double)},
  };
  return func_list;
}

template <typename T>
ResizeLinear1DCpuKernelMod::CoordinateTransformationFunc<T>
ResizeLinear1DCpuKernelMod::ChooseCoordinateTransformationFunc(
  ops::CoordinateTransformationMode coordinate_transformation_mode) const {
  const std::unordered_map<ops::CoordinateTransformationMode, CoordinateTransformationFunc<T>> coordinate_map{
    {ops::CoordinateTransformationMode::ALIGN_CORNERS, AlignCornersFunc<T>()},
    {ops::CoordinateTransformationMode::HALF_PIXEL, HalfPixelFunc<T>()}};
  return coordinate_map.at(coordinate_transformation_mode);
}

bool ResizeLinear1DCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kResizeLinear1DInputsNum || outputs.size() != kResizeLinear1DOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kResizeLinear1DInputsNum
                  << " and " << kResizeLinear1DOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }

  MS_EXCEPTION_IF_NULL(inputs[kIndex0]);
  x_type_ = inputs[kIndex0]->dtype_id();

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

void ResizeLinear1DCpuKernelMod::SetWorkSpaceSize(const std::vector<KernelTensor *> &inputs) {
  workspace_size_list_.push_back(sizeof(size_t) * out_width_);
  workspace_size_list_.push_back(sizeof(size_t) * out_width_);
  if (x_type_ == kNumberTypeFloat32) {
    workspace_size_list_.push_back(sizeof(float) * out_width_);
  } else if (x_type_ == kNumberTypeFloat64) {
    workspace_size_list_.push_back(sizeof(double) * out_width_);
  }
}

int ResizeLinear1DCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  int ret = 0;
  if ((ret = KernelMod::Resize(inputs, outputs)) != 0) {
    return ret;
  }
  std::vector<int64_t> input_shape = inputs.at(kIndex0)->GetShapeVector();
  std::vector<int64_t> output_shape = outputs.at(kIndex0)->GetShapeVector();
  batch_ = LongToSize(input_shape[kIndex0]);
  channel_ = LongToSize(input_shape[kIndex1]);
  in_width_ = LongToSize(input_shape[kIndex2]);
  out_width_ = LongToSize(output_shape[kIndex2]);

  coordinate_transformation_mode_ =
    static_cast<ops::CoordinateTransformationMode>(inputs.at(kIndex2)->GetValueWithCheck<int64_t>());
  if (coordinate_transformation_mode_ != ops::CoordinateTransformationMode::ALIGN_CORNERS &&
      coordinate_transformation_mode_ != ops::CoordinateTransformationMode::HALF_PIXEL) {
    MS_LOG_EXCEPTION << "For '" << kernel_name_ << "', coordinate_transformation_mode not support now.";
  }
  SetWorkSpaceSize(inputs);
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ResizeLinear1D,
                                 []() { return std::make_shared<ResizeLinear1DCpuKernelMod>(kResizeLinear1D); });
}  // namespace mindspore::kernel
