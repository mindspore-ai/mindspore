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
#include "mindspore/core/ops/resize_linear_1d.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "kernel/common_utils.h"

namespace mindspore::kernel {
constexpr auto kResizeLinear1D = "ResizeLinear1D";
constexpr const size_t kResizeLinear1DInputsNum = 2;
constexpr const size_t kResizeLinear1DOutputsNum = 1;
constexpr const size_t kResizeLinear1DNewShapeSize = sizeof(int64_t);

void ResizeLinear1DCpuKernelMod::ComputeInterpolationCaches(const size_t out_size, const size_t in_size,
                                                            const CoordinateTransformationFunc &func,
                                                            CachedInterpolation *interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (size_t i = 0; i <= out_size - 1; ++i) {
    const float in = func(i, in_size, out_size);
    const float in_floor = std::floor(in);
    const float in_ceil = std::ceil(in);
    interpolation[i].lower = static_cast<size_t>(in_floor > 0 ? in_floor : 0);
    interpolation[i].upper = static_cast<size_t>(in_ceil < static_cast<float>(in_size - 1) ? in_ceil : in_size - 1);
    interpolation[i].lerp = in - in_floor;
  }
}

template <typename T>
bool ResizeLinear1DCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeLinear1DInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeLinear1DOutputsNum, kernel_name_);
  T *input = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);
  auto size_input = inputs[kIndex1];
  int64_t *new_shape_data = reinterpret_cast<int64_t *>(size_input->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(new_shape_data, false);
  float *output = reinterpret_cast<float *>(outputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(output, false);

  size_t new_shape_data_size = size_input->size;
  if (new_shape_data_size != kResizeLinear1DNewShapeSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', new shape data should be " << kResizeLinear1DNewShapeSize
                  << ", but got " << new_shape_data_size;
    return false;
  }

  size_t out_width = LongToSize(new_shape_data[0]);
  if (out_width == in_width_) {
    auto task = [input, output](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        output[i] = static_cast<float>(input[i]);
      }
    };
    ParallelLaunchAutoSearch(task, inputs[kIndex0]->size / sizeof(T), this, &parallel_search_info_, pool_);
    return true;
  }

  std::vector<CachedInterpolation> xs(out_width + 1);
  ComputeInterpolationCaches(out_width, in_width_, coordinate_transformation_func_, xs.data());

  auto task = [input, output, xs, out_width, this](size_t start, size_t end) {
    for (size_t index = start; index < end; ++index) {
      for (size_t w = 0; w < out_width; ++w) {
        const size_t xs_lower = xs[w].lower;
        const size_t xs_upper = xs[w].upper;
        const float xs_lerp = static_cast<float>(xs[w].lerp);
        const float left(static_cast<float>(*(input + index * in_width_ + xs_lower)));
        const float right(static_cast<float>(*(input + index * in_width_ + xs_upper)));
        *(output + index * out_width + w) = (left + (right - left) * xs_lerp);
      }
    }
  };

  ParallelLaunchAutoSearch(task, batch_ * channel_, this, &parallel_search_info_, pool_);

  return true;
}

#define RESIZE_LINEAR_1D_CPU_REG(MS_T, T)                                                           \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32), \
    &ResizeLinear1DCpuKernelMod::LaunchKernel<T>

const std::vector<std::pair<KernelAttr, ResizeLinear1DCpuKernelMod::KernelRunFunc>>
  &ResizeLinear1DCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ResizeLinear1DCpuKernelMod::KernelRunFunc>> func_list = {
    {RESIZE_LINEAR_1D_CPU_REG(kNumberTypeInt8, int8_t)},     {RESIZE_LINEAR_1D_CPU_REG(kNumberTypeUInt8, uint8_t)},
    {RESIZE_LINEAR_1D_CPU_REG(kNumberTypeInt8, int16_t)},    {RESIZE_LINEAR_1D_CPU_REG(kNumberTypeUInt16, uint16_t)},
    {RESIZE_LINEAR_1D_CPU_REG(kNumberTypeInt32, int32_t)},   {RESIZE_LINEAR_1D_CPU_REG(kNumberTypeInt64, int64_t)},
    {RESIZE_LINEAR_1D_CPU_REG(kNumberTypeFloat16, float16)}, {RESIZE_LINEAR_1D_CPU_REG(kNumberTypeFloat32, float)},
    {RESIZE_LINEAR_1D_CPU_REG(kNumberTypeFloat64, double)},
  };
  return func_list;
}

ResizeLinear1DCpuKernelMod::CoordinateTransformationFunc ResizeLinear1DCpuKernelMod::ChooseCoordinateTransformationFunc(
  CoordinateTransformationMode coordinate_transformation_mode) {
  const std::unordered_map<CoordinateTransformationMode, CoordinateTransformationFunc> coordinate_map{
    {ALIGN_CORNERS, AlignCornersFunc()}, {HALF_PIXEL, HalfPixelFunc()}, {ASYMMETRIC, AsymmetricFunc()}};
  return coordinate_map.at(coordinate_transformation_mode);
}

bool ResizeLinear1DCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ResizeLinear1D>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kResizeLinear1DInputsNum || outputs.size() != kResizeLinear1DOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kResizeLinear1DInputsNum
                  << " and " << kResizeLinear1DOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }

  std::string coordinate_transformation_mode = kernel_ptr->get_coordinate_transformation_mode();
  if (coordinate_transformation_mode == "align_corners") {
    coordinate_transformation_mode_ = ALIGN_CORNERS;
  } else if (coordinate_transformation_mode == "half_pixel") {
    coordinate_transformation_mode_ = HALF_PIXEL;
  } else if (coordinate_transformation_mode == "asymmetric") {
    coordinate_transformation_mode_ = ASYMMETRIC;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', coordinate_transformation_mode: " << coordinate_transformation_mode
                  << " not support now.";
    return false;
  }

  coordinate_transformation_func_ = ChooseCoordinateTransformationFunc(coordinate_transformation_mode_);

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int ResizeLinear1DCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != 0) {
    return ret;
  }
  std::vector<int64_t> shape_ = inputs[kIndex0]->GetShapeVector();
  batch_ = LongToSize(shape_[kIndex0]);
  channel_ = LongToSize(shape_[kIndex1]);
  in_width_ = LongToSize(shape_[kIndex2]);
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ResizeLinear1D,
                                 []() { return std::make_shared<ResizeLinear1DCpuKernelMod>(kResizeLinear1D); });
}  // namespace mindspore::kernel
