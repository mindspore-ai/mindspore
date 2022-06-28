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

#include "plugin/device/cpu/kernel/resize_linear_1d_grad_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include <unordered_map>
#include "mindspore/core/ops/grad/resize_linear_1d_grad.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "kernel/common_utils.h"

namespace mindspore::kernel {
constexpr auto kResizeLinear1DGrad = "ResizeLinear1DGrad";
constexpr const size_t kResizeLinear1DGradInputsNum = 2;
constexpr const size_t kResizeLinear1DGradOutputsNum = 1;

void ResizeLinear1DGradCpuKernelMod::ComputeInterpolationCaches(const size_t out_size, const size_t in_size,
                                                                const CoordinateTransformationFunc &func,
                                                                CachedInterpolation *interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      const float in = func(i, in_size, out_size);
      const float in_floor = std::floor(in);
      const float in_ceil = std::ceil(in);
      interpolation[i].lower = static_cast<size_t>(in_floor > 0 ? in_floor : 0);
      interpolation[i].upper = static_cast<size_t>(in_ceil < static_cast<float>(in_size - 1) ? in_ceil : in_size - 1);
      interpolation[i].lerp = in - in_floor;
    }
  };
  ParallelLaunchAutoSearch(task, out_size, this, &parallel_search_info_, pool_);
}

template <typename T>
bool ResizeLinear1DGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &,
                                                  const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeLinear1DGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeLinear1DGradOutputsNum, kernel_name_);
  T *grad_output = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(grad_output, false);
  T *grad_input = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(grad_input, false);

  if (output_width_ == input_width_) {
    auto task = [grad_output, grad_input](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        grad_input[i] = grad_output[i];
      }
    };
    ParallelLaunchAutoSearch(task, inputs[kIndex0]->size / sizeof(T), this, &parallel_search_info_, pool_);
    return true;
  }

  if (memset_s(grad_input, outputs[kIndex0]->size, 0, outputs[kIndex0]->size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }

  std::vector<CachedInterpolation> xs(output_width_ + 1);
  ComputeInterpolationCaches(output_width_, input_width_, coordinate_transformation_func_, xs.data());

  auto task = [grad_output, grad_input, xs, this](size_t start, size_t end) {
    for (size_t index = start; index < end; ++index) {
      for (size_t w = 0; w < output_width_; ++w) {
        const size_t xs_lower = xs[w].lower;
        const size_t xs_upper = xs[w].upper;
        const float xs_lerp = static_cast<float>(xs[w].lerp);
        *(grad_input + index * input_width_ + xs_lower) +=
          static_cast<T>((*(grad_output + index * output_width_ + w)) * static_cast<T>(1 - xs_lerp));
        *(grad_input + index * input_width_ + xs_upper) +=
          static_cast<T>((*(grad_output + index * output_width_ + w)) * static_cast<T>(xs_lerp));
      }
    }
  };

  ParallelLaunchAutoSearch(task, batch_ * channel_, this, &parallel_search_info_, pool_);
  return true;
}

#define RESIZE_LINEAR_1D_GRAD_CPU_REG(MS_T, T)                            \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_T).AddOutputAttr(MS_T), \
    &ResizeLinear1DGradCpuKernelMod::LaunchKernel<T>

const std::vector<std::pair<KernelAttr, ResizeLinear1DGradCpuKernelMod::KernelRunFunc>>
  &ResizeLinear1DGradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ResizeLinear1DGradCpuKernelMod::KernelRunFunc>> func_list = {
    {RESIZE_LINEAR_1D_GRAD_CPU_REG(kNumberTypeFloat16, float16)},
    {RESIZE_LINEAR_1D_GRAD_CPU_REG(kNumberTypeFloat32, float)},
    {RESIZE_LINEAR_1D_GRAD_CPU_REG(kNumberTypeFloat64, double)},
  };
  return func_list;
}

ResizeLinear1DGradCpuKernelMod::CoordinateTransformationFunc
ResizeLinear1DGradCpuKernelMod::ChooseCoordinateTransformationFunc(
  CoordinateTransformationMode coordinate_transformation_mode) {
  const std::unordered_map<CoordinateTransformationMode, CoordinateTransformationFunc> coordinate_map{
    {ALIGN_CORNERS, AlignCornersFunc()}, {HALF_PIXEL, HalfPixelFunc()}, {ASYMMETRIC, AsymmetricFunc()}};
  return coordinate_map.at(coordinate_transformation_mode);
}

bool ResizeLinear1DGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ResizeLinear1DGrad>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kResizeLinear1DGradInputsNum || outputs.size() != kResizeLinear1DGradOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kResizeLinear1DGradInputsNum
                  << " and " << kResizeLinear1DGradOutputsNum << ", but got " << inputs.size() << " and "
                  << outputs.size();
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

int ResizeLinear1DGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != 0) {
    return ret;
  }

  std::vector<int64_t> grad_shape = inputs[kIndex0]->GetShapeVector();
  auto grad_batch = LongToSize(grad_shape[kIndex0]);
  auto grad_channel = LongToSize(grad_shape[kIndex1]);
  output_width_ = LongToSize(grad_shape[kIndex2]);

  std::vector<int64_t> shape_ = inputs[kIndex1]->GetShapeVector();
  batch_ = LongToSize(shape_[kIndex0]);
  channel_ = LongToSize(shape_[kIndex1]);
  input_width_ = LongToSize(shape_[kIndex2]);

  if (grad_batch != batch_ || grad_channel != channel_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', grad batch is : " << grad_batch
                  << ", while input batch is : " << batch_ << "; "
                  << "grad channel is : " << grad_channel << ", while input channel is : " << channel_;
    return KRET_RESIZE_FAILED;
  }

  return KRET_OK;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ResizeLinear1DGrad, []() {
  return std::make_shared<ResizeLinear1DGradCpuKernelMod>(kResizeLinear1DGrad);
});
}  // namespace mindspore::kernel
