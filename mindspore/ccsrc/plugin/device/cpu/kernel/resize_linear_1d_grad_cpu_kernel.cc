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

template <typename T>
void ResizeLinear1DGradCpuKernelMod::ComputeInterpolationCaches(const size_t out_size, const size_t in_size,
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
bool ResizeLinear1DGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &workspace,
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

  size_t *interp_lower = reinterpret_cast<size_t *>(workspace[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(interp_lower, false);
  size_t *interp_upper = reinterpret_cast<size_t *>(workspace[kIndex1]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(interp_upper, false);
  T *interp_lerp = reinterpret_cast<T *>(workspace[kIndex2]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(interp_lerp, false);

  auto coordinate_transformation_func = ChooseCoordinateTransformationFunc<T>(coordinate_transformation_mode_);

  ComputeInterpolationCaches(output_width_, input_width_, coordinate_transformation_func, interp_lower, interp_upper,
                             interp_lerp);

  auto task = [grad_output, grad_input, interp_lower, interp_upper, interp_lerp, this](size_t start, size_t end) {
    for (size_t index = start; index < end; ++index) {
      for (size_t w = 0; w < output_width_; ++w) {
        *(grad_input + index * input_width_ + interp_lower[w]) +=
          static_cast<T>((*(grad_output + index * output_width_ + w)) * (1.0 - interp_lerp[w]));
        *(grad_input + index * input_width_ + interp_upper[w]) +=
          static_cast<T>((*(grad_output + index * output_width_ + w)) * interp_lerp[w]);
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
    {RESIZE_LINEAR_1D_GRAD_CPU_REG(kNumberTypeFloat32, float)},
    {RESIZE_LINEAR_1D_GRAD_CPU_REG(kNumberTypeFloat64, double)},
  };
  return func_list;
}

template <typename T>
ResizeLinear1DGradCpuKernelMod::CoordinateTransformationFunc<T>
ResizeLinear1DGradCpuKernelMod::ChooseCoordinateTransformationFunc(
  CoordinateTransformationMode coordinate_transformation_mode) {
  const std::unordered_map<CoordinateTransformationMode, CoordinateTransformationFunc<T>> coordinate_map{
    {ALIGN_CORNERS, AlignCornersFunc<T>()}, {HALF_PIXEL, HalfPixelFunc<T>()}};
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
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', coordinate_transformation_mode: " << coordinate_transformation_mode
                  << " not support now.";
    return false;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

void ResizeLinear1DGradCpuKernelMod::SetWorkSpaceSize(const std::vector<KernelTensorPtr> &inputs) {
  workspace_size_list_.clear();
  workspace_size_list_.push_back(sizeof(size_t) * output_width_);
  workspace_size_list_.push_back(sizeof(size_t) * output_width_);
  auto input_data_type = inputs[kIndex0]->GetDtype();
  if (input_data_type == kNumberTypeFloat32) {
    workspace_size_list_.push_back(sizeof(float) * output_width_);
  } else if (input_data_type == kNumberTypeFloat64) {
    workspace_size_list_.push_back(sizeof(double) * output_width_);
  }
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

  SetWorkSpaceSize(inputs);
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ResizeLinear1DGrad, []() {
  return std::make_shared<ResizeLinear1DGradCpuKernelMod>(kResizeLinear1DGrad);
});
}  // namespace mindspore::kernel
