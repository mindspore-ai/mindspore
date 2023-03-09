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

#include "plugin/device/gpu/kernel/nn/resize_linear_1d_grad_gpu_kernel.h"
#include "mindspore/core/abstract/utils.h"
#include "mindspore/core/ops/grad/resize_linear_1d_grad.h"

namespace {
constexpr const size_t kResizeLinear1DGradInputsNum = 2;
constexpr const size_t kResizeLinear1DGradOutputsNum = 1;
constexpr const size_t kResizeInputDims = 3;
}  // namespace

namespace mindspore {
namespace kernel {
bool ResizeLinear1DGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL_W_RET_VAL(base_operator, false);

  auto kernel_ptr = std::dynamic_pointer_cast<ops::ResizeLinear1DGrad>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kResizeLinear1DGradInputsNum || outputs.size() != kResizeLinear1DGradOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', grad_output and grad_input size must be "
                  << kResizeLinear1DGradInputsNum << " and " << kResizeLinear1DGradOutputsNum << ", but got "
                  << inputs.size() << " and " << outputs.size();
    return false;
  }

  std::string coordinate_transformation_mode = kernel_ptr->get_coordinate_transformation_mode();
  if (coordinate_transformation_mode == "align_corners") {
    mode_ = ResizeLinearCoordinateTransformationMode::ALIGN_CORNERS;
  } else if (coordinate_transformation_mode == "half_pixel") {
    mode_ = ResizeLinearCoordinateTransformationMode::HALF_PIXEL;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', coordinate_transformation_mode: " << coordinate_transformation_mode
                  << " not support now.";
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[pair.second].second;
  return true;
}

int ResizeLinear1DGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != KRET_OK) {
    return ret;
  }

  grad_output_shape_ = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                            inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  batch_ = LongToSize(grad_output_shape_[kIndex0]);
  channel_ = LongToSize(grad_output_shape_[kIndex1]);
  out_width_ = grad_output_shape_[kIndex2];
  grad_input_shape_ = std::vector<int64_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                           outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  in_width_ = grad_input_shape_[kIndex2];

  if (grad_output_shape_.size() != kResizeInputDims) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'input_x' should be greater than or equal to 1, but got "
                  << grad_output_shape_.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

template <typename T>
bool ResizeLinear1DGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &workspace,
                                                  const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeLinear1DGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeLinear1DGradOutputsNum, kernel_name_);
  T *grad_output = GetDeviceAddress<T>(inputs, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(grad_output, false);

  T *grad_input = GetDeviceAddress<T>(outputs, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(grad_input, false);

  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemset(grad_input, 0, outputs.at(kIndex0)->size),
                                    "For ResizeLinear1DGradGpuKernelMod failed to cudaMemset");

  int64_t output_size = batch_ * channel_ * out_width_;
  ResizeLinear1DGrad(mode_, output_size, in_width_, out_width_, grad_output, grad_input, device_id_,
                     reinterpret_cast<cudaStream_t>(stream_ptr));

  return true;
}

#define RESIZE_LINEAR_1D_GRAD_GPU_REG(MS_T, T)                            \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_T).AddOutputAttr(MS_T), \
    &ResizeLinear1DGradGpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, ResizeLinear1DGradGpuKernelMod::ResizeLinear1DGradFunc>>
  ResizeLinear1DGradGpuKernelMod::func_list_ = {
    {RESIZE_LINEAR_1D_GRAD_GPU_REG(kNumberTypeFloat16, half)},
    {RESIZE_LINEAR_1D_GRAD_GPU_REG(kNumberTypeFloat32, float)},
    {RESIZE_LINEAR_1D_GRAD_GPU_REG(kNumberTypeFloat64, double)},
};

std::vector<KernelAttr> ResizeLinear1DGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ResizeLinear1DGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ResizeLinear1DGrad, ResizeLinear1DGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
