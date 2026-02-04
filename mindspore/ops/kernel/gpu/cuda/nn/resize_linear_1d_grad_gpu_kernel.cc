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

#include "kernel/gpu/cuda/nn/resize_linear_1d_grad_gpu_kernel.h"
#include "infer/ops_func_impl/resize_linear_1d_grad.h"
#include "mindapi/base/types.h"

namespace {
constexpr const size_t kResizeLinear1DGradInputsNum = 3;
constexpr const size_t kResizeLinear1DGradOutputsNum = 1;
constexpr const size_t kResizeInputDims = 3;
}  // namespace

namespace mindspore {
namespace kernel {
bool ResizeLinear1DGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kResizeLinear1DGradInputsNum || outputs.size() != kResizeLinear1DGradOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', grad_output and grad_input size must be "
                  << kResizeLinear1DGradInputsNum << " and " << kResizeLinear1DGradOutputsNum << ", but got "
                  << inputs.size() << " and " << outputs.size();
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

int ResizeLinear1DGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  int ret;
  if ((ret = KernelMod::Resize(inputs, outputs)) != KRET_OK) {
    return ret;
  }

  grad_output_shape_ = inputs[kIndex0]->GetDeviceShapeVector();
  batch_ = grad_output_shape_[kIndex0];
  channel_ = grad_output_shape_[kIndex1];
  out_width_ = grad_output_shape_[kIndex2];

  grad_input_shape_ = outputs[kIndex0]->GetDeviceShapeVector();
  in_width_ = grad_input_shape_[kIndex2];
  size_t work_space_size = SizeOf(grad_input_shape_);
  workspace_size_list_.push_back(work_space_size * sizeof(float));

  auto coordinate_transformation_mode = inputs.at(kIndex2)->GetValueWithCheck<int64_t>();
  if (coordinate_transformation_mode == static_cast<int64_t>(CoordinateTransformMode::ALIGN_CORNERS)) {
    mode_ = ResizeLinearCoordinateTransformationMode::ALIGN_CORNERS;
  } else if (coordinate_transformation_mode == static_cast<int64_t>(CoordinateTransformMode::HALF_PIXEL)) {
    mode_ = ResizeLinearCoordinateTransformationMode::HALF_PIXEL;
  } else {
    MS_LOG_EXCEPTION << "For '" << kernel_name_ << "', coordinate_transformation_mode not support now.";
  }
  return KRET_OK;
}

template <typename T>
bool ResizeLinear1DGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &workspace,
                                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *grad_output = GetDeviceAddress<T>(inputs, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(grad_output, false);
  T *grad_input = GetDeviceAddress<T>(outputs, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(grad_input, false);
  float *grad_work = GetDeviceAddress<float>(workspace, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(grad_work, false);

  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  MS_EXCEPTION_IF_NULL(cuda_stream);
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemsetAsync(grad_input, 0, outputs[kIndex0]->size(), cuda_stream),
                                    "For ResizeLinear1DGradGpuKernelMod failed to cudaMemset grad_input.");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemsetAsync(grad_work, 0, workspace.at(kIndex0)->size(), cuda_stream),
                                    "For ResizeLinear1DGradGpuKernelMod failed to cudaMemset grad_work.");

  auto status = ResizeLinear1DGrad(mode_, batch_, channel_, in_width_, out_width_, grad_output, grad_input, grad_work,
                                   device_id_, cuda_stream);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

#define RESIZE_LINEAR_1D_GRAD_GPU_REG(MS_T, T)         \
  KernelAttr()                                         \
    .AddInputAttr(MS_T)                                \
    .AddInputAttr(MS_T)                                \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
    .AddOutputAttr(MS_T),                              \
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
