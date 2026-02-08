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

#include "kernel/gpu/cuda/nn/resize_linear_1d_gpu_kernel.h"
#include "mindapi/base/types.h"

namespace {
constexpr const size_t kResizeLinear1DInputsNum = 3;
constexpr const size_t kResizeLinear1DOutputsNum = 1;
constexpr const size_t kResizeInputDims = 3;
}  // namespace

namespace mindspore {
namespace kernel {
bool ResizeLinear1DGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kResizeLinear1DInputsNum || outputs.size() != kResizeLinear1DOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kResizeLinear1DInputsNum
                  << " and " << kResizeLinear1DOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
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

int ResizeLinear1DGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  int ret;
  if ((ret = KernelMod::Resize(inputs, outputs)) != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[kIndex0]->GetDeviceShapeVector();
  batch_ = LongToSize(input_shape_[kIndex0]);
  channel_ = LongToSize(input_shape_[kIndex1]);
  in_width_ = input_shape_[kIndex2];
  output_shape_ = outputs[kIndex0]->GetDeviceShapeVector();
  out_width_ = output_shape_[kIndex2];

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
bool ResizeLinear1DGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &workspace,
                                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  MS_ERROR_IF_NULL_W_RET_VAL(output, false);
  int64_t output_size = batch_ * channel_ * out_width_;
  auto status = ResizeLinear1D(mode_, output_size, in_width_, out_width_, input, output, device_id_,
                               reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

#define RESIZE_LINEAR_1D_GPU_REG(MS_T, T)              \
  KernelAttr()                                         \
    .AddInputAttr(MS_T)                                \
    .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)  \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
    .AddOutputAttr(MS_T),                              \
    &ResizeLinear1DGpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, ResizeLinear1DGpuKernelMod::ResizeLinear1DFunc>>
  ResizeLinear1DGpuKernelMod::func_list_ = {
    {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeFloat16, half)},
    {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeFloat32, float)},
    {RESIZE_LINEAR_1D_GPU_REG(kNumberTypeFloat64, double)},
};

std::vector<KernelAttr> ResizeLinear1DGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ResizeLinear1DFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ResizeLinear1D, ResizeLinear1DGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
