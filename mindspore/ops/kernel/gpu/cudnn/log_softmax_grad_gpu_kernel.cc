/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cudnn/log_softmax_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool LogSoftmaxGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&y_desc_),
                                      kernel_name_ + "create input_descriptor failed");
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = child_func_list_[index].second;
  auto input_data_type = inputs[kIndex0]->dtype_id();
  type_id_size_ = abstract::TypeIdSize(input_data_type);
  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(input_data_type));
  return true;
}

int LogSoftmaxGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  ResetResource();
  auto input_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  shape_size_ = input_shape.size();
  algo_ = CUDNN_SOFTMAX_LOG;
  auto axis = inputs.at(kIndex2)->GetValueWithCheck<int64_t>();
  InitSizeByAxis(input_shape, axis);
  use_workspace_ = (axis_ != static_cast<int>(input_shape_.size()) - 1);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(y_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(batch_size_),
                               SizeToInt(channel_size_), SizeToInt(height_), SizeToInt(width_)),
    kernel_name_ + "set input_descriptor failed");
  InitSizeLists();
  return KRET_OK;
}

std::vector<std::pair<KernelAttr, LogSoftmaxGradGpuKernelMod::SoftmaxGradGpuLaunchFunc>>
  LogSoftmaxGradGpuKernelMod::child_func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &LogSoftmaxGradGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &LogSoftmaxGradGpuKernelMod::LaunchKernel<half>},
};

std::vector<KernelAttr> LogSoftmaxGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    child_func_list_.begin(), child_func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, LogSoftmaxGradGpuKernelMod::SoftmaxGradGpuLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LogSoftmaxGrad, LogSoftmaxGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
