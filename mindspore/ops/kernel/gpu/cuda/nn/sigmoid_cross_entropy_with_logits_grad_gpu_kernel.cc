/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/nn/sigmoid_cross_entropy_with_logits_grad_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/sigmoid_cross_entropy_with_logits_grad_impl.cuh"

namespace mindspore {
namespace kernel {
bool SigmoidCrossEntropyWithLogitsGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                         const std::vector<KernelTensor *> &outputs) {
  constexpr size_t input_num = 3;
  constexpr size_t output_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

template <typename T, typename S>
bool SigmoidCrossEntropyWithLogitsGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                                 const std::vector<KernelTensor *> &workspace,
                                                                 const std::vector<KernelTensor *> &outputs,
                                                                 void *stream_ptr) {
  T *logits_addr = GetDeviceAddress<T>(inputs, 0);
  S *labels_addr = GetDeviceAddress<S>(inputs, 1);
  T *dout_addr = GetDeviceAddress<T>(inputs, 2);
  T *outputs_addr = GetDeviceAddress<T>(outputs, 0);

  auto status = SigmoidCrossEntropyWithLogitsGrad(inputs[0]->size() / sizeof(T), logits_addr, labels_addr, dout_addr,
                                                  outputs_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<KernelAttr> SigmoidCrossEntropyWithLogitsGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SigmoidCrossEntropyWithLogitsGradLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

std::vector<
  std::pair<KernelAttr, SigmoidCrossEntropyWithLogitsGradGpuKernelMod::SigmoidCrossEntropyWithLogitsGradLaunchFunc>>
  SigmoidCrossEntropyWithLogitsGradGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SigmoidCrossEntropyWithLogitsGradGpuKernelMod::LaunchKernel<double, double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SigmoidCrossEntropyWithLogitsGradGpuKernelMod::LaunchKernel<float, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &SigmoidCrossEntropyWithLogitsGradGpuKernelMod::LaunchKernel<half, half>},
};
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SigmoidCrossEntropyWithLogitsGrad,
                      SigmoidCrossEntropyWithLogitsGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
