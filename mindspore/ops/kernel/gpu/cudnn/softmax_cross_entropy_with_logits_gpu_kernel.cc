/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cudnn/softmax_cross_entropy_with_logits_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/cross_entropy_impl.cuh"

namespace mindspore {
namespace kernel {
bool SoftmaxCrossEntropyWithLogitsGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &outputs) {
  constexpr size_t input_num = 2;
  constexpr size_t output_num = 2;

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

int SoftmaxCrossEntropyWithLogitsGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto logits_shape = inputs[kIndex0]->GetShapeVector();
  auto labels_shape = inputs[kIndex1]->GetShapeVector();

  auto ret = CheckShapeValidation(logits_shape, labels_shape);
  if (ret != KRET_OK) {
    return ret;
  }

  size_t logits_dims = logits_shape.size();
  batch_size_ = 1;
  for (size_t i = 0; i < logits_dims - 1; i++) {
    batch_size_ *= LongToSize(logits_shape[i]);
  }
  channel_size_ = LongToSize(logits_shape[logits_dims - 1]);
  height_ = 1;
  width_ = 1;
  logits_size_ = sizeof(float) * batch_size_ * channel_size_ * height_ * width_;

  output1_size_ = logits_size_ / LongToSize(logits_shape[logits_dims - 1]);
  output2_size_ = logits_size_;
  softmax_output_logits_size_ = logits_size_;

  workspace_size_list_.push_back(softmax_output_logits_size_);
  return KRET_OK;
}

template <typename T, typename S>
bool SoftmaxCrossEntropyWithLogitsGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                             const std::vector<KernelTensor *> &workspace,
                                                             const std::vector<KernelTensor *> &outputs,
                                                             void *stream_ptr) {
  T *logits_addr = GetDeviceAddress<T>(inputs, 0);
  S *labels_addr = GetDeviceAddress<S>(inputs, 1);
  T *loss_addr = GetDeviceAddress<T>(outputs, 0);
  T *dlogits_addr = GetDeviceAddress<T>(outputs, 1);
  T *workspace_addr = GetDeviceAddress<T>(workspace, 0);

  CrossEntropy(logits_addr, labels_addr, batch_size_, channel_size_, loss_addr, dlogits_addr, workspace_addr,
               reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<
  std::pair<KernelAttr, SoftmaxCrossEntropyWithLogitsGpuKernelMod::SoftmaxCrossEntropyWithLogitsGpuLaunchFunc>>
  SoftmaxCrossEntropyWithLogitsGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SoftmaxCrossEntropyWithLogitsGpuKernelMod::LaunchKernel<float, float>},
};

std::vector<KernelAttr> SoftmaxCrossEntropyWithLogitsGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](
      const std::pair<KernelAttr, SoftmaxCrossEntropyWithLogitsGpuKernelMod::SoftmaxCrossEntropyWithLogitsGpuLaunchFunc>
        &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogitsGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
