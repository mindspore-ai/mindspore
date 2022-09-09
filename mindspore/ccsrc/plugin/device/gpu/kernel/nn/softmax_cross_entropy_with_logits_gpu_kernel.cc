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

#include "plugin/device/gpu/kernel/nn/softmax_cross_entropy_with_logits_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cross_entropy_impl.cuh"

namespace mindspore {
namespace kernel {
bool SoftmaxCrossEntropyWithLogitsGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                     const std::vector<KernelTensorPtr> &inputs,
                                                     const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t input_num = 2;
  constexpr size_t output_num = 2;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&logits_descriptor_),
                                      kernel_name_ + " cudnnCreateTensorDescriptor failed.");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&softmax_output_descriptor_),
                                      kernel_name_ + " cudnnCreateTensorDescriptor failed.");
  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs.at(kIndex0)->GetDtype()));
  return true;
}

int SoftmaxCrossEntropyWithLogitsGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                      const std::vector<KernelTensorPtr> &inputs,
                                                      const std::vector<KernelTensorPtr> &outputs,
                                                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
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

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(logits_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch_size_, channel_size_,
                               height_, width_),
    kernel_name_ + " cudnnSetTensor4dDescriptor failed.");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensor4dDescriptor(softmax_output_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch_size_,
                               channel_size_, height_, width_),
    kernel_name_ + " cudnnSetTensor4dDescriptor failed.");

  workspace_size_list_.push_back(softmax_output_logits_size_);
  return KRET_OK;
}

template <typename T, typename S>
bool SoftmaxCrossEntropyWithLogitsGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                             const std::vector<AddressPtr> &workspace,
                                                             const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *logits_addr = GetDeviceAddress<T>(inputs, 0);
  S *labels_addr = GetDeviceAddress<S>(inputs, 1);
  T *loss_addr = GetDeviceAddress<T>(outputs, 0);
  T *dlogits_addr = GetDeviceAddress<T>(outputs, 1);
  T *softmax_output_logits = GetDeviceAddress<T>(workspace, 0);

  const float alpha = 1;
  const float beta = 0;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSoftmaxForward(cudnn_handle_, algo_, mode_, &alpha, logits_descriptor_, logits_addr, &beta,
                        softmax_output_descriptor_, softmax_output_logits),
    kernel_name_ + " cudnnSoftmaxForward failed.");

  CrossEntropy(softmax_output_logits, labels_addr, batch_size_, channel_size_, loss_addr, dlogits_addr,
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
