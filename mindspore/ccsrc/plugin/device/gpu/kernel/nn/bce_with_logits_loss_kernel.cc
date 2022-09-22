/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nn/bce_with_logits_loss_kernel.h"
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bce_with_logits_loss_impl.cuh"

namespace mindspore {
namespace kernel {
bool BCEWithLogitsLossKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t input_num = 4;
  constexpr size_t output_num = 1;
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
  auto input_data_type = inputs.at(kIndex0)->GetDtype();
  type_id_size_ = abstract::TypeIdSize(input_data_type);
  return true;
}

int BCEWithLogitsLossKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[kIndex0]->GetShapeVector();
  weight_shape_ = inputs[kIndex2]->GetShapeVector();
  pos_weight_shape_ = inputs[kIndex3]->GetShapeVector();

  if (input_shape_.size() < 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of logits cannot be less than 1, but got "
                  << input_shape_.size();
    return KRET_RESIZE_FAILED;
  }
  if (weight_shape_.size() < 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of weight cannot be less than 1, but got "
                  << weight_shape_.size();
    return KRET_RESIZE_FAILED;
  }
  if (pos_weight_shape_.size() < 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of pos_weight cannot be less than 1, but got "
                  << pos_weight_shape_.size();
    return KRET_RESIZE_FAILED;
  }
  if (input_shape_.size() > MAX_LOGITS_DIMENSION) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of logits cannot be greater than "
                  << MAX_LOGITS_DIMENSION << ", but got " << input_shape_.size();
    return KRET_RESIZE_FAILED;
  }
  input_size_ = SizeOf(input_shape_);
  // weight shape
  weight_size_ = SizeOf(weight_shape_);
  weight_need_broadcast_ = NeedBroadcast(&weight_shape_, input_shape_);
  // pos_weight shape
  pos_weight_size_ = SizeOf(pos_weight_shape_);
  pos_weight_need_broadcast_ = NeedBroadcast(&pos_weight_shape_, input_shape_);
  InitWorkSpaceSizeLists();
  return KRET_OK;
}

template <typename T>
bool BCEWithLogitsLossKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspace,
                                              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *predict = GetDeviceAddress<T>(inputs, kIndex0);
  T *target = GetDeviceAddress<T>(inputs, kIndex1);
  T *weight = GetDeviceAddress<T>(inputs, kIndex2);
  T *pos_weight = GetDeviceAddress<T>(inputs, kIndex3);
  size_t *input_shape = GetDeviceAddress<size_t>(workspace, kIndex0);
  size_t *weight_shape = GetDeviceAddress<size_t>(workspace, kIndex1);
  size_t *pos_weight_shape = GetDeviceAddress<size_t>(workspace, kIndex2);
  T *shape_broadcasted = GetDeviceAddress<T>(workspace, kIndex3);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(input_shape, &input_shape_[0], input_shape_.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    kernel_name_ + " cudaMemcpyAsync input_shape_ failed");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(weight_shape, &weight_shape_[0], weight_shape_.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    kernel_name_ + " cudaMemcpyAsync weight_shape_ failed");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(pos_weight_shape, &pos_weight_shape_[0], pos_weight_shape_.size() * sizeof(size_t),
                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
    kernel_name_ + " cudaMemcpyAsync pos_weight_shape_ failed");
  CalBCEWithLogitsLoss(input_size_, predict, target, input_shape, input_shape_.size(), weight, weight_shape,
                       weight_need_broadcast_, pos_weight, pos_weight_shape, pos_weight_need_broadcast_,
                       shape_broadcasted, output, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<std::pair<KernelAttr, BCEWithLogitsLossKernelMod::BCEWithLogitsLossLaunchFunc>>
  BCEWithLogitsLossKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &BCEWithLogitsLossKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &BCEWithLogitsLossKernelMod::LaunchKernel<float>},
};

std::vector<KernelAttr> BCEWithLogitsLossKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BCEWithLogitsLossKernelMod::BCEWithLogitsLossLaunchFunc> &pair) {
                         return pair.first;
                       });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BCEWithLogitsLoss, BCEWithLogitsLossKernelMod);
}  // namespace kernel
}  // namespace mindspore
