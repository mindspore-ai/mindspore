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

#include "kernel/gpu/cuda/nn/adam_weight_decay_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool AdamWeightDecayGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  constexpr size_t input_num = 10;
  constexpr size_t output_num = 3;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  MS_EXCEPTION_IF_NULL(inputs[kIndex0]);
  auto var_data_type = inputs.at(kIndex0)->dtype_id();
  s_type_id_size_ = abstract::TypeIdSize(var_data_type);
  MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
  auto m_data_type = inputs.at(kIndex1)->dtype_id();
  t_type_id_size_ = abstract::TypeIdSize(m_data_type);
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

void AdamWeightDecayGpuKernelMod::InitSizeLists() {
  output_size_list_.push_back(0);
  output_size_list_.push_back(0);
  output_size_list_.push_back(0);
}

int AdamWeightDecayGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  output_size_list_.clear();

  variable_size_ = s_type_id_size_;
  m_size_ = t_type_id_size_;
  v_size_ = t_type_id_size_;
  learning_rate_size_ = sizeof(float);
  beta1_size_ = sizeof(float);
  beta2_size_ = sizeof(float);
  epsilon_size_ = sizeof(float);
  decay_size_ = sizeof(float);
  gradient_size_ = s_type_id_size_;

  constexpr size_t input_num = 10;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  MS_EXCEPTION_IF_NULL(inputs[kIndex0]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex2]);
  auto variable_shape = inputs[kIndex0]->GetShapeVector();
  auto m_shape = inputs[kIndex1]->GetShapeVector();
  auto v_shape = inputs[kIndex2]->GetShapeVector();
  auto gradient_shape = inputs[kIndex8]->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(variable_shape, kernel_name_, "var") ||
                   CHECK_SHAPE_NULL(m_shape, kernel_name_, "m") || CHECK_SHAPE_NULL(v_shape, kernel_name_, "v") ||
                   CHECK_SHAPE_NULL(gradient_shape, kernel_name_, "gradient");
  if (is_null_input_) {
    return true;
  }
  variable_size_ *= SizeOf(variable_shape);
  m_size_ *= SizeOf(m_shape);
  v_size_ *= SizeOf(v_shape);
  gradient_size_ *= SizeOf(gradient_shape);

  InitSizeLists();
  return KRET_OK;
}

template <typename T, typename S>
bool AdamWeightDecayGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &,
                                               const std::vector<KernelTensor *> &outputs) {
  if (is_null_input_) {
    return true;
  }
  S *variable = GetDeviceAddress<S>(inputs, kIndex0);
  T *m = GetDeviceAddress<T>(inputs, kIndex1);
  T *v = GetDeviceAddress<T>(inputs, kIndex2);
  float *lr = GetDeviceAddress<float>(inputs, kIndex3);
  float *beta1 = GetDeviceAddress<float>(inputs, kIndex4);
  float *beta2 = GetDeviceAddress<float>(inputs, kIndex5);
  float *epsilon = GetDeviceAddress<float>(inputs, kIndex6);
  float *decay = GetDeviceAddress<float>(inputs, kIndex7);
  S *gradient = GetDeviceAddress<S>(inputs, kIndex8);
  auto status = AdamWeightDecayOp(inputs[0]->size() / s_type_id_size_, gradient, lr, beta1, beta2, epsilon, decay,
                                  variable, m, v, stream_ptr_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

const std::vector<std::pair<KernelAttr, AdamWeightDecayGpuKernelMod::KernelRunFunc>>
  &AdamWeightDecayGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, AdamWeightDecayGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &AdamWeightDecayGpuKernelMod::LaunchKernel<float, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &AdamWeightDecayGpuKernelMod::LaunchKernel<half, half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &AdamWeightDecayGpuKernelMod::LaunchKernel<float, half>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AdamWeightDecay, AdamWeightDecayGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
