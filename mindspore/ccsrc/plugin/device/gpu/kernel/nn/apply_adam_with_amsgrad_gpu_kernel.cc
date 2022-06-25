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

#include "plugin/device/gpu/kernel/nn/apply_adam_with_amsgrad_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/apply_adam_with_amsgrad_impl.cuh"
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "include/curand.h"
#include "mindspore/core/ops/apply_adam_with_amsgrad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kApplyAdamWithAmsgradInputsNum = 8;
constexpr size_t kApplyAdamWithAmsgradOutputsNum = 4;
constexpr size_t kScalarIndex = 0;
constexpr size_t kIndexVar = 0;
constexpr size_t kIndexM = 1;
constexpr size_t kIndexV = 2;
constexpr size_t kIndexVhat = 3;
constexpr size_t kIndexBeta1Power = 4;
constexpr size_t kIndexBeta2Power = 5;
constexpr size_t kIndexLr = 6;
constexpr size_t kIndexGrad = 7;
}  // namespace

bool ApplyAdamWithAmsgradGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::ApplyAdamWithAmsgrad>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr_, false);

  if (inputs.size() != kApplyAdamWithAmsgradInputsNum || outputs.size() != kApplyAdamWithAmsgradOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size should be " << kApplyAdamWithAmsgradInputsNum
                  << " and " << kApplyAdamWithAmsgradOutputsNum << ", but got " << inputs.size() << " and "
                  << outputs.size();
    return false;
  }

  kernel_name_ = kernel_ptr_->name();
  beta1_ = kernel_ptr_->get_beta1();
  beta2_ = kernel_ptr_->get_beta2();
  epsilon_ = kernel_ptr_->get_epsilon();
  batch_rank_ = base_operator->get_batch_rank();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' dose not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  t_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  return true;
}

int ApplyAdamWithAmsgradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> variable_shape_ = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                              inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  t_elements_ = std::accumulate(variable_shape_.begin(), variable_shape_.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (t_elements_ == 0);
  if (is_null_input_) {
    return 0;
  }

  std::vector<int64_t> var_shape = inputs[kIndexVar]->GetShapeVector();
  std::vector<int64_t> m_shape = inputs[kIndexM]->GetShapeVector();
  std::vector<int64_t> v_shape = inputs[kIndexV]->GetShapeVector();
  std::vector<int64_t> vhat_shape = inputs[kIndexVhat]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kIndexLr]->GetShapeVector();
  std::vector<int64_t> grad_shape = inputs[kIndexGrad]->GetShapeVector();

  if (var_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'var' must be at least 1-D, but got scalar or None.";
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(var_shape, m_shape) || !IsSameShape(var_shape, v_shape) || !IsSameShape(var_shape, vhat_shape) ||
      !IsSameShape(var_shape, grad_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shapes of 'm/v/vhat/grad/var' must be the same, "
                  << "but get the shapes of 'm': " << Vector2Str(m_shape) << ", 'v': " << Vector2Str(v_shape)
                  << ", 'vhat': " << Vector2Str(vhat_shape) << ", 'grad': " << Vector2Str(grad_shape)
                  << " and 'var': " << Vector2Str(var_shape);
    return KRET_RESIZE_FAILED;
  }

  if (batch_rank_ < 0 || lr_shape.size() < static_cast<size_t>(batch_rank_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape size of 'lr' must be larger than or equal to 'batch_rank', "
                     "but got the shape of 'lr': "
                  << Vector2Str(lr_shape) << " and 'batch_rank': " << batch_rank_;
    return KRET_RESIZE_FAILED;
  }

  if (!lr_shape.empty()) {
    batch_size_ = std::accumulate(lr_shape.begin(), lr_shape.end(), int64_t(1), std::multiplies<int64_t>());
  }

  if (batch_size_ <= 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }

  size_t var_size_ = t_elements_ * t_size_ * batch_size_;
  size_t m_size_ = t_elements_ * t_size_ * batch_size_;
  size_t v_size_ = t_elements_ * t_size_ * batch_size_;
  size_t vhat_size_ = t_elements_ * t_size_ * batch_size_;
  size_t beta1_power_size_ = t_elements_ * t_size_ * batch_size_;
  size_t beta2_power_size_ = t_elements_ * t_size_ * batch_size_;
  size_t lr_size_ = t_elements_ * t_size_ * batch_size_;
  size_t grad_size_ = t_elements_ * t_size_ * batch_size_;
  input_size_list_.emplace_back(var_size_);
  input_size_list_.emplace_back(m_size_);
  input_size_list_.emplace_back(v_size_);
  input_size_list_.emplace_back(vhat_size_);
  input_size_list_.emplace_back(beta1_power_size_);
  input_size_list_.emplace_back(beta2_power_size_);
  input_size_list_.emplace_back(lr_size_);
  input_size_list_.emplace_back(grad_size_);
  output_size_list_.emplace_back(var_size_);
  output_size_list_.emplace_back(m_size_);
  output_size_list_.emplace_back(v_size_);
  output_size_list_.emplace_back(vhat_size_);

  return KRET_OK;
}

void ApplyAdamWithAmsgradGpuKernelMod::ResetResource() noexcept {
  t_elements_ = 0;
  is_null_input_ = false;
  input_size_list_.clear();
  output_size_list_.clear();
}

template <typename T>
bool ApplyAdamWithAmsgradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &workspace,
                                                    const std::vector<AddressPtr> &outputs) {
  T *var = GetDeviceAddress<T>(inputs, 0);
  T *m = GetDeviceAddress<T>(inputs, 1);
  T *v = GetDeviceAddress<T>(inputs, 2);
  T *vhat = GetDeviceAddress<T>(inputs, 3);
  T *beta1_power = GetDeviceAddress<T>(inputs, 4);
  T *beta2_power = GetDeviceAddress<T>(inputs, 5);
  T *lr = GetDeviceAddress<T>(inputs, 6);
  T *grad = GetDeviceAddress<T>(inputs, 7);

  ApplyAdamWithAmsgrad(t_elements_ * batch_size_, var, m, v, vhat, beta1_power, beta2_power, lr, grad, beta1_, beta2_,
                       epsilon_, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr_));

  return true;
}

std::vector<std::pair<KernelAttr, ApplyAdamWithAmsgradGpuKernelMod::ApplyAdamWithAmsgradFunc>>
  ApplyAdamWithAmsgradGpuKernelMod::func_list_ = {{KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddOutputAttr(kNumberTypeFloat64)
                                                     .AddOutputAttr(kNumberTypeFloat64)
                                                     .AddOutputAttr(kNumberTypeFloat64)
                                                     .AddOutputAttr(kNumberTypeFloat64)
                                                     .AddOutInRef(0, 0)
                                                     .AddOutInRef(1, 1)
                                                     .AddOutInRef(2, 2)
                                                     .AddOutInRef(3, 3),
                                                   &ApplyAdamWithAmsgradGpuKernelMod::LaunchKernel<double>},
                                                  {KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutInRef(0, 0)
                                                     .AddOutInRef(1, 1)
                                                     .AddOutInRef(2, 2)
                                                     .AddOutInRef(3, 3),
                                                   &ApplyAdamWithAmsgradGpuKernelMod::LaunchKernel<float>},
                                                  {KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddOutputAttr(kNumberTypeFloat16)
                                                     .AddOutputAttr(kNumberTypeFloat16)
                                                     .AddOutputAttr(kNumberTypeFloat16)
                                                     .AddOutputAttr(kNumberTypeFloat16)
                                                     .AddOutInRef(0, 0)
                                                     .AddOutInRef(1, 1)
                                                     .AddOutInRef(2, 2)
                                                     .AddOutInRef(3, 3),
                                                   &ApplyAdamWithAmsgradGpuKernelMod::LaunchKernel<half>}};

std::vector<KernelAttr> ApplyAdamWithAmsgradGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ApplyAdamWithAmsgradFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApplyAdamWithAmsgrad, ApplyAdamWithAmsgradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
