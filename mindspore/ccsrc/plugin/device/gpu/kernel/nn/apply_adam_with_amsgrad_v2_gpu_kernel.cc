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

#include <algorithm>
#include "plugin/device/gpu/kernel/nn/apply_adam_with_amsgrad_v2_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/apply_adam_with_amsgrad_v2_impl.cuh"
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "include/curand.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kApplyAdamWithAmsgradV2InputsNum = 11;
constexpr size_t kApplyAdamWithAmsgradV2OutputsNum = 4;
}  // namespace

bool ApplyAdamWithAmsgradV2GpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs) {
  if (inputs.size() != kApplyAdamWithAmsgradV2InputsNum || outputs.size() != kApplyAdamWithAmsgradV2OutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size should be "
                  << kApplyAdamWithAmsgradV2InputsNum << " and " << kApplyAdamWithAmsgradV2OutputsNum << ", but got "
                  << inputs.size() << " and " << outputs.size();
    return false;
  }

  kernel_name_ = base_operator->name();
  batch_rank_ = base_operator->get_batch_rank();
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::ApplyAdamWithAmsgradV2>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr_, false);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' dose not support this kernel type: " << kernel_attr;
    return false;
  }

  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  return true;
}

int ApplyAdamWithAmsgradV2GpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs,
                                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  input_elements_ = 0;

  std::vector<int64_t> var_shape = inputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> m_shape = inputs[kIndex1]->GetShapeVector();
  std::vector<int64_t> v_shape = inputs[kIndex2]->GetShapeVector();
  std::vector<int64_t> vhat_shape = inputs[kIndex3]->GetShapeVector();
  std::vector<int64_t> beta1_power_shape = inputs[kIndex4]->GetShapeVector();
  std::vector<int64_t> beta2_power_shape = inputs[kIndex5]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kIndex6]->GetShapeVector();
  std::vector<int64_t> grad_shape = inputs[kIndex10]->GetShapeVector();

  if (var_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'var' must be at least 1-D, but got scalar or None.";
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(var_shape, m_shape) || !IsSameShape(var_shape, v_shape) || !IsSameShape(var_shape, vhat_shape) ||
      !IsSameShape(var_shape, grad_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shapes of 'm/v/vhat/grad/var' must be the same, "
                  << "but get the shapes of 'm': " << m_shape << ", 'v': " << v_shape << ", 'vhat': " << vhat_shape
                  << ", 'grad': " << grad_shape << " and 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(beta1_power_shape, beta2_power_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shapes of 'beta1_power' and 'beta2_power' must be the same, "
                  << "but get the shapes of 'beta1_power': " << beta1_power_shape
                  << " and 'beta2_power': " << beta2_power_shape;
    return KRET_RESIZE_FAILED;
  }

  if (batch_rank_ < 0 || lr_shape.size() != static_cast<size_t>(batch_rank_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape size of 'lr' must be equal to 'batch_rank', "
                     "but got the shape of 'lr': "
                  << lr_shape << " and 'batch_rank': " << batch_rank_;
    return KRET_RESIZE_FAILED;
  }

  batch_size_ = 1;
  if (!lr_shape.empty()) {
    batch_size_ = std::accumulate(lr_shape.begin(), lr_shape.end(), batch_size_, std::multiplies<int64_t>());
  }
  if (batch_size_ <= 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }

  input_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), 1, std::multiplies<int64_t>());
  input_elements_ = input_elements_ / batch_size_;
  if (batch_rank_ > 1) {
    if (var_shape.size() < lr_shape.size()) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the shape size of 'var' must be greater than 'lr_shape', but got the shape of 'var': "
                    << var_shape << " and 'lr_shape': " << lr_shape;
      return KRET_RESIZE_FAILED;
    }
    std::vector<int64_t> var_batch_shape(var_shape.begin(), var_shape.begin() + batch_rank_);
    if (!IsSameShape(lr_shape, var_batch_shape)) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the batch shape of 'var' must be the same as the shape of 'lr', "
                       "but got the batch shape of 'var': "
                    << var_batch_shape << " and the shape of 'lr': " << lr_shape;
      return KRET_RESIZE_FAILED;
    }
  }

  return KRET_OK;
}

bool ApplyAdamWithAmsgradV2GpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &workspace,
                                                const std::vector<kernel::AddressPtr> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  kernel_func_(this, inputs, outputs, stream_ptr);
  return true;
}

template <typename T>
bool ApplyAdamWithAmsgradV2GpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                      const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto *var = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto *m = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  auto *v = reinterpret_cast<T *>(inputs[kIndex2]->addr);
  auto *vhat = reinterpret_cast<T *>(inputs[kIndex3]->addr);
  auto *beta1_power = reinterpret_cast<T *>(inputs[kIndex4]->addr);
  auto *beta2_power = reinterpret_cast<T *>(inputs[kIndex5]->addr);
  auto *lr = reinterpret_cast<T *>(inputs[kIndex6]->addr);
  auto *beta1 = reinterpret_cast<T *>(inputs[kIndex7]->addr);
  auto *beta2 = reinterpret_cast<T *>(inputs[kIndex8]->addr);
  auto *epsilon = reinterpret_cast<T *>(inputs[kIndex9]->addr);
  auto *grad = reinterpret_cast<T *>(inputs[kIndex10]->addr);

  auto *output_var = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  auto *output_m = reinterpret_cast<T *>(outputs[kIndex1]->addr);
  auto *output_v = reinterpret_cast<T *>(outputs[kIndex2]->addr);
  auto *output_vhat = reinterpret_cast<T *>(outputs[kIndex3]->addr);

  auto status = CalApplyAdamWithAmsgradV2(input_elements_, batch_size_, var, m, v, vhat, beta1_power, beta2_power, lr,
                                          grad, beta1, beta2, epsilon, output_var, output_m, output_v, output_vhat,
                                          device_id_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, ApplyAdamWithAmsgradV2GpuKernelMod::KernelFunc>>
  ApplyAdamWithAmsgradV2GpuKernelMod::func_list_ = {{KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat64)
                                                       .AddInputAttr(kNumberTypeFloat64)
                                                       .AddInputAttr(kNumberTypeFloat64)
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
                                                     &ApplyAdamWithAmsgradV2GpuKernelMod::LaunchKernel<double>},
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
                                                     &ApplyAdamWithAmsgradV2GpuKernelMod::LaunchKernel<float>},
                                                    {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
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
                                                     &ApplyAdamWithAmsgradV2GpuKernelMod::LaunchKernel<half>}};

std::vector<KernelAttr> ApplyAdamWithAmsgradV2GpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApplyAdamWithAmsgradV2, ApplyAdamWithAmsgradV2GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
