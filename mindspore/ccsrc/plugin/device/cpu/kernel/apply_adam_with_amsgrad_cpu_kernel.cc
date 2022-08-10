/**
 * Copyright 2022-2022 Huawei Technologies Co., Ltd
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

#include <map>
#include <functional>
#include <algorithm>

#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/apply_adam_with_amsgrad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.h"
#include "mindspore/core/ops/apply_adam_with_amsgrad.h"
#include "utils/ms_utils.h"

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

bool ApplyAdamWithAmsgradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ApplyAdamWithAmsgrad>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  dtype_ = inputs[0]->GetDtype();
  batch_rank_ = base_operator->get_batch_rank();

  if (inputs.size() != kApplyAdamWithAmsgradInputsNum || outputs.size() != kApplyAdamWithAmsgradOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size should be " << kApplyAdamWithAmsgradInputsNum
                  << " and " << kApplyAdamWithAmsgradOutputsNum << ", but got " << inputs.size() << " and "
                  << outputs.size();
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }

  beta1_ = kernel_ptr->get_beta1();
  beta2_ = kernel_ptr->get_beta2();
  epsilon_ = kernel_ptr->get_epsilon();

  return true;
}

int ApplyAdamWithAmsgradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, others)) != 0) {
    return ret;
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

  if (batch_rank_ < 0 || lr_shape.size() != static_cast<size_t>(batch_rank_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape size of 'lr' must be equal to 'batch_rank', "
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

  int64_t temp_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), int64_t(1), std::multiplies<int64_t>());
  input_elements_ = static_cast<size_t>(temp_elements_ / batch_size_);

  return 0;
}

template <typename T>
void ApplyAdamWithAmsgradCpuKernelMod::LaunchApplyAdamWithAmsgrad(const std::vector<AddressPtr> &inputs,
                                                                  const std::vector<AddressPtr> &) {
  T *var = reinterpret_cast<T *>(inputs[kIndexVar]->addr);
  T *m = reinterpret_cast<T *>(inputs[kIndexM]->addr);
  T *v = reinterpret_cast<T *>(inputs[kIndexV]->addr);
  T *vhat = reinterpret_cast<T *>(inputs[kIndexVhat]->addr);
  T *beta1_power = reinterpret_cast<T *>(inputs[kIndexBeta1Power]->addr);
  T *beta2_power = reinterpret_cast<T *>(inputs[kIndexBeta2Power]->addr);
  T *lr = reinterpret_cast<T *>(inputs[kIndexLr]->addr);
  T *gradient = reinterpret_cast<T *>(inputs[kIndexGrad]->addr);

  T beta1 = static_cast<T>(beta1_);
  T beta2 = static_cast<T>(beta2_);
  T epsilon = static_cast<T>(epsilon_);

  T ONE = static_cast<T>(1.0);
  for (int64_t b = 0; b < batch_size_; b++) {
    // multithreading
    T new_lr = lr[b] * static_cast<T>(std::sqrt(static_cast<double>(ONE - beta2_power[b]))) / (ONE - beta1_power[b]);
    auto task = [this, &var, &m, &v, &vhat, &gradient, new_lr, beta1, beta2, epsilon](size_t start, size_t end) {
      T one = static_cast<T>(1.0);
      for (size_t i = start; i < end; i++) {
        m[i] += (gradient[i] - m[i]) * (one - beta1);
        v[i] += (gradient[i] * gradient[i] - v[i]) * (one - beta2);
        vhat[i] = std::max(vhat[i], v[i]);
        T sqrt_vhat = static_cast<T>(std::sqrt(static_cast<double>(vhat[i])));
        var[i] -= new_lr * m[i] / (sqrt_vhat + epsilon);
      }
    };
    ParallelLaunchAutoSearch(task, input_elements_, this, &parallel_search_info_);
    var = var + input_elements_;
    m = m + input_elements_;
    v = v + input_elements_;
    vhat = vhat + input_elements_;
    gradient = gradient + input_elements_;
  }
}

bool ApplyAdamWithAmsgradCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &workspace,
                                              const std::vector<AddressPtr> &outputs) {
  if (inputs[kIndexVar]->size != inputs[kIndexM]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape and dtype of 'm' and 'var' should be same, but got the memory size of 'm': "
                      << inputs[kIndexM]->size << " and 'var': " << inputs[kIndexVar]->size;
  }
  if (inputs[kIndexVar]->size != inputs[kIndexV]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape and dtype of 'v' and 'var' should be same, but got the memory size of 'v': "
                      << inputs[kIndexV]->size << " and 'var': " << inputs[kIndexVar]->size;
  }
  if (inputs[kIndexVar]->size != inputs[kIndexVhat]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape and dtype of 'vhat' and 'var' should be same, but got the size of 'vhat': "
                      << inputs[kIndexVhat]->size << " and 'var': " << inputs[kIndexVar]->size;
  }
  if (inputs[kIndexVar]->size != inputs[kIndexGrad]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape and dtype of 'gradient' and 'var' should be same, but got "
                         "the memory size of 'gradient': "
                      << inputs[kIndexGrad]->size << " and 'var': " << inputs[kIndexVar]->size;
  }

  if (dtype_ == kNumberTypeFloat32) {
    LaunchApplyAdamWithAmsgrad<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchApplyAdamWithAmsgrad<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'var' should be float16 or float32, but get "
                      << TypeIdToType(dtype_)->ToString();
  }

  return true;
}

std::vector<KernelAttr> ApplyAdamWithAmsgradCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
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
                                                 KernelAttr()
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
                                                   .AddOutInRef(3, 3)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyAdamWithAmsgrad, ApplyAdamWithAmsgradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
