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
#include "mindspore/core/ops/adam.h"
#include "plugin/device/cpu/kernel/adam_cpu_kernel.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kAdamInputsNum = 10;
constexpr size_t kAdamOutputsNum = 3;
constexpr size_t kScalarIndex = 0;
constexpr size_t kIndexVar = 0;
constexpr size_t kIndexM = 1;
constexpr size_t kIndexV = 2;
constexpr size_t kIndexBeta1Power = 3;
constexpr size_t kIndexBeta2Power = 4;
constexpr size_t kIndexLr = 5;
constexpr size_t kIndexBeta1 = 6;
constexpr size_t kIndexBeta2 = 7;
constexpr size_t kIndexEpsilon = 8;
constexpr size_t kIndexGrad = 9;
constexpr float kAdamBlock = 1000;
}  // namespace

template <typename T>
void AdamCpuKernelMod::LaunchAdam(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &) {
  T *var = reinterpret_cast<T *>(inputs[kIndexVar]->addr);
  T *m = reinterpret_cast<T *>(inputs[kIndexM]->addr);
  T *v = reinterpret_cast<T *>(inputs[kIndexV]->addr);
  float beta1_power = reinterpret_cast<float *>(inputs[kIndexBeta1Power]->addr)[kScalarIndex];
  float beta2_power = reinterpret_cast<float *>(inputs[kIndexBeta2Power]->addr)[kScalarIndex];
  float lr = reinterpret_cast<float *>(inputs[kIndexLr]->addr)[kScalarIndex];
  T beta1 = static_cast<T>(reinterpret_cast<float *>(inputs[kIndexBeta1]->addr)[kScalarIndex]);
  T beta2 = static_cast<T>(reinterpret_cast<float *>(inputs[kIndexBeta2]->addr)[kScalarIndex]);
  T epsilon = static_cast<T>(reinterpret_cast<float *>(inputs[kIndexEpsilon]->addr)[kScalarIndex]);
  T *gradient = reinterpret_cast<T *>(inputs[kIndexGrad]->addr);
  constexpr float ONE = 1.0;
  if (common::IsFloatEqual(beta1_power, ONE)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'beta1_power' can't be set 1.";
  }
  T new_lr = static_cast<T>(lr * std::sqrt(ONE - beta2_power) / (ONE - beta1_power));
  // multithreading
  size_t lens = inputs[kIndexVar]->size > 0 ? static_cast<size_t>(inputs[kIndexVar]->size / sizeof(T)) : 1;
  auto task = [this, &var, &m, &v, &gradient, new_lr, beta1, beta2, epsilon](size_t start, size_t end) {
    T one = static_cast<T>(1.0);
    for (size_t i = start; i < end; i++) {
      m[i] += (gradient[i] - m[i]) * (one - beta1);
      v[i] += (gradient[i] * gradient[i] - v[i]) * (one - beta2);
      T sqrt_v = static_cast<T>(std::sqrt(static_cast<double>(v[i])));
      if (use_nesterov_) {
        var[i] -= new_lr * (m[i] * beta1 + (one - beta1) * gradient[i]) / (sqrt_v + epsilon);
      } else {
        var[i] -= new_lr * m[i] / (sqrt_v + epsilon);
      }
    }
  };
  ParallelLaunch(task, lens, kAdamBlock, this);
}

void AdamCpuKernelMod::LaunchAdamNnacl(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &) {
  float *var = reinterpret_cast<float *>(inputs[kIndexVar]->addr);
  float *m = reinterpret_cast<float *>(inputs[kIndexM]->addr);
  float *v = reinterpret_cast<float *>(inputs[kIndexV]->addr);
  float beta1_power = reinterpret_cast<float *>(inputs[kIndexBeta1Power]->addr)[kScalarIndex];
  float beta2_power = reinterpret_cast<float *>(inputs[kIndexBeta2Power]->addr)[kScalarIndex];
  float lr = reinterpret_cast<float *>(inputs[kIndexLr]->addr)[kScalarIndex];
  float beta1 = reinterpret_cast<float *>(inputs[kIndexBeta1]->addr)[kScalarIndex];
  float beta2 = reinterpret_cast<float *>(inputs[kIndexBeta2]->addr)[kScalarIndex];
  float epsilon = reinterpret_cast<float *>(inputs[kIndexEpsilon]->addr)[kScalarIndex];
  float *gradient = reinterpret_cast<float *>(inputs[kIndexGrad]->addr);
  constexpr float ONE = 1.0;
  if (common::IsFloatEqual(beta1_power, ONE)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'beta1_power' can't be set 1.";
  }
  float new_lr = lr * std::sqrt(ONE - beta2_power) / (ONE - beta1_power);

  // multithreading
  size_t lens = inputs[kIndexVar]->size > 0 ? static_cast<size_t>(inputs[kIndexVar]->size / sizeof(float)) : 1;
  auto task = [this, &var, &m, &v, &gradient, new_lr, beta1, beta2, epsilon](size_t start, size_t end) {
    int ret = AdamFp32(var, m, v, new_lr, beta1, beta2, epsilon, gradient, start, end, use_nesterov_);
    if (ret != NNACL_OK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', AdamFp32 failed. Error no: " << ret;
    }
  };
  ParallelLaunch(task, lens, kAdamBlock, this);
}

bool AdamCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  dtype_ = inputs.at(kIndex0)->GetDtype();
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAdamInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAdamOutputsNum, kernel_name_);
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::Adam>(base_operator);
  use_nesterov_ = kernel_ptr_->get_use_nesterov();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

template <typename T>
bool AdamCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &workspace,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs[kIndexVar]->size != inputs[kIndexM]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape and dtype of 'm' and 'var' must be the same, but got the memory size of 'm': "
                      << inputs[kIndexM]->size << " and 'var': " << inputs[kIndexVar]->size;
  }
  if (inputs[kIndexVar]->size != inputs[kIndexV]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape and dtype of 'v' and 'var' must be the same, but got the memory size of 'v': "
                      << inputs[kIndexV]->size << " and 'var': " << inputs[kIndexVar]->size;
  }

  size_t f_size = sizeof(float);
  if (inputs[kIndexBeta1Power]->size != f_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'beta1_power' must be float, but got 'beta1_power': " << inputs[kIndexBeta1Power];
  }
  if (inputs[kIndexBeta2Power]->size != f_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'beta2_power' must be float, but got 'beta2_power': " << inputs[kIndexBeta2Power];
  }
  if (inputs[kIndexLr]->size != f_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'lr' must be float, but got 'lr': " << inputs[kIndexLr];
  }
  if (inputs[kIndexBeta1]->size != f_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'beta1' must be float, but got 'beta1': " << inputs[kIndexBeta1];
  }
  if (inputs[kIndexBeta2]->size != f_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'beta2' must be float, but got 'beta2': " << inputs[kIndexBeta2];
  }
  if (inputs[kIndexEpsilon]->size != f_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'epsilon' must be float, but got 'epsilon': " << inputs[kIndexEpsilon];
  }

  if (dtype_ == kNumberTypeFloat32) {
    LaunchAdamNnacl(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchAdam<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'var' must be Float16 or Float32, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

std::vector<std::pair<KernelAttr, AdamCpuKernelMod::AdamFunc>> AdamCpuKernelMod::func_list_ = {
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
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &AdamCpuKernelMod::LaunchKernel<float>},
};

std::vector<KernelAttr> AdamCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, AdamFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Adam, AdamCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
