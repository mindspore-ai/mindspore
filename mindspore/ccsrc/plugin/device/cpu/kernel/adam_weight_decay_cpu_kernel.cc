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

#include "plugin/device/cpu/kernel/adam_weight_decay_cpu_kernel.h"

#include <cmath>

#include "plugin/device/cpu/kernel/nnacl/errorcode.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSizeFloat32 = sizeof(float);
constexpr size_t kScalarIndex = 0;
constexpr size_t kAdamWeightDecayInputsNum = 9;
constexpr size_t kAdamWeightDecayOutputsNum = 3;
}  // namespace

template <typename T>
void AdamWeightDecayCpuKernelMod::LaunchAdamWeightDecay(const std::vector<AddressPtr> &inputs,
                                                        const std::vector<AddressPtr> &) {
  T *var = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  T *m = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  T *v = reinterpret_cast<T *>(inputs[kIndex2]->addr);
  T lr = static_cast<T>(reinterpret_cast<float *>(inputs[kIndex3]->addr)[kScalarIndex]);
  T beta1 = static_cast<T>(reinterpret_cast<float *>(inputs[kIndex4]->addr)[kScalarIndex]);
  T beta2 = static_cast<T>(reinterpret_cast<float *>(inputs[kIndex5]->addr)[kScalarIndex]);
  T epsilon = static_cast<T>(reinterpret_cast<float *>(inputs[kIndex6]->addr)[kScalarIndex]);
  T decay = static_cast<T>(reinterpret_cast<float *>(inputs[kIndex7]->addr)[kScalarIndex]);
  T *gradient = reinterpret_cast<T *>(inputs[kIndex8]->addr);
  const T one = static_cast<T>(1.0);
  const T beta1_minus = one - beta1;
  const T beta2_minus = one - beta2;

  // multithreading
  size_t lens = inputs[kIndex0]->size > 0 ? static_cast<size_t>(inputs[kIndex0]->size / sizeof(T)) : 1;
  std::function<void(size_t, size_t)> task;
  task = [&](size_t start, size_t end) {
    // remaining
    for (size_t i = start; i < end; i++) {
      m[i] += (gradient[i] - m[i]) * beta1_minus;
      v[i] += (gradient[i] * gradient[i] - v[i]) * beta2_minus;
      T sqrt_v = static_cast<T>(std::sqrt(static_cast<double>(v[i])));
      auto update = m[i] / (sqrt_v + epsilon) + decay * var[i];
      var[i] -= lr * update;
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
}

void AdamWeightDecayCpuKernelMod::LaunchAdamWeightDecayNnacl(const std::vector<AddressPtr> &inputs,
                                                             const std::vector<AddressPtr> &) {
  auto var = reinterpret_cast<float *>(inputs[kIndex0]->addr);
  auto m = reinterpret_cast<float *>(inputs[kIndex1]->addr);
  auto v = reinterpret_cast<float *>(inputs[kIndex2]->addr);
  auto lr = reinterpret_cast<float *>(inputs[kIndex3]->addr)[kScalarIndex];
  auto beta1 = reinterpret_cast<float *>(inputs[kIndex4]->addr)[kScalarIndex];
  auto beta2 = reinterpret_cast<float *>(inputs[kIndex5]->addr)[kScalarIndex];
  auto epsilon = reinterpret_cast<float *>(inputs[kIndex6]->addr)[kScalarIndex];
  auto decay = reinterpret_cast<float *>(inputs[kIndex7]->addr)[kScalarIndex];
  auto gradient = reinterpret_cast<float *>(inputs[kIndex8]->addr);

  // multithreading
  size_t lens = inputs[kIndex0]->size > 0 ? static_cast<size_t>(inputs[kIndex0]->size / sizeof(float)) : 1;
  std::function<void(size_t, size_t)> task;
  task = [&](size_t start, size_t end) {
    int ret = AdamWeightDecayFp32(var, m, v, lr, beta1, beta2, epsilon, decay, gradient, start, end);
    if (ret != NNACL_OK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', AdamWeightDecayFp32 failed. Error no: " << ret;
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
}

bool AdamWeightDecayCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAdamWeightDecayInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAdamWeightDecayOutputsNum, kernel_name_);
  dtype_ = inputs[kIndex0]->GetDtype();
  return true;
}

bool AdamWeightDecayCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs[kIndex0]->size != inputs[kIndex1]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype and shape of 'm' and 'var' must be the same, but got the memory size of 'm': "
                      << inputs[kIndex1]->size << " and 'var': " << inputs[kIndex0]->size;
  }
  if (inputs[kIndex0]->size != inputs[kIndex2]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype and shape of 'v' and 'var' must be the same, but got the memory size of 'v': "
                      << inputs[kIndex2]->size << " and 'var': " << inputs[kIndex0]->size;
  }
  if (inputs[kIndex0]->size != inputs[kIndex8]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype and shape of 'grad' and 'var' must be the same, "
                         "but got the memory size of 'grad': "
                      << inputs[kIndex8]->size << " and 'var': " << inputs[kIndex0]->size;
  }
  if (inputs[kIndex3]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'lr' must be float, but got 'lr': " << inputs[kIndex3];
  }
  if (inputs[kIndex4]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'beta1' must be float, but got 'beta1': " << inputs[kIndex4];
  }
  if (inputs[kIndex5]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'beta2' must be float, but got 'beta2': " << inputs[kIndex5];
  }
  if (inputs[kIndex6]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'epsilon' must be float, but got 'epsilon': " << inputs[kIndex6];
  }
  if (inputs[kIndex7]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'decay' must be float, but got 'decay': " << inputs[kIndex7];
  }
  if (dtype_ == kNumberTypeFloat32) {
    LaunchAdamWeightDecayNnacl(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchAdamWeightDecay<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'var' must be Float16 or Float32, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AdamWeightDecay, AdamWeightDecayCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
