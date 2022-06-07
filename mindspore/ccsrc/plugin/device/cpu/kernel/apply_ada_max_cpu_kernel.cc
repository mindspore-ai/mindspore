/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <cmath>
#include <map>
#include "plugin/device/cpu/kernel/apply_ada_max_cpu_kernel.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace {
const size_t kZero = 0;
const size_t kOne = 1;
const size_t kTwo = 2;

constexpr size_t kScalarIndex = 0;
constexpr size_t kIndexVar = 0;
constexpr size_t kIndexM = 1;
constexpr size_t kIndexV = 2;
constexpr size_t kIndexBeta1Power = 3;
constexpr size_t kIndexLr = 4;
constexpr size_t kIndexBeta1 = 5;
constexpr size_t kIndexBeta2 = 6;
constexpr size_t kIndexEpsilon = 7;
constexpr size_t kIndexGrad = 8;

constexpr size_t kApplyAdaMaxInputsNum = 9;
constexpr size_t kApplyAdaMaxOutputsNum = 3;
}  // namespace

namespace mindspore {
namespace kernel {
bool ApplyAdaMaxCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  dtype_ = inputs[0]->GetDtype();
  return true;
}

int ApplyAdaMaxCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != 0) {
    return ret;
  }
  return ret;
}

bool ApplyAdaMaxCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyAdaMaxInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kApplyAdaMaxOutputsNum, kernel_name_);
  if (inputs[kIndexVar]->size != inputs[kIndexM]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype and shape of 'm' and 'var' must be the same, but got the memory size of 'm': "
                      << inputs[kIndexM]->size << " and 'var': " << inputs[kIndexVar]->size;
  }
  if (inputs[kIndexVar]->size != inputs[kIndexV]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype and shape of 'v' and 'var' must be the same, but got the memory size of 'v': "
                      << inputs[kIndexV]->size << " and 'var': " << inputs[kIndexVar]->size;
  }
  if (inputs[kIndexVar]->size != inputs[kIndexGrad]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype and shape of 'grad' and 'var' must be the same, "
                         "but got the memory size of 'grad': "
                      << inputs[kIndexGrad]->size << " and 'var': " << inputs[kIndexVar]->size;
  }
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', input dtype only support float16 and float32, but got ["
                            << dtype_ << "].";
  }
  return true;
}

template <typename T>
void ApplyAdaMaxCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &outputs) {
  T *var = reinterpret_cast<T *>(inputs[kIndexVar]->addr);
  T *m = reinterpret_cast<T *>(inputs[kIndexM]->addr);
  T *v = reinterpret_cast<T *>(inputs[kIndexV]->addr);
  T beta1_power = static_cast<T>(reinterpret_cast<float *>(inputs[kIndexBeta1Power]->addr)[kScalarIndex]);
  T lr = static_cast<T>(reinterpret_cast<float *>(inputs[kIndexLr]->addr)[kScalarIndex]);
  T beta1 = static_cast<T>(reinterpret_cast<float *>(inputs[kIndexBeta1]->addr)[kScalarIndex]);
  T beta2 = static_cast<T>(reinterpret_cast<float *>(inputs[kIndexBeta2]->addr)[kScalarIndex]);
  T epsilon = static_cast<T>(reinterpret_cast<float *>(inputs[kIndexEpsilon]->addr)[kScalarIndex]);
  T *grad = reinterpret_cast<T *>(inputs[kIndexGrad]->addr);

  auto one = static_cast<T>(1);
  if (beta1_power == one) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'beta1_power' can't be set 1.";
  }

  // multithreading
  size_t length = inputs[kZero]->size / sizeof(T);
  auto task = [this, &var, &m, &v, &beta1_power, &lr, &beta1, &beta2, &epsilon, &grad](size_t start, size_t end) {
    T one = static_cast<T>(1.0);
    for (size_t i = start; i < end; i++) {
      m[i] = static_cast<T>(beta1 * m[i] + (one - beta1) * grad[i]);
      auto zero = static_cast<T>(0);
      auto grad_abs = (grad[i] > zero) ? grad[i] : -grad[i];
      v[i] = std::max(beta2 * v[i], grad_abs);
      var[i] = var[i] - (lr / (one - beta1_power)) * (m[i] / (v[i] + epsilon));
    }
  };
  CPUKernelUtils::ParallelForAutoSearch(task, length, &parallel_search_info_);

  // Copy result to output tensor
  auto output_var = reinterpret_cast<T *>(outputs[kZero]->addr);
  auto output_m = reinterpret_cast<T *>(outputs[kOne]->addr);
  auto output_v = reinterpret_cast<T *>(outputs[kTwo]->addr);
  auto ret = memcpy_s(output_var, outputs[kZero]->size, var, inputs[kZero]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', launch kernel error: memcpy failed. Error no: " << ret;
  }
  ret = memcpy_s(output_m, outputs[kOne]->size, m, inputs[kOne]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', launch kernel error: memcpy failed. Error no: " << ret;
  }
  ret = memcpy_s(output_v, outputs[kTwo]->size, v, inputs[kTwo]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', launch kernel error: memcpy failed. Error no: " << ret;
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyAdaMax, ApplyAdaMaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
