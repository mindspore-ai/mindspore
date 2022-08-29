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

#include "plugin/device/cpu/kernel/apply_addsign_cpu_kernel.h"
#include <thread>
#include <vector>
#include <map>

namespace mindspore {
namespace kernel {
namespace {
const size_t kVar = 0;
const size_t kM = 1;
const size_t kLr = 2;
const size_t kAlpha = 3;
const size_t kSignDecay = 4;
const size_t kBeta = 5;
const size_t kGrad = 6;
constexpr size_t kSizeFloat16 = 2;
constexpr size_t kSizeFloat32 = 4;
constexpr size_t kSizeFloat64 = 8;
constexpr size_t kApplyAddsignInputsNum = 7;
constexpr size_t kApplyAddsignOutputsNum = 2;
}  // namespace

bool ApplyAddsignCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  dtype_ = inputs[0]->GetDtype();
  return true;
}

bool ApplyAddsignCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                      const std::vector<AddressPtr> &outputs) {
  CheckParam(inputs, outputs);
  switch (dtype_) {
    case kNumberTypeFloat16:
      LaunchKernel<float16>(inputs, outputs);
      break;
    case kNumberTypeFloat32:
      LaunchKernel<float>(inputs, outputs);
      break;
    case kNumberTypeFloat64:
      LaunchKernel<double>(inputs, outputs);
      break;
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', can not support the data type "
                              << TypeIdToString(dtype_);
  }
  return true;
}

int ApplyAddsignCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) != 0) {
    MS_LOG(WARNING) << kernel_name_ << "reinit failed.";
    return ret;
  }
  return 0;
}

void ApplyAddsignCpuKernelMod::CheckParam(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &outputs) const {
  // inputs: var, m, lr, alpha, sign_decay, beta, gradient
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyAddsignInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kApplyAddsignOutputsNum, kernel_name_);
  if (inputs[kVar]->size != inputs[kM]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape and dtype of 'var' and 'm' must be the same, "
                         "but got the memory size of 'var': "
                      << inputs[kVar]->size << " and 'm': " << inputs[kM]->size;
  }
  if (inputs[kVar]->size != inputs[kGrad]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape and dtype of 'var' and 'gradient' must be the same, "
                         "but got the memory size of 'var': "
                      << inputs[kVar]->size << " and 'gradient': " << inputs[kGrad]->size;
  }
}

template <typename T>
void ApplyAddsignCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &outputs) {
  auto *var = static_cast<T *>(inputs[kVar]->addr);
  auto *m = static_cast<T *>(inputs[kM]->addr);
  const auto *lr = static_cast<T *>(inputs[kLr]->addr);
  const auto *alpha = static_cast<T *>(inputs[kAlpha]->addr);
  const auto *sign_decay = static_cast<T *>(inputs[kSignDecay]->addr);
  const auto *beta = static_cast<T *>(inputs[kBeta]->addr);
  const auto *gradient = static_cast<T *>(inputs[kGrad]->addr);

  size_t length = inputs[kVar]->size / sizeof(T);
  auto task = [this, &var, &m, &lr, &alpha, &sign_decay, &beta, &gradient](size_t start, size_t end) {
    LaunchApplyAddsign(var, m, lr, alpha, sign_decay, beta, gradient, start, end);
  };
  CPUKernelUtils::ParallelForAutoSearch(task, length, &parallel_search_info_);

  auto output_var = static_cast<T *>(outputs[kVar]->addr);
  auto output_m = static_cast<T *>(outputs[kM]->addr);
  auto ret_var = memcpy_s(output_var, outputs[kVar]->size, var, inputs[kVar]->size);
  if (ret_var != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', launch kernel error: memcpy failed. Error no: " << ret_var;
  }
  auto ret_m = memcpy_s(output_m, outputs[kM]->size, m, inputs[kM]->size);
  if (ret_m != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', launch kernel error: memcpy failed. Error no: " << ret_m;
  }
}

template <typename T>
T sign(const T &value) {
  auto zero = static_cast<T>(0);
  if (value > zero) {
    return static_cast<T>(1.0);
  }
  if (zero > value) {
    return static_cast<T>(-1.0);
  }
  return static_cast<T>(0.0);
}

template <typename T>
void ApplyAddsignCpuKernelMod::LaunchApplyAddsign(T *var, T *m, const T *lr, const T *alpha, const T *sign_decay,
                                                  const T *beta, const T *gradient, size_t start, size_t end) const {
  auto one = static_cast<T>(1.0);
  for (size_t i = start; i < end; ++i) {
    m[i] = beta[0] * m[i] + (one - beta[0]) * gradient[i];
    var[i] -= lr[0] * (alpha[0] + sign_decay[0] * sign(gradient[i]) * sign(m[i])) * gradient[i];
  }
}

std::vector<KernelAttr> ApplyAddsignCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> kernel_attr_list = {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddOutputAttr(kNumberTypeFloat16)
                                                .AddOutputAttr(kNumberTypeFloat16),
                                              KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddOutputAttr(kNumberTypeFloat32)
                                                .AddOutputAttr(kNumberTypeFloat32),
                                              KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddOutputAttr(kNumberTypeFloat64)
                                                .AddOutputAttr(kNumberTypeFloat64)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyAddSign, ApplyAddsignCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
