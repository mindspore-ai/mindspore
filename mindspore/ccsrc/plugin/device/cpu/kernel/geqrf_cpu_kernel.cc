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

#include "plugin/device/cpu/kernel/geqrf_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 2;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kOutputIndex0 = 0;
constexpr size_t kOutputIndex1 = 1;
}  // namespace

bool GeqrfCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For" << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int GeqrfCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  std::vector<int64_t> input0_tensor_shape = inputs[0]->GetShapeVector();
  num_m = static_cast<size_t>(input0_tensor_shape[0]);
  num_n = static_cast<size_t>(input0_tensor_shape[1]);
  return KRET_OK;
}

template <typename T>
void GeqrfCpuKernelMod::Larfg(size_t n, size_t vm, size_t vn, T *x, T *tau) {
  T zero = static_cast<T>(0);
  if (n <= 1) {
    *tau = zero;
    return;
  }
  T xnorm = zero;
  for (size_t i = vm + 1; i < vm + n; i++) {
    xnorm = xnorm + (*(x + i * num_n + vn) * *(x + i * num_n + vn));
  }
  xnorm = static_cast<T>(sqrt(xnorm));
  if (xnorm == zero) {
    *tau = zero;
    return;
  } else {
    T beta = sqrt((*(x + vm * num_n + vn) * *(x + vm * num_n + vn)) + xnorm * xnorm);
    if (*(x + vm * num_n + vn) > zero) {
      beta = -beta;
    }
    if (beta == zero) {
      return;
    }
    *tau = (beta - *(x + vm * num_n + vn)) / beta;
    auto scal = *(x + vm * num_n + vn) - beta;
    for (size_t i = vm + 1; i < vm + n; i++) {
      *(x + i * num_n + vn) /= scal;
    }
    *(x + vm * num_n + vn) = beta;
  }
}

template <typename T>
std::unique_ptr<T[]> GeqrfCpuKernelMod::Larf(size_t m, size_t n, T *x, T *tau, std::unique_ptr<T[]> workspace,
                                             size_t cm, size_t cn) {
  if (m <= 0 || n <= 0) {
    return std::move(workspace);
  }
  for (size_t i = 0; i < n; i++) {
    workspace[i] = static_cast<T>(0);
  }
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      workspace[j] += *(x + ((cm + i) * num_n) + (cn - 1)) * *(x + ((cm + i) * num_n) + (cn + j));
    }
  }
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      *(x + ((cm + i) * num_n) + (cn + j)) -= (*tau) * *(x + ((cm + i) * num_n) + (cn - 1)) * workspace[j];
    }
  }
  return std::move(workspace);
}

template <typename T>
void GeqrfCpuKernelMod::Geqrf(size_t num_m_, size_t num_n_, T *x, T *tau) {
  if (num_m_ < 0 || num_n_ < 0) {
    return;
  }
  size_t k = std::min(num_m_, num_n_);
  T one = static_cast<T>(1);
  std::unique_ptr<T[]> workspace = std::make_unique<T[]>(num_n_);
  for (size_t i = 0; i < k; i++) {
    Larfg<T>(num_m_ - i, i, i, x, tau + i);
    T aii = *(x + i * num_n_ + i);
    *(x + i * num_n_ + i) = one;
    workspace = Larf<T>(num_m_ - i, num_n_ - i - 1, x, tau + i, std::move(workspace), i, i + 1);
    *(x + i * num_n_ + i) = aii;
  }
}

template <typename T>
bool GeqrfCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  T *x = static_cast<T *>(inputs[kInputIndex0]->addr);
  T *y = static_cast<T *>(outputs[kOutputIndex0]->addr);
  T *tau = static_cast<T *>(outputs[kOutputIndex1]->addr);
  for (size_t i = 0; i < num_m; i++) {
    for (size_t j = 0; j < num_n; j++) {
      *(y + i * num_n + j) = *(x + i * num_n + j);
    }
  }
  Geqrf<T>(num_m, num_n, y, tau);
  return true;
}

std::vector<std::pair<KernelAttr, GeqrfCpuKernelMod::GeqrfLaunchFunc>> GeqrfCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &GeqrfCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &GeqrfCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> GeqrfCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GeqrfLaunchFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Geqrf, GeqrfCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
