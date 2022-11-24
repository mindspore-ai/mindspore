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
#include "plugin/device/cpu/kernel/apply_gradient_descent_cpu_kernel.h"
#include <functional>
#include <complex>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace {
const size_t kZero = 0;
const size_t kOne = 1;
const size_t kTwo = 2;
constexpr size_t kApplyGradientDescentInputsNum = 3;
constexpr size_t kApplyGradientDescentOutputsNum = 1;
}  // namespace

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool ApplyGradientDescentCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  batch_rank_ = base_operator->get_batch_rank();
  dtype_ = inputs[kZero]->GetDtype();

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyGradientDescentInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kApplyGradientDescentOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  return true;
}

int ApplyGradientDescentCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }

  // get input size and the inner input size for one batch.
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  input_size_ = std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  if (batch_rank_ != 0) {
    inner_input_size_ =
      std::accumulate(input_shape.begin() + batch_rank_, input_shape.end(), size_t(1), std::multiplies<size_t>());
  }
  return ret;
}

template <typename T>
bool ApplyGradientDescentCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &outputs) {
  auto var_addr = reinterpret_cast<T *>(inputs[kZero]->addr);
  auto alpha_addr = reinterpret_cast<T *>(inputs[kOne]->addr);
  auto delta_addr = reinterpret_cast<T *>(inputs[kTwo]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[kZero]->addr);
  auto task = [this, &var_addr, &alpha_addr, &delta_addr, &output_addr](size_t start, size_t end) {
    for (size_t pos = start; pos < end; pos++) {
      size_t batch_index = inner_input_size_ <= 0 ? 0 : pos / inner_input_size_;
      const T alpha_value = alpha_addr[batch_index];
      var_addr[pos] -= alpha_value * delta_addr[pos];
      output_addr[pos] = var_addr[pos];
    }
  };
  ParallelLaunch(task, input_size_, 0, this, pool_);
  return true;
}

std::vector<std::pair<KernelAttr, ApplyGradientDescentCpuKernelMod::ApplyGradientDescentLaunchFunc>>
  ApplyGradientDescentCpuKernelMod::func_list_ = {{KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   &ApplyGradientDescentCpuKernelMod::LaunchKernel<float>},
                                                  {KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddInputAttr(kNumberTypeFloat16)
                                                     .AddOutputAttr(kNumberTypeFloat16),
                                                   &ApplyGradientDescentCpuKernelMod::LaunchKernel<float16>},
                                                  {KernelAttr()
                                                     .AddInputAttr(kNumberTypeInt8)
                                                     .AddInputAttr(kNumberTypeInt8)
                                                     .AddInputAttr(kNumberTypeInt8)
                                                     .AddOutputAttr(kNumberTypeInt8),
                                                   &ApplyGradientDescentCpuKernelMod::LaunchKernel<int8_t>},
                                                  {KernelAttr()
                                                     .AddInputAttr(kNumberTypeUInt8)
                                                     .AddInputAttr(kNumberTypeUInt8)
                                                     .AddInputAttr(kNumberTypeUInt8)
                                                     .AddOutputAttr(kNumberTypeUInt8),
                                                   &ApplyGradientDescentCpuKernelMod::LaunchKernel<uint8_t>},
                                                  {KernelAttr()
                                                     .AddInputAttr(kNumberTypeInt16)
                                                     .AddInputAttr(kNumberTypeInt16)
                                                     .AddInputAttr(kNumberTypeInt16)
                                                     .AddOutputAttr(kNumberTypeInt16),
                                                   &ApplyGradientDescentCpuKernelMod::LaunchKernel<int16_t>},
                                                  {KernelAttr()
                                                     .AddInputAttr(kNumberTypeUInt16)
                                                     .AddInputAttr(kNumberTypeUInt16)
                                                     .AddInputAttr(kNumberTypeUInt16)
                                                     .AddOutputAttr(kNumberTypeUInt16),
                                                   &ApplyGradientDescentCpuKernelMod::LaunchKernel<uint16_t>},
                                                  {KernelAttr()
                                                     .AddInputAttr(kNumberTypeUInt32)
                                                     .AddInputAttr(kNumberTypeUInt32)
                                                     .AddInputAttr(kNumberTypeUInt32)
                                                     .AddOutputAttr(kNumberTypeUInt32),
                                                   &ApplyGradientDescentCpuKernelMod::LaunchKernel<uint32_t>},
                                                  {KernelAttr()
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeInt64),
                                                   &ApplyGradientDescentCpuKernelMod::LaunchKernel<int64_t>},
                                                  {KernelAttr()
                                                     .AddInputAttr(kNumberTypeUInt64)
                                                     .AddInputAttr(kNumberTypeUInt64)
                                                     .AddInputAttr(kNumberTypeUInt64)
                                                     .AddOutputAttr(kNumberTypeUInt64),
                                                   &ApplyGradientDescentCpuKernelMod::LaunchKernel<uint64_t>},
                                                  {KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddOutputAttr(kNumberTypeFloat64),
                                                   &ApplyGradientDescentCpuKernelMod::LaunchKernel<double>},
                                                  {KernelAttr()
                                                     .AddInputAttr(kNumberTypeComplex64)
                                                     .AddInputAttr(kNumberTypeComplex64)
                                                     .AddInputAttr(kNumberTypeComplex64)
                                                     .AddOutputAttr(kNumberTypeComplex64),
                                                   &ApplyGradientDescentCpuKernelMod::LaunchKernel<complex64>},
                                                  {KernelAttr()
                                                     .AddInputAttr(kNumberTypeComplex128)
                                                     .AddInputAttr(kNumberTypeComplex128)
                                                     .AddInputAttr(kNumberTypeComplex128)
                                                     .AddOutputAttr(kNumberTypeComplex128),
                                                   &ApplyGradientDescentCpuKernelMod::LaunchKernel<complex128>}};

std::vector<KernelAttr> ApplyGradientDescentCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, ApplyGradientDescentCpuKernelMod::ApplyGradientDescentLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyGradientDescent, ApplyGradientDescentCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
