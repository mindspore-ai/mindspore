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
bool ApplyGradientDescentCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  batch_rank_ = base_operator->get_batch_rank();
  dtype_ = inputs[kZero]->GetDtype();
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

bool ApplyGradientDescentCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyGradientDescentInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kApplyGradientDescentOutputsNum, kernel_name_);
  if (input_size_ == 0) {
    return true;
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
void ApplyGradientDescentCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
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
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyGradientDescent, ApplyGradientDescentCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
