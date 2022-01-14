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

#include "backend/kernel_compiler/cpu/apply_adagrad_cpu_kernel.h"

#include <thread>
#include <vector>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSizeFloat16 = 2;
constexpr size_t kSizeFloat32 = 4;
constexpr size_t kApplyAdagradInputsNum = 4;
constexpr size_t kApplyAdagradOutputsNum = 2;
}  // namespace

void ApplyAdagradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  update_slots_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "update_slots");
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

bool ApplyAdagradCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                      const std::vector<AddressPtr> &outputs) {
  CheckParam(inputs, outputs);
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'var' should be Float16 or Float32, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

void ApplyAdagradCpuKernelMod::CheckParam(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &outputs) const {
  // inputs: var, accum, lr, gradient
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyAdagradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kApplyAdagradOutputsNum, kernel_name_);
  if (inputs[0]->size != inputs[1]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape and dtype of 'accum' and 'var' should be same, "
                         "but got the memory size of 'accum': "
                      << inputs[1]->size << " and 'var': " << inputs[0]->size;
  }
  if (inputs[0]->size != inputs[3]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape and dtype of 'grad' and 'var' should be same, "
                         "but got the memory size of 'grad': "
                      << inputs[3]->size << " and 'var': " << inputs[0]->size;
  }
  if (inputs[2]->size != kSizeFloat16 && inputs[2]->size != kSizeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'lr' should be float16(memory size: 2) or float32(memory size:4), but got 'lr': "
                      << inputs[2] << ", with memory size: " << inputs[2]->size << " bytes.";
  }
}

template <typename T>
void ApplyAdagradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &outputs) {
  auto *var = reinterpret_cast<T *>(inputs[0]->addr);
  auto *accum = reinterpret_cast<T *>(inputs[1]->addr);
  const auto *lr = reinterpret_cast<T *>(inputs[2]->addr);
  const auto *gradient = reinterpret_cast<T *>(inputs[3]->addr);

  // multithreading
  size_t length = inputs[0]->size / sizeof(T);
  auto task = [this, &var, &accum, &lr, &gradient](size_t start, size_t end) {
    LaunchApplyAdagrad(var, accum, lr, gradient, start, end);
  };
  CPUKernelUtils::ParallelForAutoSearch(task, length, &parallel_search_info_);

  // Copy result to output tensor
  auto output_var = reinterpret_cast<T *>(outputs[0]->addr);
  auto output_accum = reinterpret_cast<T *>(outputs[1]->addr);
  auto ret = memcpy_s(output_var, outputs[0]->size, var, inputs[0]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', launch kernel error: memcpy failed. Error no: " << ret;
  }
  ret = memcpy_s(output_accum, outputs[1]->size, accum, inputs[1]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', launch kernel error: memcpy failed. Error no: " << ret;
  }
}

template <typename T>
void ApplyAdagradCpuKernelMod::LaunchApplyAdagrad(T *var, T *accum, const T *lr, const T *gradient, size_t start,
                                                  size_t end) const {
  // DataType can only be float32 or float16, so eps will not be zero.
  auto one = static_cast<T>(1);
  auto eps = static_cast<T>(1e-6);
  for (size_t i = start; i < end; ++i) {
    // update accum: accum += grad * grad
    if (update_slots_) {
      accum[i] += gradient[i] * gradient[i];
    }
    // update var: var -= lr * grad * \frac{1}{\sqrt{accum}}
    var[i] -= lr[0] * gradient[i] * (one / sqrt(accum[i] + eps));
  }
}
}  // namespace kernel
}  // namespace mindspore
