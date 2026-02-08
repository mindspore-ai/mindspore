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

#include "kernel/cpu/native/apply_adagrad_da_cpu_kernel.h"
#include <thread>
#include <vector>
#include <algorithm>
#include <map>
#include <functional>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace apply_adagrad_da_cpu {
namespace {
constexpr size_t kSizeFloat16 = 2;
constexpr size_t kSizeFloat32 = 4;
constexpr size_t kSizeInt32 = 4;
constexpr size_t kSizeInt64 = 8;
constexpr size_t kApplyAdagradDAInputsNum = 8;
constexpr size_t kApplyAdagradDAOutputsNum = 1;
constexpr size_t kVarIndex = 0;
constexpr size_t kAccIndex = 1;
constexpr size_t kSquarAccIndex = 2;
constexpr size_t kGradIndex = 3;
constexpr size_t kLRIndex = 4;
constexpr size_t kL1Index = 5;
constexpr size_t kL2Index = 6;
constexpr size_t kStepIndex = 7;
}  // namespace

bool ApplyAdagradDACpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty()) {
    MS_EXCEPTION(ValueError) << "ApplyAdagradDA input is empty";
  }
  dtype_ = inputs[0]->dtype_id();
  batch_rank_ = ops::get_batch_rank(primitive_);
  return true;
}

int ApplyAdagradDACpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyAdagradDAInputsNum, kernel_name_);
  std::vector<int64_t> var_shape = inputs[kVarIndex]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kLRIndex]->GetShapeVector();

  if (!lr_shape.empty()) {
    batch_size_ = std::accumulate(lr_shape.begin(), lr_shape.end(), int64_t(1), std::multiplies<int64_t>());
  }

  if (batch_size_ > 0) {
    input_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), int64_t(1), std::multiplies<int64_t>());
    input_elements_ /= batch_size_;
    return ret;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }
}

bool ApplyAdagradDACpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                        const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kApplyAdagradDAOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'var' must be Float16 or Float32, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

template <typename T>
T Sign(T num) {
  if (num > static_cast<T>(0.0)) {
    return static_cast<T>(1.0);
  } else if (num == static_cast<T>(0.0)) {
    return static_cast<T>(0.0);
  } else {
    return static_cast<T>(-1.0);
  }
}

template <typename T>
T abs(T num) {
  if (num >= static_cast<T>(0.0)) {
    return static_cast<T>(num);
  } else {
    return static_cast<T>(-num);
  }
}

template <typename T>
T max(T num1, T num2) {
  if (num1 >= num2) {
    return static_cast<T>(num1);
  } else {
    return static_cast<T>(num2);
  }
}

template <typename T>
void ApplyAdagradDACpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyAdagradDAInputsNum, kernel_name_);
  auto *var = reinterpret_cast<T *>(inputs[kVarIndex]->device_ptr());
  auto *gradient_accumulator = reinterpret_cast<T *>(inputs[kAccIndex]->device_ptr());
  auto *gradient_squared_accumulator = reinterpret_cast<T *>(inputs[kSquarAccIndex]->device_ptr());
  const auto *grad = reinterpret_cast<T *>(inputs[kGradIndex]->device_ptr());
  const auto *lr = reinterpret_cast<T *>(inputs[kLRIndex]->device_ptr());
  const auto *l1 = reinterpret_cast<T *>(inputs[kL1Index]->device_ptr());
  const auto *l2 = reinterpret_cast<T *>(inputs[kL2Index]->device_ptr());
  const int *global_step = reinterpret_cast<int *>(inputs[kStepIndex]->device_ptr());

  for (int64_t b = 0; b < batch_size_; b++) {
    // Multithreading
    auto task = [this, &var, &gradient_accumulator, &gradient_squared_accumulator, &grad, &lr, &l1, &l2, &global_step](
                  size_t start, size_t end) {
      LaunchApplyAdagradDA(var, gradient_accumulator, gradient_squared_accumulator, grad, lr, l1, l2, global_step,
                           start, end);
    };
    ParallelLaunch(task, input_elements_, 0, this, pool_);
    var += input_elements_;
    gradient_accumulator += input_elements_;
    gradient_squared_accumulator += input_elements_;
    grad += input_elements_;
    ++lr;
    ++l1;
    ++l2;
    ++global_step;
  }
}

template <typename T>
void ApplyAdagradDACpuKernelMod::LaunchApplyAdagradDA(T *var, T *gradient_accumulator, T *gradient_squared_accumulator,
                                                      const T *grad, const T *lr, const T *l1, const T *l2,
                                                      const int *global_step, size_t start, size_t end) const {
  auto minus_one = static_cast<T>(-1);
  auto zeros = static_cast<T>(0);
  for (size_t i = start; i < end; i++) {
    gradient_accumulator[i] += grad[i];
    gradient_squared_accumulator[i] += grad[i] * grad[i];
    auto tmp_val =
      l1[0] > zeros
        ? Sign<T>(gradient_accumulator[i]) *
            max<T>(abs<T>(gradient_accumulator[i] - static_cast<T>(l1[0]) * static_cast<T>(global_step[0])), zeros)
        : gradient_accumulator[i];
    auto x_value = minus_one * lr[0] * tmp_val;
    auto y_value = static_cast<T>(l2[0]) * static_cast<T>(global_step[0]) * static_cast<T>(lr[0]) +
                   sqrt(gradient_squared_accumulator[i]);
    // Update var
    var[i] = static_cast<T>(x_value) / static_cast<T>(y_value);
  }
}

std::vector<KernelAttr> ApplyAdagradDACpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeFloat32)
                                                       .AddOutInRef(0, 0),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeFloat16)
                                                       .AddOutInRef(0, 0)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyAdagradDA, ApplyAdagradDACpuKernelMod);
}  // namespace apply_adagrad_da_cpu
}  // namespace kernel
}  // namespace mindspore
