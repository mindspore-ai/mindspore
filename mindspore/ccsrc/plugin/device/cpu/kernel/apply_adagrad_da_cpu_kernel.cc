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

#include <thread>
#include <vector>
#include <algorithm>
#include <map>
#include <functional>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/apply_adagrad_da_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSizeFloat16 = 2;
constexpr size_t kSizeFloat32 = 4;
constexpr size_t kSizeInt32 = 4;
constexpr size_t kSizeInt64 = 8;
constexpr size_t kApplyAdagradDAInputsNum = 8;
constexpr size_t kApplyAdagradDAOutputsNum = 3;
constexpr size_t kVarIndex = 0;
constexpr size_t kAccIndex = 1;
constexpr size_t kSquarAccIndex = 2;
constexpr size_t kGradIndex = 3;
constexpr size_t kLRIndex = 4;
constexpr size_t kL1Index = 5;
constexpr size_t kL2Index = 6;
constexpr size_t kStepIndex = 7;
}  // namespace

bool ApplyAdagradDACpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  dtype_ = inputs[0]->GetDtype();
  batch_rank_ = base_operator->get_batch_rank();
  return true;
}

void ApplyAdagradDACpuKernelMod::CheckDType(const std::vector<KernelTensorPtr> &inputs) const {
  auto LRDtype = inputs[kLRIndex]->GetDtype();
  auto L1Dtype = inputs[kL1Index]->GetDtype();
  auto L2Dtype = inputs[kL2Index]->GetDtype();
  auto StepDtype = inputs[kStepIndex]->GetDtype();
  if (LRDtype != kNumberTypeFloat16 && LRDtype != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'lr' should be float16 or float32, but got " << LRDtype
                      << " .";
  }
  if (L1Dtype != kNumberTypeFloat16 && L1Dtype != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'l1' should be float16 or float32, but got " << L1Dtype
                      << " .";
  }
  if (L2Dtype != kNumberTypeFloat16 && L2Dtype != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'l2' should be float16 or float32, but got " << L2Dtype
                      << " .";
  }
  if (StepDtype != kNumberTypeInt32 && StepDtype != kNumberTypeInt64) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'step' should be int32 or int64, but got " << StepDtype
                      << " .";
  }
}
int ApplyAdagradDACpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  CheckDType(inputs);
  std::vector<int64_t> var_shape = inputs[kVarIndex]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kLRIndex]->GetShapeVector();

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

  if (batch_size_ > 0) {
    input_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), int64_t(1), std::multiplies<int64_t>());
    input_elements_ = input_elements_ / batch_size_;

    return ret;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }
}

bool ApplyAdagradDACpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &outputs) {
  CheckParam(inputs, outputs);
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

void ApplyAdagradDACpuKernelMod::CheckShapeAndDtypeEqual(int64_t size_a, int64_t size_b, const char *name_a,
                                                         const char *name_b) const {
  if (size_a != size_b) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape and dtype of '" << name_a << "' and '" << name_b
                      << "' must be the same, "
                         "but got the memory size of '"
                      << name_a << "': " << size_a << " and '" << name_b << "': " << size_b;
  }
}

void ApplyAdagradDACpuKernelMod::CheckParam(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &outputs) const {
  // Inputs: var, gradient_accumulator, gradient_squared_accumulator, grad, lr, l1, l2, global_step
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyAdagradDAInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kApplyAdagradDAOutputsNum, kernel_name_);
  CheckShapeAndDtypeEqual(inputs[kAccIndex]->size, inputs[kVarIndex]->size, "gradient_accumulator", "var");
  CheckShapeAndDtypeEqual(inputs[kSquarAccIndex]->size, inputs[kVarIndex]->size, "gradient_squared_accumulator", "var");
  CheckShapeAndDtypeEqual(inputs[kGradIndex]->size, inputs[kVarIndex]->size, "grad", "var");
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
void ApplyAdagradDACpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &) {
  auto *var = reinterpret_cast<T *>(inputs[kVarIndex]->addr);
  auto *gradient_accumulator = reinterpret_cast<T *>(inputs[kAccIndex]->addr);
  auto *gradient_squared_accumulator = reinterpret_cast<T *>(inputs[kSquarAccIndex]->addr);
  const auto *grad = reinterpret_cast<T *>(inputs[kGradIndex]->addr);
  const auto *lr = reinterpret_cast<T *>(inputs[kLRIndex]->addr);
  const auto *l1 = reinterpret_cast<T *>(inputs[kL1Index]->addr);
  const auto *l2 = reinterpret_cast<T *>(inputs[kL2Index]->addr);
  const int *global_step = reinterpret_cast<int *>(inputs[kStepIndex]->addr);

  for (int64_t b = 0; b < batch_size_; b++) {
    // Multithreading
    auto task = [this, &var, &gradient_accumulator, &gradient_squared_accumulator, &grad, &lr, &l1, &l2, &global_step](
                  size_t start, size_t end) {
      LaunchApplyAdagradDA(var, gradient_accumulator, gradient_squared_accumulator, grad, lr, l1, l2, global_step,
                           start, end);
    };
    CPUKernelUtils::ParallelForAutoSearch(task, input_elements_, &parallel_search_info_);
    var = var + input_elements_;
    gradient_accumulator = gradient_accumulator + input_elements_;
    gradient_squared_accumulator = gradient_squared_accumulator + input_elements_;
    grad = grad + input_elements_;
    lr++;
    l1++;
    l2++;
    global_step++;
  }
}

template <typename T>
void ApplyAdagradDACpuKernelMod::LaunchApplyAdagradDA(T *var, T *gradient_accumulator, T *gradient_squared_accumulator,
                                                      const T *grad, const T *lr, const T *l1, const T *l2,
                                                      const int *global_step, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    gradient_accumulator[i] += grad[i];
    gradient_squared_accumulator[i] += grad[i] * grad[i];
    auto minus_one = static_cast<T>(-1);
    auto zeros = static_cast<T>(0);
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
                                                       .AddOutputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeFloat32)
                                                       .AddOutInRef(0, 0)
                                                       .AddOutInRef(1, 1)
                                                       .AddOutInRef(2, 2),
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
                                                       .AddOutputAttr(kNumberTypeFloat16)
                                                       .AddOutputAttr(kNumberTypeFloat16)
                                                       .AddOutInRef(0, 0)
                                                       .AddOutInRef(1, 1)
                                                       .AddOutInRef(2, 2)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyAdagradDA, ApplyAdagradDACpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
