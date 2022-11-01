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

#include "plugin/device/cpu/kernel/apply_adagrad_v2_cpu_kernel.h"

#include <algorithm>
#include <functional>
#include <map>
#include <utility>
#include <thread>
#include <vector>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSizeFloat16 = 2;
constexpr size_t kSizeFloat32 = 4;
constexpr size_t kApplyAdagradV2InputsNum = 4;
constexpr size_t kApplyAdagradV2OutputsNum = 2;
constexpr size_t kVarIndex = 0;
constexpr size_t kAccumIndex = 1;
constexpr size_t kLRIndex = 2;
constexpr size_t kGradIndex = 3;
}  // namespace

bool ApplyAdagradV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  batch_rank_ = base_operator->get_batch_rank();

  auto kernel_ptr = std::dynamic_pointer_cast<ops::ApplyAdagradV2>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Cast ApplyAdagradV2 ops failed!";
    return false;
  }
  epsilon_ = kernel_ptr->get_epsilon();
  update_slots_ = kernel_ptr->get_update_slots();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int ApplyAdagradV2CpuKernelMod::CheckParam(const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) const {
  // inputs: var, accum, lr, gradient
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kApplyAdagradV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kApplyAdagradV2OutputsNum, kernel_name_);
  auto var_shape = inputs[kVarIndex]->GetShapeVector();
  auto accum_shape = inputs[kAccumIndex]->GetShapeVector();
  auto grad_shape = inputs[kGradIndex]->GetShapeVector();
  auto lr_dtype = inputs[kLRIndex]->GetDtype();
  if (var_shape != accum_shape) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape of 'accum' and 'var' should be the same, "
                         "but got shape of 'accum': "
                      << accum_shape << " and shape of 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }
  if (var_shape != grad_shape) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape of 'grad' and 'var' should be same, "
                         "but got the shape of 'grad': "
                      << grad_shape << " and shape of'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }
  if (lr_dtype != kNumberTypeFloat16 && lr_dtype != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'lr' should be float16 or float32, but got 'lr': " << inputs[kLRIndex]
                      << ", with type " << lr_dtype << " .";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

int ApplyAdagradV2CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }

  ret = CheckParam(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  // get inner input size.
  if (batch_rank_ != 0) {
    auto input_shape = inputs[kIndex0]->GetShapeVector();
    inner_input_size_ =
      std::accumulate(input_shape.begin() + batch_rank_, input_shape.end(), size_t(1), std::multiplies<size_t>());
  }
  return ret;
}

template <typename T>
bool ApplyAdagradV2CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  auto *var = reinterpret_cast<T *>(inputs[kVarIndex]->addr);
  auto *accum = reinterpret_cast<T *>(inputs[kAccumIndex]->addr);
  const auto *lr = reinterpret_cast<T *>(inputs[kLRIndex]->addr);
  const auto *gradient = reinterpret_cast<T *>(inputs[kGradIndex]->addr);

  // multithreading
  size_t length = inputs[0]->size / sizeof(T);
  const float &epsilon = this->epsilon_;
  const bool &update_slots = this->update_slots_;
  auto task = [this, var, accum, lr, gradient, &epsilon, &update_slots](size_t start, size_t end) {
    // DataType can only be float32 or float16, so eps will not be zero.
    const T zero = static_cast<T>(0);
    const T one = static_cast<T>(1);
    const T eps_if_zero = static_cast<T>(1e-6);
    const T eps_param = static_cast<T>(epsilon);
    for (size_t i = start; i < end; ++i) {
      size_t batch_index = inner_input_size_ <= 0 ? 0 : i / inner_input_size_;
      // update accum: accum += grad * grad
      if (update_slots) {
        accum[i] += gradient[i] * gradient[i];
      }
      T dividend = sqrt(accum[i]) + eps_param;
      // if dividend is zero, add a small number to avoid division by zero
      if (std::equal_to<T>()(dividend, zero)) {
        dividend = dividend + eps_if_zero;
      }
      // update var: var -= lr * grad * \frac{1}{\sqrt{accum} * eps}
      var[i] -= lr[batch_index] * gradient[i] * (one / dividend);
    }
  };
  ParallelLaunch(task, length, 0, this, pool_);
  return true;
}

std::vector<KernelAttr> ApplyAdagradV2CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ApplyAdagradV2Func> &pair) { return pair.first; });
  return support_list;
}

std::vector<std::pair<KernelAttr, ApplyAdagradV2CpuKernelMod::ApplyAdagradV2Func>>
  ApplyAdagradV2CpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutInRef(0, 0)
       .AddOutInRef(1, 1),
     &ApplyAdagradV2CpuKernelMod::LaunchKernel<float>},
};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyAdagradV2, ApplyAdagradV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
