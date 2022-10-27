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

#include "plugin/device/cpu/kernel/multi_margin_loss_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/grad/multi_margin_loss_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMultiMarginLossGradInputNumWithWeight = 4;
constexpr size_t kMultiMarginLossGradInputNumWithoutWeight = 3;
constexpr size_t kMultiMarginLossGradOutputsNum = 1;
const size_t kZero = 0;
const size_t kOne = 1;
const size_t kTwo = 2;
const size_t kThree = 3;
constexpr char kKernelName[] = "MultiMarginLossGrad";
}  // namespace

bool MultiMarginLossGradCPUKernelMod::Init(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();

  auto kernel_ptr = std::dynamic_pointer_cast<ops::MultiMarginLossGrad>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  reduction = kernel_ptr->get_reduction();
  p = kernel_ptr->get_p();
  margin = kernel_ptr->get_margin();

  dtype_ = inputs[kZero]->GetDtype();
  input_num = inputs.size();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int MultiMarginLossGradCPUKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto x_shape = inputs[kOne]->GetShapeVector();
  batch_size = LongToSize(x_shape[kZero]);
  dims = LongToSize(x_shape[kOne]);
  return KRET_OK;
}

bool MultiMarginLossGradCPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &workspace,
                                                   const std::vector<AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernelFP16<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernelFP32AndFP64<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernelFP32AndFP64<double>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "Data type is " << TypeIdLabel(dtype_) << " which is not supported.";
  }
  return true;
}

const std::vector<std::pair<KernelAttr, MultiMarginLossGradCPUKernelMod::KernelRunFunc>>
  &MultiMarginLossGradCPUKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, MultiMarginLossGradCPUKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &MultiMarginLossGradCPUKernelMod::LaunchKernel},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &MultiMarginLossGradCPUKernelMod::LaunchKernel},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &MultiMarginLossGradCPUKernelMod::LaunchKernel},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &MultiMarginLossGradCPUKernelMod::LaunchKernel},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &MultiMarginLossGradCPUKernelMod::LaunchKernel},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &MultiMarginLossGradCPUKernelMod::LaunchKernel},
  };
  return func_list;
}

template <typename T>
void MultiMarginLossGradCPUKernelMod::LaunchKernelFP32AndFP64(const std::vector<kernel::AddressPtr> &inputs,
                                                              const std::vector<kernel::AddressPtr> &outputs) {
  auto y_grad_addr = reinterpret_cast<T *>(inputs[kZero]->addr);
  auto x_addr = reinterpret_cast<T *>(inputs[kOne]->addr);
  auto target_addr = reinterpret_cast<int64_t *>(inputs[kTwo]->addr);
  for (size_t i = 0; i < batch_size; i++) {
    if (target_addr[i] < 0 || target_addr[i] >= SizeToLong(dims)) {
      MS_EXCEPTION(ValueError) << "Target out of range.";
    }
  }
  T *weight_addr = nullptr;
  bool weight_defined_ = (input_num == 4);
  if (weight_defined_) {
    weight_addr = reinterpret_cast<T *>(inputs[kThree]->addr);
  }
  auto x_grad_addr = reinterpret_cast<T *>(outputs[kZero]->addr);
  auto task = [&](size_t start, size_t end) {
    start *= dims;
    end *= dims;
    std::vector<T> calc(dims);
    auto calc_data = calc.data();
    size_t once_compute_thread_size = end - start;
    for (size_t i = 0; i < once_compute_thread_size / dims; i++) {
      size_t m = start / dims;
      size_t target_idx = LongToSize(target_addr[m]);
      T input_target = x_addr[start + target_idx];
      T grad_input_target = static_cast<T>(0);
      for (size_t d = 0; d < dims; d++) {
        calc_data[d] = static_cast<T>(margin) + x_addr[start + d] - input_target;
        if (d == target_idx) {
          continue;
        }
        if (calc_data[d] > static_cast<T>(0)) {
          auto weights = reduction == MEAN ? (static_cast<T>(1) / (static_cast<T>(dims) * static_cast<T>(batch_size)))
                                           : (static_cast<T>(1) / static_cast<T>(dims));
          calc_data[d] = (p == 1) ? weights : static_cast<T>(2) * weights * calc_data[d];
          if (weight_defined_) {
            calc_data[d] *= static_cast<T>(weight_addr[target_idx]);
          }
          grad_input_target -= calc_data[d];
          *(x_grad_addr + start + d) = calc_data[d];
        } else {
          *(x_grad_addr + start + d) = static_cast<T>(0);
        }
      }
      *(x_grad_addr + start + target_idx) = grad_input_target;
      start += dims;
    }
  };
  CPUKernelUtils::ParallelFor(task, batch_size);
  T y_grad_value = static_cast<T>(1);
  auto y_grad_data = (y_grad_addr == nullptr) ? &y_grad_value : y_grad_addr;
  if (reduction != NONE || y_grad_dims == 0) {
    for (size_t i = 0; i < batch_size * dims; i++) {
      *(x_grad_addr + i) *= *(y_grad_data);
    }
  } else {
    for (size_t i = 0; i < batch_size; i++) {
      for (size_t j = 0; j < dims; j++) {
        *(x_grad_addr + i * dims + j) *= *(y_grad_data + i);
      }
    }
  }
}

template <typename T>
void MultiMarginLossGradCPUKernelMod::LaunchKernelFP16(const std::vector<kernel::AddressPtr> &inputs,
                                                       const std::vector<kernel::AddressPtr> &outputs) {
  auto y_grad_addr = reinterpret_cast<T *>(inputs[kZero]->addr);
  auto x_addr = reinterpret_cast<T *>(inputs[kOne]->addr);
  auto target_addr = reinterpret_cast<int64_t *>(inputs[kTwo]->addr);
  for (size_t i = 0; i < batch_size; i++) {
    if (target_addr[i] < 0 || target_addr[i] >= SizeToLong(dims)) {
      MS_EXCEPTION(ValueError) << "Target out of range.";
    }
  }
  T *weight_addr = nullptr;
  bool weight_defined_ = (input_num == 4);
  if (weight_defined_) {
    weight_addr = reinterpret_cast<T *>(inputs[kThree]->addr);
  }
  auto x_grad_addr = reinterpret_cast<T *>(outputs[kZero]->addr);
  auto task = [&](size_t start, size_t end) {
    start *= dims;
    end *= dims;
    std::vector<float> calc(dims);
    auto calc_data = calc.data();
    size_t once_compute_thread_size = end - start;
    for (size_t i = 0; i < once_compute_thread_size / dims; i++) {
      size_t m = start / dims;
      size_t target_idx = LongToSize(target_addr[m]);
      float input_target = static_cast<float>(x_addr[start + target_idx]);
      float grad_input_target = static_cast<float>(0);
      for (size_t d = 0; d < dims; d++) {
        calc_data[d] = margin + static_cast<float>(x_addr[start + d]) - input_target;
        if (d == target_idx) {
          continue;
        }
        if (calc_data[d] > static_cast<float>(0)) {
          auto weights = reduction == MEAN
                           ? (static_cast<float>(1) / (static_cast<float>(dims) * static_cast<float>(batch_size)))
                           : (static_cast<float>(1) / static_cast<float>(dims));
          calc_data[d] = (p == 1) ? weights : static_cast<float>(2) * weights * calc_data[d];
          if (weight_defined_) {
            calc_data[d] *= static_cast<float>(weight_addr[target_idx]);
          }
          grad_input_target -= calc_data[d];
          *(x_grad_addr + start + d) = static_cast<T>(calc_data[d]);
        } else {
          *(x_grad_addr + start + d) = static_cast<T>(0);
        }
      }
      *(x_grad_addr + start + target_idx) = static_cast<T>(grad_input_target);
      start += dims;
    }
  };
  CPUKernelUtils::ParallelFor(task, batch_size);
  T y_grad_value = static_cast<T>(1);
  auto y_grad_data = (y_grad_addr == nullptr) ? &y_grad_value : y_grad_addr;
  if (reduction != NONE || y_grad_dims == 0) {
    for (size_t i = 0; i < batch_size * dims; i++) {
      *(x_grad_addr + i) = static_cast<T>(static_cast<float>(*(x_grad_addr + i)) * static_cast<float>(*(y_grad_data)));
    }
  } else {
    for (size_t i = 0; i < batch_size; i++) {
      for (size_t j = 0; j < dims; j++) {
        *(x_grad_addr + i * dims + j) =
          static_cast<T>(static_cast<float>(*(x_grad_addr + i * dims + j)) * static_cast<float>(*(y_grad_data + i)));
      }
    }
  }
}

void MultiMarginLossGradCPUKernelMod::CheckParam(const CNodePtr &kernel_node) const {
  size_t inputs_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (inputs_num != kMultiMarginLossGradInputNumWithoutWeight && inputs_num != kMultiMarginLossGradInputNumWithWeight) {
    MS_LOG(EXCEPTION) << "Invalid input numbers, expect input number 3 or 4, but actual input number " << inputs_num;
  }
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kMultiMarginLossGradOutputsNum, kKernelName);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MultiMarginLossGrad, MultiMarginLossGradCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
