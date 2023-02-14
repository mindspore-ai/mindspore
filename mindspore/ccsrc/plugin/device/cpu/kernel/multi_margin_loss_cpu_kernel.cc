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
 * once_compute_thread_sizetributed under the License is
 * once_compute_thread_sizetributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

#include "plugin/device/cpu/kernel/multi_margin_loss_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/multi_margin_loss.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMultiMarginLossInputNumWithWeight = 3;
constexpr size_t kMultiMarginLossInputNumWithoutWeight = 2;
constexpr size_t kMultiMarginLossOutputsNum = 1;
const size_t kZero = 0;
const size_t kOne = 1;
const size_t kTwo = 2;
constexpr char kKernelName[] = "MultiMarginLoss";
}  // namespace

bool MultiMarginLossCPUKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();

  auto kernel_ptr = std::dynamic_pointer_cast<ops::MultiMarginLoss>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  reduction = kernel_ptr->get_reduction();
  p = kernel_ptr->get_p();
  margin = kernel_ptr->get_margin();

  dtype_ = inputs[kZero]->GetDtype();
  input_num = inputs.size();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int MultiMarginLossCPUKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto x_shape = inputs[kZero]->GetShapeVector();
  batch_size = LongToSize(x_shape[kZero]);
  dims = LongToSize(x_shape[kOne]);
  return KRET_OK;
}

bool MultiMarginLossCPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
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

const std::vector<std::pair<KernelAttr, MultiMarginLossCPUKernelMod::KernelRunFunc>>
  &MultiMarginLossCPUKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, MultiMarginLossCPUKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &MultiMarginLossCPUKernelMod::LaunchKernel},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &MultiMarginLossCPUKernelMod::LaunchKernel},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &MultiMarginLossCPUKernelMod::LaunchKernel},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
     &MultiMarginLossCPUKernelMod::LaunchKernel},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &MultiMarginLossCPUKernelMod::LaunchKernel},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
     &MultiMarginLossCPUKernelMod::LaunchKernel},
  };
  return func_list;
}

template <typename T>
void MultiMarginLossCPUKernelMod::LaunchKernelFP32AndFP64(const std::vector<kernel::AddressPtr> &inputs,
                                                          const std::vector<kernel::AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[kZero]->addr);
  auto target_addr = reinterpret_cast<int64_t *>(inputs[kOne]->addr);
  for (size_t i = 0; i < batch_size; i++) {
    if (target_addr[i] < 0 || target_addr[i] >= SizeToLong(dims)) {
      MS_EXCEPTION(ValueError) << "Target out of range.";
    }
  }
  T *weight_addr = nullptr;
  bool weight_defined_ = (input_num == 3);
  if (weight_defined_) {
    weight_addr = reinterpret_cast<T *>(inputs[kTwo]->addr);
  }
  auto y_addr = reinterpret_cast<T *>(outputs[kZero]->addr);
  std::vector<T> tmp_loss(batch_size);
  auto task = [&](size_t start, size_t end) {
    start *= dims;
    end *= dims;
    size_t once_compute_thread_size = (end - start);
    std::vector<T> calc(dims);
    auto calc_data = calc.data();
    for (size_t m = 0; m < (once_compute_thread_size) / dims; m++) {
      size_t i = start / dims;
      for (size_t d = 0; d < dims; d++) {
        if (d == LongToSize(target_addr[i])) {
          continue;
        }
        calc_data[d] = static_cast<T>(margin) + x_addr[start + d] - x_addr[start + LongToSize(target_addr[i])];
        if (calc_data[d] > static_cast<T>(0)) {
          calc_data[d] = (p == 1) ? calc_data[d] : calc_data[d] * calc_data[d];
          if (weight_defined_) {
            calc_data[d] *= static_cast<T>(weight_addr[target_addr[i]]);
          }
          tmp_loss[i] += calc_data[d];
        }
      }
      tmp_loss[i] = tmp_loss[i] / static_cast<T>(dims);
      start += dims;
    }
  };
  CPUKernelUtils::ParallelFor(task, batch_size);
  if (reduction == MEAN) {
    *y_addr = static_cast<T>(0);
    for (size_t i = 0; i < batch_size; i++) {
      *y_addr += tmp_loss[i];
    }
    *y_addr /= static_cast<T>(batch_size);
  }
  if (reduction == SUM) {
    *y_addr = static_cast<T>(0);
    for (size_t i = 0; i < batch_size; i++) {
      *y_addr += tmp_loss[i];
    }
  }
  if (reduction == NONE) {
    for (size_t t = 0; t < batch_size; t++) {
      *(y_addr + t) = tmp_loss[t];
    }
  }
}

template <typename T>
void MultiMarginLossCPUKernelMod::LaunchKernelFP16(const std::vector<kernel::AddressPtr> &inputs,
                                                   const std::vector<kernel::AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[kZero]->addr);
  auto target_addr = reinterpret_cast<int64_t *>(inputs[kOne]->addr);
  for (size_t i = 0; i < batch_size; i++) {
    if (target_addr[i] < 0 || target_addr[i] >= SizeToLong(dims)) {
      MS_EXCEPTION(ValueError) << "Target out of range.";
    }
  }
  T *weight_addr = nullptr;
  bool weight_defined_ = (input_num == 3);
  if (weight_defined_) {
    weight_addr = reinterpret_cast<T *>(inputs[kTwo]->addr);
  }
  auto y_addr = reinterpret_cast<T *>(outputs[kZero]->addr);
  std::vector<float> tmp_loss(batch_size);
  auto task = [&](size_t start, size_t end) {
    start *= dims;
    end *= dims;
    size_t once_compute_thread_size = (end - start);
    std::vector<float> calc(dims);
    auto calc_data = calc.data();
    for (size_t m = 0; m < (once_compute_thread_size) / dims; m++) {
      size_t i = start / dims;
      for (size_t d = 0; d < dims; d++) {
        if (d == LongToSize(target_addr[i])) {
          continue;
        }
        calc_data[d] = margin + static_cast<float>(x_addr[start + d]) -
                       static_cast<float>(x_addr[start + LongToSize(target_addr[i])]);
        if (calc_data[d] > 0) {
          calc_data[d] = (p == 1) ? calc_data[d] : calc_data[d] * calc_data[d];
          if (weight_defined_) {
            calc_data[d] *= static_cast<float>(weight_addr[target_addr[i]]);
          }
          tmp_loss[i] += calc_data[d];
        }
      }
      tmp_loss[i] = tmp_loss[i] / static_cast<float>(dims);
      start += dims;
    }
  };
  CPUKernelUtils::ParallelFor(task, batch_size);
  if (reduction == NONE) {
    for (size_t t = 0; t < batch_size; t++) {
      *(y_addr + t) = static_cast<T>(tmp_loss[t]);
    }
  } else {
    float tmp_loss_sum = 0.0f;
    for (size_t i = 0; i < batch_size; i++) {
      tmp_loss_sum += tmp_loss[i];
    }
    if (reduction == MEAN) {
      *y_addr = static_cast<T>(tmp_loss_sum / batch_size);
    } else if (reduction == SUM) {
      *y_addr = static_cast<T>(tmp_loss_sum);
    }
  }
}

void MultiMarginLossCPUKernelMod::CheckParam(const CNodePtr &kernel_node) {
  input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != kMultiMarginLossInputNumWithoutWeight && input_num != kMultiMarginLossInputNumWithWeight) {
    MS_LOG(EXCEPTION) << "Invalid input numbers, expect input number 2 or 3, but actual input number " << input_num;
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kMultiMarginLossOutputsNum, kKernelName);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MultiMarginLoss, MultiMarginLossCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
