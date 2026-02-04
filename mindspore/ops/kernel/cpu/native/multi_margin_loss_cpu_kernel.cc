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

#include "kernel/cpu/native/multi_margin_loss_cpu_kernel.h"

#include "mindspore/ops/infer/multi_margin_loss.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
namespace multi_margin_loss_cpu {
namespace {
constexpr size_t kMultiMarginLossInputNumWithWeight = 3;
constexpr size_t kMultiMarginLossInputNumWithoutWeight = 2;
constexpr size_t kMultiMarginLossOutputsNum = 1;
const size_t kZero = 0;
const size_t kOne = 1;
const size_t kTwo = 2;
constexpr char kKernelName[] = "MultiMarginLoss";
}  // namespace

bool MultiMarginLossCPUKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  reduction_ = GetValue<std::string>(primitive_->GetAttr(ops::kReduction));
  p_ = GetValue<int64_t>(primitive_->GetAttr(ops::kP));
  margin_ = GetValue<float>(primitive_->GetAttr(ops::kMargin));

  dtype_ = inputs[kZero]->dtype_id();
  input_num_ = inputs.size();
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int MultiMarginLossCPUKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto x_shape = inputs[kZero]->GetShapeVector();
  batch_size_ = LongToSize(x_shape.at(kZero));
  dims_ = LongToSize(x_shape.at(kOne));
  auto type = inputs[kTwo]->GetType();
  weight_defined_ = !type->isa<TypeNone>();
  return KRET_OK;
}

bool MultiMarginLossCPUKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &,
                                               const std::vector<KernelTensor *> &outputs) {
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

const std::vector<std::pair<KernelAttr, MultiMarginLossCPUKernelMod::KernelRunFunc>> &
MultiMarginLossCPUKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, MultiMarginLossCPUKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOptionalInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &MultiMarginLossCPUKernelMod::LaunchKernel},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOptionalInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &MultiMarginLossCPUKernelMod::LaunchKernel},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOptionalInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &MultiMarginLossCPUKernelMod::LaunchKernel}};
  return func_list;
}

template <typename T>
void MultiMarginLossCPUKernelMod::LaunchKernelFP32AndFP64(const std::vector<kernel::KernelTensor *> &inputs,
                                                          const std::vector<kernel::KernelTensor *> &outputs) {
  if (batch_size_ == 0) {
    return;
  }
  auto x_addr = static_cast<T *>(inputs[kZero]->device_ptr());
  MS_EXCEPTION_IF_NULL(x_addr);
  auto target_addr = static_cast<int64_t *>(inputs[kOne]->device_ptr());
  MS_EXCEPTION_IF_NULL(target_addr);
  for (size_t i = 0; i < batch_size_; i++) {
    if (target_addr[i] < 0 || target_addr[i] >= SizeToLong(dims_)) {
      MS_EXCEPTION(ValueError) << "Target out of range.";
    }
  }
  T *weight_addr = nullptr;
  if (weight_defined_) {
    weight_addr = static_cast<T *>(inputs[kTwo]->device_ptr());
    MS_EXCEPTION_IF_NULL(weight_addr);
  }
  auto y_addr = static_cast<T *>(outputs[kZero]->device_ptr());
  std::vector<T> tmp_loss(batch_size_);
  auto task = [&](size_t start, size_t end) {
    start *= dims_;
    end *= dims_;
    size_t once_compute_thread_size = (end - start);
    std::vector<T> calc(dims_);
    auto calc_data = calc.data();
    for (size_t m = 0; m < (once_compute_thread_size) / dims_; m++) {
      size_t i = start / dims_;
      for (size_t d = 0; d < dims_; d++) {
        if (d == LongToSize(target_addr[i])) {
          continue;
        }
        calc_data[d] = static_cast<T>(margin_) + x_addr[start + d] - x_addr[start + LongToSize(target_addr[i])];
        if (calc_data[d] > static_cast<T>(0)) {
          calc_data[d] = (p_ == 1) ? calc_data[d] : calc_data[d] * calc_data[d];
          if (weight_defined_) {
            calc_data[d] *= static_cast<T>(weight_addr[target_addr[i]]);
          }
          tmp_loss[i] += calc_data[d];
        }
      }
      tmp_loss[i] = tmp_loss[i] / static_cast<T>(dims_);
      start += dims_;
    }
  };
  CPUKernelUtils::ParallelFor(task, batch_size_);
  if (reduction_ == MEAN) {
    *y_addr = static_cast<T>(0);
    for (size_t i = 0; i < batch_size_; i++) {
      *y_addr += tmp_loss[i];
    }
    *y_addr /= static_cast<T>(batch_size_);
  }
  if (reduction_ == SUM) {
    *y_addr = static_cast<T>(0);
    for (size_t i = 0; i < batch_size_; i++) {
      *y_addr += tmp_loss[i];
    }
  }
  if (reduction_ == NONE) {
    for (size_t t = 0; t < batch_size_; t++) {
      *(y_addr + t) = tmp_loss[t];
    }
  }
}

template <typename T>
void MultiMarginLossCPUKernelMod::LaunchKernelFP16(const std::vector<kernel::KernelTensor *> &inputs,
                                                   const std::vector<kernel::KernelTensor *> &outputs) {
  if (batch_size_ == 0) {
    return;
  }
  auto x_addr = reinterpret_cast<T *>(inputs[kZero]->device_ptr());
  MS_EXCEPTION_IF_NULL(x_addr);
  auto target_addr = reinterpret_cast<int64_t *>(inputs[kOne]->device_ptr());
  MS_EXCEPTION_IF_NULL(target_addr);
  for (size_t i = 0; i < batch_size_; i++) {
    if (target_addr[i] < 0 || target_addr[i] >= SizeToLong(dims_)) {
      MS_EXCEPTION(ValueError) << "Target out of range.";
    }
  }
  T *weight_addr = nullptr;
  bool weight_defined_ = (input_num_ == 3);
  if (weight_defined_) {
    weight_addr = reinterpret_cast<T *>(inputs[kTwo]->device_ptr());
    MS_EXCEPTION_IF_NULL(weight_addr);
  }
  auto y_addr = reinterpret_cast<T *>(outputs[kZero]->device_ptr());
  std::vector<float> tmp_loss(batch_size_);
  auto task = [&](size_t start, size_t end) {
    start *= dims_;
    end *= dims_;
    size_t once_compute_thread_size = (end - start);
    std::vector<float> calc(dims_);
    auto calc_data = calc.data();
    for (size_t m = 0; m < (once_compute_thread_size) / dims_; m++) {
      size_t i = start / dims_;
      for (size_t d = 0; d < dims_; d++) {
        if (d == LongToSize(target_addr[i])) {
          continue;
        }
        calc_data[d] = margin_ + static_cast<float>(x_addr[start + d]) -
                       static_cast<float>(x_addr[start + LongToSize(target_addr[i])]);
        if (calc_data[d] > 0) {
          calc_data[d] = (p_ == 1) ? calc_data[d] : calc_data[d] * calc_data[d];
          if (weight_defined_) {
            calc_data[d] *= static_cast<float>(weight_addr[target_addr[i]]);
          }
          tmp_loss[i] += calc_data[d];
        }
      }
      tmp_loss[i] = tmp_loss[i] / static_cast<float>(dims_);
      start += dims_;
    }
  };
  CPUKernelUtils::ParallelFor(task, batch_size_);
  if (reduction_ == NONE) {
    for (size_t t = 0; t < batch_size_; t++) {
      *(y_addr + t) = static_cast<T>(tmp_loss[t]);
    }
  } else {
    float tmp_loss_sum = 0.0f;
    for (size_t i = 0; i < batch_size_; i++) {
      tmp_loss_sum += tmp_loss[i];
    }
    if (reduction_ == MEAN) {
      *y_addr = static_cast<T>(tmp_loss_sum / batch_size_);
    } else if (reduction_ == SUM) {
      *y_addr = static_cast<T>(tmp_loss_sum);
    }
  }
}

void MultiMarginLossCPUKernelMod::CheckParam(const CNodePtr &kernel_node) {
  input_num_ = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num_ != kMultiMarginLossInputNumWithoutWeight && input_num_ != kMultiMarginLossInputNumWithWeight) {
    MS_LOG(EXCEPTION) << "Invalid input numbers, expect input number 2 or 3, but actual input number " << input_num_;
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kMultiMarginLossOutputsNum, kKernelName);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MultiMarginLoss, MultiMarginLossCPUKernelMod);
}  // namespace multi_margin_loss_cpu
}  // namespace kernel
}  // namespace mindspore
