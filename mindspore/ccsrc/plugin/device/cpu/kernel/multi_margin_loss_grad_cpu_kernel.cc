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

void MultiMarginLossGradCPUKernelMod::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  ShapeVector x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kOne);
  if (IsDynamic({x_shape})) {
    return;
  }
  batch_size = LongToSize(x_shape[kZero]);
  dims = LongToSize(x_shape[kOne]);
  reduction = common::AnfAlgo::GetNodeAttr<string>(kernel_node, REDUCTION);
  p = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "p");
  margin = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "margin");
  y_grad_dims = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kZero).size();
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kZero);
  input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
}

bool MultiMarginLossGradCPUKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernelFP16<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "Data type is " << TypeIdLabel(dtype_) << " which is not supported.";
  }
  return true;
}

template <typename T>
void MultiMarginLossGradCPUKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
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
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != kMultiMarginLossGradInputNumWithoutWeight && input_num != kMultiMarginLossGradInputNumWithWeight) {
    MS_LOG(EXCEPTION) << "Invalid input numbers, expect input number 3 or 4, but actual input number " << input_num;
  }
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kMultiMarginLossGradOutputsNum, kKernelName);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MultiMarginLossGrad, MultiMarginLossGradCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
