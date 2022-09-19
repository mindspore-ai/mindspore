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
#include <string>
#include <utility>

#include "mindspore/core/ops/dynamic_broadcast_gradient_args.h"
#include "plugin/device/cpu/kernel/dynamic_broadcast_grad_args_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDynamicBroadcastGradientArgsInputsNum = 2;
constexpr size_t kDynamicBroadcastGradientArgsOutputsNum = 2;
constexpr char kKernelName[] = "DynamicBroadcastGradientArgs";
using KernelRunFunc = DynamicBroadcastGradientArgsCpuKernelMod::KernelRunFunc;
}  // namespace

template <typename T>
void AddGradReduceIdx(std::vector<std::vector<T>> *grad_reduce_idx, std::vector<bool> cur_one, bool none_one,
                      const size_t max_rank, size_t j) {
  MS_EXCEPTION_IF_NULL(grad_reduce_idx);
  for (size_t i = 0; i < kDynamicBroadcastGradientArgsInputsNum; i++) {
    if (cur_one[i] && !none_one) {
      (void)(*grad_reduce_idx)[i].emplace_back(SizeToLong(max_rank - 1 - j));
    }
  }
}

template <typename T>
std::vector<std::vector<T>> GetGradIndex(const std::vector<std::vector<T>> &revers_shapes, const size_t max_rank) {
  std::vector<std::vector<T>> grad_reduce_index(kDynamicBroadcastGradientArgsInputsNum);
  std::vector<bool> pre_one(kDynamicBroadcastGradientArgsInputsNum);
  std::vector<bool> cur_one(kDynamicBroadcastGradientArgsInputsNum);
  for (size_t i = 0; i < kDynamicBroadcastGradientArgsInputsNum; i++) {
    pre_one[i] = false;
    cur_one[i] = false;
  }
  bool set_one = false;
  for (size_t j = 0; j < max_rank; j++) {
    int out_dim = -1;
    bool out_dim_set = false;
    bool none_one = true;
    for (size_t i = 0; i < kDynamicBroadcastGradientArgsInputsNum; i++) {
      if (revers_shapes[i][j] == 1) {
        cur_one[i] = true;
        none_one = false;
      } else {
        cur_one[i] = false;
        if (!out_dim_set || revers_shapes[i][j] == static_cast<T>(out_dim)) {
          out_dim = static_cast<int>(revers_shapes[i][j]);
          out_dim_set = true;
        } else {
          MS_LOG(EXCEPTION) << "Can not broadcast inputs[0] and inputs[1].";
        }
      }
    }
    if (!out_dim_set) {
      for (size_t i = 0; i < kDynamicBroadcastGradientArgsInputsNum; i++) {
        (void)grad_reduce_index[i].emplace_back(max_rank - 1 - j);
      }
      continue;
    } else if (std::equal(cur_one.begin(), cur_one.end(), pre_one.begin()) && set_one) {
      AddGradReduceIdx(&grad_reduce_index, cur_one, none_one, max_rank, j);
    } else {
      AddGradReduceIdx(&grad_reduce_index, cur_one, none_one, max_rank, j);
    }
    set_one = true;
    for (size_t i = 0; i < kDynamicBroadcastGradientArgsInputsNum; i++) {
      pre_one[i] = cur_one[i];
    }
  }
  return grad_reduce_index;
}

template <typename T, typename S>
size_t SetOuputValue(S *addr, const std::vector<T> &grad_reduce_idx, size_t input_num) {
  size_t index_num = grad_reduce_idx.size();
  for (size_t i = 0; i < index_num; i++) {
    addr[i] = static_cast<S>(grad_reduce_idx[index_num - 1 - i]);
  }

  return index_num;
}

template <typename T, typename S>
bool DynamicBroadcastGradientArgsCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                            const std::vector<kernel::AddressPtr> &,
                                                            const std::vector<kernel::AddressPtr> &outputs) {
  const T *s0_addr = reinterpret_cast<T *>(inputs[0]->addr);
  const T *s1_addr = reinterpret_cast<T *>(inputs[1]->addr);
  S *r0_addr = reinterpret_cast<S *>(outputs[0]->addr);
  S *r1_addr = reinterpret_cast<S *>(outputs[1]->addr);
  std::vector<size_t> ranks = {input_size_list_[0] / sizeof(T), input_size_list_[1] / sizeof(T)};

  std::vector<std::vector<T>> grad_reduce_idx(kDynamicBroadcastGradientArgsInputsNum);
  bool all_equal = true;
  size_t max_rank = ranks[0] > ranks[1] ? ranks[0] : ranks[1];
  size_t min_rank = ranks[0] < ranks[1] ? ranks[0] : ranks[1];
  for (size_t i = 0; i < min_rank; i++) {
    if (s0_addr[i] != s1_addr[i]) {
      all_equal = false;
      break;
    }
  }
  if (!all_equal) {
    // Reverse shapes
    std::vector<std::vector<T>> reverse_shapes(kDynamicBroadcastGradientArgsInputsNum);
    for (size_t j = 0; j < ranks[0]; j++) {
      reverse_shapes[0].push_back(s0_addr[ranks[0] - j - 1]);
    }
    if (reverse_shapes[0].size() < max_rank) {
      reverse_shapes[0].resize(max_rank, 1);
    }

    for (size_t j = 0; j < ranks[1]; j++) {
      reverse_shapes[1].push_back(s1_addr[ranks[1] - j - 1]);
    }
    if (reverse_shapes[1].size() < max_rank) {
      reverse_shapes[1].resize(max_rank, 1);
    }

    grad_reduce_idx = GetGradIndex(reverse_shapes, max_rank);
  }

  r0_size_ = SetOuputValue(r0_addr, grad_reduce_idx[0], input_size_list_[0] / sizeof(T));
  r1_size_ = SetOuputValue(r1_addr, grad_reduce_idx[1], input_size_list_[1] / sizeof(T));

  return true;
}

bool DynamicBroadcastGradientArgsCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                    const std::vector<KernelTensorPtr> &inputs,
                                                    const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::DynamicBroadcastGradientArgs>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast DynamicBroadcastGradientArgs ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();

  if (inputs.size() != kDynamicBroadcastGradientArgsInputsNum ||
      outputs.size() != kDynamicBroadcastGradientArgsOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kDynamicBroadcastGradientArgsInputsNum
                  << " and " << kDynamicBroadcastGradientArgsOutputsNum << ", but get " << inputs.size() << " and "
                  << outputs.size();
    return false;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  is_need_retrieve_output_shape_ = true;
  return true;
}

int DynamicBroadcastGradientArgsCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                     const std::vector<KernelTensorPtr> &inputs,
                                                     const std::vector<KernelTensorPtr> &outputs,
                                                     const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost) == static_cast<int>(KRET_RESIZE_FAILED)) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return static_cast<int>(KRET_RESIZE_FAILED);
  }
  // get input_shape
  outputs_ = outputs;

  return static_cast<int>(KRET_OK);
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &DynamicBroadcastGradientArgsCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &DynamicBroadcastGradientArgsCpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &DynamicBroadcastGradientArgsCpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &DynamicBroadcastGradientArgsCpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &DynamicBroadcastGradientArgsCpuKernelMod::LaunchKernel<int64_t, int64_t>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DynamicBroadcastGradientArgs, DynamicBroadcastGradientArgsCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
