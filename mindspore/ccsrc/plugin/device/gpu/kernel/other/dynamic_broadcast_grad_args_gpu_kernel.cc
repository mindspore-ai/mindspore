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
#include "plugin/device/gpu/kernel/other/dynamic_broadcast_grad_args_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 2;
constexpr size_t kOutputsNum = 2;
constexpr char kKernelName[] = "DynamicBroadcastGradientArgs";
using KernelRunFunc = DynamicBroadcastGradientArgsGpuKernelMod::KernelRunFunc;
}  // namespace

template <typename T>
void AddGradReduceIdx(std::vector<std::vector<T>> *grad_reduce_idx, std::vector<bool> cur_one, bool none_one,
                      const size_t max_rank, size_t j) {
  MS_EXCEPTION_IF_NULL(grad_reduce_idx);
  for (size_t i = 0; i < kInputsNum; i++) {
    if (cur_one[i] && !none_one) {
      (void)(*grad_reduce_idx)[i].emplace_back(SizeToLong(max_rank - 1 - j));
    }
  }
}

template <typename T>
std::vector<std::vector<T>> GetGradIndex(const std::vector<std::vector<T>> &reverse_shapes, const size_t max_rank) {
  std::vector<std::vector<T>> grad_reduce_index(kInputsNum);
  std::vector<bool> pre_one(kInputsNum);
  std::vector<bool> cur_one(kInputsNum);
  for (size_t i = 0; i < kInputsNum; i++) {
    pre_one[i] = false;
    cur_one[i] = false;
  }
  bool set_one = false;
  for (size_t j = 0; j < max_rank; j++) {
    int out_dim = -1;
    bool out_dim_set = false;
    bool none_one = true;
    for (size_t i = 0; i < kInputsNum; i++) {
      if (reverse_shapes[i][j] == 1) {
        cur_one[i] = true;
        none_one = false;
      } else {
        cur_one[i] = false;
        if (!out_dim_set || reverse_shapes[i][j] == static_cast<T>(out_dim)) {
          out_dim = static_cast<int>(reverse_shapes[i][j]);
          out_dim_set = true;
        } else {
          MS_LOG(EXCEPTION) << "Can not broadcast inputs[0] and inputs[1].";
        }
      }
    }
    if (!out_dim_set) {
      for (size_t i = 0; i < kInputsNum; i++) {
        (void)grad_reduce_index[i].emplace_back(max_rank - 1 - j);
      }
      continue;
    } else if (std::equal(cur_one.begin(), cur_one.end(), pre_one.begin()) && set_one) {
      AddGradReduceIdx(&grad_reduce_index, cur_one, none_one, max_rank, j);
    } else {
      AddGradReduceIdx(&grad_reduce_index, cur_one, none_one, max_rank, j);
    }
    set_one = true;
    for (size_t i = 0; i < kInputsNum; i++) {
      pre_one[i] = cur_one[i];
    }
  }
  return grad_reduce_index;
}

template <typename T, typename S>
size_t SetOuputValue(S *addr, const std::vector<T> &grad_reduce_idx, cudaStream_t cuda_stream) {
  std::vector<S> output;
  size_t index_num = grad_reduce_idx.size();
  for (size_t i = 0; i < index_num; i++) {
    output.push_back(static_cast<S>(grad_reduce_idx[index_num - 1 - i]));
  }
  size_t out_size = index_num;
  if (index_num == 0) {
    return out_size;
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(addr, &output[0], out_size * sizeof(S), cudaMemcpyHostToDevice, cuda_stream),
    "DynamicBroadcastGradientArgs copy output failed");
  return out_size;
}

template <typename T, typename S>
bool DynamicBroadcastGradientArgsGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                            const std::vector<AddressPtr> &,
                                                            const std::vector<kernel::AddressPtr> &outputs) {
  std::vector<std::vector<T>> reverse_shapes(kInputsNum);
  std::vector<size_t> ranks = {input_size_list_[0] / sizeof(T), input_size_list_[1] / sizeof(T)};
  std::vector<T> x0_value(input_size_list_[0] / sizeof(T), 0);
  if (!is_null_input0_) {
    auto s0_addr = GetDeviceAddress<T>(inputs, 0);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(&x0_value[0], s0_addr, input_size_list_[0], cudaMemcpyDeviceToHost, cuda_stream_),
      "DynamicBroadcastGradientArgs copy s0 value failed");
    for (size_t j = 0; j < ranks[0]; j++) {
      reverse_shapes[0].push_back(x0_value[ranks[0] - j - 1]);
    }
  } else {
    ranks[0] = 0;
  }
  std::vector<T> x1_value(input_size_list_[1] / sizeof(T), 0);
  if (!is_null_input1_) {
    auto s1_addr = GetDeviceAddress<T>(inputs, 1);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(&x1_value[0], s1_addr, input_size_list_[1], cudaMemcpyDeviceToHost, cuda_stream_),
      "DynamicBroadcastGradientArgs copy s1 value failed");
    for (size_t j = 0; j < ranks[1]; j++) {
      reverse_shapes[1].push_back(x1_value[ranks[1] - j - 1]);
    }
  } else {
    ranks[1] = 0;
  }
  auto r0_addr = GetDeviceAddress<S>(outputs, 0);
  auto r1_addr = GetDeviceAddress<S>(outputs, 1);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_),
                                     "DynamicBroadcastGradientArgs cudaStreamSynchronized failed");
  std::vector<std::vector<T>> grad_reduce_idx(kInputsNum);
  size_t max_rank = ranks[0] > ranks[1] ? ranks[0] : ranks[1];
  if (reverse_shapes[0].size() < max_rank) {
    reverse_shapes[0].resize(max_rank, 1);
  }
  if (reverse_shapes[1].size() < max_rank) {
    reverse_shapes[1].resize(max_rank, 1);
  }

  if (reverse_shapes[0] != reverse_shapes[1]) {
    grad_reduce_idx = GetGradIndex(reverse_shapes, max_rank);
  }
  r0_size_ = SetOuputValue(r0_addr, grad_reduce_idx[0], cuda_stream_);
  r1_size_ = SetOuputValue(r1_addr, grad_reduce_idx[1], cuda_stream_);

  return true;
}

bool DynamicBroadcastGradientArgsGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                    const std::vector<KernelTensorPtr> &inputs,
                                                    const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::DynamicBroadcastGradientArgs>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast DynamicBroadcastGradientArgs ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();

  if (inputs.size() != kInputsNum || outputs.size() != kOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kInputsNum << " and " << kOutputsNum
                  << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  is_need_retrieve_output_shape_ = true;
  return true;
}

int DynamicBroadcastGradientArgsGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                     const std::vector<KernelTensorPtr> &inputs,
                                                     const std::vector<KernelTensorPtr> &outputs,
                                                     const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost) == static_cast<int>(KRET_RESIZE_FAILED)) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return static_cast<int>(KRET_RESIZE_FAILED);
  }
  // get input_shape
  auto input_0_shape = inputs[0]->GetShapeVector();
  auto input_1_shape = inputs[1]->GetShapeVector();
  is_null_input0_ = CHECK_NULL_INPUT(input_0_shape);
  is_null_input1_ = CHECK_NULL_INPUT(input_1_shape);
  outputs_ = outputs;

  return static_cast<int>(KRET_OK);
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &DynamicBroadcastGradientArgsGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &DynamicBroadcastGradientArgsGpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &DynamicBroadcastGradientArgsGpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &DynamicBroadcastGradientArgsGpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &DynamicBroadcastGradientArgsGpuKernelMod::LaunchKernel<int64_t, int64_t>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, DynamicBroadcastGradientArgs, DynamicBroadcastGradientArgsGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
