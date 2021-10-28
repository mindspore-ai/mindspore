/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/host/dynamic_broadcast_gradient_args_kernel.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
namespace {
const int kInputNum = 2;
const size_t one = 1;

void UpdatePreIsOne(std::vector<bool> *prev_is_one, std::vector<bool> current_is_one) {
  for (size_t i = 0; i < kInputNum; ++i) {
    (*prev_is_one)[i] = current_is_one[i];
  }
}

void AddElementToGradReduceIdx(std::vector<std::vector<int64_t>> *grad_reduce_idx, std::vector<bool> current_is_one,
                               bool none_is_one, const size_t largest_rank, size_t j) {
  MS_EXCEPTION_IF_NULL(grad_reduce_idx);
  for (size_t i = 0; i < kInputNum; ++i) {
    if (current_is_one[i] && !none_is_one) {
      (void)(*grad_reduce_idx)[i].emplace_back(SizeToLong(largest_rank - one - j));
    }
  }
}

std::vector<std::vector<int64_t>> GetGradientIndices(const std::vector<std::vector<int64_t>> &reverse_shape,
                                                     const size_t largest_rank) {
  std::vector<std::vector<int64_t>> grad_reduce_idx(kInputNum);
  // indices of j-th component of each input.
  std::vector<bool> prev_is_one(kInputNum);
  std::vector<bool> current_is_one(kInputNum);
  for (size_t i = 0; i < kInputNum; ++i) {
    prev_is_one[i] = false;
    current_is_one[i] = false;
  }

  bool set_one = false;
  for (size_t j = 0; j < largest_rank; ++j) {
    int output_dim = -1;
    bool output_dim_set = false;
    bool none_is_one = true;
    // Find which indices are 1.
    for (size_t i = 0; i < kInputNum; ++i) {
      if (reverse_shape[i][j] == 1) {
        current_is_one[i] = true;
        none_is_one = false;
      } else {
        current_is_one[i] = false;
        if (!output_dim_set || reverse_shape[i][j] == static_cast<int64_t>(output_dim)) {
          output_dim = LongToInt(reverse_shape[i][j]);
          output_dim_set = true;
        } else {
          MS_LOG(EXCEPTION) << "Input[0] and input[1] Cannot broadcast!";
        }
      }
    }
    // All dimensions are 1.
    if (!output_dim_set) {
      for (size_t i = 0; i < kInputNum; ++i) {
        (void)grad_reduce_idx[i].emplace_back(SizeToLong(largest_rank - one - j));
      }
      continue;
    } else if (std::equal(current_is_one.begin(), current_is_one.end(), prev_is_one.begin()) && set_one) {
      AddElementToGradReduceIdx(&grad_reduce_idx, current_is_one, none_is_one, largest_rank, j);
    } else {
      AddElementToGradReduceIdx(&grad_reduce_idx, current_is_one, none_is_one, largest_rank, j);
    }
    set_one = true;
    UpdatePreIsOne(&prev_is_one, current_is_one);
  }
  return grad_reduce_idx;
}

std::vector<std::vector<int64_t>> CalculateOutput(const std::vector<std::vector<int64_t>> &x) {
  std::vector<std::vector<int64_t>> grad_reduce_idx(kInputNum);
  bool all_equal = true;
  size_t largest_rank = 0;
  for (size_t i = 0; i < kInputNum; ++i) {
    if (x[i] != x[0]) {
      all_equal = false;
    }
    if (x[i].size() > largest_rank) {
      largest_rank = x[i].size();
    }
  }
  if (all_equal) {
    return grad_reduce_idx;
  }

  // Reverse input the shapes
  std::vector<std::vector<int64_t>> reverse_shape(kInputNum);
  for (size_t i = 0; i < kInputNum; ++i) {
    reverse_shape[i] = x[i];
    std::reverse(reverse_shape[i].begin(), reverse_shape[i].end());
  }

  // 1-extend and align all vectors.
  for (size_t i = 0; i < kInputNum; ++i) {
    if (reverse_shape[i].size() < largest_rank) {
      reverse_shape[i].resize(largest_rank, 1);
    }
  }
  grad_reduce_idx = GetGradientIndices(reverse_shape, largest_rank);
  return grad_reduce_idx;
}

std::vector<int64_t> GetInputShape(const CNodePtr &cnode, size_t index) {
  auto address_x = AnfAlgo::GetPrevNodeMutableOutputAddr(cnode, index);
  auto shape_x = AnfAlgo::GetPrevNodeOutputInferShape(cnode, index);
  auto type_x = AnfAlgo::GetOutputInferDataType(cnode, index);
  if (type_x != TypeId::kNumberTypeInt64) {
    MS_LOG(EXCEPTION) << "Input x type must be int64, but :" << type_x;
  }
  if (shape_x.size() != 1) {
    MS_LOG(EXCEPTION) << "Input" << index << " must be [1-D], but " << shape_x.size() << "-D.";
  }

  size_t x_num = shape_x[0];
  std::vector<int64_t> x{SizeToLong(x_num)};

  auto x_shape_value = std::make_shared<tensor::Tensor>(type_x, x);
  // The second parameter must be false, otherwise the device address cannot be released and allocated, and the
  // address size will be wrong in the dynamic shape scenario.
  MS_EXCEPTION_IF_NULL(x_shape_value);
  x_shape_value->set_device_address(address_x, false);
  x_shape_value->data_sync();

  auto x_value = reinterpret_cast<int64_t *>(x_shape_value->data_c());
  MS_EXCEPTION_IF_NULL(x_value);
  std::vector<int64_t> input_shape = {x_value, x_value + x_num};
  return input_shape;
}

size_t SetOutputValue(const CNodePtr &cnode, const std::vector<std::vector<int64_t>> &grad_reduce_idx, size_t index,
                      size_t input_num) {
  std::vector<int64_t> output;
  size_t idx_num = grad_reduce_idx[index].size();

  for (size_t k = 0; k < idx_num; ++k) {
    output.push_back(grad_reduce_idx[index][idx_num - 1 - k]);
  }

  auto out_addr = AnfAlgo::GetOutputAddr(cnode, index);
  MS_EXCEPTION_IF_NULL(out_addr);

  size_t out_size = idx_num;
  if (idx_num == 0) {
    out_size = input_num;
    for (size_t k = 0; k < input_num; ++k) {
      output.push_back(k);
    }
  }

  std::vector<int64_t> out_shape{SizeToLong(out_size)};
  auto output_type = TypeId::kNumberTypeInt64;
  auto tensor_for_sync = std::make_shared<tensor::Tensor>(output_type, out_shape);
  MS_EXCEPTION_IF_NULL(tensor_for_sync);

  auto data_ptr = static_cast<int64_t *>(tensor_for_sync->data_c());
  for (size_t i = 0; i < out_size; ++i) {
    MS_LOG(DEBUG) << "DEBUG r" << index << "_output_shape[" << i << "]:" << output[i];
    *(data_ptr + i) = output[i];
  }

  if (!out_addr->SyncHostToDevice(out_shape, LongToSize(tensor_for_sync->data().nbytes()), tensor_for_sync->data_type(),
                                  tensor_for_sync->data_c(), tensor_for_sync->device_info().host_format_)) {
    MS_LOG(EXCEPTION) << "Output Value SyncHostToDevice failed.";
  }
  return out_size;
}
}  // namespace

void DynamicBroadcastGradientArgsKernel::Execute() {
  MS_LOG(INFO) << "Execute DynamicBroadcastGradientArgsKernel Start";
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_num = AnfAlgo::GetInputTensorNum(cnode);
  if (input_num != kInputNum) {
    MS_LOG(EXCEPTION) << "Invalid Input Num:" << input_num;
  }

  std::vector<std::vector<int64_t>> input_shapes(kInputNum);
  input_shapes[0] = GetInputShape(cnode, 0);
  input_shapes[1] = GetInputShape(cnode, 1);
  auto grad_reduce_idx = CalculateOutput(input_shapes);

  auto r0_size = SetOutputValue(cnode, grad_reduce_idx, 0, input_shapes[0].size());
  auto r1_size = SetOutputValue(cnode, grad_reduce_idx, 1, input_shapes[1].size());

  std::vector<size_t> r0_shp{r0_size};
  std::vector<size_t> r1_shp{r1_size};
  auto output_type = TypeId::kNumberTypeInt64;
  AnfAlgo::SetOutputInferTypeAndShape({output_type, output_type}, {r0_shp, r1_shp}, cnode.get());
  MS_LOG(INFO) << "Execute DynamicBroadcastGradientArgsKernel End";
}

device::DynamicKernelPtr DynamicBroadcastGradientArgsKernelMod::GenDynamicKernel(const CNodePtr &cnode_ptr,
                                                                                 void *stream_ptr) {
  return std::make_shared<DynamicBroadcastGradientArgsKernel>(stream_ptr, cnode_ptr);
}
}  // namespace kernel
}  // namespace mindspore
