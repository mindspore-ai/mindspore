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

#include "plugin/device/ascend/kernel/host/dynamic_broadcast_gradient_args_kernel.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/hal/device/ascend_kernel_runtime.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "utils/trace_base.h"

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

std::vector<int64_t> GetInputShape(const CNodePtr &cnode, const std::vector<AddressPtr> &inputs, size_t index) {
  auto shape_x = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, index);
  auto type_x = common::AnfAlgo::GetOutputInferDataType(cnode, index);
  if (type_x != TypeId::kNumberTypeInt64) {
    MS_LOG(EXCEPTION) << "Input x type must be int64, but got " << type_x << trace::DumpSourceLines(cnode);
  }
  if (shape_x.size() != 1) {
    MS_LOG(EXCEPTION) << "Input" << index << " must be [1-D], but got " << shape_x.size()
                      << trace::DumpSourceLines(cnode);
  }
  auto all_input_formats = AnfAlgo::GetAllInputFormats(cnode);
  if (index >= all_input_formats.size()) {
    MS_LOG(EXCEPTION) << "Input index is" << index << ", but the node only has " << all_input_formats.size()
                      << " inputs.";
  }
  device::ascend::AscendDeviceAddressPtr address_x = std::make_shared<device::ascend::AscendDeviceAddress>(
    inputs[index]->addr, inputs[index]->size, all_input_formats[index], type_x);

  auto x_num = shape_x[0];
  std::vector<int64_t> x{x_num};

  auto x_shape_value = std::make_shared<tensor::Tensor>(type_x, x);
  // The second parameter must be false, otherwise the device address cannot be released and allocated, and the
  // address size will be wrong in the dynamic shape scenario.
  MS_EXCEPTION_IF_NULL(x_shape_value);
  x_shape_value->set_device_address(address_x, false);
  x_shape_value->data_sync();

  auto x_value = static_cast<int64_t *>(x_shape_value->data_c());
  MS_EXCEPTION_IF_NULL(x_value);
  std::vector<int64_t> input_shape = {x_value, x_value + x_num};
  return input_shape;
}

size_t SetOutputValue(const CNodePtr &cnode, const std::vector<std::vector<int64_t>> &grad_reduce_idx, size_t index) {
  std::vector<int64_t> output;
  size_t out_size = grad_reduce_idx[index].size();
  for (size_t k = 0; k < out_size; ++k) {
    output.push_back(grad_reduce_idx[index][out_size - 1 - k]);
  }
  if (out_size == 0) {
    return out_size;
  }
  auto out_addr = AnfAlgo::GetOutputAddr(cnode, index);
  MS_EXCEPTION_IF_NULL(out_addr);

  std::vector<int64_t> out_shape{SizeToLong(out_size)};
  auto output_type = TypeId::kNumberTypeInt64;
  auto tensor_for_sync = std::make_shared<tensor::Tensor>(output_type, out_shape);
  MS_EXCEPTION_IF_NULL(tensor_for_sync);

  auto data_ptr = static_cast<int64_t *>(tensor_for_sync->data_c());
  for (size_t i = 0; i < out_size; ++i) {
    MS_LOG(DEBUG) << "DEBUG r" << index << "_output_shape[" << i << "]:" << output[i];
    *(data_ptr + i) = output[i];
  }

  auto runtime_instance = device::KernelRuntimeManager::Instance().GetCurrentKernelRuntime();
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto ret = runtime_instance->SyncStream();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Sync stream error!";
  }
  if (!out_addr->SyncHostToDevice(out_shape, LongToSize(tensor_for_sync->data().nbytes()), tensor_for_sync->data_type(),
                                  tensor_for_sync->data_c(), tensor_for_sync->device_info().host_format_)) {
    MS_LOG(EXCEPTION) << "Output Value SyncHostToDevice failed.";
  }
  return out_size;
}
}  // namespace

void DynamicBroadcastGradientArgsKernelMod::Execute(const std::vector<AddressPtr> &inputs) const {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  if (input_num != kInputNum) {
    MS_LOG(EXCEPTION) << "Invalid input num, should be " << kInputNum << ", but got " << input_num
                      << trace::DumpSourceLines(cnode);
  }

  std::vector<std::vector<int64_t>> input_shapes(kInputNum);
  input_shapes[0] = GetInputShape(cnode, inputs, 0);
  input_shapes[1] = GetInputShape(cnode, inputs, 1);
  auto grad_reduce_idx = CalculateOutput(input_shapes);

  auto r0_size = SetOutputValue(cnode, grad_reduce_idx, 0);
  auto r1_size = SetOutputValue(cnode, grad_reduce_idx, 1);

  ShapeVector r0_shp{SizeToLong(r0_size)};
  ShapeVector r1_shp{SizeToLong(r1_size)};
  auto output_type = TypeId::kNumberTypeInt64;
  common::AnfAlgo::SetOutputInferTypeAndShape({output_type, output_type}, {r0_shp, r1_shp}, cnode.get());
}

bool DynamicBroadcastGradientArgsKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &,
                                                   const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  try {
    Execute(inputs);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "DynamicBroadcastGradientArgsKernel Launch failed. node: " << cnode->fullname_with_scope()
                  << ", Error message is " << e.what();
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
