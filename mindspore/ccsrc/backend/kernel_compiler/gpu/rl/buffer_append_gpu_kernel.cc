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
#include "backend/kernel_compiler/gpu/rl/buffer_append_gpu_kernel.h"

#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/gpu/cuda_impl/rl/rl_buffer_impl.cuh"
#include "runtime/device/gpu/gpu_common.h"

namespace mindspore {
namespace kernel {
BufferAppendKernel::BufferAppendKernel() : element_nums_(0), exp_batch_(0), capacity_(0) {}

BufferAppendKernel::~BufferAppendKernel() {}

void BufferAppendKernel::ReleaseResource() {}

const std::vector<size_t> &BufferAppendKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &BufferAppendKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &BufferAppendKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool BufferAppendKernel::Init(const CNodePtr &kernel_node) {
  kernel_node_ = kernel_node;
  auto shapes = GetAttr<std::vector<int64_t>>(kernel_node, "buffer_elements");
  auto types = GetAttr<std::vector<TypePtr>>(kernel_node, "buffer_dtype");
  capacity_ = GetAttr<int64_t>(kernel_node, "capacity");
  exp_batch_ = GetAttr<int64_t>(kernel_node, "exp_batch");
  element_nums_ = shapes.size();
  for (size_t i = 0; i < element_nums_; i++) {
    exp_element_list.push_back(shapes[i] * UnitSizeInBytes(types[i]->type_id()));
  }
  // buffer size
  for (auto i : exp_element_list) {
    input_size_list_.push_back(i * capacity_);
  }
  // exp size
  for (auto i : exp_element_list) {
    input_size_list_.push_back(i * exp_batch_);
  }
  // count and head
  input_size_list_.push_back(sizeof(int));
  input_size_list_.push_back(sizeof(int));
  output_size_list_.push_back(0);
  workspace_size_list_.push_back(sizeof(int));
  return true;
}

void BufferAppendKernel::InitSizeLists() { return; }

bool BufferAppendKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                const std::vector<AddressPtr> &, void *stream) {
  int *count_addr = GetDeviceAddress<int>(inputs, 2 * element_nums_);
  int *head_addr = GetDeviceAddress<int>(inputs, 2 * element_nums_ + 1);
  int *index_addr = GetDeviceAddress<int>(workspace, 0);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  IncreaseCount(capacity_, LongToInt(exp_batch_), count_addr, head_addr, index_addr, cuda_stream);
  for (size_t i = 0; i < element_nums_; i++) {
    auto buffer_addr = GetDeviceAddress<unsigned char>(inputs, i);
    auto exp_addr = GetDeviceAddress<unsigned char>(inputs, i + element_nums_);
    size_t one_exp_len = input_size_list_[i + element_nums_];
    BufferAppend(capacity_, one_exp_len, index_addr, LongToInt(exp_batch_), buffer_addr, exp_addr, cuda_stream);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
