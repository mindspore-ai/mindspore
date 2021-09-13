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
#include "backend/kernel_compiler/gpu/rl/buffer_get_gpu_kernel.h"

#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/gpu/cuda_impl/rl/rl_buffer_impl.cuh"
#include "runtime/device/gpu/gpu_common.h"

namespace mindspore {
namespace kernel {
BufferGetKernel::BufferGetKernel() : element_nums_(0), capacity_(0) {}

BufferGetKernel::~BufferGetKernel() {}

void BufferGetKernel::ReleaseResource() {}

const std::vector<size_t> &BufferGetKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &BufferGetKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &BufferGetKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool BufferGetKernel::Init(const CNodePtr &kernel_node) {
  kernel_node_ = kernel_node;
  auto shapes = GetAttr<std::vector<int64_t>>(kernel_node, "buffer_elements");
  auto types = GetAttr<std::vector<TypePtr>>(kernel_node, "buffer_dtype");
  capacity_ = GetAttr<int64_t>(kernel_node, "capacity");
  element_nums_ = shapes.size();
  for (size_t i = 0; i < element_nums_; i++) {
    exp_element_list.push_back(shapes[i] * UnitSizeInBytes(types[i]->type_id()));
  }
  // buffer size
  for (auto i : exp_element_list) {
    input_size_list_.push_back(i * capacity_);
    output_size_list_.push_back(i);
  }
  // count, head, index
  input_size_list_.push_back(sizeof(int));
  input_size_list_.push_back(sizeof(int));
  input_size_list_.push_back(sizeof(int));
  workspace_size_list_.push_back(sizeof(int));
  return true;
}

void BufferGetKernel::InitSizeLists() { return; }

bool BufferGetKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                             const std::vector<AddressPtr> &outputs, void *stream) {
  int *count_addr = GetDeviceAddress<int>(inputs, element_nums_);
  int *head_addr = GetDeviceAddress<int>(inputs, element_nums_ + 1);
  int *origin_index_addr = GetDeviceAddress<int>(inputs, element_nums_ + 2);
  int *index_addr = GetDeviceAddress<int>(workspace, 0);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  ReMappingIndex(count_addr, head_addr, origin_index_addr, index_addr, cuda_stream);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return false;
  }
  for (size_t i = 0; i < element_nums_; i++) {
    auto buffer_addr = GetDeviceAddress<unsigned char>(inputs, i);
    auto item_addr = GetDeviceAddress<unsigned char>(outputs, i);
    size_t one_exp_len = output_size_list_[i];
    BufferGetItem(one_exp_len, index_addr, one_exp_len, buffer_addr, item_addr, cuda_stream);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
