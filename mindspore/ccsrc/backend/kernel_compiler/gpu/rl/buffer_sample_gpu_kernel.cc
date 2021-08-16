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
#include "backend/kernel_compiler/gpu/rl/buffer_sample_gpu_kernel.h"

#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <chrono>

#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/gpu/cuda_impl/rl/rl_buffer_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/topk_impl.cuh"
#include "runtime/device/gpu/gpu_common.h"

namespace mindspore {
namespace kernel {

BufferSampleKernel::BufferSampleKernel() : element_nums_(0), capacity_(0), batch_size_(0), seed_(0) {}

BufferSampleKernel::~BufferSampleKernel() {}

void BufferSampleKernel::ReleaseResource() {}

const std::vector<size_t> &BufferSampleKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &BufferSampleKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &BufferSampleKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool BufferSampleKernel::Init(const CNodePtr &kernel_node) {
  kernel_node_ = kernel_node;
  auto shapes = GetAttr<std::vector<int64_t>>(kernel_node, "buffer_elements");
  auto types = GetAttr<std::vector<TypePtr>>(kernel_node, "buffer_dtype");
  capacity_ = GetAttr<int64_t>(kernel_node, "capacity");
  seed_ = GetAttr<int64_t>(kernel_node, "seed");
  batch_size_ = LongToSize(GetAttr<int64_t>(kernel_node, "batch_size"));
  element_nums_ = shapes.size();
  for (size_t i = 0; i < element_nums_; i++) {
    auto element = shapes[i] * UnitSizeInBytes(types[i]->type_id());
    exp_element_list.push_back(element);
    input_size_list_.push_back(capacity_ * element);
    output_size_list_.push_back(batch_size_ * element);
  }
  // count and head
  input_size_list_.push_back(sizeof(int));
  input_size_list_.push_back(sizeof(int));
  workspace_size_list_.push_back(capacity_ * sizeof(curandState));
  workspace_size_list_.push_back(capacity_ * sizeof(float));
  workspace_size_list_.push_back(capacity_ * sizeof(int));
  workspace_size_list_.push_back(capacity_ * sizeof(float));
  return true;
}

void BufferSampleKernel::InitSizeLists() { return; }

bool BufferSampleKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
                                const std::vector<AddressPtr> &outputs, void *stream) {
  int *count_addr = GetDeviceAddress<int>(inputs, element_nums_);
  int *head_addr = GetDeviceAddress<int>(inputs, element_nums_ + 1);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CheckBatchSize(count_addr, head_addr, batch_size_, capacity_, cuda_stream);
  int k_cut = 0;
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                             cudaMemcpyAsync(&k_cut, count_addr, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream),
                             "sync dev to host failed");
  // 1 Generate random floats
  auto States = GetDeviceAddress<void *>(workspaces, 0);
  auto random_f = GetDeviceAddress<float>(workspaces, 1);
  auto indexes = GetDeviceAddress<int>(workspaces, 2);
  auto useless_out = GetDeviceAddress<float>(workspaces, 3);
  int seedc = 0;
  if (seed_ == 0) {
    generator_.seed(std::chrono::system_clock::now().time_since_epoch().count());
    seedc = generator_();
  } else {
    seedc = seed_;
  }

  float init_k = std::numeric_limits<float>::lowest();
  curandState *devStates = reinterpret_cast<curandState *>(States);
  RandomGen(k_cut, devStates, seedc, random_f, cuda_stream);
  // 2 Sort the random floats, and get the sorted indexes as the random indexes
  FastTopK(1, k_cut, random_f, k_cut, useless_out, indexes, init_k, cuda_stream);
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaDeviceSynchronize(), "cudaDeviceSync failed, sample-topk");
  for (size_t i = 0; i < element_nums_; i++) {
    auto buffer_addr = GetDeviceAddress<unsigned char>(inputs, i);
    auto out_addr = GetDeviceAddress<unsigned char>(outputs, i);
    size_t size = batch_size_ * exp_element_list[i];
    BufferSample(size, exp_element_list[i], indexes, buffer_addr, out_addr, cuda_stream);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
