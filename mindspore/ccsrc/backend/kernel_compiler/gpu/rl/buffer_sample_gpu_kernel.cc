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
BufferSampleKernel::BufferSampleKernel()
    : element_nums_(0), capacity_(0), batch_size_(0), seed_(0), states_init_(false), unique_(false) {}

BufferSampleKernel::~BufferSampleKernel() {
  if (devStates_ != nullptr) {
    device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(static_cast<void *>(devStates_));
  }
}

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
  unique_ = GetAttr<bool>(kernel_node, "unique");
  batch_size_ = LongToSize(GetAttr<int64_t>(kernel_node, "batch_size"));
  element_nums_ = shapes.size();
  // Set default seed, if seed == 0
  if (seed_ == 0) {
    generator_.seed(std::chrono::system_clock::now().time_since_epoch().count());
    seed_ = generator_();
  }
  auto indexes_size = batch_size_;
  if (unique_) indexes_size = capacity_;
  // Keep the device memory for curandstate
  const size_t cap_state_size = sizeof(curandState) * indexes_size;
  void *dev_state = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(cap_state_size);
  if (dev_state == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to alloc dev_state, size is " << cap_state_size;
  }
  devStates_ = reinterpret_cast<curandState *>(dev_state);

  for (size_t i = 0; i < element_nums_; i++) {
    auto element = shapes[i] * UnitSizeInBytes(types[i]->type_id());
    exp_element_list.push_back(element);
    input_size_list_.push_back(capacity_ * element);
    output_size_list_.push_back(batch_size_ * element);
  }
  // count and head
  input_size_list_.push_back(sizeof(int));
  input_size_list_.push_back(sizeof(int));
  workspace_size_list_.push_back(indexes_size * sizeof(unsigned int));
  if (unique_) {
    workspace_size_list_.push_back(indexes_size * sizeof(unsigned int));
  }
  return true;
}

void BufferSampleKernel::InitSizeLists() { return; }

bool BufferSampleKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
                                const std::vector<AddressPtr> &outputs, void *stream) {
  int *count_addr = GetDeviceAddress<int>(inputs, element_nums_);
  int *head_addr = GetDeviceAddress<int>(inputs, element_nums_ + 1);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CheckBatchSize(count_addr, head_addr, batch_size_, capacity_, cuda_stream);
  int k_num = 0;
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                             cudaMemcpyAsync(&k_num, count_addr, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream),
                             "sync dev to host failed");
  // 1 Init curandState for the first time
  if (!states_init_) {
    RandInit(unique_ ? capacity_ : batch_size_, seed_, devStates_, cuda_stream);
    states_init_ = true;
  }
  // 2 Generate random indexes by kernel
  auto indexes = GetDeviceAddress<unsigned int>(workspaces, 0);
  if (unique_) {
    auto key = GetDeviceAddress<unsigned int>(workspaces, 1);
    RandomGen(k_num, devStates_, indexes, key, cuda_stream);
  } else {
    RandomGenUniform(batch_size_, devStates_, k_num, indexes, cuda_stream);
  }
  // 3 Sample data by indexes
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
