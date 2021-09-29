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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDPERM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDPERM_GPU_KERNEL_H_

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class RandpermGpuKernel : public GpuKernel {
 public:
  // initialize rng here to minimize how many times we seed it.
  RandpermGpuKernel() : rng_(std::random_device()()) { ResetResource(); }

  ~RandpermGpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input_device = GetDeviceAddress<T>(inputs, 0);
    T *output_device = GetDeviceAddress<T>(outputs, 0);

    int32_t n = 0;
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(&n, input_device, sizeof(int32_t), cudaMemcpyDeviceToHost,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "Failed to copy error code to host.");

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaDeviceSynchronize(), "cudaDeviceSyncFailed in RandpermGpuKernel");

    if (static_cast<size_t>(n) > max_length_) {
      MS_LOG(EXCEPTION) << "RandpermGpuKernel: n (" << n << ") cannot exceed max_length_ (" << max_length_ << ")";
    }

    // might not be a significant performance gain if this kernel is executed in cuda,
    // so we do the calculations on host and copy to device afterwards.
    std::vector<T> output_host(max_length_);
    std::iota(output_host.begin(), output_host.begin() + n, 0);
    std::fill(output_host.begin() + n, output_host.end(), pad_);
    std::shuffle(output_host.begin(), output_host.begin() + n, rng_);

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(output_device, &output_host[0], max_length_ * sizeof(T),
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync for output_host failed");

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_count = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_count != 1) {
      MS_LOG(ERROR) << input_count << " inputs were provided, but RandpermGpuKernel expects 1.";
      return false;
    }

    size_t output_count = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_count != 1) {
      MS_LOG(ERROR) << "Number of outputs is " << output_count << ", but should be 1 for RandpermGpuKernel.";
      return false;
    }

    max_length_ = static_cast<size_t>(GetAttr<int64_t>(kernel_node, "max_length"));
    if (max_length_ < 1) {
      MS_LOG(ERROR) << "For 'RandpermGpuKernel', the max_length cannot be less than 1, but got " << max_length_;
      return false;
    }
    pad_ = static_cast<T>(GetAttr<int64_t>(kernel_node, "pad"));

    InitSizeLists();

    return true;
  }

  void ResetResource() noexcept override {
    max_length_ = 1;
    pad_ = -1;

    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(sizeof(int32_t));
    output_size_list_.push_back(max_length_ * sizeof(T));
  }

 private:
  std::mt19937 rng_;
  size_t max_length_;
  T pad_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDPERM_GPU_KERNEL_H_
