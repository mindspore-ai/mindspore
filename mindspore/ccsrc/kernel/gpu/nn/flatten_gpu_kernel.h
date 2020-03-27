/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_FLATTEN_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_FLATTEN_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class FlattenGpuFwdKernel : public GpuKernel {
 public:
  FlattenGpuFwdKernel() : input_size_(0), output_size_(0), workspace_size_(0) {}
  ~FlattenGpuFwdKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);
    cudaError_t ret =
      cudaMemcpyAsync(output, input, input_size_, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr));
    if (ret) {
      MS_LOG(ERROR) << "cudaMemcpyAsync error in FlattenGpuFwdKernel::Launch, error code is " << ret;
      return false;
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < shape.size(); ++i) {
      if (input_size_ == 0) {
        input_size_ = 1;
      }
      input_size_ *= shape[i];
    }
    input_size_ = input_size_ * sizeof(T);

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_ = input_size_;
    output_size_list_.push_back(output_size_);
  }

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_FLATTEN_GPU_KERNEL_H_
