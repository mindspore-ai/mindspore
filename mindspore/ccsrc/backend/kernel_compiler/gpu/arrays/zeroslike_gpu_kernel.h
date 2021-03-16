/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ZEROSLIKE_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ZEROSLIKE_GPU_KERNEL_H

#include <vector>

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class ZerosLikeGpuKernel : public GpuKernel {
 public:
  ZerosLikeGpuKernel() { ResetResource(); }
  ~ZerosLikeGpuKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *output_device_address = GetDeviceAddress<T>(outputs, 0);

    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      // have to use a float literal instead of an int literal because of ambiguous half() overload.
      cudaMemsetAsync(output_device_address, static_cast<T>(0.0), input_size_ * sizeof(T),
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemset failed");

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;

    std::vector<size_t> input_shape = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }

    InitSizeLists();

    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 1;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    // allocate space for input even though we don't need to do anything with the input
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(input_size_ * sizeof(T));
  }

 private:
  size_t input_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ZEROSLIKE_GPU_KERNEL_H
