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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DEBUG_PRINT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DEBUG_PRINT_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class PrintGpuKernel : public GpuKernel {
 public:
  PrintGpuKernel() { ResetResource(); }
  ~PrintGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    VARIABLE_NOT_USED(outputs);
    for (size_t i = 0; i < inputs.size(); i++) {
      input_device_data_[i] = GetDeviceAddress<T>(inputs, i);
    }
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemcpy(&input_host_data_[0], &input_device_data_[0], input_size_ * sizeof(T), cudaMemcpyDeviceToHost),
      "cudaMemcpy output failed");
    for (size_t i = 0; i < input_num_.size(); i++) {
      for (size_t j = 0; j < input_num_[i]; j++) {
        std::cout << input_host_data_[i][j];
      }
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    size_t input_tensor_num = AnfAlgo::GetInputTensorNum(kernel_node);
    input_device_data_ = std::make_unique<T *[]>(input_tensor_num);
    input_host_data_ = std::make_unique<T *[]>(input_tensor_num);
    for (size_t i = 0; i < input_tensor_num; i++) {
      size_t counter = 0;
      auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i);
      for (size_t j = 0; j < input_shape.size(); j++) {
        input_size_ *= input_shape[j];
        counter++;
      }
      input_num_.push_back(counter);
    }
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 1;
    input_device_data_ = nullptr;
    input_host_data_ = nullptr;
    input_num_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override { input_size_list_.push_back(input_size_ * sizeof(T)); }

 private:
  size_t input_size_;
  std::unique_ptr<T *[]> input_device_data_;
  std::unique_ptr<T *[]> input_host_data_;
  std::vector<size_t> input_num_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};  // namespace kernel
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DEBUG_PRINT_GPU_KERNEL_H_
