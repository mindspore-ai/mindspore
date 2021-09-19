/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_LINSPACE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_LINSPACE_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <iostream>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/linspace.cuh"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class LinSpaceGpuKernel : public GpuKernel {
 public:
  LinSpaceGpuKernel() { ResetResource(); }
  ~LinSpaceGpuKernel() = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);
    T *start_addr = GetDeviceAddress<T>(inputs, 0);
    T *stop_addr = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    calLinSpace(start_addr, stop_addr, value_count_, output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 3) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but DynamicLinSpace needs 3 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but DynamicLinSpace needs 1 output.";
      return false;
    }
    auto input_1 = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    auto input_2 = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 1);
    auto value_count = AnfAlgo::GetOutputRealDeviceShapeIfExist(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_1) || CHECK_NULL_INPUT(input_2) || CHECK_NULL_INPUT(value_count);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'LinspaceGpuKernel', input is null";
      InitSizeLists();
      return true;
    }
    // error checking input data
    if ((input_1.size() != 0) || (input_2.size() != 0)) {
      MS_LOG(ERROR) << "For LinShape "
                    << "both start and end must be 0-D Tensors. Got " << input_1.size() << " and " << input_2.size()
                    << ".";
      return false;
    }

    if (value_count.size() != 1) {
      MS_LOG(ERROR) << "For LinShape, output shape incorrect rank. Expect Rank: 1, got Rank: " << value_count.size()
                    << ".";
      return false;
    }
    value_count_ = value_count[0];
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    value_count_ = 0;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(sizeof(T));  // Scalar tensor
    input_size_list_.push_back(sizeof(T));  // Scalar tensor
    output_size_list_.push_back(value_count_ * sizeof(T));
  }

 private:
  size_t value_count_ = 0;
  bool is_null_input_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_LINSPACE_GPU_KERNEL_H_
