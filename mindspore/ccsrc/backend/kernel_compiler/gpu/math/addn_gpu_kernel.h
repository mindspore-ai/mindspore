/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_ADDN_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_ADDN_GPU_KERNEL_H_

#include <memory>
#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/math/broadcast_gpu_kernel.h"
#include "backend/kernel_compiler/gpu/cuda_impl/slice_impl.cuh"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class AddNGpuFwdKernel : public GpuKernel {
 public:
  AddNGpuFwdKernel() : input_size_(0), output_size_(0), workspace_size_(0), is_null_input_(false), num_input_(0) {}
  ~AddNGpuFwdKernel() override {}
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    auto work_addr = output_addr;
    for (size_t i = 0; i < num_input_; i++) {
      if (output_addr == GetDeviceAddress<T>(inputs, i)) {
        work_addr = GetDeviceAddress<T>(workspace, 0);
        break;
      }
    }
    FillDeviceArray(outputs[0]->size / sizeof(T), output_addr, 0.0f, reinterpret_cast<cudaStream_t>(stream_ptr));
    FillDeviceArray(outputs[0]->size / sizeof(T), work_addr, 0.0f, reinterpret_cast<cudaStream_t>(stream_ptr));
    for (size_t i = 0; i < num_input_; i++) {
      T *input_addr = GetDeviceAddress<T>(inputs, i);
      ElewiseArith(outputs[0]->size / sizeof(T), BROADCAST_TYPE_ADD, input_addr, work_addr, work_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    if (work_addr != output_addr) {
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(output_addr, work_addr, outputs[0]->size, cudaMemcpyDeviceToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "Addn cudaMemcpyAsync outputs failed");
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    num_input_ = GetAttr<int64_t>(kernel_node, "n");
    if (num_input_ != input_num) {
      MS_LOG(ERROR) << "Input number is " << num_input_ << " in attr, but got " << input_num << "input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but cudnnAddTensor needs 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "AddNGpuFwdKernel input is null";
      InitSizeLists();
      return true;
    }
    input_size_ = sizeof(T);
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    for (size_t i = 0; i < num_input_; i++) {
      input_size_list_.push_back(input_size_);
    }
    output_size_list_.push_back(input_size_);
    workspace_size_list_.push_back(input_size_);
  }

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  bool is_null_input_;
  size_t num_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_ADDN_GPU_KERNEL_H_
