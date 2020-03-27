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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CONCATV2_GPU_KERNEL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_CONCATV2_GPU_KERNEL_H

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/concatv2_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class ConcatV2GpuFwdKernel : public GpuKernel {
 public:
  ConcatV2GpuFwdKernel() : axis_(0), input0_size_(0), input1_size_(0), output_size_(0), workspace_size_(0) {}
  ~ConcatV2GpuFwdKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    T *input_0 = GetDeviceAddress<T>(inputs, 0);
    T *input_1 = GetDeviceAddress<T>(inputs, 1);
    T *output = GetDeviceAddress<T>(outputs, 0);

    CalConcatV2(output_size_ / sizeof(T), w_[0], w_[1], input_0, input_1, output,
                reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    if (!CheckParam(kernel_node)) {
      return false;
    }

    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    input0_size_ = sizeof(T);
    for (size_t i = 0; i < input_shape.size(); i++) {
      input0_size_ *= input_shape[i];
    }
    auto input_shape1 = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    input1_size_ = sizeof(T);
    for (size_t i = 0; i < input_shape1.size(); i++) {
      input1_size_ *= input_shape1[i];
    }
    output_size_ = input0_size_ + input1_size_;
    axis_ = GetAttr<int>(kernel_node, "axis");
    if (axis_ < 0) {
      axis_ += SizeToInt(input_shape.size());
    }
    w_[0] = 1;
    w_[1] = 1;
    for (size_t i = IntToSize(axis_); i < input_shape.size(); i++) {
      w_[0] *= SizeToInt(input_shape[i]);
      w_[1] *= SizeToInt(input_shape1[i]);
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input0_size_);
    input_size_list_.push_back(input1_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but ConcatV2GpuFwdKernel needs 2 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but ConcatV2GpuFwdKernel needs 1 output.";
      return false;
    }
    return true;
  }
  int w_[2] = {1};
  int axis_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  size_t input0_size_;
  size_t input1_size_;
  size_t output_size_;
  size_t workspace_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CONCATV2_GPU_KERNEL_H
