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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPLIT_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPLIT_GPU_KERNEL_H

#include <vector>
#include <memory>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/split_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class SplitGpuFwdKernel : public GpuKernel {
 public:
  SplitGpuFwdKernel() { ResetResource(); }
  ~SplitGpuFwdKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    T **outputs_device = GetDeviceAddress<T *>(workspace, 0);
    for (size_t i = 0; i < outputs.size(); i++) {
      outputs_host_[i] = GetDeviceAddress<T>(outputs, i);
    }
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(outputs_device, outputs_host_.get(), sizeof(T *) * output_num_,
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "Split opt cudaMemcpyAsync outputs failed");
    SplitKernel(input_size_, axis_step_, all_size_before_axis_, all_size_axis_, input, outputs_device,
                reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    auto input_shape = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'SplitGpuKernel', input is null";
      InitSizeLists();
      return true;
    }
    int dims = SizeToInt(input_shape.size());
    axis_ = static_cast<int64_t>(GetAttr<int64_t>(kernel_node, "axis"));
    if (axis_ < -dims || axis_ >= dims) {
      MS_LOG(EXCEPTION) << "axis must be in the range [-rank, rank)";
    }
    if (axis_ < 0) {
      axis_ += dims;
    }

    auto origin_data_format = AnfAlgo::GetOriginDataFormat(kernel_node);
    auto input_format = AnfAlgo::GetInputFormat(kernel_node, 0);
    axis_ = AxisTransform(origin_data_format, input_format, axis_);

    output_num_ = static_cast<int64_t>(GetAttr<int64_t>(kernel_node, "output_num"));

    if (!CheckParam(kernel_node)) {
      return false;
    }
    input_size_ = 1;
    all_size_before_axis_ = 1;
    all_size_axis_ = 1;

    for (int i = 0; i < SizeToInt(input_shape.size()); i++) {
      input_size_ *= input_shape[i];
      if (i > axis_) {
        all_size_before_axis_ *= input_shape[i];
        all_size_axis_ *= input_shape[i];
      }
      if (i == axis_) {
        all_size_before_axis_ *= input_shape[i];
      }
    }
    input_size_list_.push_back(input_size_ * sizeof(T));
    axis_step_ = input_shape[axis_] / output_num_;

    for (int i = 0; i < output_num_; i++) {
      size_t output_size = 1;
      auto output_shape = AnfAlgo::GetOutputRealDeviceShapeIfExist(kernel_node, i);
      is_null_input_ = CHECK_NULL_INPUT(output_shape);
      if (is_null_input_) {
        MS_LOG(WARNING) << "SplitGpuKernel output is null";
        InitSizeLists();
        return true;
      }
      for (size_t j = 0; j < output_shape.size(); j++) {
        output_size *= output_shape[j];
      }
      output_size_list_.push_back(output_size * sizeof(T));
    }
    workspace_size_list_.push_back(sizeof(T *) * output_num_);
    InitSizeLists();
    outputs_host_ = std::make_unique<T *[]>(output_num_);
    return true;
  }

  void ResetResource() noexcept override {
    axis_ = 0;
    output_num_ = 1;
    input_size_ = 1;
    axis_step_ = 1;
    all_size_before_axis_ = 1;
    all_size_axis_ = 1;
    is_null_input_ = false;
    outputs_host_ = nullptr;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {}

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    auto input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    int dims = SizeToInt(input_shape.size());
    int output_num = SizeToInt(AnfAlgo::GetOutputTensorNum(kernel_node));
    if (output_num <= 0) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", must > 0.";
      return false;
    }
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but Split needs 1 input.";
      return false;
    }
    if (dims == 0) {
      MS_LOG(ERROR) << "Input dims is " << dims << ", scalar is not supported.";
      return false;
    }
    if (axis_ < -dims || axis_ >= dims) {
      MS_LOG(ERROR) << "Attr axis " << axis_ << " must be in " << -dims << "~" << dims;
      return false;
    }
    if (output_num_ > SizeToInt(input_shape[axis_])) {
      MS_LOG(ERROR) << "Attr output_num " << output_num_ << "must less than" << input_shape[axis_];
      return false;
    }
    if (output_num_ != output_num) {
      MS_LOG(ERROR) << "Output num is " << output_num << ", but need " << output_num_;
      return false;
    }
    return true;
  }
  int axis_;
  int output_num_;
  size_t input_size_;
  int axis_step_;
  int all_size_before_axis_;
  int all_size_axis_;
  bool is_null_input_;
  std::unique_ptr<T *[]> outputs_host_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPLIT_GPU_KERNEL_H
