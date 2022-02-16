/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <string>
#include <memory>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/split_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class SplitFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  SplitFwdGpuKernelMod() { ResetResource(); }
  ~SplitFwdGpuKernelMod() override = default;

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
    kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    auto input_shape = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    int dims = SizeToInt(input_shape.size());
    axis_ = static_cast<int64_t>(GetAttr<int64_t>(kernel_node, "axis"));
    if (axis_ < -dims || axis_ >= dims) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' should be in the range [-" << dims << "," << dims
                        << "), but got " << axis_;
    }
    if (axis_ < 0) {
      axis_ += dims;
    }

    auto origin_data_format = AnfAlgo::GetOriginDataFormat(kernel_node);
    auto input_format = AnfAlgo::GetInputFormat(kernel_node, 0);
    axis_ = AxisTransform(origin_data_format, input_format, axis_);

    output_num_ = static_cast<int64_t>(GetAttr<int64_t>(kernel_node, "output_num"));

    (void)CheckParam(kernel_node);
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
      is_null_input_ = CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
      if (is_null_input_) {
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
    kernel_name_ = "Split";
    outputs_host_ = nullptr;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {}

 private:
  void CheckParam(const CNodePtr &kernel_node) {
    auto input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    int dims = SizeToInt(input_shape.size());
    int output_num = SizeToInt(AnfAlgo::GetOutputTensorNum(kernel_node));
    if (output_num <= 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be greater than 0, but got "
                        << output_num;
    }
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << input_num;
    }
    if (dims == 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be 0, but got " << dims;
    }
    if (axis_ < -dims || axis_ >= dims) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' should be in the range [-" << dims << "," << dims
                        << "), but got " << axis_;
    }
    if (output_num_ > SizeToInt(input_shape[axis_])) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs cannot be greater than "
                        << SizeToInt(input_shape[axis_]) << ", but got " << output_num_;
    }
    if (output_num_ != output_num) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be " << output_num_
                        << ", but got " << output_num;
    }
  }
  int axis_;
  int output_num_;
  size_t input_size_;
  int axis_step_;
  int all_size_before_axis_;
  int all_size_axis_;
  bool is_null_input_;
  std::unique_ptr<T *[]> outputs_host_;

  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPLIT_GPU_KERNEL_H
