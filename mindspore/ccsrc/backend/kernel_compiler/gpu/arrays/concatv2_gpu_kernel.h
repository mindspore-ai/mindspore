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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_CONCATV2_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_CONCATV2_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/concatv2_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class ConcatV2GpuFwdKernel : public GpuKernel {
 public:
  ConcatV2GpuFwdKernel()
      : axis_(0),
        input_num_(1),
        output_size_(0),
        all_size_before_axis_(1),
        all_size_axis_(1),
        inputs_host_(nullptr),
        len_axis_(nullptr) {}
  ~ConcatV2GpuFwdKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *output = GetDeviceAddress<T>(outputs, 0);
    T **inputs_device = GetDeviceAddress<T *>(workspace, 0);
    int *len_axis_device = GetDeviceAddress<int>(workspace, 1);
    for (size_t i = 0; i < inputs.size(); i++) {
      inputs_host_[i] = GetDeviceAddress<T>(inputs, i);
    }
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(inputs_device, inputs_host_.get(), sizeof(T *) * input_num_,
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "ConcatV2 opt cudaMemcpyAsync inputs failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(len_axis_device, len_axis_.get(), sizeof(int) * input_num_,
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "ConcatV2 opt cudaMemcpyAsync length on axis failed");
    ConcatKernel(output_size_, input_num_, all_size_before_axis_, all_size_axis_, len_axis_device, inputs_device,
                 output, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    if (!CheckParam(kernel_node)) {
      return false;
    }
    axis_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "axis"));
    if (axis_ < 0) {
      auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
      axis_ += SizeToInt(input_shape.size());
    }
    auto origin_data_format = AnfAlgo::GetOriginDataFormat(kernel_node);
    auto input_format = AnfAlgo::GetInputFormat(kernel_node, 0);
    axis_ = AxisTransform(origin_data_format, input_format, axis_);

    input_num_ = SizeToInt(AnfAlgo::GetInputTensorNum(kernel_node));
    inputs_host_ = std::make_unique<T *[]>(input_num_);
    len_axis_ = std::make_unique<int[]>(input_num_);
    for (int i = 0; i < input_num_; i++) {
      size_t input_size = 1;
      auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i);
      for (size_t j = 0; j < input_shape.size(); j++) {
        input_size *= input_shape[j];
      }
      input_size_list_.push_back(input_size * sizeof(T));
      len_axis_[i] = SizeToInt(input_shape[axis_]);
    }
    workspace_size_list_.push_back(sizeof(T *) * input_num_);
    workspace_size_list_.push_back(sizeof(int) * input_num_);

    auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
    output_size_ = 1;
    for (int i = 0; i < SizeToInt(output_shape.size()); i++) {
      output_size_ *= output_shape[i];
      if (i > axis_) {
        all_size_before_axis_ *= output_shape[i];
        all_size_axis_ *= output_shape[i];
      }
      if (i == axis_) {
        all_size_before_axis_ *= output_shape[i];
      }
    }
    output_size_list_.push_back(output_size_ * sizeof(T));
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {}

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but ConcatV2GpuFwdKernel needs 1 output.";
      return false;
    }
    return true;
  }
  int axis_;
  int input_num_;
  size_t output_size_;
  int all_size_before_axis_;
  int all_size_axis_;
  std::unique_ptr<T *[]> inputs_host_;
  std::unique_ptr<int[]> len_axis_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_CONCATV2_GPU_KERNEL_H_
