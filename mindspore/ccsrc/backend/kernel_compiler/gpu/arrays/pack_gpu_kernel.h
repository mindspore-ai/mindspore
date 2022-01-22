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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_PACK_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_PACK_GPU_KERNEL_H

#include <vector>
#include <string>
#include <memory>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/pack.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class PackFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  PackFwdGpuKernelMod()
      : axis_(0),
        is_null_input_(false),
        input_num_(1),
        output_size_(0),
        dims_behind_axis_(1),
        inputs_host_(nullptr),
        kernel_name_("Pack") {}
  ~PackFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *output = GetDeviceAddress<T>(outputs, 0);
    T **inputs_array = GetDeviceAddress<T *>(workspace, 0);
    for (size_t i = 0; i < inputs.size(); i++) {
      inputs_host_[i] = GetDeviceAddress<T>(inputs, i);
    }
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(inputs_array,  // NOLINT
                                               inputs_host_.get(), sizeof(T *) * input_num_, cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "Pack opt cudaMemcpyAsync inputs failed");
    PackKernel(output_size_, input_num_, dims_behind_axis_, inputs_array, output,
               reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    (void)CheckParam(kernel_node);
    axis_ = static_cast<int32_t>(GetAttr<int64_t>(kernel_node, "axis"));
    if (axis_ < 0) {
      auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
      axis_ += (SizeToInt(input_shape.size()) + 1);
    }
    auto origin_data_format = AnfAlgo::GetOriginDataFormat(kernel_node);
    auto input_format = AnfAlgo::GetInputFormat(kernel_node, 0);
    axis_ = AxisTransform(origin_data_format, input_format, axis_);

    input_num_ = AnfAlgo::GetInputTensorNum(kernel_node);
    inputs_host_ = std::make_unique<T *[]>(input_num_);
    for (size_t i = 0; i < input_num_; i++) {
      size_t input_size = 1;
      auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i);
      is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
      if (is_null_input_) {
        InitSizeLists();
        return true;
      }
      for (size_t j = 0; j < input_shape.size(); j++) {
        input_size *= input_shape[j];
        if (i == 0 && j >= IntToSize(axis_)) {
          dims_behind_axis_ *= input_shape[j];
        }
      }
      input_size_list_.push_back(input_size * sizeof(T));
    }
    workspace_size_list_.push_back(sizeof(T *) * input_num_);

    auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    output_size_ = 1;
    for (size_t i = 0; i < output_shape.size(); i++) {
      output_size_ *= output_shape[i];
    }
    output_size_list_.push_back(output_size_ * sizeof(T));
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {}

 private:
  void CheckParam(const CNodePtr &kernel_node) {
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
  }
  int axis_;
  bool is_null_input_;
  size_t input_num_;
  size_t output_size_;
  size_t dims_behind_axis_;
  std::unique_ptr<T *[]> inputs_host_;
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_PACK_GPU_KERNEL_H
