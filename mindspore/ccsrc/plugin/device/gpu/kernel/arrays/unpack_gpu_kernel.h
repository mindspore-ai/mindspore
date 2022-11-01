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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNPACK_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNPACK_GPU_KERNEL_H

#include <vector>
#include <string>
#include <memory>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unpack.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class UnpackFwdGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  UnpackFwdGpuKernelMod()
      : axis_(0), is_null_input_(false), output_num_(0), input_size_(1), dims_after_axis_(1), outputs_host_(nullptr) {}
  ~UnpackFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    T **outputs_array = GetDeviceAddress<T *>(workspace, 0);
    for (size_t i = 0; i < outputs.size(); i++) {
      outputs_host_[i] = GetDeviceAddress<T>(outputs, i);
    }
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(outputs_array,  // NOLINT
                                               outputs_host_.get(), sizeof(T *) * output_num_, cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "Unpack opt cudaMemcpyAsync outputs failed");
    UnpackKernel(input_size_, output_num_, dims_after_axis_, outputs_array, input,
                 reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    (void)CheckParam(kernel_node);
    axis_ = static_cast<int32_t>(GetAttr<int64_t>(kernel_node, "axis"));
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    int32_t shape_size = SizeToInt(input_shape.size());
    if (axis_ < -shape_size || axis_ >= shape_size) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the `axis` should be in [" << -shape_size << ", "
                        << shape_size << "), but got " << axis_;
    }
    if (axis_ < 0) {
      axis_ += SizeToInt(input_shape.size());
    }
    auto origin_data_format = AnfAlgo::GetOriginDataFormat(kernel_node);
    auto input_format = AnfAlgo::GetInputFormat(kernel_node, 0);
    axis_ = AxisTransform(origin_data_format, input_format, axis_);

    output_num_ = LongToSize(GetAttr<int64_t>(kernel_node, "num"));
    outputs_host_ = std::make_unique<T *[]>(output_num_);
    for (size_t i = 0; i < output_num_; i++) {
      size_t _size = 1;
      auto _shape = AnfAlgo::GetOutputDeviceShape(kernel_node, i);
      is_null_input_ = CHECK_SHAPE_NULL(_shape, kernel_name_, "output");
      if (is_null_input_) {
        InitSizeLists();
        return true;
      }
      for (size_t j = 0; j < _shape.size(); j++) {
        _size *= static_cast<size_t>(_shape[j]);
      }
      output_size_list_.push_back(_size * sizeof(T));
    }
    workspace_size_list_.push_back(sizeof(T *) * output_num_);

    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= static_cast<size_t>(input_shape[i]);
      if (i > IntToSize(axis_)) {
        dims_after_axis_ *= static_cast<size_t>(input_shape[i]);
      }
    }
    input_size_list_.push_back(input_size_ * sizeof(T));
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {}

 private:
  void ResetResource() noexcept override {
    axis_ = 0;
    is_null_input_ = false;
    output_num_ = 0;
    input_size_ = 1;
    dims_after_axis_ = 1;
    outputs_host_ = nullptr;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }
  void CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1, but got " << input_num;
    }
  }
  int axis_;
  bool is_null_input_;
  size_t output_num_;
  size_t input_size_;
  size_t dims_after_axis_;
  std::unique_ptr<T *[]> outputs_host_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNPACK_GPU_KERNEL_H
