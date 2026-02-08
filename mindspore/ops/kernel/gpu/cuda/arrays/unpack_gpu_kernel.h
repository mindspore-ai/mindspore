/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_UNPACK_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_UNPACK_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/unpack.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class UnpackFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  UnpackFwdGpuKernelMod()
      : axis_(0), is_null_input_(false), output_num_(0), input_size_(1), dims_after_axis_(1), outputs_host_(nullptr) {}
  ~UnpackFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    T **outputs_array = GetDeviceAddress<T *>(workspace, 0);
    for (size_t i = 0; i < outputs.size(); i++) {
      outputs_host_[i] = GetDeviceAddress<T>(outputs, i);
    }
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(outputs_array,  // NOLINT
                      outputs_host_.get(), sizeof(T *) * output_num_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "Unpack opt cudaMemcpyAsync outputs failed");
    auto status = UnpackKernel(input_size_, output_num_, dims_after_axis_, outputs_array, input,
                               reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    constexpr size_t input_num = 1;
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
      return ret;
    }
    axis_ = static_cast<int32_t>(GetValue<int64_t>(primitive_->GetAttr("axis")));
    auto input_shape = inputs[kIndex0]->GetDeviceShapeVector();
    if (axis_ < 0) {
      axis_ += SizeToInt(input_shape.size());
    }
    output_num_ = LongToSize(GetValue<int64_t>(primitive_->GetAttr("num")));
    outputs_host_ = std::make_unique<T *[]>(output_num_);
    ResetResource();

    for (size_t i = 0; i < output_num_; i++) {
      size_t _size = 1;
      auto _shape = outputs[i]->GetDeviceShapeVector();
      is_null_input_ = CHECK_SHAPE_NULL(_shape, kernel_name_, "output");
      if (is_null_input_) {
        return KRET_OK;
      }
      for (size_t j = 0; j < _shape.size(); j++) {
        _size *= static_cast<size_t>(_shape[j]);
      }
      output_size_list_.push_back(_size * sizeof(T));
    }
    workspace_size_list_.push_back(sizeof(T *) * output_num_);

    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      return KRET_OK;
    }
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= static_cast<size_t>(input_shape[i]);
      if (i > IntToSize(axis_)) {
        dims_after_axis_ *= static_cast<size_t>(input_shape[i]);
      }
    }

    return KRET_OK;
  }

 private:
  void ResetResource() noexcept {
    is_null_input_ = false;
    input_size_ = 1;
    dims_after_axis_ = 1;
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  int axis_{0};
  bool is_null_input_{false};
  size_t output_num_{0};
  size_t input_size_;
  size_t dims_after_axis_;
  std::unique_ptr<T *[]> outputs_host_;
  std::string origin_data_format_{};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_UNPACK_GPU_KERNEL_H_
