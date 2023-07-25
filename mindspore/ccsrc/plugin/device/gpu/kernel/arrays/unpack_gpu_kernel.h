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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_UNPACK_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_UNPACK_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unpack.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class UnpackFwdGpuKernelMod : public NativeGpuKernelMod {
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

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    MS_EXCEPTION_IF_NULL(base_operator);
    kernel_name_ = base_operator->name();
    constexpr size_t input_num = 1;
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override {
    if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
      return ret;
    }

    auto prim = base_operator->GetPrim();
    MS_EXCEPTION_IF_NULL(prim);
    axis_ = static_cast<int32_t>(GetValue<int64_t>(prim->GetAttr("axis")));
    origin_data_format_ = GetValue<std::string>(prim->GetAttr("operator_origin_format"));
    auto input_shape = inputs[kIndex0]->GetDeviceShapeAdaptively();
    if (axis_ < 0) {
      axis_ += SizeToInt(input_shape.size());
    }

    auto input_format = FormatEnumToString(inputs[0]->GetFormat());
    axis_ = AxisTransform(origin_data_format_, input_format, axis_);
    output_num_ = LongToSize(GetValue<int64_t>(prim->GetAttr("num")));
    outputs_host_ = std::make_unique<T *[]>(output_num_);

    ResetResource();

    for (size_t i = 0; i < output_num_; i++) {
      size_t _size = 1;
      auto _shape = outputs[i]->GetDeviceShapeAdaptively();
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
    input_size_list_.push_back(input_size_ * sizeof(T));

    return KRET_OK;
  }

 private:
  void ResetResource() noexcept {
    is_null_input_ = false;
    input_size_ = 1;
    dims_after_axis_ = 1;
    input_size_list_.clear();
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
