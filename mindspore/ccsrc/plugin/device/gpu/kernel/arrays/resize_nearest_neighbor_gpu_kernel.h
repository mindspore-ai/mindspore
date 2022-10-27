/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_RESIZE_NEAREST_NEIGHBOR_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_RESIZE_NEAREST_NEIGHBOR_GPU_KERNEL_H_

#include <vector>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/resize_nearest_neighbor_impl.cuh"
#include "mindspore/core/ops/resize_nearest_neighbor.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInputNumTwo = 2;
constexpr size_t kSecondInputSize = 2;

template <typename T>
class ResizeNearestNeighborGpuKernelMod : public NativeGpuKernelMod {
 public:
  ResizeNearestNeighborGpuKernelMod() = default;
  ~ResizeNearestNeighborGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    auto output_size = outputs[kIndex0]->size;
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);
    int size = SizeToInt(output_size / sizeof(T));
    float h_scale = Scaling(input_shape_[2], output_shape_[2], align_corners_);
    float w_scale = Scaling(input_shape_[3], output_shape_[3], align_corners_);
    CalResizeNearestNeighbor(size, input, input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3], output,
                             output_shape_[0], output_shape_[1], output_shape_[2], output_shape_[3], align_corners_,
                             h_scale, w_scale, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    MS_EXCEPTION_IF_NULL(base_operator);
    kernel_name_ = base_operator->name();
    input_num_ = inputs.size();
    if (input_num_ != 1 && input_num_ != kInputNumTwo) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 1 or " << kInputNumTwo
                    << ", but got " << input_num_;
      return false;
    }
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), 1, kernel_name_);
    auto kernel_ptr = std::dynamic_pointer_cast<ops::ResizeNearestNeighbor>(base_operator);
    MS_EXCEPTION_IF_NULL(kernel_ptr);
    align_corners_ = kernel_ptr->get_align_corners();
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override {
    if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
      return ret;
    }
    auto input_shape = inputs[kIndex0]->GetShapeVector();
    auto output_shape = outputs[kIndex0]->GetShapeVector();
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      return KRET_OK;
    }
    input_shape_.clear();
    for (size_t i = 0; i < input_shape.size(); ++i) {
      input_shape_.push_back(LongToInt(input_shape[i]));
    }
    output_shape_.clear();
    for (size_t i = 0; i < output_shape.size(); ++i) {
      output_shape_.push_back(LongToInt(output_shape[i]));
    }
    if (input_num_ == kInputNumTwo) {
      input_size_list_.push_back(sizeof(int32_t) * kSecondInputSize);
    }
    return KRET_OK;
  }

 private:
  float Scaling(const int in_size, const int out_size, bool align_corners) {
    return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                           : in_size / static_cast<float>(out_size);
  }

  bool align_corners_{false};
  bool is_null_input_{false};
  std::vector<int> input_shape_{};
  std::vector<int> output_shape_{};
  size_t input_num_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_RESIZE_NEAREST_NEIGHBOR_GPU_KERNEL_H_
