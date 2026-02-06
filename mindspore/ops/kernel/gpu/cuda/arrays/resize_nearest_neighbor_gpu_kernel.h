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

#include <map>
#include <vector>
#include "kernel/gpu/cuda_impl/cuda_ops/resize_nearest_neighbor_impl.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
constexpr size_t kResizeNearestNeighborV2InputNum = 4;
constexpr size_t kSecondInputSize = 2;

template <typename T>
class ResizeNearestNeighborGpuKernelMod : public NativeGpuKernelMod {
 public:
  ResizeNearestNeighborGpuKernelMod() = default;
  ~ResizeNearestNeighborGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    MS_EXCEPTION_IF_NULL(input);
    T *output = GetDeviceAddress<T>(outputs, 0);
    MS_EXCEPTION_IF_NULL(output);
    auto output_size = outputs[kIndex0]->size();
    int size = SizeToInt(output_size / sizeof(T));
    float h_scale = Scaling(input_shape_[2], output_shape_[2], align_corners_);
    float w_scale = Scaling(input_shape_[3], output_shape_[3], align_corners_);
    auto status =
      CalResizeNearestNeighbor(size, input, input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3], output,
                               output_shape_[0], output_shape_[1], output_shape_[2], output_shape_[3], align_corners_,
                               h_scale, w_scale, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    input_num_ = inputs.size();
    if (input_num_ != kResizeNearestNeighborV2InputNum) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be " << kResizeNearestNeighborV2InputNum
                    << ", but got " << input_num_;
      return false;
    }
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), 1, kernel_name_);
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
      return ret;
    }
    auto input_shape = inputs.at(kIndex0)->GetShapeVector();
    auto output_shape = outputs.at(kIndex0)->GetShapeVector();
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
    align_corners_ = inputs.at(kIndex2)->GetValueWithCheck<bool>();
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
