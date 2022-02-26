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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARGMAX_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARGMAX_GPU_KERNEL_H_

#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/argmax_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T, typename S>
class ArgmaxGpuKernelMod : public NativeGpuKernelMod {
 public:
  ArgmaxGpuKernelMod()
      : input_size_(0),
        output_size_(0),
        workspace_size_(0),
        bound_(0),
        outer_size_(0),
        inner_size_(0),
        is_null_input_(false),
        kernel_name_("Argmax") {}
  ~ArgmaxGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    S *output = GetDeviceAddress<S>(outputs, 0);
    MS_EXCEPTION_IF_NULL(input);
    MS_EXCEPTION_IF_NULL(output);
    CalArgmax(input, bound_, outer_size_, inner_size_, output, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ =
      CHECK_SHAPE_NULL(shape, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    int64_t dims = shape.size();
    int64_t axis = GetAttr<int64_t>(kernel_node, "axis");
    if (axis < -dims || axis >= dims) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' should be in the range [-" << dims << "," << dims
                        << "), but got " << axis;
    }

    if (axis < 0) {
      axis += dims;
    }
    input_size_ = sizeof(T);
    for (auto x : shape) {
      input_size_ *= x;
    }
    output_size_ = sizeof(S);
    for (auto x : output_shape) {
      output_size_ *= x;
    }
    bound_ = static_cast<S>(shape[axis]);
    if (shape[axis] != static_cast<size_t>(bound_)) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of shape[axis] should be "
                        << static_cast<size_t>(bound_) << ", but got " << shape[axis];
    }
    outer_size_ = 1;
    for (int64_t i = axis - 1; i >= 0; i--) {
      outer_size_ *= shape[i];
    }
    inner_size_ = 1;
    for (int64_t i = axis + 1; i < dims; i++) {
      inner_size_ *= shape[i];
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  S bound_;
  size_t outer_size_;
  size_t inner_size_;
  bool is_null_input_;
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARGMAX_GPU_KERNEL_H_
