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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ONEHOT_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ONEHOT_GPU_KERNEL_H

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/one_hot_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class OneHotFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  OneHotFwdGpuKernelMod()
      : input_size_(1), output_size_(1), depth_(0), left_dim_size_(1), right_dim_size_(1), is_null_input_(false) {}
  ~OneHotFwdGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);
    const S *indices = GetDeviceAddress<S>(inputs, 0);
    const T *on_value = GetDeviceAddress<T>(inputs, 1);
    const T *off_value = GetDeviceAddress<T>(inputs, 2);
    T *output = GetDeviceAddress<T>(outputs, 0);
    OneHot(indices, depth_, on_value, off_value, left_dim_size_, right_dim_size_, output,
           reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    int64_t axis = GetAttr<int64_t>(kernel_node, "axis");
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shape, kernel_name, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    int64_t input_dims = static_cast<int64_t>(input_shape.size());
    int64_t output_dims = static_cast<int64_t>(output_shape.size());
    if (axis >= input_dims || axis >= output_dims) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the 'axis' should be less than the dimension of input and output"
                        << ", but got 'axis': " << axis << ", the dimension of input: " << input_dims
                        << ", the dimension of output: " << output_dims;
    }
    const int64_t default_axis = -1;

    // Compress arbitrary tensor dimensions into three dimensions (left_dims, depth, right_dims).
    for (size_t i = 0; i < input_shape.size(); i++) {
      auto dim_size = input_shape[i];
      if (axis == default_axis || i < IntToSize(axis)) {
        left_dim_size_ *= dim_size;
      }
      if (axis != default_axis && i >= IntToSize(axis)) {
        right_dim_size_ *= dim_size;
      }
    }
    for (auto size : input_shape) {
      input_size_ *= size;
    }
    for (auto size : output_shape) {
      output_size_ *= size;
    }
    if (axis == default_axis) {
      depth_ = output_shape[output_shape.size() - 1];
    } else {
      depth_ = output_shape[IntToSize(axis)];
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    // inputs: indices, depth
    input_size_list_.push_back((input_size_ + 1) * sizeof(S));
    output_size_list_.push_back(output_size_ * sizeof(T));
  }

 private:
  size_t input_size_;
  size_t output_size_;

  size_t depth_;
  size_t left_dim_size_;
  size_t right_dim_size_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ONEHOT_GPU_KERNEL_H
