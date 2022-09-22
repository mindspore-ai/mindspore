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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/one_hot_impl.cuh"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
constexpr int DynamicInputNum = 4;
template <typename T, typename S, typename G = int>
class OneHotFwdGpuKernelMod : public DeprecatedNativeGpuKernelMod {
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
    size_t on_value_idx = 1;
    size_t off_value_idx = 2;
    if (is_dynamic_shape_) {
      on_value_idx++;
      off_value_idx++;
    }
    const S *indices = GetDeviceAddress<S>(inputs, 0);
    const T *on_value = GetDeviceAddress<T>(inputs, on_value_idx);
    const T *off_value = GetDeviceAddress<T>(inputs, off_value_idx);
    T *output = GetDeviceAddress<T>(outputs, 0);
    OneHot(indices, depth_, on_value, off_value, left_dim_size_, right_dim_size_, output, device_id_,
           reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    device_id_ = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    int64_t axis = GetAttr<int64_t>(kernel_node, "axis");
    auto input_shape_signed = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shape_signed = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    if (AnfAlgo::IsShapesDynamic({input_shape_signed, output_shape_signed})) {
      return true;
    }
    auto input_shape = Convert2SizeTClipNeg(input_shape_signed);
    auto output_shape = Convert2SizeTClipNeg(output_shape_signed);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num == DynamicInputNum) {
      is_dynamic_shape_ = true;
    }
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shape, kernel_name, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    int64_t input_dims = static_cast<int64_t>(input_shape.size());
    int64_t output_dims = static_cast<int64_t>(output_shape.size());
    if (axis > input_dims || axis >= output_dims) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the 'axis' must be less than the dimension of input and output"
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
  void ResetResource() noexcept override {
    is_dynamic_shape_ = false;
    input_size_ = 1;
    output_size_ = 1;
    depth_ = 0;
    left_dim_size_ = 1;
    right_dim_size_ = 1;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    is_dynamic_shape_ = false;
  }

 protected:
  void InitSizeLists() override {
    // inputs: indices, depth
    input_size_list_.push_back((input_size_ + 1) * sizeof(S));
    if (is_dynamic_shape_) {
      input_size_list_.push_back(sizeof(int64_t));
    }
    output_size_list_.push_back(output_size_ * sizeof(T));
  }

 private:
  size_t input_size_;
  size_t output_size_;

  bool is_dynamic_shape_ = false;
  size_t depth_;
  size_t left_dim_size_;
  size_t right_dim_size_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ONEHOT_GPU_KERNEL_H
