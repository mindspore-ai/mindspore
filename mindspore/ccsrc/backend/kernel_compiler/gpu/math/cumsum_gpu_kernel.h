/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CUMSUM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CUMSUM_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/cumsum_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr int kMaxDimsSize = 3;
template <typename T>
class CumSumGpuKernel : public GpuKernel {
 public:
  CumSumGpuKernel()
      : exclusive_(false),
        reverse_(false),
        is_null_input_(false),
        axis_(0),
        input_size_0_(0),
        stride_(0),
        stride2_(0) {}
  ~CumSumGpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    T *ws_addr = GetDeviceAddress<T>(workspace, 0);
    CumSum(input_addr, output_addr, ws_addr, dims_[0], dims_[1], dims_[2], stride_, stride2_, exclusive_, reverse_,
           reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "Argument number is " << input_num << ", but CumSumGpuKernel needs 1.";
    }
    input_size_0_ = sizeof(T);
    shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(shape_);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'CumSumGpuKernel', input is null";
      InitSizeLists();
      return true;
    }
    axis_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "axis"));
    exclusive_ = GetAttr<bool>(kernel_node, "exclusive");
    reverse_ = GetAttr<bool>(kernel_node, "reverse");
    int input_dim_length = SizeToInt(shape_.size());
    if (axis_ >= input_dim_length) {
      MS_LOG(EXCEPTION) << "Axis is: " << axis_ << " out of bounds.";
    }
    while (axis_ < 0) {
      axis_ += input_dim_length;
    }
    for (size_t i = 0; i < shape_.size(); i++) {
      input_size_0_ *= shape_[i];
    }
    Reshape();
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_0_);
    output_size_list_.push_back(input_size_0_);
    workspace_size_list_.push_back(input_size_0_);
  }

 private:
  void Reshape() {
    dims_[0] = 1;
    dims_[1] = shape_[IntToSize(axis_)];
    dims_[2] = 1;
    for (size_t i = 0; i < IntToSize(axis_); i++) {
      dims_[0] *= shape_[i];
    }
    for (size_t i = IntToSize(axis_) + 1; i < shape_.size(); i++) {
      dims_[2] *= shape_[i];
    }
    stride_ = dims_[1] * dims_[2];
    stride2_ = dims_[2];
    return;
  }
  bool exclusive_;
  bool reverse_;
  bool is_null_input_;
  int axis_;
  size_t input_size_0_;
  size_t stride_;
  size_t stride2_;
  size_t dims_[kMaxDimsSize] = {};
  std::vector<size_t> shape_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CUMSUM_GPU_KERNEL_H_
