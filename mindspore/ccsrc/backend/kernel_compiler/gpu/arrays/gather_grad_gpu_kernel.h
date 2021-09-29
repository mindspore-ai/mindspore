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

#ifndef MINDSPORE_GATHER_GRAD_GPU_KERNEL_H
#define MINDSPORE_GATHER_GRAD_GPU_KERNEL_H

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/gather_grad.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class GatherGradGpuKernel : public GpuKernel {
 public:
  GatherGradGpuKernel() : axis_(0), is_null_input_(false) {}
  ~GatherGradGpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);
    T *index_addr = GetDeviceAddress<T>(inputs, 0);
    S *grad_addr = GetDeviceAddress<S>(inputs, 1);
    S *output_addr = GetDeviceAddress<S>(outputs, 0);

    GatherGrad(index_addr, grad_addr, output_addr, dims_[0], dims_[1], dims_[2], dims_[3],
               reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "Argument number is " << input_num << ", but GatherGradGpuKernel needs 2.";
    }
    index_shapes_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    grad_shapes_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    output_shapes_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ =
      CHECK_NULL_INPUT(index_shapes_) || CHECK_NULL_INPUT(grad_shapes_) || CHECK_NULL_INPUT(output_shapes_);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'GatherGradGpuKernel', input or output is null";
      InitSizeLists();
      return true;
    }
    if (grad_shapes_.size() != index_shapes_.size() || grad_shapes_.size() != output_shapes_.size()) {
      MS_LOG(ERROR) << "The shape of grad, index and output should be same.";
      return false;
    }
    int dims = SizeToInt(grad_shapes_.size());
    axis_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "dim"));
    if (axis_ < -dims || axis_ >= dims) {
      MS_LOG(ERROR) << "axis must be in the range [-rank, rank)";
      return false;
    }
    if (axis_ < 0) {
      axis_ += dims;
    }

    Reshape();
    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {}
  void InitSizeLists() override {
    size_t size = GetSize(index_shapes_, true);
    input_size_list_.push_back(size);

    size = GetSize(grad_shapes_, false);
    input_size_list_.push_back(size);

    size = GetSize(output_shapes_, false);
    output_size_list_.push_back(size);
  }

 private:
  void Reshape() {
    size_t dim_before_axis = 1;
    for (size_t i = 0; i < IntToSize(axis_); i++) {
      dim_before_axis *= output_shapes_[i];
    }
    size_t dim_at_axis_index = index_shapes_[IntToSize(axis_)];
    size_t dim_at_axis_output = output_shapes_[IntToSize(axis_)];
    size_t dim_after_axis = 1;
    for (size_t i = IntToSize(axis_) + 1; i < output_shapes_.size(); i++) {
      dim_after_axis *= output_shapes_[i];
    }

    dims_[0] = dim_before_axis;
    dims_[1] = dim_at_axis_index;
    dims_[2] = dim_at_axis_output;
    dims_[3] = dim_after_axis;
    return;
  }
  size_t GetSize(const std::vector<size_t> &shape, const bool flag = true) const {
    size_t result = flag ? sizeof(T) : sizeof(S);
    for (size_t i = 0; i < shape.size(); i++) {
      result *= shape[i];
    }
    return result;
  }

  std::vector<size_t> index_shapes_;
  std::vector<size_t> grad_shapes_;
  std::vector<size_t> output_shapes_;

  size_t dims_[4] = {};
  int axis_;
  bool is_null_input_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_GATHER_GRAD_GPU_KERNEL_H
