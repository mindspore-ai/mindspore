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

#ifndef MINDSPORE_GATHER_GRAD_GPU_KERNEL_H
#define MINDSPORE_GATHER_GRAD_GPU_KERNEL_H

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gather_grad.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class GatherGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  GatherGradGpuKernelMod() : axis_(0), is_null_input_(false) {}
  ~GatherGradGpuKernelMod() = default;

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
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    InitResource();
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 2, but got " << input_num;
    }
    index_shapes_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    grad_shapes_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    output_shapes_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(index_shapes_, kernel_name, "index") ||
                     CHECK_SHAPE_NULL(grad_shapes_, kernel_name, "grad") ||
                     CHECK_SHAPE_NULL(output_shapes_, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    if (grad_shapes_.size() != index_shapes_.size() || grad_shapes_.size() != output_shapes_.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name
                        << "', the dimension of grad, index and output should be the same, but got the dimension of "
                        << "grad: " << grad_shapes_.size() << ", the dimension of index: " << index_shapes_.size()
                        << ", the dimension of output: " << output_shapes_.size();
    }
    int dims = SizeToInt(grad_shapes_.size());
    axis_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "dim"));
    if (axis_ < -dims || axis_ >= dims) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' should be in the range [-" << dims << "," << dims
                        << "), but got " << axis_;
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
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_GATHER_GRAD_GPU_KERNEL_H
