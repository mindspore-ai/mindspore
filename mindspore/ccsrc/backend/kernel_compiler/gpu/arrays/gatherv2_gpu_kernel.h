/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_GATHER_GPU_KERNEL_H
#define MINDSPORE_GATHER_GPU_KERNEL_H

#include <vector>
#include <algorithm>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/gatherv2.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class GatherV2GpuFwdKernel : public GpuKernel {
 public:
  GatherV2GpuFwdKernel() { ResetResource(); }
  ~GatherV2GpuFwdKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    S *indices_addr = GetDeviceAddress<S>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    if (is_dynamic_shape_) {
      // if we are in dynamic shape mode, we don't know dims_, so we need to store the input_shape_ and indices_shape_,
      // and axis_ in the workspace to calculate dims_
      size_t *input_shape_device_address = GetDeviceAddress<size_t>(workspace, 0);
      size_t *indices_shape_device_address = GetDeviceAddress<size_t>(workspace, 1);
      int64_t *axis_device_address = GetDeviceAddress<int64_t>(workspace, 2);

      CHECK_CUDA_RET_WITH_EXCEPT(
        cudaMemcpyAsync(input_shape_device_address, input_shapes_.data(), workspace_size_list_[0],
                        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemcpyAsync input_shape failed");
      CHECK_CUDA_RET_WITH_EXCEPT(
        cudaMemcpyAsync(indices_shape_device_address, indices_shapes_.data(), workspace_size_list_[1],
                        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemcpyAsync indices_shape failed");
      CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(axis_device_address, &axis_, workspace_size_list_[2],
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync axis_ failed");

      // output shape will be here for us to copy back to host
      size_t *output_shape_device_address = GetDeviceAddress<size_t>(workspace, 3);
      CalGatherV2DynamicShape(input_addr, indices_addr, output_addr, input_shape_device_address, input_shapes_.size(),
                              indices_shape_device_address, indices_shapes_.size(), axis_device_address,
                              output_shape_device_address, max_output_size_,
                              reinterpret_cast<cudaStream_t>(stream_ptr));

      size_t output_rank = input_shapes_.size() - 1 + indices_shapes_.size();
      real_output_shape_.resize(output_rank);
      CHECK_CUDA_RET_WITH_ERROR(
        cudaMemcpyAsync(&real_output_shape_[0], output_shape_device_address, output_rank * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_ptr)),
        "Failed to copy gpu memory.");

    } else {
      auto input_dim1 = input_shapes_[IntToSize(axis_)];
      CalGatherV2StaticShape(input_addr, indices_addr, output_addr, dims_[0], dims_[1], dims_[2], input_dim1,
                             reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num == 3) {
      is_dynamic_shape_ = true;
    } else if (input_num != 2) {
      MS_LOG(EXCEPTION) << "Argument number is " << input_num << ", but GatherGpuV2FwdKernel needs 2.";
    }

    input_shapes_ = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    indices_shapes_ = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 1);
    output_shapes_ = AnfAlgo::GetOutputRealDeviceShapeIfExist(kernel_node, 0);

    if (is_dynamic_shape_) {
      c_node_ptr_ = kernel_node;
      size_t input_shape_min = *std::min_element(input_shapes_.begin(), input_shapes_.end());
      max_output_size_ = (GetSize(input_shapes_) / input_shape_min) * GetSize(indices_shapes_);
    } else {
      axis_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "axis"));
      if (axis_ < 0) {
        axis_ = axis_ + SizeToInt(input_shapes_.size());
      }

      Reshape();
    }

    InitSizeLists();
    return true;
  }
  void ResetResource() noexcept override {
    is_dynamic_shape_ = false;
    max_output_size_ = -1;
    input_shapes_.clear();
    indices_shapes_.clear();
    output_shapes_.clear();
    std::fill(dims_, dims_ + 3, 0);
    axis_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    size_t size = GetSize(input_shapes_);
    input_size_list_.push_back(size);

    size = GetSize(indices_shapes_);
    input_size_list_.push_back(size);

    if (is_dynamic_shape_) {
      // add by chenweifeng
      input_size_list_.push_back(sizeof(S));

      // allocate maximum size needed
      output_size_list_.push_back(max_output_size_);

      // allocate workspace memory for input, indices, axis, and output shape respectively
      size = GetSize(input_shapes_);
      workspace_size_list_.push_back(size);

      size = GetSize(indices_shapes_);
      workspace_size_list_.push_back(size);

      size = sizeof(int32_t);
      workspace_size_list_.push_back(size);

      size = GetSize(input_shapes_);
      workspace_size_list_.push_back(size);
    } else {
      size = GetSize(output_shapes_);
      output_size_list_.push_back(size);
    }
  }

 private:
  void Reshape() {
    size_t dim_before_axis = 1;
    for (size_t i = 0; i < IntToSize(axis_); i++) {
      dim_before_axis *= output_shapes_[i];
    }

    size_t dim_of_indices = 1;
    for (size_t i = 0; i < indices_shapes_.size(); i++) {
      dim_of_indices *= indices_shapes_[i];
    }

    size_t dim_after_indices = 1;
    for (size_t i = IntToSize(axis_) + indices_shapes_.size(); i < output_shapes_.size(); i++) {
      dim_after_indices *= output_shapes_[i];
    }

    dims_[0] = dim_before_axis;
    dims_[1] = dim_of_indices;
    dims_[2] = dim_after_indices;
    return;
  }
  size_t GetSize(const std::vector<size_t> &shape) const {
    if (shape.size() == 0) {
      return 0;
    }
    size_t result = sizeof(T);
    for (size_t i = 0; i < shape.size(); i++) {
      result *= shape[i];
    }
    return result;
  }

  std::vector<size_t> input_shapes_;
  std::vector<size_t> indices_shapes_;
  std::vector<size_t> output_shapes_;

  size_t dims_[3] = {};
  int64_t axis_;
  bool is_dynamic_shape_;
  int max_output_size_;
  std::vector<size_t> real_output_shape_;
  CNodePtr c_node_ptr_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_GATHER_GPU_KERNEL_H
