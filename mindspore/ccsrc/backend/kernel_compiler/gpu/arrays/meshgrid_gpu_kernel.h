/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MESHGRID_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MESHGRID_GPU_KERNEL_H

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backend/kernel_compiler/gpu/cuda_impl/broadcast_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/oneslike_impl.cuh"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/math/broadcast_gpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename T>
class MeshgridGpuKernel : public GpuKernel {
 public:
  MeshgridGpuKernel() { ResetResource(); }
  ~MeshgridGpuKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *ones_device = GetDeviceAddress<T>(workspace, 0);
    CalOnesLike(output_size_, static_cast<T *>(nullptr), ones_device, reinterpret_cast<cudaStream_t>(stream_ptr));

    std::vector<size_t> broadcasted_ones_shape(MAX_DIMS, 1);
    for (size_t i = 0; i < output_shape_.size(); i++) {
      broadcasted_ones_shape[i] = output_shape_[i];
    }

    for (size_t i = 0; i < outputs.size(); i++) {
      T *input_device = GetDeviceAddress<T>(inputs, i);
      T *output_device = GetDeviceAddress<T>(outputs, i);
      std::vector<size_t> broadcasted_input_shape(MAX_DIMS, 1);
      broadcasted_input_shape[i] = input_shapes_[i];

      if (swap_indexing_ && i < 2) {
        std::swap(broadcasted_input_shape[0], broadcasted_input_shape[1]);
      }

      BroadcastArith(broadcasted_input_shape, broadcasted_ones_shape, output_shape_, BROADCAST_TYPE_MUL, input_device,
                     ones_device, output_device, reinterpret_cast<cudaStream_t>(stream_ptr));
    }

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    std::string indexing = GetAttr<std::string>(kernel_node, "indexing");
    if (indexing == "xy") {
      swap_indexing_ = true;
    } else if (indexing == "ij") {
      swap_indexing_ = false;
    } else {
      MS_LOG(ERROR) << "invalid string for argument \"indexing\", must be \"xy\" or \"ij\" but got " << indexing;
      return false;
    }

    input_size_ = 1;
    input_count_ = AnfAlgo::GetInputTensorNum(kernel_node);
    for (size_t i = 0; i < input_count_; i++) {
      auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i);
      if (input_shape.size() < 1) {
        MS_LOG(ERROR) << "For 'MeshGridGpuKernel', the rank of input" << i << " cannot be less than 1.";
        return false;
      }
      size_t input_size = input_shape[0];
      input_shapes_.push_back(input_size);
      input_size_ *= input_size;
    }

    output_size_ = 1;
    output_count_ = AnfAlgo::GetOutputTensorNum(kernel_node);

    // inferred shape swaps output shape for us if needed
    output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(output_shape_);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'MeshGridGpuKernel', output is null.";
      InitSizeLists();
      return true;
    }

    if (output_count_ != input_count_) {
      MS_LOG(ERROR) << "output count is " << output_count_ << ", but MeshgridGpuKernel needs " << input_count_
                    << " output(s).";
      return false;
    }

    for (size_t i = 0; i < output_shape_.size(); i++) {
      output_size_ *= output_shape_[i];
    }

    // need to pad output shape with ones for broadcast kernel
    int need_broadcast_size = MAX_DIMS - output_shape_.size();
    for (int i = 0; i < need_broadcast_size; i++) {
      output_shape_.push_back(1);
    }

    InitSizeLists();

    return true;
  }

  void ResetResource() noexcept override {
    input_shapes_.clear();
    output_shape_.clear();
    input_size_ = 0;
    input_count_ = 0;
    output_size_ = 0;
    output_count_ = 0;
    swap_indexing_ = true;
    is_null_input_ = false;

    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    for (const size_t &input_shape : input_shapes_) {
      input_size_list_.push_back(input_shape * sizeof(T));
    }

    for (size_t i = 0; i < output_count_; i++) {
      output_size_list_.push_back(output_size_ * sizeof(T));
    }

    workspace_size_list_.push_back(output_size_ * sizeof(T));
  }

 private:
  std::vector<size_t> input_shapes_;
  std::vector<size_t> output_shape_;
  size_t input_size_;
  size_t input_count_;
  size_t output_size_;
  size_t output_count_;
  bool swap_indexing_;
  bool is_null_input_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MESHGRID_GPU_KERNEL_H
