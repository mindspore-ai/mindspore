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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BROADCAST_TO_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BROADCAST_TO_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/broadcast_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t SHAPE_SIZE = 4;
template <typename T>
class BroadcastToGpuKernel : public GpuKernel {
 public:
  BroadcastToGpuKernel() {}
  ~BroadcastToGpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    BroadcastTo(input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3], output_shape_[0], output_shape_[1],
                output_shape_[2], output_shape_[3], input_addr, output_addr,
                reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    auto input_shapes = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shapes = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shapes) || CHECK_NULL_INPUT(output_shapes);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'BroadcastToGpuKernel', input or output is null";
      InitSizeLists();
      return true;
    }
    if (input_shapes.size() > SHAPE_SIZE || output_shapes.size() > SHAPE_SIZE) {
      MS_LOG(EXCEPTION) << "BroadcastTo operation not support dim greater than " << SHAPE_SIZE;
    }

    if (output_shapes.size() < input_shapes.size()) {
      MS_LOG(EXCEPTION) << "The rank of BroadcastTo's output cannot be smaller than the rank of the input.";
    }

    size_t offset = output_shapes.size() - input_shapes.size();
    for (size_t i = 0; i < input_shapes.size(); i++) {
      input_shape_[i + offset] = input_shapes[i];
    }

    for (size_t j = 0; j < output_shapes.size(); j++) {
      output_shape_[j] = output_shapes[j];
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_shape_[0] * input_shape_[1] * input_shape_[2] * input_shape_[3] * sizeof(T));
    output_size_list_.push_back(output_shape_[0] * output_shape_[1] * output_shape_[2] * output_shape_[3] * sizeof(T));
  }

 private:
  size_t input_shape_[4] = {1, 1, 1, 1};
  size_t output_shape_[4] = {1, 1, 1, 1};
  bool is_null_input_ = false;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BROADCAST_TO_GPU_KERNEL_H_
