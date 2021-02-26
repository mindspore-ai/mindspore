/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BINARY_CROSS_ENTROPY_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BINARY_CROSS_ENTROPY_KERNEL_H

#include <vector>
#include <string>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/loss_with_reduction_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class BinaryCrossEntropyGpuKernel : public GpuKernel {
 public:
  BinaryCrossEntropyGpuKernel() : input_size_(1), reduction_(1) {}
  ~BinaryCrossEntropyGpuKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input_x = GetDeviceAddress<T>(inputs, 0);
    T *input_y = GetDeviceAddress<T>(inputs, 1);
    T *weight = GetDeviceAddress<T>(inputs, 2);
    T *loss = GetDeviceAddress<T>(outputs, 0);
    T *tmp_loss = GetDeviceAddress<T>(workspace, 0);
    if (input_size_ > 0) {
      BinaryCrossEntropyLoss(input_size_, reduction_, input_x, input_y, weight, loss, tmp_loss,
                             reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }
    string reduction = GetAttr<string>(kernel_node, "reduction");
    if (reduction == "none") {
      reduction_ = 0;
    } else if (reduction == "sum") {
      reduction_ = 2;
    }
    workspace_size_ = sizeof(T);
    if (reduction_ != 0) {
      workspace_size_ *= input_size_;
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    input_size_list_.push_back(input_size_ * sizeof(T));
    input_size_list_.push_back(input_size_ * sizeof(T));
    if (reduction_ == 0) {
      output_size_list_.push_back(input_size_ * sizeof(T));
    } else {
      output_size_list_.push_back(sizeof(T));
    }
    workspace_size_list_.push_back(workspace_size_);
  }

 private:
  size_t input_size_;
  int reduction_;
  size_t workspace_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BINARY_CROSS_ENTROPY_H
