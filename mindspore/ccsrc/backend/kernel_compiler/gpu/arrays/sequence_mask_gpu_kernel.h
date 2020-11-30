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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SEQUENCE_MASK_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SEQUENCE_MASK_GPU_KERNEL_H_

#include "backend/kernel_compiler/gpu/cuda_impl/sequence_mask_impl.cuh"

#include <cuda_runtime.h>

#include <vector>

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class SequenceMaskGpuKernel : public GpuKernel {
 public:
  SequenceMaskGpuKernel() { ResetResource(); }
  ~SequenceMaskGpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *lengths_device_address = GetDeviceAddress<T>(inputs, 0);
    T *maxlen_device_address = GetDeviceAddress<T>(inputs, 1);
    S *output_device_address = GetDeviceAddress<S>(outputs, 0);

    CalSequenceMask(lengths_device_address, maxlen_device_address, output_device_address, output_size_,
                    reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_count = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_count != 2) {
      MS_LOG(EXCEPTION) << input_count << " inputs were provided, but SequenceMaskGpuKernel expects 2.";
    }

    input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    for (const int &e : input_shape_) {
      lengths_size_ *= e;
    }

    std::vector<size_t> inferred_output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    for (const size_t &e : inferred_output_shape) {
      output_size_ *= e;
    }

    InitSizeLists();

    return true;
  }

  void ResetResource() noexcept override {
    output_size_ = 1;
    lengths_size_ = 1;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(lengths_size_ * sizeof(T));
    input_size_list_.push_back(sizeof(T));
    output_size_list_.push_back(output_size_);
  }

 private:
  std::vector<size_t> input_shape_;
  size_t lengths_size_;
  size_t output_size_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SEQUENCE_MASK_GPU_KERNEL_H_
