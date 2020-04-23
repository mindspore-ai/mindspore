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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CORRECTIONMUL_GPUKERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CORRECTIONMUL_GPUKERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/correction_mul_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class CorrectionMulGpuKernel : public GpuKernel {
 public:
  CorrectionMulGpuKernel() : batch_size_(0), channel_(0), height_(0), width_(0) {}
  ~CorrectionMulGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    auto *weight = GetDeviceAddress<T>(inputs, 0);
    auto *gamma = GetDeviceAddress<T>(inputs, 1);
    auto *running_std = GetDeviceAddress<T>(inputs, 2);
    auto *output = GetDeviceAddress<T>(outputs, 0);

    CalCorrectionMul(weight, gamma, running_std, batch_size_, channel_, height_, width_, output,
                     reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();

    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 3) {
      MS_LOG(ERROR) << "Argument number is " << input_num << ", but CorrectionMulGpuKernel needs 3.";
      return false;
    }

    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (input_shape.size() != 4) {
      MS_LOG(ERROR) << "CorrectionMulGpuKernel input shape needs (N,C,H,W).";
      return false;
    }
    batch_size_ = input_shape[0];
    channel_ = input_shape[1];
    height_ = input_shape[2];
    width_ = input_shape[3];

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    size_t input_size = batch_size_ * channel_ * height_ * width_ * sizeof(T);
    size_t weight_size = batch_size_ * sizeof(T);
    input_size_list_.push_back(input_size);   // weight
    input_size_list_.push_back(weight_size);  // gamma
    input_size_list_.push_back(weight_size);  // running_std
    size_t workspace_size = 0;
    output_size_list_.push_back(input_size);
    workspace_size_list_.push_back(workspace_size);
  }
  void InitResource() override {}

 private:
  void DestroyResource() noexcept {}

  size_t batch_size_;
  size_t channel_;
  size_t height_;
  size_t width_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CORRECTIONMUL_GPUKERNEL_H_
