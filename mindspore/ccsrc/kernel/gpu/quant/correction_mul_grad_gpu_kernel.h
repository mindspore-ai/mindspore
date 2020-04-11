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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CORRECTIONMULGRAD_GPUKERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CORRECTIONMULGRAD_GPUKERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/correction_mul_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class CorrectionMulGradGpuKernel : public GpuKernel {
 public:
  CorrectionMulGradGpuKernel() : batch_size_(0), channel_(0), height_(0), width_(0) {}
  ~CorrectionMulGradGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) {
    auto *d_out = GetDeviceAddress<T>(inputs, 0);
    auto *weight = GetDeviceAddress<T>(inputs, 1);
    auto *gamma = GetDeviceAddress<T>(inputs, 2);
    auto *running_std = GetDeviceAddress<T>(inputs, 3);
    auto *d_weight = GetDeviceAddress<T>(outputs, 0);
    auto *d_gamma = GetDeviceAddress<T>(outputs, 1);
    auto *tmp = GetDeviceAddress<T>(workspace, 0);

    CalCorrectionMul(d_out, gamma, running_std, batch_size_, channel_, height_, width_, d_weight,
                     reinterpret_cast<cudaStream_t>(stream_ptr));
    CalCorrectionMulGrad(d_out, weight, running_std, batch_size_, channel_, height_, width_, d_gamma, tmp,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) {
    InitResource();

    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 4) {
      MS_LOG(ERROR) << "Argument number is " << input_num << ", but CorrectionMulGradGpuKernel needs 4.";
      return false;
    }

    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);

    if (input_shape.size() != 4) {
      MS_LOG(ERROR) << "CorrectionMulGradGpuKernel input shape needs (N,C,H,W).";
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
  void InitSizeLists() {
    size_t input_size = batch_size_ * channel_ * height_ * width_ * sizeof(T);
    size_t weight_size = batch_size_ * sizeof(T);
    input_size_list_.push_back(input_size);      // d_out
    input_size_list_.push_back(input_size);      // weight
    input_size_list_.push_back(weight_size);     // gamma
    input_size_list_.push_back(weight_size);     // running_std
    output_size_list_.push_back(input_size);     // d_weight
    output_size_list_.push_back(weight_size);    // d_gamma
    workspace_size_list_.push_back(input_size);  // tmp d_out * weight
  }
  void InitResource() {}

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

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CORRECTIONMULGRAD_GPUKERNEL_H_
