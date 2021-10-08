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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_IN_TOP_K_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_IN_TOP_K_GPU_KERNEL_H_

#include <cstdint>
#include <limits>
#include <vector>

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/cast_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/in_top_k_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/topk_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class InTopKGpuKernel : public GpuKernel {
 public:
  InTopKGpuKernel() { ResetResource(); }
  ~InTopKGpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *predictions_device = GetDeviceAddress<T>(inputs, 0);
    int32_t *targets_device = GetDeviceAddress<int32_t>(inputs, 1);

    bool *output_device = GetDeviceAddress<bool>(outputs, 0);

    if (k_ <= 0) {
      CHECK_CUDA_RET_WITH_EXCEPT(
        kernel_node_, cudaMemsetAsync(output_device, false, outer_size_, reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemsetAsync failed.");

      return true;
    }

    if (k_ >= static_cast<int64_t>(inner_size_)) {
      CHECK_CUDA_RET_WITH_EXCEPT(
        kernel_node_, cudaMemsetAsync(output_device, true, outer_size_, reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemsetAsync failed.");

      return true;
    }

    T *top_k_output_device = GetDeviceAddress<T>(workspace, 0);
    int32_t *top_k_indices_device = GetDeviceAddress<int32_t>(workspace, 1);

    if (std::is_same<T, half>::value) {
      // remove later! urgent fix for bug: topk has incorrect output for float16
      float top_k_init = std::numeric_limits<float>::lowest();

      // cast to float32
      float *casted_float32_input = GetDeviceAddress<float>(workspace, 2);
      float *top_k_output_device_float32 = GetDeviceAddress<float>(workspace, 3);

      Cast(input_size_, predictions_device, casted_float32_input, reinterpret_cast<cudaStream_t>(stream_ptr));

      FastTopK(outer_size_, inner_size_, casted_float32_input, static_cast<int32_t>(k_), top_k_output_device_float32,
               top_k_indices_device, top_k_init, reinterpret_cast<cudaStream_t>(stream_ptr));

      CalInTopK(casted_float32_input, targets_device, output_device, top_k_output_device_float32, input_shape_[0],
                input_shape_[1], k_, reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      // topk sorts the input along the last dimension
      FastTopK(outer_size_, inner_size_, predictions_device, static_cast<int32_t>(k_), top_k_output_device,
               top_k_indices_device, top_k_init_, reinterpret_cast<cudaStream_t>(stream_ptr));

      CalInTopK(predictions_device, targets_device, output_device, top_k_output_device, input_shape_[0],
                input_shape_[1], k_, reinterpret_cast<cudaStream_t>(stream_ptr));
    }

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    size_t input_count = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_count != 2) {
      MS_LOG(ERROR) << input_count << " inputs were provided, but InTopKGpuKernel expects 2.";
      return false;
    }

    size_t output_count = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_count != 1) {
      MS_LOG(ERROR) << "Number of outputs is " << output_count << ", but should be 1 for InTopKGpuKernel.";
      return false;
    }

    input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    if (input_shape_.size() < 2) {
      MS_LOG(EXCEPTION) << "For 'InTopKGpuKernel', the rank of input cannot be less than 2, but got "
                        << input_shape_.size();
    }
    is_null_input_ = CHECK_NULL_INPUT(input_shape_);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'InTopKGpuKernel', input is null.";
      InitSizeLists();
      return true;
    }
    input_rank_ = input_shape_.size();
    input_size_ = 1;
    for (size_t i = 0; i < input_rank_; i++) {
      input_size_ *= input_shape_[i];
    }

    k_ = GetAttr<int64_t>(kernel_node, "k");

    inner_size_ = input_shape_[1];
    outer_size_ = input_shape_[0];

    if (std::is_same<T, half>::value) {
      // min value representable by float16, std::numeric_limits doesn't support half
      top_k_init_ = static_cast<half>(-65504.);
    } else {
      top_k_init_ = std::numeric_limits<T>::lowest();
    }

    InitSizeLists();

    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 0;
    k_ = 0;
    input_shape_.clear();
    input_rank_ = 0;
    outer_size_ = 0;
    inner_size_ = 0;
    is_null_input_ = false;
    top_k_init_ = static_cast<T>(0.);
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    input_size_list_.push_back(input_shape_[0] * sizeof(int32_t));
    output_size_list_.push_back(input_shape_[0] * sizeof(bool));
    if (k_ > 0) {
      workspace_size_list_.push_back(input_shape_[0] * k_ * sizeof(T));
      workspace_size_list_.push_back(input_shape_[0] * k_ * sizeof(int32_t));
    }

    // remove later! urgent fix for bug: topk has incorrect output for float16
    if (std::is_same<T, half>::value) {
      workspace_size_list_.push_back(input_size_ * sizeof(float));
      if (k_ > 0) {
        workspace_size_list_.push_back(input_shape_[0] * k_ * sizeof(float));
      }
    }
  }

 private:
  size_t input_size_;
  T top_k_init_;
  int64_t k_;
  std::vector<size_t> input_shape_;
  size_t input_rank_;

  // for topk
  size_t outer_size_;
  size_t inner_size_;
  bool is_null_input_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_IN_TOP_K_GPU_KERNEL_H_
