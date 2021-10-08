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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SORT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SORT_GPU_KERNEL_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/topk_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/transpose_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/unary_op_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class SortGpuKernel : public GpuKernel {
 public:
  SortGpuKernel() { ResetResource(); }
  ~SortGpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_device = GetDeviceAddress<T>(inputs, 0);

    T *output_device = GetDeviceAddress<T>(outputs, 0);
    int32_t *indices_device = GetDeviceAddress<int32_t>(outputs, 1);

    T *temp_output_device = GetDeviceAddress<T>(workspace, 0);
    int32_t *temp_indices_device = GetDeviceAddress<int32_t>(workspace, 1);
    size_t *input_shape_device = GetDeviceAddress<size_t>(workspace, 2);
    size_t *perm_device = GetDeviceAddress<size_t>(workspace, 3);
    size_t *transposed_shape_device = GetDeviceAddress<size_t>(workspace, 4);

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(input_shape_device, &input_shape_[0], workspace_size_list_[2],
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync for input_shape_ failed");

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(perm_device, &perm_[0], workspace_size_list_[3], cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync for perm_ failed");

    // Sort is implemented using a combination of Neg, Transpose, and TopK. It's
    // Not safe to treat Transpose and TopK as inplace operators, so we alternate
    // between using temp_output_device and output_device for intermediate calculations,
    // this way only a constant number of allocations is needed instead of needing to
    // allocate once for each intermediate calculation.
    T *intermediate_input_device = input_device;
    T *intermediate_output_device = output_device;

    // if sort not in descending order, negate input and negate back after sorting
    if (!descending_) {
      Negative(intermediate_input_device, intermediate_output_device, input_size_,
               reinterpret_cast<cudaStream_t>(stream_ptr));
      intermediate_input_device = output_device;
      intermediate_output_device = temp_output_device;
    }

    // transpose so that desired dimension to sort along becomes the last one
    CalTranspose(input_size_, intermediate_input_device, input_shape_device, perm_device, input_rank_,
                 intermediate_output_device, reinterpret_cast<cudaStream_t>(stream_ptr));
    intermediate_input_device = intermediate_output_device;
    intermediate_output_device = intermediate_input_device == output_device ? temp_output_device : output_device;

    // topk sorts the input along the last dimension
    FastTopK(outer_size_, inner_size_, intermediate_input_device, static_cast<int32_t>(input_shape_[axis_]),
             intermediate_output_device, temp_indices_device, topk_init_, reinterpret_cast<cudaStream_t>(stream_ptr));
    std::swap(intermediate_input_device, intermediate_output_device);

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(transposed_shape_device, &transposed_shape_[0], workspace_size_list_[4],
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync for transposed_shape_ failed");

    // transpose the sorted output back to the original input shape
    CalTranspose(input_size_, intermediate_input_device, transposed_shape_device, perm_device, input_rank_,
                 intermediate_output_device, reinterpret_cast<cudaStream_t>(stream_ptr));

    // transpose the indices back to the original input shape
    CalTranspose(input_size_, temp_indices_device, transposed_shape_device, perm_device, input_rank_, indices_device,
                 reinterpret_cast<cudaStream_t>(stream_ptr));

    // negate back the sorted values if we negated prior to sorting
    if (!descending_) {
      std::swap(intermediate_input_device, intermediate_output_device);
      Negative(intermediate_input_device, intermediate_output_device, input_size_,
               reinterpret_cast<cudaStream_t>(stream_ptr));
    }

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_count = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_count != 1) {
      MS_LOG(ERROR) << input_count << " inputs were provided, but SortGpuKernel expects 1.";
      return false;
    }

    size_t output_count = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_count != 2) {
      MS_LOG(ERROR) << "Number of outputs is " << output_count << ", but should be 2 for SortGpuKernel.";
      return false;
    }

    input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape_);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'SortGpuKernel', input is null.";
      InitSizeLists();
      return true;
    }

    input_rank_ = input_shape_.size();
    if (input_rank_ > TRANSPOSE_MAX_DIMENSION || input_rank_ < 1) {
      MS_LOG(ERROR) << "For 'SortGpuKernel', the rank of input cannot be more than " << TRANSPOSE_MAX_DIMENSION
                    << " dimensions or less than 1 dimension.";
      return false;
    }

    input_size_ = 1;
    for (size_t i = 0; i < input_rank_; i++) {
      input_size_ *= input_shape_[i];
    }

    descending_ = GetAttr<bool>(kernel_node, "descending");
    axis_ = GetAttr<int64_t>(kernel_node, "axis");

    if (axis_ < 0) {
      axis_ += input_rank_;
    }
    if ((size_t)axis_ >= input_rank_) {
      MS_LOG(ERROR) << "For 'SortGpuKernel', axis should be less than the rank of input, bot got axis: " << axis_
                    << " the rank of input: " << input_rank_;
      return false;
    }

    perm_.resize(input_rank_);
    std::iota(perm_.begin(), perm_.end(), 0);
    std::swap(perm_[input_rank_ - 1], perm_[axis_]);

    transposed_shape_ = input_shape_;
    std::swap(transposed_shape_[input_rank_ - 1], transposed_shape_[axis_]);

    inner_size_ = input_shape_[axis_];
    outer_size_ = input_size_ / inner_size_;

    if (std::is_same<T, half>::value) {
      // min value representable by float16, std::numeric_limits doesn't support half
      topk_init_ = static_cast<half>(-65504.);
    } else {
      topk_init_ = std::numeric_limits<T>::lowest();
    }

    InitSizeLists();

    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 0;
    axis_ = 0;
    descending_ = false;
    is_null_input_ = false;
    input_shape_.clear();
    input_rank_ = 0;
    transposed_shape_.clear();
    perm_.clear();
    outer_size_ = 0;
    inner_size_ = 0;
    topk_init_ = static_cast<T>(0.);
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    size_t input_bytes = input_size_ * sizeof(T);
    size_t indices_bytes = input_size_ * sizeof(int32_t);
    input_size_list_.push_back(input_bytes);

    // outputs: sorted values, indices
    output_size_list_.push_back(input_bytes);
    output_size_list_.push_back(indices_bytes);

    // workspace: temp output, temp indices, input shape, perm, transposed_shape
    workspace_size_list_.push_back(input_bytes);
    workspace_size_list_.push_back(indices_bytes);
    workspace_size_list_.push_back(input_rank_ * sizeof(size_t));
    workspace_size_list_.push_back(input_rank_ * sizeof(size_t));
    workspace_size_list_.push_back(input_rank_ * sizeof(size_t));
  }

 private:
  size_t input_size_;
  int64_t axis_;
  bool descending_;
  bool is_null_input_;
  std::vector<size_t> input_shape_;
  size_t input_rank_;

  // for transpose
  std::vector<size_t> transposed_shape_;
  std::vector<size_t> perm_;

  // for topk
  size_t outer_size_;
  size_t inner_size_;
  T topk_init_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SORT_GPU_KERNEL_H_
