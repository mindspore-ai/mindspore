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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_REVERSE_SEQUENCE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_REVERSE_SEQUENCE_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <iostream>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/cuda_common.h"
#include "backend/kernel_compiler/gpu/cuda_impl/reverse_sequence_impl.cuh"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class ReverseSequenceGpuFwdKernel : public GpuKernel {
 public:
  ReverseSequenceGpuFwdKernel()
      : shape_size_(0),
        input_size_(0),
        batch_dim_(0),
        seq_dim_(0),
        is_null_input_(false),
        seq_len_size_(0),
        total_index_dim_(0),
        output_size_(0),
        workspace_size_(0) {}
  ~ReverseSequenceGpuFwdKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    S *seq_len = GetDeviceAddress<S>(inputs, 1);
    size_t *input_shape_ptr = GetDeviceAddress<size_t>(workspace, 0);
    size_t *input_cum_shape_ptr = GetDeviceAddress<size_t>(workspace, 1);
    size_t *cur_pos_arr = GetDeviceAddress<size_t>(workspace, 2);
    T *output = GetDeviceAddress<T>(outputs, 0);
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(input_shape_ptr, &input_shape_[0], input_shape_.size() * sizeof(size_t),
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync input_shape_ failed");
    CalReverseSequence(input_size_, input, seq_len, batch_dim_, seq_dim_, cur_pos_arr, input_shape_ptr,
                       input_cum_shape_ptr, shape_size_, output, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    batch_dim_ = GetAttr<int64_t>(kernel_node, "batch_dim");
    seq_dim_ = GetAttr<int64_t>(kernel_node, "seq_dim");
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but ReverseSequence needs 2 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but ReverseSequence needs 1 output.";
      return false;
    }
    input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto seq_len_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    is_null_input_ = CHECK_NULL_INPUT(input_shape_) || CHECK_NULL_INPUT(seq_len_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'ReverseSequenceGpuKernel', input is null.";
      InitSizeLists();
      return true;
    }
    if (input_shape_.size() < 1) {
      MS_LOG(EXCEPTION) << "For 'ReverseSequenceGpuKernel', the rank of input cannot be less than 1, but got "
                        << input_shape_.size();
    }
    input_size_ = 1;
    shape_size_ = input_shape_.size();  // required for calls
    for (size_t i = 0; i < shape_size_; i++) {
      input_size_ *= input_shape_[i];
    }
    // get seq len shape
    seq_len_size_ = seq_len_shape.size();
    output_size_ = input_size_;  // size does not change
    // Allocate workspace memory to use for storing indices for each thread to compute with
    size_t total_threads = GET_BLOCKS(input_size_) * GET_THREADS;
    total_index_dim_ = total_threads * shape_size_;
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    input_size_list_.push_back(seq_len_size_ * sizeof(S));
    workspace_size_list_.push_back(shape_size_ * sizeof(size_t));       // input_shape
    workspace_size_list_.push_back(shape_size_ * sizeof(size_t));       // cumulative shape
    workspace_size_list_.push_back(total_index_dim_ * sizeof(size_t));  // scratch memory for holding indices per thread
    output_size_list_.push_back(output_size_ * sizeof(T));
  }

 private:
  size_t shape_size_;
  size_t input_size_;
  int64_t batch_dim_;
  int64_t seq_dim_;
  bool is_null_input_;
  size_t seq_len_size_;
  size_t total_index_dim_;
  size_t output_size_;
  size_t workspace_size_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_REVERSE_SEQUENCE_GPU_KERNEL_H_
