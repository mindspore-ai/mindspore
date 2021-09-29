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

#ifndef MINDSPORE_MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TENSOR_STRIDE_UPDATE_GPU_KERNEL_H_
#define MINDSPORE_MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TENSOR_STRIDE_UPDATE_GPU_KERNEL_H_

#include <algorithm>
#include <vector>
#include <numeric>
#include <functional>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/gpu/cuda_impl/slice_copy_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class TensorCopySlicesGpuKernel : public GpuKernel {
 public:
  TensorCopySlicesGpuKernel() : input_size_(0), update_size_(0), output_size_(0), is_null_input_(false) {}
  ~TensorCopySlicesGpuKernel() {}

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *update_addr = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(output_addr, input_addr, inputs[0]->size, cudaMemcpyDeviceToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "TensorCopySlices cudaMemcpyAsync outputs failed");
    CopySlices(update_shape_, begin_, strides_, output_shape_, update_addr, output_addr,
               reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;

    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but TensorCopySlices needs 2 inputs.";
      return false;
    }

    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but TensorCopySlices has 1 output.";
      return false;
    }

    input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto update_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    is_null_input_ = CHECK_NULL_INPUT(input_shape_) || CHECK_NULL_INPUT(update_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'TensorCopySlicesGpuKernel', input or output is null.";
      InitSizeLists();
      return true;
    }
    if (input_shape_.size() > kMaxDims) {
      MS_LOG(ERROR) << "StridedSlice support dims no more than " << kMaxDims << ", but the input shape is "
                    << input_shape_.size();
      return false;
    }

    begin_ = GetAttr<std::vector<int64_t>>(kernel_node, kAttrBegin);
    end_ = GetAttr<std::vector<int64_t>>(kernel_node, kAttrEnd);
    strides_ = GetAttr<std::vector<int64_t>>(kernel_node, kAttrStrides);

    if (begin_.size() > input_shape_.size()) {
      MS_LOG(ERROR) << "For 'TensorCopySlicesGpuKernel', the rank of begin attr cannot be more than the rank of input, "
                    << "but got the rank of begin attr: " << begin_.size()
                    << ", the rank of input: " << input_shape_.size();
      return false;
    }

    FillEmptyDims(kernel_node);
    output_shape_ = input_shape_;
    FillUpdateDim();
    CheckAtrrAndShapeValid(kernel_node);

    GetSize();
    InitSizeLists();
    return true;
  }

 protected:
  void CheckAtrrAndShapeValid(const CNodePtr &kernel_node) {
    auto update_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    size_t total_update_num = std::accumulate(update_shape.begin(), update_shape.end(), 1, std::multiplies<size_t>());
    if (begin_.size() != end_.size() || end_.size() != strides_.size()) {
      MS_LOG(EXCEPTION) << "Invalid attr begin:" << begin_ << " end:" << end_ << " strides:" << strides_;
    }
    auto len = begin_.size();
    size_t total_input_num = 1;
    for (size_t i = 0; i < len; ++i) {
      MS_EXCEPTION_IF_ZERO("strides_[i]", strides_[i]);
      total_input_num *= ((end_[i] - begin_[i]) / strides_[i]);
    }
    if (total_input_num != total_update_num) {
      MS_LOG(EXCEPTION) << "Invalid update_shape:" << update_shape << ". Maybe you need to broadcast it.";
    }
  }

  void GetSize() {
    input_size_ = sizeof(T);
    for (size_t i = 0; i < input_shape_.size(); i++) {
      input_size_ *= input_shape_[i];
    }

    update_size_ = sizeof(T);
    for (size_t i = 0; i < update_shape_.size(); i++) {
      update_size_ *= update_shape_[i];
    }
    output_size_ = sizeof(T);
    for (size_t i = 0; i < output_shape_.size(); i++) {
      output_size_ *= output_shape_[i];
    }
  }

  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(update_size_);
    output_size_list_.push_back(output_size_);
    return;
  }

  void FillEmptyDims(const CNodePtr &kernel_node) {
    for (size_t i = 0; i < kMaxDims; i++) {
      if (i < begin_.size()) {
        int64_t dim = input_shape_[i];
        begin_[i] = std::min(begin_[i] < 0 ? std::max(begin_[i] + dim, static_cast<int64_t>(0)) : begin_[i], dim - 1);
      } else {
        begin_.push_back(0);
      }

      if (i < end_.size()) {
        int64_t dim = input_shape_[i];
        end_[i] = std::max(end_[i] < 0 ? end_[i] + dim : std::min(end_[i], dim), static_cast<int64_t>(-1));
      } else {
        end_.push_back(i < input_shape_.size() ? input_shape_[i] : 1);
      }

      if (i >= strides_.size()) {
        strides_.push_back(1);
      }

      if (i >= input_shape_.size()) {
        input_shape_.push_back(1);
      }
    }
  }

  void FillUpdateDim() {
    for (size_t i = 0; i < kMaxDims; i++) {
      if (begin_[i] <= end_[i] && strides_[i] > 0) {
        update_shape_.push_back((end_[i] - 1 - begin_[i]) / strides_[i] + 1);
      } else if (begin_[i] > end_[i] && strides_[i] < 0) {
        MS_EXCEPTION_IF_ZERO("strides_[i] + 1", strides_[i] + 1);
        update_shape_.push_back((end_[i] - begin_[i] + 1) / strides_[i] + 1);
      } else {
        update_shape_.push_back(0);
      }
    }
  }

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  std::vector<size_t> input_shape_;
  std::vector<size_t> update_shape_;
  std::vector<size_t> output_shape_;

  std::vector<int64_t> begin_;
  std::vector<int64_t> end_;
  std::vector<int64_t> strides_;

  size_t input_size_;
  size_t update_size_;
  size_t output_size_;
  inline static size_t kMaxDims = 8;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TENSOR_STRIDE_UPDATE_GPU_KERNEL_H_
