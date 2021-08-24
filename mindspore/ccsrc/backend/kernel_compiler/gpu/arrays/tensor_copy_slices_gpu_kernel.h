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
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace kernel {
template <typename T>
class TensorCopySlicesGpuKernel : public GpuKernel {
 public:
  TensorCopySlicesGpuKernel() : input_size_(0), update_size_(0), output_size_(0), offset_(0), copy_size_(0) {}
  ~TensorCopySlicesGpuKernel() {}

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *update_addr = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    if (inputs[1]->size != copy_size_) {
      MS_LOG(EXCEPTION) << "Invalid update size:" << inputs[1]->size << " copy_size_:" << copy_size_;
    }

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(output_addr, input_addr, inputs[0]->size, cudaMemcpyDeviceToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "TensorCopySlices cudaMemcpyAsync outputs failed");

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(output_addr + offset_, update_addr, inputs[1]->size,
                                               cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "TensorCopySlices cudaMemcpyAsync outputs failed");

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

    auto input_shapes = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto update_shapes = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto output_shapes = AnfAlgo::GetOutputInferShape(kernel_node, 0);

    CastShapeSizeToLong(input_shapes, &input_shapes_);
    CastShapeSizeToLong(update_shapes, &update_shapes_);
    CastShapeSizeToLong(output_shapes, &output_shapes_);

    GetSize();
    InitSizeLists();

    auto begin = GetAttr<std::vector<int64_t>>(kernel_node, kAttrBegin);
    auto end = GetAttr<std::vector<int64_t>>(kernel_node, kAttrEnd);
    auto strides = GetAttr<std::vector<int64_t>>(kernel_node, kAttrStrides);

    CheckSliceValid(begin, end, strides, input_shapes_);
    auto dim_offset = CalDimOffset(input_shapes_);
    offset_ = CalOffset(begin, end, dim_offset);
    copy_size_ = GetCopySize(dim_offset, begin, end) * sizeof(T);
    return true;
  }

 protected:
  void GetSize() {
    input_size_ = sizeof(T);
    for (size_t i = 0; i < input_shapes_.size(); i++) {
      input_size_ *= LongToSize(input_shapes_[i]);
    }

    update_size_ = sizeof(T);
    for (size_t i = 0; i < update_shapes_.size(); i++) {
      update_size_ *= LongToSize(update_shapes_[i]);
    }
    output_size_ = sizeof(T);
    for (size_t i = 0; i < output_shapes_.size(); i++) {
      output_size_ *= LongToSize(output_shapes_[i]);
    }
  }

  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(update_size_);
    output_size_list_.push_back(output_size_);
    return;
  }

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  std::vector<int64_t> input_shapes_;
  std::vector<int64_t> update_shapes_;
  std::vector<int64_t> output_shapes_;

  size_t input_size_;
  size_t update_size_;
  size_t output_size_;

  size_t offset_;
  size_t copy_size_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TENSOR_STRIDE_UPDATE_GPU_KERNEL_H_
