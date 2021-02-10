/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SLICE_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SLICE_GRAD_GPU_KERNEL_H_

#include <vector>
#include <algorithm>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/slice_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class SliceGradGpuKernel : public GpuKernel {
 public:
  SliceGradGpuKernel() : is_strided_slice_(false), input_size_(0), output_size_(0), workspace_size_(0) {}
  ~SliceGradGpuKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *dy = GetDeviceAddress<T>(inputs, 0);
    T *dx = GetDeviceAddress<T>(outputs, 0);
    FillDeviceArray(outputs[0]->size / sizeof(T), dx, 0.f, reinterpret_cast<cudaStream_t>(stream_ptr));
    CalSliceGrad(output_size_ / sizeof(T), dy, input_shape_, begin_, size_, dx,
                 reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    if (!CheckParam(kernel_node)) {
      return false;
    }
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    if (kernel_name == "StridedSliceGrad") {
      is_strided_slice_ = true;
      std::vector<int64_t> shapex = GetAttr<std::vector<int64_t>>(kernel_node, "shapex");
      for (auto x : shapex) {
        input_shape_.push_back(static_cast<size_t>(x));
      }
      for (auto i = input_shape_.size(); i < 4; i++) {
        (void)input_shape_.insert(input_shape_.begin(), 1);
      }
      strides_ = GetAttr<std::vector<int64_t>>(kernel_node, "strides");
      for (auto i = strides_.size(); i < 4; i++) {
        (void)strides_.insert(strides_.begin(), 1);
      }
      size_ = GetAttr<std::vector<int64_t>>(kernel_node, "end");
    } else {
      auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
      ShapeNdTo4d(input_shape, &input_shape_);
      size_ = GetAttr<std::vector<int64_t>>(kernel_node, "size");
    }

    auto dy_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    ShapeNdTo4d(dy_shape, &dy_shape_);
    begin_ = GetAttr<std::vector<int64_t>>(kernel_node, "begin");
    DealParam();
    input_size_ = input_shape_[0] * input_shape_[1] * input_shape_[2] * input_shape_[3] * sizeof(T);

    output_size_ = sizeof(T);
    for (auto x : dy_shape_) {
      output_size_ = output_size_ * x;
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(output_size_);
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(input_size_);
  }

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but SliceGradGpuKernel needs 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (input_shape.size() > 4) {
      MS_LOG(ERROR) << "Input dims is " << input_shape.size() << ", but SliceGradGpuKernel only support 4d or lower.";
      return false;
    }
    if (input_shape.size() == 0) {
      MS_LOG(ERROR) << "Input dims is " << input_shape.size() << ", scalar is not supported.";
      return false;
    }
    return true;
  }
  void DealParam() {
    for (auto i = begin_.size(); i < 4; i++) {
      (void)begin_.insert(begin_.begin(), 0);
    }
    for (auto i = size_.size(); i < 4; i++) {
      (void)size_.insert(size_.begin(), 1);
    }
    for (size_t i = 0; i < begin_.size(); i++) {
      if (begin_[i] < 0) {
        begin_[i] = begin_[i] + input_shape_[i];
      }
    }
    for (size_t i = 0; i < size_.size(); i++) {
      if (size_[i] < 0) {
        size_[i] = (size_[i] + input_shape_[i]) > 0 ? (size_[i] + input_shape_[i]) : 0;
      }
    }
  }
  std::vector<int64_t> begin_;
  std::vector<int64_t> size_;
  std::vector<int64_t> strides_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> dy_shape_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  bool is_strided_slice_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
};  // namespace kernel
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SLICE_GRAD_GPU_KERNEL_H_
