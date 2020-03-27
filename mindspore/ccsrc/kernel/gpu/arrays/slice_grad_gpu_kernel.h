/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_SLICE_GRAD_GPU_KERNEL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_SLICE_GRAD_GPU_KERNEL_H

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/slice_impl.cuh"

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
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    T *dy = GetDeviceAddress<T>(inputs, 0);
    T *dx = GetDeviceAddress<T>(outputs, 0);
    FillDeviceArray(outputs[0]->size / sizeof(T), dx, 0.f, reinterpret_cast<cudaStream_t>(stream_ptr));
    if (is_strided_slice_) {
      CalStridedSliceGrad(output_size_ / sizeof(T), dy, input_shape_, begin_, size_, strides_, dx,
                          reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      CalSliceGrad(output_size_ / sizeof(T), dy, input_shape_, begin_, size_, dx,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    if (!CheckParam(kernel_node)) {
      return false;
    }
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    if (kernel_name == "StridedSliceGrad") {
      is_strided_slice_ = true;
      input_shape_ = GetAttr<std::vector<int>>(kernel_node, "shapex");
      for (auto i = input_shape_.size(); i < 4; i++) {
        (void)input_shape_.insert(input_shape_.begin(), 1);
      }
      strides_ = GetAttr<std::vector<int>>(kernel_node, "strides");
      for (auto i = strides_.size(); i < 4; i++) {
        (void)strides_.insert(strides_.begin(), 1);
      }
      size_ = GetAttr<std::vector<int>>(kernel_node, "end");
    } else {
      auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
      input_shape_.push_back(input_shape.size() < 4 ? 1 : SizeToInt(input_shape[input_shape.size() - 4]));
      input_shape_.push_back(input_shape.size() < 3 ? 1 : SizeToInt(input_shape[input_shape.size() - 3]));
      input_shape_.push_back(input_shape.size() < 2 ? 1 : SizeToInt(input_shape[input_shape.size() - 2]));
      input_shape_.push_back(SizeToInt(input_shape[input_shape.size() - 1]));
      size_ = GetAttr<std::vector<int>>(kernel_node, "size");
    }

    auto dy_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    dy_shape_.push_back(dy_shape.size() < 4 ? 1 : SizeToInt(dy_shape[dy_shape.size() - 4]));
    dy_shape_.push_back(dy_shape.size() < 3 ? 1 : SizeToInt(dy_shape[dy_shape.size() - 3]));
    dy_shape_.push_back(dy_shape.size() < 2 ? 1 : SizeToInt(dy_shape[dy_shape.size() - 2]));
    dy_shape_.push_back(SizeToInt(dy_shape[dy_shape.size() - 1]));

    begin_ = GetAttr<std::vector<int>>(kernel_node, "begin");
    DealParam();
    input_size_ = IntToSize(input_shape_[0] * input_shape_[1] * input_shape_[2] * input_shape_[3]) * sizeof(T);

    output_size_ = sizeof(T);
    for (auto x : dy_shape_) {
      output_size_ = output_size_ * IntToSize(x);
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
  std::vector<int> begin_;
  std::vector<int> size_;
  std::vector<int> strides_;
  std::vector<int> input_shape_;
  std::vector<int> dy_shape_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  bool is_strided_slice_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_SLICE_GRAD_GPU_KERNEL_H
