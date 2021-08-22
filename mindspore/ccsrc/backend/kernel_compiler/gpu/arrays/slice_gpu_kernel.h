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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SLICE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SLICE_GPU_KERNEL_H_

#include <vector>
#include <utility>
#include <algorithm>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/slice_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class SliceGpuFwdKernel : public GpuKernel {
 public:
  SliceGpuFwdKernel()
      : is_null_input_(false), is_1d_to_4d_input_(false), input_size_(0), output_size_(0), workspace_size_(0) {}
  ~SliceGpuFwdKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);
    if (is_1d_to_4d_input_) {
      Slice4DKernel(begin_[0], begin_[1], begin_[2], begin_[3], size_[0], size_[1], size_[2], size_[3], input_shape_[0],
                    input_shape_[1], input_shape_[2], input_shape_[3], input, output,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      Slice5DKernel(begin_[0], begin_[1], begin_[2], begin_[3], begin_[4], size_[0], size_[1], size_[2], size_[3],
                    size_[4], input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3], input_shape_[4],
                    input, output, reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    if (!CheckParam(kernel_node)) {
      return false;
    }
    auto data_format = AnfAlgo::GetInputFormat(kernel_node, 0);
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    is_1d_to_4d_input_ = input_shape.size() <= 4;
    if (is_1d_to_4d_input_) {
      ShapeNdTo4d(input_shape, &input_shape_);
    } else {
      ShapeNdTo5d(input_shape, &input_shape_);
    }

    for (auto i = begin_.size(); i < input_shape_.size(); i++) {
      (void)begin_.insert(begin_.begin(), 0);
    }
    for (size_t i = size_.size(); i < input_shape_.size(); i++) {
      (void)size_.insert(size_.begin(), 1);
    }

    if (is_1d_to_4d_input_) {
      input_size_ = input_shape_[0] * input_shape_[1] * input_shape_[2] * input_shape_[3] * sizeof(T);
    } else {
      input_size_ = input_shape_[0] * input_shape_[1] * input_shape_[2] * input_shape_[3] * input_shape_[4] * sizeof(T);
    }
    auto out_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);

    output_size_ = sizeof(T);
    for (size_t x : out_shape) {
      output_size_ = output_size_ * x;
    }
    // transpose begin and size for NHWC data
    if (data_format == "NHWC") {
      std::swap(begin_[1], begin_[3]);
      std::swap(begin_[1], begin_[2]);
      std::swap(size_[1], size_[3]);
      std::swap(size_[1], size_[2]);
    } else if (data_format == "NDHWC") {
      std::swap(begin_[1], begin_[4]);
      std::swap(begin_[1], begin_[3]);
      std::swap(begin_[1], begin_[2]);
      std::swap(size_[1], size_[4]);
      std::swap(size_[1], size_[3]);
      std::swap(size_[1], size_[2]);
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  // expand Nd Shape to 5d (N in [0,5])
  void ShapeNdTo5d(const std::vector<size_t> &src, std::vector<size_t> *dst) {
    if (src.size() > 5) {
      MS_EXCEPTION(ValueError) << src.size() << "-D data is not supported!";
    }
    dst->push_back(src.size() < 5 ? 1 : src[src.size() - 5]);
    dst->push_back(src.size() < 4 ? 1 : src[src.size() - 4]);
    dst->push_back(src.size() < 3 ? 1 : src[src.size() - 3]);
    dst->push_back(src.size() < 2 ? 1 : src[src.size() - 2]);
    dst->push_back(src.size() == 0 ? 1 : src[src.size() - 1]);
  }

  bool CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but SliceGpuFwdKernel needs 1 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but SliceGpuFwdKernel needs 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (input_shape.size() > 5) {
      MS_LOG(ERROR) << "Input dims is " << input_shape.size() << ", but SliceGpuFwdKernel olny support 5d or lower.";
      return false;
    }
    if (input_shape.size() == 0) {
      MS_LOG(ERROR) << "Input dims is " << input_shape.size() << ", scalar is not supported.";
      return false;
    }
    size_ = GetAttr<std::vector<int64_t>>(kernel_node, "size");
    begin_ = GetAttr<std::vector<int64_t>>(kernel_node, "begin");

    for (size_t i = 0; i < input_shape.size(); i++) {
      if (input_shape[i] <= 0 || size_[i] <= 0) {
        MS_LOG(WARNING) << "Slice output is null.";
        is_null_input_ = true;
      }
    }
    return true;
  }
  std::vector<int64_t> begin_;
  std::vector<int64_t> size_;
  std::vector<size_t> input_shape_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  bool is_null_input_;
  bool is_1d_to_4d_input_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SLICE_GPU_KERNEL_H_
