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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PAD_GPU_FWD_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PAD_GPU_FWD_KERNEL_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/pad_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class PadGpuFwdKernel : public GpuKernel {
 public:
  PadGpuFwdKernel() : shape_size_(0), temp(0), input_size_(1), output_size_(1), workspace_size_(0) {}
  ~PadGpuFwdKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);
    size_t size = output_size_ / sizeof(T);
    int pad_left = paddings[3][0];
    int pad_top = paddings[2][0];
    int pad_channel_before = paddings[1][0];
    int pad_channel_after = paddings[1][1];
    const T pad_value = 0.0;
    CalPadGeneral(size, input, input_shape_[0], input_shape_[1], pad_channel_before, pad_channel_after, input_shape_[2],
                  input_shape_[3], output_shape_[2], output_shape_[3], pad_top, pad_left, pad_value, output,
                  reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    if (!CheckIONumber(kernel_node)) {
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    shape_size_ = input_shape.size();
    if (shape_size_ == 4) {  // shape adjustement from 2d/3d to 4d
    } else if (shape_size_ == 3) {
      auto it = input_shape.begin();
      input_shape.insert(it, 1);  // batch padding
      shape_size_ = 4;
    } else if (shape_size_ == 2) {
      auto it = input_shape.begin();
      input_shape.insert(it, 2, 1);  // channel padding
      shape_size_ = 4;
    }
    std::vector<std::vector<int64_t>> paddings_me =
      GetValue<std::vector<std::vector<int64_t>>>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("paddings"));
    (void)std::transform(paddings_me.begin(), paddings_me.end(), std::back_inserter(paddings),
                         [](const std::vector<int64_t> &values) {
                           std::vector<int> shape;
                           (void)std::transform(values.begin(), values.end(), std::back_inserter(shape),
                                                [](const int64_t &value) { return static_cast<int>(value); });
                           return shape;
                         });
    if (paddings.size() == 4) {  // shape adjustement from 2d/3d to 4d
    } else if (paddings.size() == 3) {
      auto it = paddings.begin();
      paddings.insert(it, 1, {0, 0});  // batch padding
    } else if (paddings.size() == 2) {
      auto it = paddings.begin();
      paddings.insert(it, 2, {0, 0});  // channel padding
    }
    for (size_t i = 0; i < shape_size_; i++) {
      input_size_ *= input_shape[i];
      input_shape_.push_back(input_shape[i]);
    }
    input_size_ *= sizeof(T);
    for (size_t i = 0; i < shape_size_; i++) {
      temp = input_shape[i] + (paddings[i][0] + paddings[i][1]);  // compute new dim size
      output_size_ *= temp;
      output_shape_.push_back(temp);  // correct new dimension size
    }
    output_size_ *= sizeof(T);
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  bool CheckIONumber(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but Pad needs 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but Pad needs 1 output.";
      return false;
    }
    return true;
  }

  size_t shape_size_;
  size_t temp;
  std::vector<std::vector<int>> paddings;  // list of paddings (tuple of tuple in python)
  std::vector<int> input_shape_;           // dims of the input data
  std::vector<int> output_shape_;          // dims of the output data
  // default
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PAD_GPU_FWD_KERNEL_H_
