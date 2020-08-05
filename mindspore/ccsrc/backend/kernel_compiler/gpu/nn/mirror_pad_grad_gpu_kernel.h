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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MIRROR_PAD_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MIRROR_PAD_GRAD_GPU_KERNEL_H_

#include <iostream>
#include <vector>
#include <string>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/mirror_pad_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class MirrorPadGpuBackKernel : public GpuKernel {
 public:
  MirrorPadGpuBackKernel()
      : num_input_(0), num_paddings_(0), mode_(0), input_size_(1), output_size_(1), workspace_size_(0) {}
  ~MirrorPadGpuBackKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    int *paddings = GetDeviceAddress<int>(inputs, 1);
    T *output = GetDeviceAddress<T>(outputs, 0);

    size_t size = output_size_ / sizeof(T);
    int dim_offset = output_shape_.size() - 2;

    CalMirrorPadGrad(size, input, input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3],
                     output_shape_[dim_offset + 0], output_shape_[dim_offset + 1], num_paddings_, paddings, mode_,
                     output, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but MirrorPadGrad needs 2 input.";
      return false;
    }
    // check number of output -> should be 1
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but MirrorPadGrad needs 1 output.";
      return false;
    }

    string mode = GetValue<string>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("mode"));

    if (mode == "REFLECT") {
      mode_ = 0;  // reflected mirroring
    } else {
      mode_ = 1;  // symmetric mirroring
    }

    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    // shape adjustement -> from 2d/3d to 4d to standardize
    if (input_shape.size() == 4) {
    } else if (input_shape.size() == 3) {
      auto it = input_shape.begin();
      input_shape.insert(it, 1);  // batch padding
    } else if (input_shape.size() == 2) {
      auto it = input_shape.begin();
      input_shape.insert(it, 2, 1);  // channel padding
    }

    for (auto in_shape : input_shape) {
      input_size_ *= in_shape;
      input_shape_.push_back(in_shape);
    }
    num_input_ = input_size_;
    input_size_ *= sizeof(T);

    auto padding_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    num_paddings_ = padding_shape[0];
    input_size_ += +(2 * num_paddings_ * sizeof(int));

    output_size_ = sizeof(T);
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    for (auto x : output_shape) {
      output_size_ *= x;
      output_shape_.push_back(x);
    }

    int max_width = input_shape_[3];
    int max_height = input_shape_[2];

    // basic error check for padding value
    if (mode_ == 1) {  // symmetric
      max_width = max_width + (2 * max_width);
      max_height = max_height + (2 * max_height);
    } else {  // reflect
      max_width = max_width + (2 * (max_width - 1));
      max_height = max_height + (2 * (max_height - 1));
    }

    if (output_shape_[(output_shape_.size() - 2) + 0] > max_width ||
        output_shape_[(output_shape_.size() - 2) + 1] > max_width) {
      MS_LOG(ERROR) << "ERROR: Padding value too high for input Tensor on 1 or more DIMS";
      return false;
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(num_input_ * sizeof(T));
    input_size_list_.push_back(2 * num_paddings_ * sizeof(int));
    output_size_list_.push_back(output_size_);
  }

 private:
  size_t num_input_;
  int num_paddings_;
  int mode_;
  std::vector<int> input_shape_;   // dims of the input data
  std::vector<int> output_shape_;  // dims of the output data
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
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MIRROR_PAD_GRAD_GPU_KERNEL_H_
