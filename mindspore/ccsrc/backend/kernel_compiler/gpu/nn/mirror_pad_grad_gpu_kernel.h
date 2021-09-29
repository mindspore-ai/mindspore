/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
      : num_input_(0),
        num_paddings_(0),
        mode_(0),
        is_null_input_(false),
        input_size_(1),
        output_size_(1),
        workspace_size_(0) {}
  ~MirrorPadGpuBackKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    int64_t *paddings = GetDeviceAddress<int64_t>(inputs, 1);
    T *interim = GetDeviceAddress<T>(workspace, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);

    size_t dx_size = output_size_ / sizeof(T);
    size_t interim_dy_size = workspace_size_ / sizeof(T);
    CalMirrorPadGrad(dx_size, interim_dy_size, input, interim, output_shape_[0], output_shape_[1], output_shape_[2],
                     output_shape_[3], input_shape_[2], input_shape_[3], num_paddings_, paddings, mode_, output,
                     reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but MirrorPadGrad needs 2 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but MirrorPadGrad needs 1 output.";
      return false;
    }
    auto prim = AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    string mode = GetValue<string>(prim->GetAttr("mode"));
    if (mode == "REFLECT") {
      mode_ = 0;  // reflected mirroring
    } else {
      mode_ = 1;  // symmetric mirroring
    }

    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto padding_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape) || CHECK_NULL_INPUT(padding_shape) || CHECK_NULL_INPUT(output_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'MirrorPadGradGpuKernel', input or output is null.";
      InitSizeLists();
      return true;
    }
    // shape adjustment -> from 2d/3d to 4d to standardize
    if (input_shape.size() == 3) {
      auto it = input_shape.begin();
      (void)input_shape.insert(it, 1);  // batch padding
    } else if (input_shape.size() == 2) {
      auto it = input_shape.begin();
      (void)input_shape.insert(it, 2, 1);  // channel padding
    }
    if (input_shape.size() < 4) {
      MS_LOG(EXCEPTION) << "For 'MirrorPadGradGpuKernel', the rank of input should be greater than or equal to 4, "
                        << "but got the rank of input: " << input_shape.size();
    }
    input_size_ = sizeof(T);
    for (auto in_shape : input_shape) {
      input_size_ *= in_shape;
      input_shape_.push_back(in_shape);
    }
    num_input_ = input_size_;

    // account for paddings in input size -> passed as int64_ts

    num_paddings_ = padding_shape[0];
    input_size_ += (2 * num_paddings_ * sizeof(int64_t));

    if (output_shape.size() == 4) {
    } else if (output_shape.size() == 3) {
      auto it = output_shape.begin();
      (void)output_shape.insert(it, 1);  // batch padding
    } else if (output_shape.size() == 2) {
      auto it = output_shape.begin();
      (void)output_shape.insert(it, 2, 1);  // channel padding
    }
    if (output_shape.size() < 2) {
      MS_LOG(EXCEPTION) << "For 'MirrorPadGradGpuKernel', the rank of output should be greater than or equal to 2, "
                        << "but got the rank of output: " << output_shape.size();
    }
    output_size_ = sizeof(T);
    for (auto x : output_shape) {
      output_size_ *= x;
      output_shape_.push_back(x);
    }

    // calc workspace size
    // store dy values with accumulation across batch and channel only
    workspace_size_ = sizeof(T);
    for (int i = 0; i < 2; i++) {
      workspace_size_ *= output_shape[i];     // BATCH, CHANNEL -> Output size
      workspace_size_ *= input_shape[i + 2];  // WIDTH, HEIGHT -> Input Size
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
    input_size_list_.push_back(2 * num_paddings_ * sizeof(int64_t));  // for 64 bit int defined in API
    workspace_size_list_.push_back(workspace_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  size_t num_input_;
  int num_paddings_;
  int mode_;
  bool is_null_input_;
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;
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
