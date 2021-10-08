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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RESIZE_BILINEAR_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RESIZE_BILINEAR_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/resize_bilinear_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class ResizeBilinearGpuKernel : public GpuKernel {
 public:
  ResizeBilinearGpuKernel() { ResetResource(); }
  ~ResizeBilinearGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);
    float h_scale = Scaling(input_h_, output_h_, align_corners_);
    float w_scale = Scaling(input_w_, output_w_, align_corners_);
    CalResizeBilinear(input, n_, c_, input_h_, input_w_, output_h_, output_w_, h_scale, w_scale, output,
                      reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but ResizeBilinear needs 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but ResizeBilinear has 1 output.";
      return false;
    }
    std::vector<size_t> input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    std::vector<size_t> output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape) || CHECK_NULL_INPUT(output_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'ResizeBilinearGpuKernel', input or output is null.";
      InitSizeLists();
      return true;
    }
    if (input_shape.size() != 4 || output_shape.size() != 4) {
      MS_LOG(ERROR) << "For 'ResizeBilinear', the rank of input and output must be 4, but got the rank of input: "
                    << input_shape.size() << ", the rank of output: " << output_shape.size();
      return false;
    }
    n_ = SizeToInt(input_shape[0]);
    c_ = SizeToInt(input_shape[1]);
    input_h_ = SizeToInt(input_shape[2]);
    input_w_ = SizeToInt(input_shape[3]);
    output_h_ = SizeToInt(output_shape[2]);
    output_w_ = SizeToInt(output_shape[3]);
    input_size_ = sizeof(T);
    for (auto x : input_shape) {
      input_size_ *= x;
    }
    output_size_ = sizeof(T);
    for (auto x : output_shape) {
      output_size_ *= x;
    }
    align_corners_ = GetAttr<bool>(kernel_node, "align_corners");
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    align_corners_ = false;
    is_null_input_ = false;
    n_ = 0;
    c_ = 0;
    input_h_ = 0;
    input_w_ = 0;
    output_h_ = 0;
    output_w_ = 0;
    input_size_ = 0;
    output_size_ = 0;
    workspace_size_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  float Scaling(const int in_size, const int out_size, bool align_corners) {
    return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                           : in_size / static_cast<float>(out_size);
  }

  bool align_corners_;
  bool is_null_input_;
  int n_;
  int c_;
  int input_h_;
  int input_w_;
  int output_h_;
  int output_w_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RESIZE_BILINEAR_GPU_KERNEL_H_
