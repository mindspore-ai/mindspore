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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RESIZE_BILINEAR_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RESIZE_BILINEAR_GRAD_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/resize_bilinear_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class ResizeBilinearGradGpuKernel : public GpuKernel {
 public:
  ResizeBilinearGradGpuKernel() { ResetResource(); }
  ~ResizeBilinearGradGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *dy = GetDeviceAddress<T>(inputs, 0);
    float *interim = GetDeviceAddress<float>(workspace, 0);
    T *dx = GetDeviceAddress<T>(outputs, 0);
    float h_scale = Scaling(dx_h_, dy_h_, align_corners_);
    float w_scale = Scaling(dx_w_, dy_w_, align_corners_);
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemsetAsync(dx, 0, dx_size_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemsetAsync dx failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemsetAsync(interim, 0, workspace_size_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemsetAsync dx_interim failed");
    CalResizeBilinearGrad(dy, n_, c_, dy_h_, dy_w_, dx_h_, dx_w_, h_scale, w_scale, dx, interim,
                          reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but ResizeBilinearGrad needs 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but ResizeBilinearGrad has 1 output.";
      return false;
    }
    std::vector<size_t> dy_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    std::vector<size_t> x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    std::vector<size_t> dx_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(dy_shape) || CHECK_NULL_INPUT(x_shape) || CHECK_NULL_INPUT(dx_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'ResizeBilinearGradGpuKernel', input or output is null.";
      InitSizeLists();
      return true;
    }
    if (dy_shape.size() != 4) {
      MS_LOG(ERROR) << "Input is " << dy_shape.size() << "-D, but ResizeBilinearGrad supports only 4-D inputs.";
      return false;
    }
    if (x_shape.size() != 4) {
      MS_LOG(ERROR) << "Input is " << x_shape.size() << "-D, but ResizeBilinearGrad supports only 4-D inputs.";
      return false;
    }
    if (dx_shape.size() != 4) {
      MS_LOG(ERROR) << "For 'ResizeBilinearGradGpuKernel', the rank of output must be 4, but got " << dx_shape.size();
      return false;
    }
    n_ = SizeToInt(dy_shape[0]);
    c_ = SizeToInt(dy_shape[1]);
    dy_h_ = SizeToInt(dy_shape[2]);
    dy_w_ = SizeToInt(dy_shape[3]);
    dx_h_ = SizeToInt(dx_shape[2]);
    dx_w_ = SizeToInt(dx_shape[3]);
    dy_size_ = sizeof(T);
    for (auto x : dy_shape) {
      dy_size_ *= x;
    }
    dx_size_ = sizeof(T);
    for (auto x : dx_shape) {
      dx_size_ *= x;
    }
    workspace_size_ = (dx_size_ / sizeof(T)) * sizeof(float);
    align_corners_ = GetAttr<bool>(kernel_node, "align_corners");
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    align_corners_ = false;
    is_null_input_ = false;
    n_ = 0;
    c_ = 0;
    dy_h_ = 0;
    dy_w_ = 0;
    dx_h_ = 0;
    dx_w_ = 0;
    dy_size_ = 0;
    dx_size_ = 0;
    workspace_size_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(dy_size_);
    workspace_size_list_.push_back(workspace_size_);
    output_size_list_.push_back(dx_size_);
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
  int dy_h_;
  int dy_w_;
  int dx_h_;
  int dx_w_;
  size_t dy_size_;
  size_t dx_size_;
  size_t workspace_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RESIZE_BILINEAR_GRAD_GPU_KERNEL_H_
