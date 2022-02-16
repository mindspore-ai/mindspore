/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/resize_bilinear_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t kInputsNum = 2;
constexpr size_t kDyShapeSize = 4;
constexpr size_t kxShapeSize = 4;
constexpr size_t kDxShapeSize = 4;
constexpr size_t kDyIndexForN = 0;
constexpr size_t kDyIndexForC = 1;
constexpr size_t kDyIndexForH = 2;
constexpr size_t kDyIndexForW = 3;
constexpr size_t kDxIndexForH = 2;
constexpr size_t kDxIndexForW = 3;

template <typename T>
class ResizeBilinearGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  ResizeBilinearGradGpuKernelMod() { ResetResource(); }
  ~ResizeBilinearGradGpuKernelMod() override = default;

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
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != kInputsNum) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 2, but got " << input_num;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }
    std::vector<size_t> dy_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    std::vector<size_t> x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    std::vector<size_t> dx_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(dy_shape, kernel_name, "dy") || CHECK_SHAPE_NULL(x_shape, kernel_name, "x") ||
                     CHECK_SHAPE_NULL(dx_shape, kernel_name, "dx");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    if (dy_shape.size() != kDyShapeSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of dy should be equal to 4, but got "
                        << dy_shape.size();
    }
    if (x_shape.size() != kxShapeSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of x should be equal to 4, but got "
                        << x_shape.size();
    }
    if (dx_shape.size() != kDxShapeSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of dx should be equal to 4, but got "
                        << dx_shape.size();
    }
    n_ = SizeToInt(dy_shape[kDyIndexForN]);
    c_ = SizeToInt(dy_shape[kDyIndexForC]);
    dy_h_ = SizeToInt(dy_shape[kDyIndexForH]);
    dy_w_ = SizeToInt(dy_shape[kDyIndexForW]);
    dx_h_ = SizeToInt(dx_shape[kDxIndexForH]);
    dx_w_ = SizeToInt(dx_shape[kDxIndexForW]);
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
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RESIZE_BILINEAR_GRAD_GPU_KERNEL_H_
