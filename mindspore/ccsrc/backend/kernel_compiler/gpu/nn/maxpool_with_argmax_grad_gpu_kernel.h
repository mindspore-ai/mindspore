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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOLWITHARGMAX_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOLWITHARGMAX_GRAD_GPU_KERNEL_H_

#include <algorithm>
#include <vector>
#include <string>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/maxpool_with_argmax_grad_impl.cuh"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class MaxPoolWithArgmaxGradGpuKernel : public GpuKernel {
 public:
  MaxPoolWithArgmaxGradGpuKernel()
      : n_(0),
        c_(0),
        x_height_(0),
        x_width_(0),
        dy_height_(0),
        dy_width_(0),
        x_size_(0),
        dy_size_(0),
        index_size_(0),
        dx_size_(0) {}
  ~MaxPoolWithArgmaxGradGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    T *dy_addr = GetDeviceAddress<T>(inputs, 1);
    S *index_addr = GetDeviceAddress<S>(inputs, 2);
    T *dx_addr = GetDeviceAddress<T>(outputs, 0);
    CalMaxPoolWithArgmaxGrad(dy_addr, index_addr, n_, c_, x_height_, x_width_, dy_height_, dy_width_, dx_addr,
                             reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 3) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but MaxPoolGradWithArgmax needs 3 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but MaxPoolGradWithArgmax needs 1 output.";
      return false;
    }
    auto x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto dy_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto index_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto dx_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    x_size_ = sizeof(T);
    for (auto x : x_shape) {
      x_size_ *= x;
    }
    dy_size_ = sizeof(T);
    for (auto x : dy_shape) {
      dy_size_ *= x;
    }
    index_size_ = sizeof(S);
    for (auto x : index_shape) {
      index_size_ *= x;
    }
    dx_size_ = sizeof(T);
    for (auto x : dx_shape) {
      dx_size_ *= x;
    }
    n_ = SizeToInt(x_shape[0]);
    c_ = SizeToInt(x_shape[1]);
    x_height_ = SizeToInt(x_shape[2]);
    x_width_ = SizeToInt(x_shape[3]);
    dy_height_ = SizeToInt(dy_shape[2]);
    dy_width_ = SizeToInt(dy_shape[3]);

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(dy_size_);
    input_size_list_.push_back(index_size_);
    output_size_list_.push_back(dx_size_);
  }

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  int n_;
  int c_;
  int x_height_;
  int x_width_;
  int dy_height_;
  int dy_width_;

  size_t x_size_;
  size_t dy_size_;
  size_t index_size_;
  size_t dx_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOLWITHARGMAX_GRAD_GPU_KERNEL_H_
