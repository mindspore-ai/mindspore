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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_FTRL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_FTRL_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/sparse_ftrl_impl.cuh"
namespace mindspore {
namespace kernel {
constexpr size_t INPUT_NUM = 5;
template <typename T, typename S>
class SparseFtrlGpuKernel : public GpuKernel {
 public:
  SparseFtrlGpuKernel() { ResetResource(); }
  ~SparseFtrlGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *variable = GetDeviceAddress<T>(inputs, 0);
    T *accumulation = GetDeviceAddress<T>(inputs, 1);
    T *linear = GetDeviceAddress<T>(inputs, 2);
    T *gradient = GetDeviceAddress<T>(inputs, 3);
    S *indices = GetDeviceAddress<S>(inputs, 4);
    T *variable_out = GetDeviceAddress<T>(outputs, 0);
    T *accumulation_out = GetDeviceAddress<T>(outputs, 1);
    T *linear_out = GetDeviceAddress<T>(outputs, 2);
    CalSparseApplyFtrl(gradient, indices, num_index_, n_stride_, lr_, l1_, l2_, lr_power_, use_locking_, variable,
                       accumulation, linear, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(variable_out, variable, variable_size_, cudaMemcpyDeviceToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync output failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(accumulation_out, accumulation, accumulation_size_,
                                               cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync output failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(linear_out, linear, linear_size_, cudaMemcpyDeviceToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync output failed");
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != INPUT_NUM) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but sparse ftrl needs " << INPUT_NUM << " inputs.";
      return false;
    }

    variable_size_ = sizeof(T);
    accumulation_size_ = sizeof(T);
    linear_size_ = sizeof(T);
    gradient_size_ = sizeof(T);
    indices_size_ = sizeof(S);

    auto variable_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto accumulation_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto linear_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto gradient_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    auto indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 4);
    is_null_input_ = CHECK_NULL_INPUT(variable_shape) || CHECK_NULL_INPUT(accumulation_shape) ||
                     CHECK_NULL_INPUT(linear_shape) || CHECK_NULL_INPUT(gradient_shape) ||
                     CHECK_NULL_INPUT(indices_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'SparseFTRLGpuKernel', input is null";
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < variable_shape.size(); i++) {
      variable_size_ *= variable_shape[i];
      if (i > 0) {
        n_stride_ *= variable_shape[i];
      }
    }

    for (size_t i = 0; i < accumulation_shape.size(); i++) {
      accumulation_size_ *= accumulation_shape[i];
    }

    for (size_t i = 0; i < linear_shape.size(); i++) {
      linear_size_ *= linear_shape[i];
    }

    for (size_t i = 0; i < gradient_shape.size(); i++) {
      gradient_size_ *= gradient_shape[i];
    }

    for (size_t i = 0; i < indices_shape.size(); i++) {
      indices_size_ *= indices_shape[i];
    }

    lr_ = GetAttr<float>(kernel_node, "lr");
    l1_ = GetAttr<float>(kernel_node, "l1");
    l2_ = GetAttr<float>(kernel_node, "l2");
    lr_power_ = GetAttr<float>(kernel_node, "lr_power");
    use_locking_ = GetAttr<bool>(kernel_node, "use_locking");
    num_index_ = indices_shape[0];

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(variable_size_);
    input_size_list_.push_back(accumulation_size_);
    input_size_list_.push_back(linear_size_);
    input_size_list_.push_back(gradient_size_);
    input_size_list_.push_back(indices_size_);
    output_size_list_.push_back(variable_size_);
    output_size_list_.push_back(accumulation_size_);
    output_size_list_.push_back(linear_size_);
  }

  void ResetResource() noexcept override {
    variable_size_ = 0;
    accumulation_size_ = 0;
    linear_size_ = 0;
    gradient_size_ = 0;
    indices_size_ = 0;
    lr_ = 0.0f;
    l1_ = 0.0f;
    l2_ = 0.0f;
    lr_power_ = 0.0f;
    use_locking_ = false;
    is_null_input_ = false;
    num_index_ = 0;
    n_stride_ = 1;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 private:
  size_t variable_size_;
  size_t accumulation_size_;
  size_t linear_size_;
  size_t gradient_size_;
  size_t indices_size_;
  float lr_;
  float l1_;
  float l2_;
  float lr_power_;
  bool use_locking_;
  bool is_null_input_;
  int num_index_;
  size_t n_stride_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_FTRL_GPU_KERNEL_H_
