/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the
 * "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the
 * License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in
 * writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions
 * and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_APPLY_PROXIMAL_ADAGRAD_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_APPLY_PROXIMAL_ADAGRAD_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/sparse_apply_proximal_adagrad_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t INPUT_NUM = 7;
template <typename T>
class SparseApplyProximalAdagradKernel : public GpuKernel {
 public:
  SparseApplyProximalAdagradKernel() { ResetResource(); }
  ~SparseApplyProximalAdagradKernel() override = default;

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
    T *learning_rate = GetDeviceAddress<T>(inputs, 2);
    T *l1_regularization = GetDeviceAddress<T>(inputs, 3);
    T *l2_regularization = GetDeviceAddress<T>(inputs, 4);
    T *gradient = GetDeviceAddress<T>(inputs, 5);
    int *indices = GetDeviceAddress<int>(inputs, 6);
    T *variable_out = GetDeviceAddress<T>(outputs, 0);
    T *accumulation_out = GetDeviceAddress<T>(outputs, 1);

    CalSparseApplyProximalAdagrad(inputs[0]->size / sizeof(T), indices_size_ / sizeof(int), learning_rate,
                                  l1_regularization, l2_regularization, gradient, indices, variable, accumulation,
                                  variable_out, accumulation_out, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != INPUT_NUM) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but SparseApplyProximalAdagrad needs " << INPUT_NUM
                    << " inputs.";
      return false;
    }

    variable_size_ = sizeof(T);
    accumulation_size_ = sizeof(T);
    learning_rate_size_ = sizeof(T);
    l1_regularization_size_ = sizeof(T);
    l2_regularization_size_ = sizeof(T);
    gradient_size_ = sizeof(T);
    indices_size_ = sizeof(int);

    auto variable_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto accumulation_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto learning_rate_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto gradient_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 5);
    auto indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 6);
    is_null_input_ = CHECK_NULL_INPUT(variable_shape) || CHECK_NULL_INPUT(accumulation_shape) ||
                     CHECK_NULL_INPUT(learning_rate_shape) || CHECK_NULL_INPUT(gradient_shape) ||
                     CHECK_NULL_INPUT(indices_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'SparseApplyProximalAdagradGpuKernel', input is null";
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < variable_shape.size(); i++) {
      variable_size_ *= variable_shape[i];
    }

    for (size_t i = 0; i < accumulation_shape.size(); i++) {
      accumulation_size_ *= accumulation_shape[i];
    }

    for (size_t i = 0; i < learning_rate_shape.size(); i++) {
      learning_rate_size_ *= learning_rate_shape[i];
    }

    for (size_t i = 0; i < gradient_shape.size(); i++) {
      gradient_size_ *= gradient_shape[i];
    }

    for (size_t i = 0; i < indices_shape.size(); i++) {
      indices_size_ *= indices_shape[i];
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(variable_size_);
    input_size_list_.push_back(accumulation_size_);
    input_size_list_.push_back(learning_rate_size_);
    input_size_list_.push_back(l1_regularization_size_);
    input_size_list_.push_back(l2_regularization_size_);
    input_size_list_.push_back(gradient_size_);
    input_size_list_.push_back(indices_size_);
    output_size_list_.push_back(variable_size_);
    output_size_list_.push_back(accumulation_size_);
  }

  void ResetResource() noexcept override {
    is_null_input_ = false;
    variable_size_ = 0;
    accumulation_size_ = 0;
    learning_rate_size_ = 0;
    l1_regularization_size_ = 0;
    l2_regularization_size_ = 0;
    gradient_size_ = 0;
    indices_size_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 private:
  bool is_null_input_;
  size_t variable_size_;
  size_t accumulation_size_;
  size_t learning_rate_size_;
  size_t l1_regularization_size_;
  size_t l2_regularization_size_;
  size_t gradient_size_;
  size_t indices_size_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_APPLY_PROXIMAL_ADAGRAD_KERNEL_H_
