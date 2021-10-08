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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ADAM_WEIGHT_DECAY_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ADAM_WEIGHT_DECAY_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/adam_impl.cuh"
namespace mindspore {
namespace kernel {
constexpr size_t INPUT_NUM = 9;
template <typename T>
class AdamWeightDecayGpuKernel : public GpuKernel {
 public:
  AdamWeightDecayGpuKernel()
      : variable_size_(0),
        m_size_(0),
        v_size_(0),
        learning_rate_size_(0),
        beta1_size_(0),
        beta2_size_(0),
        epsilon_size_(0),
        decay_size_(0),
        gradient_size_(0),
        is_null_input_(false) {}

  ~AdamWeightDecayGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *variable = GetDeviceAddress<T>(inputs, 0);
    T *m = GetDeviceAddress<T>(inputs, 1);
    T *v = GetDeviceAddress<T>(inputs, 2);
    float *lr = GetDeviceAddress<float>(inputs, 3);
    float *beta1 = GetDeviceAddress<float>(inputs, 4);
    float *beta2 = GetDeviceAddress<float>(inputs, 5);
    float *epsilon = GetDeviceAddress<float>(inputs, 6);
    float *decay = GetDeviceAddress<float>(inputs, 7);
    T *gradient = GetDeviceAddress<T>(inputs, 8);
    AdamWeightDecayOp(inputs[0]->size / sizeof(T), gradient, lr, beta1, beta2, epsilon, decay, variable, m, v,
                      reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != INPUT_NUM) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but adam needs " << INPUT_NUM << " inputs.";
      return false;
    }

    variable_size_ = sizeof(T);
    m_size_ = sizeof(T);
    v_size_ = sizeof(T);
    learning_rate_size_ = sizeof(float);
    beta1_size_ = sizeof(float);
    beta2_size_ = sizeof(float);
    epsilon_size_ = sizeof(float);
    decay_size_ = sizeof(float);
    gradient_size_ = sizeof(T);

    auto variable_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto m_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto v_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto gradient_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 8);
    is_null_input_ = CHECK_NULL_INPUT(variable_shape) || CHECK_NULL_INPUT(m_shape) || CHECK_NULL_INPUT(v_shape) ||
                     CHECK_NULL_INPUT(gradient_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'AdamWeightDecayGpuKernel', input is null";
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < variable_shape.size(); i++) {
      variable_size_ *= variable_shape[i];
    }

    for (size_t i = 0; i < m_shape.size(); i++) {
      m_size_ *= m_shape[i];
    }

    for (size_t i = 0; i < v_shape.size(); i++) {
      v_size_ *= v_shape[i];
    }

    for (size_t i = 0; i < gradient_shape.size(); i++) {
      gradient_size_ *= gradient_shape[i];
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(variable_size_);
    input_size_list_.push_back(m_size_);
    input_size_list_.push_back(v_size_);
    input_size_list_.push_back(learning_rate_size_);
    input_size_list_.push_back(beta1_size_);
    input_size_list_.push_back(beta2_size_);
    input_size_list_.push_back(epsilon_size_);
    input_size_list_.push_back(decay_size_);
    input_size_list_.push_back(gradient_size_);
    output_size_list_.push_back(0);
    output_size_list_.push_back(0);
    output_size_list_.push_back(0);
  }

 private:
  size_t variable_size_;
  size_t m_size_;
  size_t v_size_;
  size_t learning_rate_size_;
  size_t beta1_size_;
  size_t beta2_size_;
  size_t epsilon_size_;
  size_t decay_size_;
  size_t gradient_size_;
  bool is_null_input_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ADAM_WEIGHT_DECAY_GPU_KERNEL_H_
