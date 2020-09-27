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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_SCALE_MOMENTUM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_SCALE_MOMENTUM_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/momentum_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T, typename S>
class CombineMomentumGpuKernel : public GpuKernel {
 public:
  CombineMomentumGpuKernel() : element_num_(1), num_(0), max_(0), input_num_(6) {}
  ~CombineMomentumGpuKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &workspace, void *stream_ptr) override {
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto weight_decay = std::make_unique<T *[]>(input_num_ * num_);
    auto scale = std::make_unique<T *[]>(input_num_ * num_);
    auto variable = std::make_unique<T *[]>(input_num_ * num_);
    auto accumulation = std::make_unique<T *[]>(input_num_ * num_);
    auto learning_rate = std::make_unique<T *[]>(input_num_ * num_);
    auto gradient = std::make_unique<S *[]>(input_num_ * num_);
    auto momentum = std::make_unique<T *[]>(input_num_ * num_);
    if (input_num_ == 6) {
      LaunchCombineMom(inputs, workspace, stream, scale, variable, accumulation, learning_rate, gradient, momentum);
    } else {
      LaunchCombineMomWeightDecay(inputs, workspace, stream, weight_decay, scale, variable, accumulation, learning_rate,
                                  gradient, momentum);
    }

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    num_ = GetAttr<size_t>(kernel_node, "n");
    elements_ = std::make_unique<size_t[]>(num_);
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    if (kernel_name == "CombineMomentum") {
      input_num_ = 6;
    } else {
      input_num_ = 7;
      workspace_size_list_.push_back(sizeof(T *) * num_);
    }

    for (size_t i = 0; i < num_; i++) {
      element_num_ = 1;
      auto variable_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i * input_num_ + input_num_ - 4);
      for (size_t j = 0; j < variable_shape.size(); j++) {
        element_num_ *= variable_shape[j];
      }
      if (max_ < element_num_) {
        max_ = element_num_;
      }
      elements_[i] = element_num_;
      InitSizeLists();
    }
    workspace_size_list_.push_back(sizeof(T *) * num_);
    workspace_size_list_.push_back(sizeof(T *) * num_);
    workspace_size_list_.push_back(sizeof(T *) * num_);
    workspace_size_list_.push_back(sizeof(T *) * num_);
    workspace_size_list_.push_back(sizeof(S *) * num_);
    workspace_size_list_.push_back(sizeof(T *) * num_);
    workspace_size_list_.push_back(sizeof(size_t) * num_);
    return true;
  }

 protected:
  void InitSizeLists() override {
    if (input_num_ == 7) {
      input_size_list_.push_back(sizeof(T));
    }
    input_size_list_.push_back(sizeof(T));
    input_size_list_.push_back(element_num_ * sizeof(T));
    input_size_list_.push_back(element_num_ * sizeof(T));
    input_size_list_.push_back(sizeof(T));
    input_size_list_.push_back(element_num_ * sizeof(S));
    input_size_list_.push_back(sizeof(T));
    output_size_list_.push_back(element_num_ * sizeof(T));
  }

 private:
  void LaunchCombineMom(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                        const cudaStream_t &stream, const std::unique_ptr<T *[]> &scale,
                        const std::unique_ptr<T *[]> &variable, const std::unique_ptr<T *[]> &accumulation,
                        const std::unique_ptr<T *[]> &learning_rate, const std::unique_ptr<S *[]> &gradient,
                        const std::unique_ptr<T *[]> &momentum) {
    for (size_t i = 0; i < num_; i++) {
      scale[i] = GetDeviceAddress<T>(inputs, i * input_num_);
      variable[i] = GetDeviceAddress<T>(inputs, i * input_num_ + 1);
      accumulation[i] = GetDeviceAddress<T>(inputs, i * input_num_ + 2);
      learning_rate[i] = GetDeviceAddress<T>(inputs, i * input_num_ + 3);
      gradient[i] = GetDeviceAddress<S>(inputs, i * input_num_ + 4);
      momentum[i] = GetDeviceAddress<T>(inputs, i * input_num_ + 5);
    }
    T **scale_dev = GetDeviceAddress<T *>(workspace, 0);
    T **variable_dev = GetDeviceAddress<T *>(workspace, 1);
    T **accumulation_dev = GetDeviceAddress<T *>(workspace, 2);
    T **learning_rate_dev = GetDeviceAddress<T *>(workspace, 3);
    S **gradient_dev = GetDeviceAddress<S *>(workspace, 4);
    T **momentum_dev = GetDeviceAddress<T *>(workspace, 5);
    size_t *elements_dev = GetDeviceAddress<size_t>(workspace, 6);
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(scale_dev, scale.get(), sizeof(T *) * num_, cudaMemcpyHostToDevice, stream), "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(variable_dev, variable.get(), sizeof(T *) * num_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(accumulation_dev, accumulation.get(), sizeof(T *) * num_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(learning_rate_dev, learning_rate.get(), sizeof(T *) * num_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(gradient_dev, gradient.get(), sizeof(S *) * num_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(momentum_dev, momentum.get(), sizeof(T *) * num_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(elements_dev, elements_.get(), sizeof(size_t) * num_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CombineFusedScaleMomentum(max_, num_, elements_dev, scale_dev, variable_dev, accumulation_dev, learning_rate_dev,
                              gradient_dev, momentum_dev, stream);
  }
  void LaunchCombineMomWeightDecay(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const cudaStream_t &stream, const std::unique_ptr<T *[]> &weight_decay,
                                   const std::unique_ptr<T *[]> &scale, const std::unique_ptr<T *[]> &variable,
                                   const std::unique_ptr<T *[]> &accumulation,
                                   const std::unique_ptr<T *[]> &learning_rate, const std::unique_ptr<S *[]> &gradient,
                                   const std::unique_ptr<T *[]> &momentum) {
    for (size_t i = 0; i < num_; i++) {
      weight_decay[i] = GetDeviceAddress<T>(inputs, i * input_num_);
      scale[i] = GetDeviceAddress<T>(inputs, i * input_num_ + 1);
      variable[i] = GetDeviceAddress<T>(inputs, i * input_num_ + 2);
      accumulation[i] = GetDeviceAddress<T>(inputs, i * input_num_ + 3);
      learning_rate[i] = GetDeviceAddress<T>(inputs, i * input_num_ + 4);
      gradient[i] = GetDeviceAddress<S>(inputs, i * input_num_ + 5);
      momentum[i] = GetDeviceAddress<T>(inputs, i * input_num_ + 6);
    }
    T **weight_decay_dev = GetDeviceAddress<T *>(workspace, 0);
    T **scale_dev = GetDeviceAddress<T *>(workspace, 1);
    T **variable_dev = GetDeviceAddress<T *>(workspace, 2);
    T **accumulation_dev = GetDeviceAddress<T *>(workspace, 3);
    T **learning_rate_dev = GetDeviceAddress<T *>(workspace, 4);
    S **gradient_dev = GetDeviceAddress<S *>(workspace, 5);
    T **momentum_dev = GetDeviceAddress<T *>(workspace, 6);
    size_t *elements_dev = GetDeviceAddress<size_t>(workspace, 7);
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(weight_decay_dev, weight_decay.get(), sizeof(T *) * num_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(scale_dev, scale.get(), sizeof(T *) * num_, cudaMemcpyHostToDevice, stream), "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(variable_dev, variable.get(), sizeof(T *) * num_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(accumulation_dev, accumulation.get(), sizeof(T *) * num_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(learning_rate_dev, learning_rate.get(), sizeof(T *) * num_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(gradient_dev, gradient.get(), sizeof(S *) * num_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(momentum_dev, momentum.get(), sizeof(T *) * num_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(elements_dev, elements_.get(), sizeof(size_t) * num_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CombineFusedWeightDecayScaleMomentum(max_, num_, elements_dev, weight_decay_dev, scale_dev, variable_dev,
                                         accumulation_dev, learning_rate_dev, gradient_dev, momentum_dev, stream);
  }
  size_t element_num_;
  std::unique_ptr<size_t[]> elements_;
  size_t num_;
  size_t max_;
  int input_num_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_SCALE_MOMENTUM_GPU_KERNEL_H_
