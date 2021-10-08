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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_SGD_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_SGD_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/cuda_impl/sgd_impl.cuh"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class SGDGpuKernel : public GpuKernel {
 public:
  SGDGpuKernel() : size_(1), dampening_(0.0), weight_decay_(0.0), nesterov_(false), is_null_input_(false) {}
  ~SGDGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream) override {
    if (is_null_input_) {
      return true;
    }
    T *param = GetDeviceAddress<T>(inputs, 0);
    T *grad = GetDeviceAddress<T>(inputs, 1);
    T *lr = GetDeviceAddress<T>(inputs, 2);
    T *accum = GetDeviceAddress<T>(inputs, 3);
    T *momentum = GetDeviceAddress<T>(inputs, 4);
    T *stat = GetDeviceAddress<T>(inputs, 5);
    T *output_param = GetDeviceAddress<T>(outputs, 0);

    SGD(size_, dampening_, weight_decay_, nesterov_, lr, momentum, grad, param, accum, stat,
        reinterpret_cast<cudaStream_t>(stream));
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(output_param, param, sizeof(T) * size_, cudaMemcpyDeviceToDevice,
                                               reinterpret_cast<cudaStream_t>(stream)),
                               "SGD cudaMemcpyAsync params to outputs failed");
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    dampening_ = GetAttr<float>(kernel_node, "dampening");
    weight_decay_ = GetAttr<float>(kernel_node, "weight_decay");
    nesterov_ = GetAttr<bool>(kernel_node, "nesterov");

    auto input_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'SGDGpuKernel', input is null.";
      InitSizeLists();
      return true;
    }
    for (auto &dim : input_shape) {
      size_ *= dim;
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    size_t input_size = size_ * sizeof(T);
    input_size_list_.push_back(input_size);  // parameter
    input_size_list_.push_back(input_size);  // gradient
    input_size_list_.push_back(sizeof(T));   // lr
    input_size_list_.push_back(input_size);  // accum
    input_size_list_.push_back(sizeof(T));   // momentum
    input_size_list_.push_back(input_size);  // stat
    output_size_list_.push_back(input_size);
  }

 private:
  size_t size_;
  float dampening_;
  float weight_decay_;
  bool nesterov_;
  bool is_null_input_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_SGD_KERNEL_H_
