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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_RMSPROP_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_RMSPROP_KERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/rmsprop_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class RMSPropGpuKernel : public GpuKernel {
 public:
  RMSPropGpuKernel() : size_(1), use_center_(false), decay_(0.0), momentum_(0.9), epsilon_(1e-12) {}
  ~RMSPropGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream) override {
    if (!use_center_) {
      T *variable = GetDeviceAddress<T>(inputs, 0);
      T *mean_square = GetDeviceAddress<T>(inputs, 1);
      T *moment = GetDeviceAddress<T>(inputs, 2);
      T *learning_rate = GetDeviceAddress<T>(inputs, 3);
      T *gradients = GetDeviceAddress<T>(inputs, 4);

      RmsProp(learning_rate, decay_, momentum_, epsilon_, variable, mean_square, moment, gradients, size_,
              reinterpret_cast<cudaStream_t>(stream));
    } else {
      T *variable = GetDeviceAddress<T>(inputs, 0);
      T *mean_gradients = GetDeviceAddress<T>(inputs, 1);
      T *mean_square = GetDeviceAddress<T>(inputs, 2);
      T *moment = GetDeviceAddress<T>(inputs, 3);
      T *gradients = GetDeviceAddress<T>(inputs, 4);
      T *learning_rate = GetDeviceAddress<T>(inputs, 5);
      T *decay = GetDeviceAddress<T>(inputs, 6);
      T *momentum = GetDeviceAddress<T>(inputs, 7);
      T *epsilon = GetDeviceAddress<T>(inputs, 8);

      RmsPropCenter(learning_rate, decay, momentum, epsilon, variable, mean_gradients, mean_square, moment, gradients,
                    size_, reinterpret_cast<cudaStream_t>(stream));
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    auto node_name = AnfAlgo::GetCNodeName(kernel_node);
    if (node_name == "ApplyCenteredRMSProp") {
      use_center_ = true;
    }

    if (node_name == "ApplyRMSProp") {
      decay_ = GetAttr<float>(kernel_node, "rho");
      momentum_ = GetAttr<float>(kernel_node, "momentum");
      epsilon_ = GetAttr<float>(kernel_node, "epsilon");
    }
    auto input_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    for (auto &dim : input_shape) {
      size_ *= dim;
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    size_t input_size = size_ * sizeof(T);
    if (!use_center_) {
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(sizeof(T));
      input_size_list_.push_back(input_size);
      output_size_list_.push_back(input_size);
    } else {
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(sizeof(T));
      input_size_list_.push_back(sizeof(T));
      input_size_list_.push_back(sizeof(T));
      input_size_list_.push_back(sizeof(T));
      output_size_list_.push_back(input_size);
    }
  }

 private:
  size_t size_;
  bool use_center_;
  float decay_;
  float momentum_;
  float epsilon_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif
