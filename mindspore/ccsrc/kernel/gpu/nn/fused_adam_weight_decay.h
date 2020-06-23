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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_FUSED_ADAM_WEIGHT_DECAY_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_FUSED_ADAM_WEIGHT_DECAY_KERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "kernel/gpu/cuda_impl/adam_weight_decay_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class FusedAdamWeightDecayGpuKernel : public GpuKernel {
 public:
  FusedAdamWeightDecayGpuKernel() : element_nums_(0), weight_decay_(false) {}
  ~FusedAdamWeightDecayGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    auto node_name = AnfAlgo::GetCNodeName(kernel_node);
    if (node_name == "AdamWeighDecay") {
      weight_decay_ = true;
    }

    auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 7);
    element_nums_ = 1;
    for (auto i : shape) {
      element_nums_ *= i;
    }

    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    float *beta1 = GetDeviceAddress<float>(inputs, 0);
    float *one_sub_beta1 = GetDeviceAddress<float>(inputs, 1);
    float *beta2 = GetDeviceAddress<float>(inputs, 2);
    float *one_sub_beta2 = GetDeviceAddress<float>(inputs, 3);
    float *epsilon = GetDeviceAddress<float>(inputs, 4);
    float *lr = GetDeviceAddress<float>(inputs, 5);
    T *param = GetDeviceAddress<T>(inputs, 6);
    T *m = GetDeviceAddress<T>(inputs, 7);
    T *v = GetDeviceAddress<T>(inputs, 8);
    T *gradient = GetDeviceAddress<T>(inputs, 9);
    float *weight_decay = nullptr;
    if (weight_decay_) {
      weight_decay = GetDeviceAddress<float>(inputs, 10);
    }
    AdamWeightDecay(element_nums_, true, beta1, one_sub_beta1, beta2, one_sub_beta2, epsilon, lr, weight_decay, m, v,
                    param, gradient, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitResource() override{};
  void InitSizeLists() override {
    input_size_list_.push_back(sizeof(float));
    input_size_list_.push_back(sizeof(float));
    input_size_list_.push_back(sizeof(float));
    input_size_list_.push_back(sizeof(float));
    input_size_list_.push_back(element_nums_ * sizeof(T));
    input_size_list_.push_back(sizeof(float));
    input_size_list_.push_back(sizeof(float));
    input_size_list_.push_back(element_nums_ * sizeof(T));
    if (weight_decay_) {
      input_size_list_.push_back(sizeof(float));
    }
    output_size_list_.push_back(element_nums_ * sizeof(T));
  }

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  int element_nums_;
  bool weight_decay_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_FUSED_ADAM_WEIGHT_DECAY_KERNEL_H_
