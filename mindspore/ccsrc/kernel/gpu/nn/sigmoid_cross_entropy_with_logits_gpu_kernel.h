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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GPU_KERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/sigmoid_cross_entropy_with_logits_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class SigmoidCrossEntropyWithLogitsGpuKernel : public GpuKernel {
 public:
  SigmoidCrossEntropyWithLogitsGpuKernel() : logits_size_(0), labels_size_(0), outputs_size_(0) {}

  ~SigmoidCrossEntropyWithLogitsGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *logits_addr = GetDeviceAddress<T>(inputs, 0);
    S *labels_addr = GetDeviceAddress<S>(inputs, 1);
    T *outputs_addr = GetDeviceAddress<T>(outputs, 0);

    SigmoidCrossEntropyWithLogits(inputs[0]->size / sizeof(T), logits_addr, labels_addr, outputs_addr,
                                  reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but SigmoidCrossEntropyWithLogits needs 2 inputs.";
      return false;
    }
    logits_size_ = sizeof(T);
    labels_size_ = sizeof(S);
    outputs_size_ = sizeof(T);

    auto logits_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < logits_shape.size(); i++) {
      logits_size_ *= logits_shape[i];
    }

    auto labels_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    for (size_t i = 0; i < labels_shape.size(); i++) {
      labels_size_ *= labels_shape[i];
    }

    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < output_shape.size(); i++) {
      outputs_size_ *= output_shape[i];
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(logits_size_);
    input_size_list_.push_back(labels_size_);
    output_size_list_.push_back(outputs_size_);
  }

 private:
  size_t logits_size_;
  size_t labels_size_;
  size_t outputs_size_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GPU_KERNEL_H_
