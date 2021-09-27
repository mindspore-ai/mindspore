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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BCE_WITH_LOGITS_LOSS_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BCE_WITH_LOGITS_LOSS_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/bce_with_logits_loss_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class BCEWithLogitsLossKernel : public GpuKernel {
 public:
  BCEWithLogitsLossKernel() { ResetResource(); }
  ~BCEWithLogitsLossKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *predict = GetDeviceAddress<T>(inputs, 0);
    T *target = GetDeviceAddress<T>(inputs, 1);
    T *weight = GetDeviceAddress<T>(inputs, 2);
    T *pos_weight = GetDeviceAddress<T>(inputs, 3);
    size_t *input_shape = GetDeviceAddress<size_t>(workspace, 0);
    size_t *weight_shape = GetDeviceAddress<size_t>(workspace, 1);
    size_t *pos_weight_shape = GetDeviceAddress<size_t>(workspace, 2);
    T *shape_broadcasted = GetDeviceAddress<T>(workspace, 3);
    T *output = GetDeviceAddress<T>(outputs, 0);
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(input_shape, &input_shape_[0], input_shape_.size() * sizeof(size_t),
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync input_shape_ failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(weight_shape, &weight_shape_[0], weight_shape_.size() * sizeof(size_t),
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync weight_shape_ failed");
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemcpyAsync(pos_weight_shape, &pos_weight_shape_[0], pos_weight_shape_.size() * sizeof(size_t),
                      cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync pos_weight_shape_ failed");
    CalBCEWithLogitsLoss(input_size_, predict, target, input_shape, input_shape_.size(), weight, weight_shape,
                         weight_need_broadcast_, pos_weight, pos_weight_shape, pos_weight_need_broadcast_,
                         shape_broadcasted, output, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 4) {
      MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but BCEWithLogitsLoss needs 4 inputs.";
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but BCEWithLogitsLoss has 1 output.";
    }
    input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    weight_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    pos_weight_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    is_null_input_ =
      CHECK_NULL_INPUT(input_shape_) || CHECK_NULL_INPUT(weight_shape_) || CHECK_NULL_INPUT(pos_weight_shape_);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'BCEWithLogitsLossGpuKernel', input is null.";
      InitSizeLists();
      return true;
    }
    if (input_shape_.size() < 1) {
      MS_LOG(EXCEPTION) << "For 'BCEWithLogitsLossGpuKernel', the rank of input cannot be less than 1, but got "
                        << input_shape_.size();
    }
    if (weight_shape_.size() < 1) {
      MS_LOG(EXCEPTION) << "For 'BCEWithLogitsLossGpuKernel', the rank of weight cannot be less than 1, but got "
                        << weight_shape_.size();
    }
    if (pos_weight_shape_.size() < 1) {
      MS_LOG(EXCEPTION) << "For 'BCEWithLogitsLossGpuKernel', the rank of pos_weight cannot be less than 1, but got "
                        << pos_weight_shape_.size();
    }
    input_size_ = 1;
    if (input_shape_.size() > MAX_LOGITS_DIMENSION) {
      MS_LOG(EXCEPTION) << "Input dimension is " << input_shape_.size()
                        << ", but BCEWithLogitsLoss can only support up to " << MAX_LOGITS_DIMENSION << "-D.";
    }
    for (size_t i = 0; i < input_shape_.size(); i++) {
      input_size_ *= input_shape_[i];
    }
    // weight shape
    weight_size_ = 1;
    for (size_t i = 0; i < weight_shape_.size(); i++) {
      weight_size_ *= weight_shape_[i];
    }
    weight_need_broadcast_ = NeedBroadcast(&weight_shape_, input_shape_);
    // pos_weight shape
    pos_weight_size_ = 1;
    for (size_t i = 0; i < pos_weight_shape_.size(); i++) {
      pos_weight_size_ *= pos_weight_shape_[i];
    }
    pos_weight_need_broadcast_ = NeedBroadcast(&pos_weight_shape_, input_shape_);
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 1;
    weight_size_ = 1;
    pos_weight_size_ = 1;
    weight_need_broadcast_ = false;
    pos_weight_need_broadcast_ = false;
    is_null_input_ = false;
    input_shape_.clear();
    weight_shape_.clear();
    pos_weight_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    input_size_list_.push_back(input_size_ * sizeof(T));
    input_size_list_.push_back(weight_size_ * sizeof(T));
    input_size_list_.push_back(pos_weight_size_ * sizeof(T));
    workspace_size_list_.push_back(input_shape_.size() * sizeof(size_t));
    workspace_size_list_.push_back(weight_shape_.size() * sizeof(size_t));
    workspace_size_list_.push_back(pos_weight_shape_.size() * sizeof(size_t));
    // extra space for holding extra array shape of input, for broadcasted
    // weight and pos_weight
    workspace_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(input_size_ * sizeof(T));
  }

 private:
  bool NeedBroadcast(std::vector<size_t> *shape, const std::vector<size_t> &result_shape) {
    // result_shape is larger that shape
    // and shape is able to broadcasted to result_shape
    if (shape->size() < result_shape.size()) {
      size_t fill_size = result_shape.size() - shape->size();
      (void)shape->insert(shape->begin(), fill_size, 1);
      return true;
    }
    for (size_t i = 0; i < result_shape.size(); i++) {
      if (shape->at(i) != result_shape[i]) {
        return true;
      }
    }
    return false;
  }

  size_t input_size_;
  size_t weight_size_;
  size_t pos_weight_size_;
  bool weight_need_broadcast_;
  bool pos_weight_need_broadcast_;
  bool is_null_input_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> weight_shape_;
  std::vector<size_t> pos_weight_shape_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BCE_WITH_LOGITS_LOSS_KERNEL_H_
