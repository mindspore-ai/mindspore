/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bce_with_logits_loss_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class BCEWithLogitsLossKernelMod : public NativeGpuKernelMod {
 public:
  BCEWithLogitsLossKernelMod() { ResetResource(); }
  ~BCEWithLogitsLossKernelMod() override = default;

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
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 4, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
    input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    weight_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    pos_weight_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape_, kernel_name_, "logits") ||
                     CHECK_SHAPE_NULL(weight_shape_, kernel_name_, "weight") ||
                     CHECK_SHAPE_NULL(pos_weight_shape_, kernel_name_, "pos_weight");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    if (input_shape_.size() < 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of logits cannot be less than 1, but got "
                        << input_shape_.size();
    }
    if (weight_shape_.size() < 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of weight cannot be less than 1, but got "
                        << weight_shape_.size();
    }
    if (pos_weight_shape_.size() < 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of pos_weight cannot be less than 1, but got "
                        << pos_weight_shape_.size();
    }
    input_size_ = 1;
    if (input_shape_.size() > MAX_LOGITS_DIMENSION) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of logits cannot be greater than "
                        << MAX_LOGITS_DIMENSION << ", but got " << input_shape_.size();
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
    kernel_name_ = "BCEWithLogitsLoss";
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
  std::string kernel_name_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> weight_shape_;
  std::vector<size_t> pos_weight_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BCE_WITH_LOGITS_LOSS_KERNEL_H_
