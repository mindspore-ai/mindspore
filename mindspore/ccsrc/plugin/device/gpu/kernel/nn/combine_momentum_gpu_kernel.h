/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <string>
#include <memory>
#include "ops/nn_optimizer_op_name.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/momentum_impl.cuh"
namespace mindspore {
namespace kernel {
constexpr int kCombineMomentumInputsNum = 5;
constexpr int kCombineScaleMomentumInputsNum = 6;
constexpr int kCombineWeightDecayMomentumInputsNum = 7;
template <typename T, typename S, typename G>
class CombineMomentumGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  CombineMomentumGpuKernelMod()
      : element_num_(1), combine_num_(0), input_num_(0), is_null_input_(false), kernel_name_("CombineMomentum") {}
  ~CombineMomentumGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    cudaError_t status = cudaErrorNotReady;
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    for (size_t i = 0; i < combine_num_; i++) {
      if (input_num_ == kCombineMomentumInputsNum) {
        T *variable = GetDeviceAddress<T>(inputs, i * input_num_);
        T *acc = GetDeviceAddress<T>(inputs, i * input_num_ + 1);
        S *lr = GetDeviceAddress<S>(inputs, i * input_num_ + 2);
        G *grad = GetDeviceAddress<G>(inputs, i * input_num_ + 3);
        S *mom = GetDeviceAddress<S>(inputs, i * input_num_ + 4);
        status = MomentumUpdateVariable(elements_[i], variable, acc, lr, grad, mom, false, stream);
      } else if (input_num_ == kCombineScaleMomentumInputsNum) {
        S *scale = GetDeviceAddress<S>(inputs, i * input_num_);
        T *variable = GetDeviceAddress<T>(inputs, i * input_num_ + 1);
        T *acc = GetDeviceAddress<T>(inputs, i * input_num_ + 2);
        S *lr = GetDeviceAddress<S>(inputs, i * input_num_ + 3);
        G *grad = GetDeviceAddress<G>(inputs, i * input_num_ + 4);
        S *mom = GetDeviceAddress<S>(inputs, i * input_num_ + 5);
        status = FusedScaleMomentum(elements_[i], scale, variable, acc, lr, grad, mom, stream);
      } else if (input_num_ == kCombineWeightDecayMomentumInputsNum) {
        S *weight_decay = GetDeviceAddress<S>(inputs, i * input_num_);
        S *scale = GetDeviceAddress<S>(inputs, i * input_num_ + 1);
        T *variable = GetDeviceAddress<T>(inputs, i * input_num_ + 2);
        T *acc = GetDeviceAddress<T>(inputs, i * input_num_ + 3);
        S *lr = GetDeviceAddress<S>(inputs, i * input_num_ + 4);
        G *grad = GetDeviceAddress<G>(inputs, i * input_num_ + 5);
        S *mom = GetDeviceAddress<S>(inputs, i * input_num_ + 6);
        status = FusedWeightDecayScaleMomentum(elements_[i], weight_decay, scale, variable, acc, lr, grad, mom, stream);
      } else {
        MS_LOG(EXCEPTION) << "Combine kernel input num is invalid.";
      }
      CHECK_CUDA_STATUS(status, kernel_name_);
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    combine_num_ = GetAttr<size_t>(kernel_node, "combine_num");
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    if (kernel_name == kCombineMomentumOpName) {
      input_num_ = kCombineMomentumInputsNum;
    } else if (kernel_name == kCombineScaleMomentumOpName) {
      input_num_ = kCombineScaleMomentumInputsNum;
    } else if (kernel_name == kCombineWeightDecayScaleMomentumOpName) {
      input_num_ = kCombineWeightDecayMomentumInputsNum;
    } else {
      MS_LOG(EXCEPTION) << "Combine kernel name is invalid.";
    }
    for (size_t i = 0; i < combine_num_; i++) {
      auto variable_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i * input_num_ + input_num_ - 5);
      is_null_input_ = CHECK_SHAPE_NULL(variable_shape, kernel_name_,
                                        "input[" + std::to_string(i * input_num_ + input_num_ - 5) + "]");
      if (is_null_input_ || IsDynamic(variable_shape)) {
        InitSizeLists();
        return true;
      }
      element_num_ = SizeOf(variable_shape);
      elements_.push_back(element_num_);
      InitSizeLists();
    }
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
  size_t element_num_;
  std::vector<size_t> elements_;
  size_t combine_num_;
  int input_num_;
  bool is_null_input_;
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_SCALE_MOMENTUM_GPU_KERNEL_H_
