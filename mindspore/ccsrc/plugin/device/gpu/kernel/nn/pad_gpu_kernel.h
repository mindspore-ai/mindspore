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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PAD_GPU_KERNEL_H_

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/pad_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/slice_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class PadFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  PadFwdGpuKernelMod() { ResetResource(); }
  ~PadFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_device = GetDeviceAddress<T>(inputs, 0);
    T *output_device = GetDeviceAddress<T>(outputs, 0);

    float pad_value = 0.0;
    FillDeviceArray(output_size_, output_device, pad_value, reinterpret_cast<cudaStream_t>(stream_ptr));

    size_t *input_shape_device = GetDeviceAddress<size_t>(workspace, 0);
    size_t *strides_device = GetDeviceAddress<size_t>(workspace, 1);
    int32_t *paddings_device = GetDeviceAddress<int32_t>(workspace, 2);

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(input_shape_device, &input_shape_[0], workspace_size_list_[0],
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync for input_shape_ failed");

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(strides_device, &strides_[0], workspace_size_list_[1],
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync for strides_ failed");

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(paddings_device, &flattened_paddings_[0], workspace_size_list_[2],
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync for paddings_ failed");

    CalPadGeneral(input_device, output_device, input_shape_device, strides_device, paddings_device, input_size_,
                  input_rank_, reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    (void)CheckIONumber(kernel_node);

    input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    std::vector<size_t> output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shape_, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    input_rank_ = input_shape_.size();

    auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    std::vector<std::vector<int64_t>> paddings = GetValue<std::vector<std::vector<int64_t>>>(prim->GetAttr("paddings"));
    if (paddings.size() != input_rank_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'paddings' should be equal to the dimension of "
                        << "input, but got the length of 'paddings': " << paddings.size()
                        << " the dimension of input: " << input_rank_;
    }

    for (size_t i = 0; i < paddings.size(); i++) {
      if (paddings[i].size() != 2) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of element of 'paddings' should be equal to 2, "
                          << "but got the size of paddings[" << i << "]: " << paddings[i].size();
      }
      flattened_paddings_.push_back(paddings[i][0]);
      flattened_paddings_.push_back(paddings[i][1]);
    }

    input_size_ = 1;
    output_size_ = 1;
    for (size_t i = 0; i < input_rank_; i++) {
      input_size_ *= input_shape_[i];
      output_size_ *= (input_shape_[i] + flattened_paddings_[2 * i] + flattened_paddings_[(2 * i) + 1]);
    }

    if (input_rank_ == 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be equal to 0, but "
                        << "got the " << input_rank_;
    }
    if (output_shape.size() != input_rank_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input and output should be the same, but "
                        << "got the dimension of input: " << input_rank_
                        << ", the dimension of output: " << output_shape.size();
    }
    strides_.resize(input_rank_);
    strides_[input_rank_ - 1] = 1;
    for (int32_t i = input_rank_ - 2; i >= 0; i--) {
      strides_[i] = output_shape[i + 1] * strides_[i + 1];
    }

    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_rank_ = 0;
    input_size_ = 0;
    output_size_ = 0;
    workspace_size_ = 0;
    is_null_input_ = false;
    kernel_name_ = "Pad";
    flattened_paddings_.clear();
    input_shape_.clear();
    strides_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(output_size_ * sizeof(T));
    workspace_size_list_.push_back(input_rank_ * sizeof(size_t));       // input shape
    workspace_size_list_.push_back(input_rank_ * sizeof(size_t));       // strides
    workspace_size_list_.push_back(input_rank_ * sizeof(int32_t) * 2);  // paddings
  }

 private:
  void CheckIONumber(const CNodePtr &kernel_node) {
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
  }

  size_t input_rank_;
  std::vector<int32_t> flattened_paddings_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> strides_;

  // default
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  bool is_null_input_;
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PAD_GPU_KERNEL_H_
