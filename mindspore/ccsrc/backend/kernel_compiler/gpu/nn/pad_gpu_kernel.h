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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PAD_GPU_KERNEL_H_

#include <iostream>
#include <vector>
#include <algorithm>

#include "backend/kernel_compiler/gpu/cuda_impl/pad_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/slice_impl.cuh"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class PadGpuFwdKernel : public GpuKernel {
 public:
  PadGpuFwdKernel() { ResetResource(); }
  ~PadGpuFwdKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_device = GetDeviceAddress<T>(inputs, 0);
    T *output_device = GetDeviceAddress<T>(outputs, 0);

    const float pad_value = 0.0;
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
    if (!CheckIONumber(kernel_node)) {
      return false;
    }

    input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    std::vector<size_t> output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape_) || CHECK_NULL_INPUT(output_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'PadGpuKernel', input or output is null.";
      InitSizeLists();
      return true;
    }
    input_rank_ = input_shape_.size();

    auto prim = AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    std::vector<std::vector<int64_t>> paddings = GetValue<std::vector<std::vector<int64_t>>>(prim->GetAttr("paddings"));
    if (paddings.size() != input_rank_) {
      MS_LOG(EXCEPTION) << "PadGpuFwdKernel: paddings' size must be equal to the rank of the input.";
    }

    for (size_t i = 0; i < paddings.size(); i++) {
      if (paddings[i].size() != 2) {
        MS_LOG(EXCEPTION) << "PadGpuFwdKernel: each element in paddings must have size 2.";
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

    if (input_rank_ < 1) {
      MS_LOG(EXCEPTION) << "For 'PadGpuKernel', the rank of input should be greater than or equal to 1, "
                        << "but got the rank of input: " << input_rank_;
    }
    if (output_shape.size() != input_rank_) {
      MS_LOG(EXCEPTION) << "For 'PadGpuKernel', the rank of input should be equal to the rank of output, "
                        << "but got the rank of input: " << input_rank_
                        << ", the rank of output: " << output_shape.size();
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
  bool CheckIONumber(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but Pad needs 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but Pad needs 1 output.";
      return false;
    }
    return true;
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
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PAD_GPU_KERNEL_H_
