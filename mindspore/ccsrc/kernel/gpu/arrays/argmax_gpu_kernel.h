/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_ARGMAXGPUKERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_ARGMAXGPUKERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/argmax_impl.cuh"
namespace mindspore {
namespace kernel {
#define ARGMAX_MAX_DIMENSION 2
template <typename T>
class ArgmaxGpuKernel : public GpuKernel {
 public:
  ArgmaxGpuKernel() : input_size_(0), output_size_(0), workspace_size_(0), batch_size_(0), channel_size_(0), axis_(0) {}
  ~ArgmaxGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    int *output = GetDeviceAddress<int>(outputs, 0);
    CalArgmax(input, SizeToInt(batch_size_), SizeToInt(channel_size_), axis_, output,
              reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but argmax needs 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but argmax needs 1 output.";
      return false;
    }
    auto output_type = GetValue<TypePtr>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("output_type"));
    if (output_type->type_id() != TypeId::kNumberTypeInt32) {
      MS_LOG(EXCEPTION) << "Argmax only supports int32 output type.";
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (input_shape.size() > ARGMAX_MAX_DIMENSION) {
      MS_LOG(EXCEPTION) << "Input is " << input_shape.size() << "-D, but argmax supports max " << ARGMAX_MAX_DIMENSION
                        << "-D inputs.";
    }

    axis_ = GetAttr<int>(kernel_node, "axis");
    if (axis_ < 0) {
      axis_ += SizeToInt(input_shape.size());
    }
    if (input_shape.size() == 1) {
      batch_size_ = 0;
      channel_size_ = input_shape[0];
      input_size_ = sizeof(T) * channel_size_;
      output_size_ = sizeof(int);
    } else {
      batch_size_ = input_shape[0];
      channel_size_ = input_shape[1];
      input_size_ = sizeof(T) * batch_size_ * channel_size_;
      output_size_ = (axis_ == 1) ? sizeof(int) * batch_size_ : sizeof(int) * channel_size_;
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  size_t batch_size_;
  size_t channel_size_;
  int axis_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_ARGMAXGPUKERNEL_H_
