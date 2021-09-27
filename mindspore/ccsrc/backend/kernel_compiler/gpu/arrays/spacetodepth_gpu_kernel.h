
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPACETODEPTH_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPACETODEPTH_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/spacetodepth_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class SpaceToDepthFwdKernel : public GpuKernel {
 public:
  SpaceToDepthFwdKernel() { ResetResource(); }
  ~SpaceToDepthFwdKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    // get device buffer ptr
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);

    // get input size
    size_t size = input_size_ / sizeof(T);

    // call cuda kernel
    CalSpaceToDepth(size, input, in_, ic_, ih_, iw_, on_, oc_, oh_, ow_, block_size_, output,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    block_size_ = static_cast<int64_t>(GetAttr<int64_t>(kernel_node, "block_size"));
    if (block_size_ == 0) {
      MS_LOG(ERROR) << "block_size_ can not be 0.";
      return false;
    }
    // check input num and output num
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but SpaceToDepth needs 1 input.";
      return false;
    }

    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", SpaceToDepth needs 1 output.";
      return false;
    }
    // check input_shape
    auto input_shape = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'SpaceToDepthGpuKernel', input is null.";
      InitSizeLists();
      return true;
    }
    shape_size_ = input_shape.size();
    if (shape_size_ != SPACETODEPTH_BUFFER_DIMENSION) {
      MS_LOG(EXCEPTION) << "Input is " << shape_size_ << "-D, but SpaceToDepth supports 4-D tensor.";
    }
    // get input and out put information
    input_size_ = 1;
    for (size_t i = 0; i < shape_size_; i++) {
      input_size_ *= input_shape[i];
    }
    input_size_ *= sizeof(T);
    output_size_ = input_size_;

    in_ = input_shape[0];
    ic_ = input_shape[1];
    ih_ = input_shape[2];
    iw_ = input_shape[3];

    on_ = in_;
    oc_ = ic_ * block_size_ * block_size_;
    oh_ = ih_ / block_size_;
    ow_ = iw_ / block_size_;
    // Private members Initialize
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    shape_size_ = 0;
    input_size_ = 0;
    output_size_ = 0;
    block_size_ = 0;
    is_null_input_ = false;
    in_ = 0;
    ic_ = 0;
    ih_ = 0;
    iw_ = 0;
    on_ = 0;
    oc_ = 0;
    oh_ = 0;
    ow_ = 0;

    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    return;
  }

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  size_t shape_size_;
  size_t input_size_;
  size_t output_size_;
  size_t block_size_;
  bool is_null_input_;
  size_t in_;
  size_t ic_;
  size_t ih_;
  size_t iw_;
  size_t on_;
  size_t oc_;
  size_t oh_;
  size_t ow_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPACETODEPTH_KERNEL__H_
