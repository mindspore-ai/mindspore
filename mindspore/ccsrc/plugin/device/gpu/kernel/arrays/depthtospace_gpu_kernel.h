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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DEPTHTOSPACE_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DEPTHTOSPACE_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/depthtospace_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class DepthToSpaceFwdKernelMod : public NativeGpuKernelMod {
 public:
  DepthToSpaceFwdKernelMod() { ResetResource(); }
  ~DepthToSpaceFwdKernelMod() = default;

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
    CalDepthToSpace(size, input, in_, ic_, ih_, iw_, on_, oc_, oh_, ow_, block_size_, output,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    block_size_ = static_cast<int64_t>(GetAttr<int64_t>(kernel_node, "block_size"));
    if (block_size_ < 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the 'block_size' cannot be less than 2, but got "
                        << block_size_;
    }
    // check input num and output num
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 1, but got " << input_num;
    }

    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }
    // check input_shape
    auto input_shape = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    shape_size_ = input_shape.size();
    if (shape_size_ != DEPTHTOSPACE_BUFFER_DIMENSION) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of input should be "
                        << DEPTHTOSPACE_BUFFER_DIMENSION << ", but got " << shape_size_;
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
    oc_ = ic_ / block_size_ / block_size_;
    oh_ = ih_ * block_size_;
    ow_ = iw_ * block_size_;

    // Private members Initialize
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    shape_size_ = 0;
    input_size_ = 0;
    output_size_ = 0;
    block_size_ = 0;
    in_ = 0;
    ic_ = 0;
    ih_ = 0;
    iw_ = 0;
    on_ = 0;
    oc_ = 0;
    oh_ = 0;
    ow_ = 0;
    is_null_input_ = false;

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
  size_t shape_size_;
  size_t input_size_;
  size_t output_size_;
  size_t block_size_;
  size_t in_;
  size_t ic_;
  size_t ih_;
  size_t iw_;
  size_t on_;
  size_t oc_;
  size_t oh_;
  size_t ow_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DEPTHTOSPACE_KERNEL_H_
