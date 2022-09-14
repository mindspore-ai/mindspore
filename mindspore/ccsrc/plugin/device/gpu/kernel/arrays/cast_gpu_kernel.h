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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CAST_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CAST_GPU_KERNEL_H_

#include <vector>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cast_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename S, typename T>
class CastGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  CastGpuKernelMod() { ResetResource(); }
  ~CastGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    S *input_addr = GetPossiblyNullDeviceAddress<S>(inputs, 0);
    T *output_addr = GetPossiblyNullDeviceAddress<T>(outputs, 0);

    if (input_addr == nullptr && output_addr == nullptr) {
      return true;
    } else if (input_addr != nullptr && output_addr != nullptr) {
      Cast(input_size_, input_addr, output_addr, reinterpret_cast<cudaStream_t>(stream_ptr), GET_CTX_DEVICE_ID);
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the input and output device addresses must be both null or both not null";
    }

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    auto input_shapes = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shapes = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    kernel_node_ = kernel_node;
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shapes, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shapes, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    input_size_ = 1;
    for (size_t i = 0; i < input_shapes.size(); i++) {
      input_size_ *= input_shapes[i];
    }

    output_size_ = 1;
    for (size_t j = 0; j < output_shapes.size(); j++) {
      output_size_ *= output_shapes[j];
    }

    if (input_size_ != output_size_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the size of input and output must be the same, but got the size of input: "
                        << input_size_ << ", the size of output: " << output_size_;
    }
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 1;
    output_size_ = 1;
    is_null_input_ = false;
    kernel_name_ = "Cast";
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(output_size_ * sizeof(T));
  }

 private:
  int64_t input_size_;
  int64_t output_size_;
  bool is_null_input_;

  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CAST_GPU_KERNEL_H_
