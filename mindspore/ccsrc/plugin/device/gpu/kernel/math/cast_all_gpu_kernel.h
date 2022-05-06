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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CAST_ALL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CAST_ALL_GPU_KERNEL_H_

#include <memory>
#include <vector>
#include <map>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cast_all_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T, typename S>
class CastAllFwdGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  CastAllFwdGpuKernelMod() : max_(0), input_size_(0), output_size_(0), num_input_(0), is_null_input_(false) {}
  ~CastAllFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto in_addr = std::make_unique<T *[]>(num_input_);
    auto out_addr = std::make_unique<S *[]>(num_input_);
    for (size_t i = 0; i < num_input_; i++) {
      in_addr[i] = GetDeviceAddress<T>(inputs, i);
      out_addr[i] = GetDeviceAddress<S>(outputs, i);
    }
    T **inputs_dev = GetDeviceAddress<T *>(workspace, 0);
    S **outputs_dev = GetDeviceAddress<S *>(workspace, 1);
    size_t *size_dev = GetDeviceAddress<size_t>(workspace, 2);
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemcpyAsync(inputs_dev, in_addr.get(), sizeof(T *) * num_input_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemcpyAsync(outputs_dev, out_addr.get(), sizeof(S *) * num_input_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_, cudaMemcpyAsync(size_dev, size_.get(), sizeof(size_t) * num_input_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CastAllKernel(inputs_dev, outputs_dev, max_, num_input_, size_dev, stream);
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    num_input_ = GetAttr<size_t>(kernel_node, "n");
    size_ = std::make_unique<size_t[]>(num_input_);
    for (size_t i = 0; i < num_input_; i++) {
      auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i);
      is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name, "input");
      if (is_null_input_) {
        InitSizeLists();
        return true;
      }
      size_t s = SizeOf(shape);
      if (max_ < s) {
        max_ = s;
      }
      size_[i] = s;
      input_size_ = sizeof(T) * s;
      output_size_ = sizeof(S) * s;
      InitSizeLists();
    }
    workspace_size_list_.push_back(sizeof(T *) * num_input_);
    workspace_size_list_.push_back(sizeof(S *) * num_input_);
    workspace_size_list_.push_back(sizeof(size_t) * num_input_);
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

  std::unique_ptr<size_t[]> size_;
  size_t max_;
  size_t input_size_;
  size_t output_size_;
  size_t num_input_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CAST_ALL_GPU_KERNEL_H_
