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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RELU_GRAD_V2_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RELU_GRAD_V2_GPU_KERNEL_H_

#include <vector>
#include <algorithm>
#include <functional>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/relu_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class ReluGradV2GpuKernel : public GpuKernel {
 public:
  ReluGradV2GpuKernel() { ResetResource(); }
  ~ReluGradV2GpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    auto dy = GetDeviceAddress<T>(inputs, 0);
    auto mask = GetDeviceAddress<uint32_t>(inputs, 1);
    auto dx = GetDeviceAddress<T>(outputs, 0);

    ReluGradV2(element_num_, dy, mask, dx, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    MS_EXCEPTION_IF_NULL(kernel_node);
    auto shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'ReLUGradV2GpuKernel', input is null.";
      InitSizeLists();
      return true;
    }
    element_num_ = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    element_num_ = 0;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    auto size = element_num_ * sizeof(T);
    input_size_list_.push_back(size);
    output_size_list_.push_back(size);

    auto mask_size = (element_num_ + 31) / 32 * sizeof(uint32_t);
    input_size_list_.push_back(mask_size);
  }

 private:
  size_t element_num_;
  bool is_null_input_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RELU_V2_GRAD_GPU_KERNEL_H_
