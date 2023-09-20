/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DYNAMIC_SHAPE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DYNAMIC_SHAPE_GPU_KERNEL_H_

#include <cuda_runtime.h>

#include <vector>
#include <map>

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class TensorShapeGpuKernelMod : public NativeGpuKernelMod {
 public:
  TensorShapeGpuKernelMod() { ResetResource(); }
  ~TensorShapeGpuKernelMod() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_ || prev_node_output_shape_.empty()) {
      return true;
    }
    S *output_device_address = GetDeviceAddress<S>(outputs, 0);
    size_t prev_node_output_shape_size = prev_node_output_shape_.size() * sizeof(S);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output_device_address, prev_node_output_shape_.data(), prev_node_output_shape_size,
                      cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync prev_node_output_shape failed");

    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    const size_t kDynamicShapeOutputNum = 1;

    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDynamicShapeOutputNum, kernel_name_);
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    auto shape = inputs.at(kIndex0)->GetShapeVector();
    is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name_, "input");
    if (is_null_input_) {
      input_size_list_.clear();
      output_size_list_.clear();
      input_size_list_.push_back(0);
      output_size_list_.push_back(0);
      return KRET_OK;
    }
    if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
      return ret;
    }
    ResetResource();
    input_size_ = SizeOf(shape);
    for (auto x : shape) {
      prev_node_output_shape_.push_back(static_cast<S>(x));
    }
    output_size_ = prev_node_output_shape_.size();

    InitSizeLists();
    return KRET_OK;
  }

  void ResetResource() noexcept {
    input_size_ = 0;
    output_size_ = 0;
    is_null_input_ = false;
    prev_node_output_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() {
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(output_size_ * sizeof(S));
  }

 private:
  size_t input_size_;
  size_t output_size_;
  bool is_null_input_;
  std::vector<S> prev_node_output_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DYNAMIC_SHAPE_GPU_KERNEL_H_
