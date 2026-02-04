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
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/cast_all_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T, typename S>
class CastAllFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  CastAllFwdGpuKernelMod() : max_(0), output_size_(0), num_input_(0), is_null_input_(false) {}
  ~CastAllFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
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
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(inputs_dev, in_addr.get(), sizeof(T *) * num_input_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(outputs_dev, out_addr.get(), sizeof(S *) * num_input_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(size_dev, size_.get(), sizeof(size_t) * num_input_, cudaMemcpyHostToDevice, stream),
      "cudaMemCPY failed")
    auto status = CastAllKernel(inputs_dev, outputs_dev, max_, num_input_, size_dev, stream);
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
      return ret;
    }
    num_input_ = GetValue<size_t>(primitive_->GetAttr("n"));
    size_ = std::make_unique<size_t[]>(num_input_);
    output_size_list_.clear();
    for (size_t i = 0; i < num_input_; i++) {
      const auto &shape = inputs[i]->GetShapeVector();
      is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name_, "input");
      if (is_null_input_) {
        output_size_list_.push_back(output_size_);
        return true;
      }
      size_t s = SizeOf(shape);
      if (max_ < s) {
        max_ = s;
      }
      size_[i] = s;
      output_size_ = sizeof(S) * s;
      output_size_list_.push_back(output_size_);
    }
    workspace_size_list_.push_back(sizeof(T *) * num_input_);
    workspace_size_list_.push_back(sizeof(S *) * num_input_);
    workspace_size_list_.push_back(sizeof(size_t) * num_input_);
    return KRET_OK;
  }

  std::unique_ptr<size_t[]> size_;
  size_t max_;
  size_t output_size_;
  size_t num_input_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_CAST_ALL_GPU_KERNEL_H_
