/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SEQUENCE_STACK_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SEQUENCE_STACK_GPU_KERNEL_H_
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <string>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/pack.cuh"
#include "mindspore/ops/infer/stack.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
class SequenceStackGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<SequenceStackGpuKernelMod> {
 public:
  SequenceStackGpuKernelMod() = default;
  ~SequenceStackGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  using FuncList = std::vector<std::pair<KernelAttr, KernelRunFunc>>;
  const FuncList &GetFuncList() const override;

  void ResetResource() noexcept {
    axis_ = 0;
    input_num_ = 1;
    output_size_ = 0;
    dims_behind_axis_ = 1;
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 private:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  int axis_;
  int input_num_{1};
  size_t output_size_;
  size_t dims_behind_axis_;
  std::vector<int64_t> tuple_shape_;
  std::vector<ShapeVector> inputs_shape_;
  std::vector<void *> inputs_host_;
  std::string kernel_name_;
  void *stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SEQUENCE_STACK_GPU_KERNEL_H_
