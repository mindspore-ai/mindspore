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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PAD_GPU_KERNEL_H_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
class PadFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  PadFwdGpuKernelMod() = default;
  ~PadFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ResetResource() noexcept {
    input_rank_ = 0;
    input_size_ = 0;
    output_size_ = 0;
    is_null_input_ = false;
    kernel_name_ = "Pad";
    flattened_paddings_.clear();
    input_shape_.clear();
    strides_.clear();
  }

  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  using PadFunc = std::function<bool(PadFwdGpuKernelMod *, const std::vector<KernelTensor *> &,
                                     const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, PadFunc>> func_list_;
  PadFunc kernel_func_;
  size_t input_rank_;
  std::vector<size_t> input_shape_;
  std::vector<int> strides_;
  std::vector<int> flattened_paddings_;

  // default
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_PAD_GPU_KERNEL_H_
