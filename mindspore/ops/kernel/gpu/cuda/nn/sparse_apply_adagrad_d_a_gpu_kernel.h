/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_APPLY_ADAGRAD_AD_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_APPLY_ADAGRAD_AD_KERNEL_H_
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <map>

#include "mindspore/ops/infer/sparse_apply_adagrad_da.h"
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
class SparseApplyAdagradDAGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseApplyAdagradDAGpuKernelMod() = default;
  ~SparseApplyAdagradDAGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  void ResetResource() noexcept {
    workspace_size_list_.clear();
    output_size_list_.clear();
  }

 private:
  void CheckParam(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) const;
  void CheckShape(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) const;
  void CheckDType(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) const;
  template <typename T, typename S, typename S1>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  using SparseApplyAdagradDAFunc = std::function<bool(
    SparseApplyAdagradDAGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
    const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, SparseApplyAdagradDAFunc>> func_list_;
  SparseApplyAdagradDAFunc kernel_func_;

  int64_t unit_var_size_{1};
  int64_t unit_indices_size_{1};
  void *stream_ptr_{nullptr};
  size_t unit_size_{0};
  bool is_null_input_{false};
  size_t input_elements_{0};
  size_t batch_rank_;
  size_t batch_size_;
  size_t indices_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_APPLY_ADAGRAD_AD_KERNEL_H_
