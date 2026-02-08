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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UPPER_BOUND_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UPPER_BOUND_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <map>
#include "mindspore/ops/infer/upper_bound.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/upper_bound_lower_bound_impl.cuh"

namespace mindspore {
namespace kernel {
class UpperBoundGpuKernelMod : public NativeGpuKernelMod {
 public:
  UpperBoundGpuKernelMod() { ResetResource(); }
  ~UpperBoundGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  void ResetResource() noexcept {
    sorted_x_elements_ = 0;
    values_elements_ = 0;
    sorted_x_row_ = 0;
    sorted_x_col_ = 0;
    values_row_ = 0;
    values_col_ = 0;
    is_null_input_ = false;
    output_size_list_.clear();
  }

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  using UpperBoundFunc =
    std::function<bool(UpperBoundGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;

 private:
  size_t unit_size_{1};
  size_t unit_out_size_{1};
  size_t sorted_x_elements_;
  size_t values_elements_;
  size_t sorted_x_row_;
  size_t sorted_x_col_;
  size_t values_row_;
  size_t values_col_;
  UpperBoundFunc kernel_func_{};

  bool is_null_input_{false};
  void *cuda_stream_{nullptr};
  std::optional<bool> is_input_dynamic_shape_ = {};
  static std::vector<std::pair<KernelAttr, UpperBoundFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UPPER_BOUND_GPU_KERNEL_H_
