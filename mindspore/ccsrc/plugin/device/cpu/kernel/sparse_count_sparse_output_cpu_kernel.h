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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_COUNT_SPARSE_OUTPUT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_COUNT_SPARSE_OUTPUT_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SparseCountSparseOutputCpuKernelMod : public NativeCpuKernelMod {
 public:
  SparseCountSparseOutputCpuKernelMod() = default;
  ~SparseCountSparseOutputCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename I, typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                    const std::vector<kernel::KernelTensor *> &outputs);
  void CheckIndicesInBounds(const int64_t *indices_addr, const int64_t *shape_ptr, size_t indices_length, bool is_1d,
                            size_t rank, int64_t n_batches) const;
  template <typename T>
  void CheckValidValuesAndWeights(const T *values_addr, bool use_weights) const;
  using SparseCountSparseOutputFunc =
    std::function<bool(SparseCountSparseOutputCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  SparseCountSparseOutputFunc kernel_func_;
  ShapeVector indices_shape_;
  ShapeVector values_shape_;
  ShapeVector shape_shape_;
  ShapeVector weights_shape_;
  bool binary_output_{false};
  int64_t minlength_{-1};
  int64_t maxlength_{-1};
  size_t values_size_{0};
  static std::vector<std::pair<KernelAttr, SparseCountSparseOutputFunc>> func_list_;
  std::vector<TypeId> types_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_COUNT_SPARSE_OUTPUT_CPU_KERNEL_H_
