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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSEFILLEMPTYROWS_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSEFILLEMPTYROWS_CPU_KERNEL_H_
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SparseFillEmptyRowsCpuKernelMod : public NativeCpuKernelMod {
 public:
  SparseFillEmptyRowsCpuKernelMod() = default;
  ~SparseFillEmptyRowsCpuKernelMod() override = default;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                                void *stream_ptr) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);

  TypeId output_indices_type_;
  TypeId output_values_type_;
  TypeId output_empty_row_indicator_type_;
  TypeId output_reverse_index_type_;
  ShapeVector out_indice_shape_dense_rows_zero_;
  ShapeVector out_values_shape_dense_rows_zero_;
  ShapeVector out_indice_shape_;
  ShapeVector out_values_shape_;
  ShapeVector out_empty_row_indicator_shape_;
  ShapeVector out_reverse_index_shape_;
  bool dense_rows_zero{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSEFILLEMPTYROWS_CPU_KERNEL_H_
