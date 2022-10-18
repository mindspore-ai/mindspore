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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_DIAG_V3_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_DIAG_V3_CPU_KERNEL_H_

#include <map>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MatrixDiagV3CpuKernelMod : public NativeCpuKernelMod {
 public:
  MatrixDiagV3CpuKernelMod() = default;
  ~MatrixDiagV3CpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using MatrixDiagV3Func = std::function<bool(MatrixDiagV3CpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                              const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, MatrixDiagV3Func>> func_list_;
  MatrixDiagV3Func kernel_func_;

  template <typename T>
  bool DoLaunch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  std::vector<int64_t> diagonal_shape_;
  std::vector<int64_t> k_shape_;
  TypeId diagonal_data_type_;
  std::string align_{"RIGHT_LEFT"};
  bool align_superdiag_ = true;
  bool align_subdiag_ = true;
  int64_t num_batches_ = 0;
  int32_t lower_diag_index_ = 0;
  int32_t upper_diag_index_ = 0;
  int32_t num_rows_ = -1;
  int32_t num_cols_ = -1;
  int64_t max_diag_len_ = 1;
  int64_t diag_batch_base_index_ = 0;
  int64_t diag_elements_in_batch_ = 0;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_DIAG_V3_CPU_KERNEL_H_
