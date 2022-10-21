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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_SET_DIAG_V3_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_SET_DIAG_V3_CPU_KERNEL_H_

#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
class MatrixSetDiagV3CpuKernelMod : public NativeCpuKernelMod {
 public:
  MatrixSetDiagV3CpuKernelMod() = default;
  ~MatrixSetDiagV3CpuKernelMod() override = default;

  // void InitKernel(const CNodePtr &kernel_node) override;
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
  using MatrixSetDiagV3Func = std::function<bool(MatrixSetDiagV3CpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                                 const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, MatrixSetDiagV3Func>> func_list_;
  MatrixSetDiagV3Func kernel_func_;

  template <typename T>
  bool DoLaunch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename T>
  void singleCal(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  std::vector<int64_t> diagonal_shape_;
  std::vector<int64_t> k_shape_;
  std::vector<int64_t> x_shape_;
  TypeId input_dtype_;
  std::string align_;
  size_t input_columns_ = 1;
  size_t input_rows_ = 1;
  size_t diagonal_columns_ = 1;
  size_t diagonal_rows_ = 1;
  size_t k_len_ = 0;
  int64_t k_lower_ = 0;
  int64_t k_upper_ = 0;
  int64_t max_diag_len_ = 0;
  size_t input_numelements_ = 0;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_SET_DIAG_V3_CPU_KERNEL_H_
