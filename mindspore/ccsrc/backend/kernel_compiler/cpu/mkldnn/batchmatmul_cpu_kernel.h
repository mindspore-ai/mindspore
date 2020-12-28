/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BATCHMATMUL_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BATCHMATMUL_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_cpu_kernel.h"

namespace mindspore {
namespace kernel {
class BatchMatMulCPUKernel : public MKLCPUKernel {
 public:
  BatchMatMulCPUKernel() = default;
  ~BatchMatMulCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

 private:
  char trans_a_{TRANSPOSE_NO};
  char trans_b_{TRANSPOSE_NO};
  dnnl_dim_t dim_m_{0};
  dnnl_dim_t dim_n_{0};
  dnnl_dim_t dim_k_{0};
  size_t batch_{0};
  size_t size_mat_a_{0};
  size_t size_mat_b_{0};
  size_t size_mat_output_{0};
};

MS_REG_CPU_KERNEL(
  BatchMatMul,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  BatchMatMulCPUKernel);

MS_REG_CPU_KERNEL(
  BatchMatMul,
  KernelAttr().AddInputAttr(kNumberTypeFloat).AddInputAttr(kNumberTypeFloat).AddOutputAttr(kNumberTypeFloat32),
  BatchMatMulCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BATCHMATMUL_CPU_KERNEL_H_
