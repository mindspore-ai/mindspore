/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATMUL_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATMUL_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_cpu_kernel.h"
#include "backend/kernel_compiler/cpu/nnacl/matmul_parameter.h"
#include "backend/kernel_compiler/cpu/nnacl/fp32/matmul_fp32.h"

namespace mindspore {
namespace kernel {
class MatMulCPUKernel : public MKLCPUKernel {
 public:
  MatMulCPUKernel() = default;
  ~MatMulCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void InitTile();
  void InitMatrixA(const float *src_ptr);
  void InitMatrixB(const float *src_ptr);
  void InitArmKernel(bool trans_a, bool trans_b, const std::vector<size_t> &a_shape,
                     const std::vector<size_t> &o_shape);
  void InitX64Kernel(bool trans_a, bool trans_b, const std::vector<size_t> &a_shape, const std::vector<size_t> &b_shape,
                     const std::vector<size_t> &o_shape);
  void LaunchX64(const float *input_a, const float *input_b, float *output) const;
  void LaunchARM(const float *input_a, const float *input_b, float *output);
  void ParallelRun(float *output);
  int FloatRun(size_t task_id) const;
  void FreeBuffer();

  dnnl_dim_t dim_m_{0};
  dnnl_dim_t dim_n_{0};
  dnnl_dim_t dim_k_{0};
  size_t batch_{0};
  size_t rank_{0};
  size_t row_tile_{0};
  size_t col_tile_{0};
  size_t thread_count_{0};
  size_t thread_stride_{0};
  size_t size_mat_a_{0};
  size_t size_mat_b_{0};
  size_t size_mat_o_{0};
  char trans_a_{TRANSPOSE_NO};
  char trans_b_{TRANSPOSE_NO};
  bool vec_matmul_{false};
  float *a_pack_ptr_{nullptr};
  float *b_pack_ptr_{nullptr};
  float *batch_a_ptr_{nullptr};
  float *batch_b_ptr_{nullptr};
  float *batch_o_ptr_{nullptr};
  MatMulParameter param_{};
};

MS_REG_CPU_KERNEL(
  MatMul,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  MatMulCPUKernel);

MS_REG_CPU_KERNEL(
  BatchMatMul,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  MatMulCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATMUL_CPU_KERNEL_H_
