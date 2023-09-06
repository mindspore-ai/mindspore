/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "cpu_kernel/ms_kernel/cholesky_grad.h"

#include <algorithm>
#include <iostream>
#include <map>

#include "cpu_kernel/common/cpu_kernel_utils.h"

namespace {
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
const char *CholeskyGrad = "CholeskyGrad";
}  // namespace

namespace aicpu {
uint32_t CholeskyGradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "CholeskyGrad check input and output number failed.");
  Tensor *input0 = ctx.Input(0);
  Tensor *input1 = ctx.Input(1);
  Tensor *output0 = ctx.Output(0);
  if (input0->GetDataSize() == 0 || input1->GetDataSize() == 0) {
    KERNEL_LOG_ERROR("[%s] Input tensor is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto shape0 = input0->GetTensorShape();
  auto shape1 = input1->GetTensorShape();
  auto shape2 = output0->GetTensorShape();
  if (shape0->GetDims() != shape1->GetDims() || shape1->GetDims() != shape2->GetDims()) {
    KERNEL_LOG_ERROR("[%s] Inputs and Output tensors should have same dims.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto dims = shape0->GetDims();
  if (shape0->GetDimSize(dims - 1) != shape0->GetDimSize(dims - 2)) {
    KERNEL_LOG_ERROR("[%s] Tensor input0 is not square.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if ((shape0->GetDimSize(dims - 1) != shape1->GetDimSize(dims - 1)) ||
      (shape0->GetDimSize(dims - 2) != shape1->GetDimSize(dims - 2))) {
    KERNEL_LOG_ERROR("[%s] Tensor input0&input1's shape mismatch.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if ((shape0->GetDimSize(dims - 1) != shape2->GetDimSize(dims - 1)) ||
      (shape0->GetDimSize(dims - 2) != shape2->GetDimSize(dims - 2))) {
    KERNEL_LOG_ERROR("[%s] Tensor input0&output0's shape mismatch.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto data_type_0 = input0->GetDataType();
  auto data_type_1 = input1->GetDataType();
  auto data_type_2 = output0->GetDataType();
  if (data_type_0 != data_type_1 || data_type_0 != data_type_2) {
    KERNEL_LOG_ERROR("[%s] Tensor data type mismatch.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (data_type_0 != DT_FLOAT && data_type_0 != DT_DOUBLE) {
    KERNEL_LOG_ERROR("CholeskyGrad kernel data type [%u] not support.", data_type_0);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (data_type_0 == DT_FLOAT) {
    return ComputeKernel<float>(ctx, true);
  } else {
    return ComputeKernel<double>(ctx, true);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CholeskyGradCpuKernel::ComputeKernel(const CpuKernelContext &ctx, const bool &reverse) {
  auto dims = ctx.Input(0)->GetTensorShape()->GetDims();
  auto lptr = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto gradptr = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto outputptr = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int n = ctx.Input(0)->GetTensorShape()->GetDimSize(dims - 1);
  int64_t data_num = ctx.Input(0)->NumElements();
  const int64_t mat_size = static_cast<int64_t>(n * n);
  const int64_t batch = data_num / mat_size;
  const int64_t kParallelDataNum = 16 * mat_size;
  const int64_t kParallelDataNumMid = 72 * mat_size;
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    auto sharder_cholesky_grad = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        ComputeMatrix(lptr + i * mat_size, gradptr + i * mat_size, outputptr + i * mat_size, n);
      }
    };

    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, batch, batch / max_core_num, sharder_cholesky_grad),
                        "CholeskyGrad Compute failed.");

  } else {
    for (int64_t i = 0; i < batch; i++) {
      ComputeMatrix(lptr + i * mat_size, gradptr + i * mat_size, outputptr + i * mat_size, n);
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void CholeskyGradCpuKernel::ComputeMatrix(T *lptr, T *gradptr, T *outputptr, int64_t n) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigengrad(n, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigenl(n, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> output_matrix(n, n);
  for (int i = 0; i < n * n; i++) {
    *(eigengrad.data() + i) = *(gradptr + i);
    *(eigenl.data() + i) = *(lptr + i);
  }

  // Algorithm only depends on lower triangular half on input_matrix_l.
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> input_matrix_l =
    eigenl.template triangularView<Eigen::Lower>();
  // Algorithm only depends on lower triangular half on input_matrix_grad.
  output_matrix = eigengrad.template triangularView<Eigen::Lower>();

  const int64_t kMatrixSize = input_matrix_l.rows();
  const int64_t kMaxBlockSize = 32;

  for (int64_t block_end = kMatrixSize; block_end > 0; block_end -= kMaxBlockSize) {
    const int64_t block_begin = std::max(int64_t{0}, block_end - kMaxBlockSize);
    const int64_t block_size = block_end - block_begin;
    const int64_t trailing_size = kMatrixSize - block_end;

    auto B = input_matrix_l.block(block_end, 0, trailing_size, block_begin);
    auto B_bar = output_matrix.block(block_end, 0, trailing_size, block_begin);

    auto C = input_matrix_l.block(block_end, block_begin, trailing_size, block_size);
    auto C_bar = output_matrix.block(block_end, block_begin, trailing_size, block_size);

    auto D = input_matrix_l.block(block_begin, block_begin, block_size, block_size);
    auto D_bar = output_matrix.block(block_begin, block_begin, block_size, block_size);

    auto R = input_matrix_l.block(block_begin, 0, block_size, block_begin);
    auto R_bar = output_matrix.block(block_begin, 0, block_size, block_begin);

    C_bar = D.adjoint().template triangularView<Eigen::Upper>().solve(C_bar.adjoint()).adjoint();
    D_bar -= (C_bar.adjoint() * C).template triangularView<Eigen::Lower>();
    B_bar -= C_bar * R;
    R_bar -= C_bar.adjoint() * B;
    CholeskyGradUnblocked<T>(D, D_bar);
    R_bar -= (D_bar + D_bar.adjoint()) * R;
  }
  output_matrix = (0.5 * (output_matrix + output_matrix.transpose())).eval();
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      *(outputptr + i * n + j) = output_matrix(i, j);
    }
  }
}

template <typename T>
void CholeskyGradCpuKernel::CholeskyGradUnblocked(
  const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &l_block,
  Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> grad_block) {
  const int64_t kMatrixSize = l_block.rows();
  for (int64_t k = kMatrixSize - 1; k >= 0; k--) {
    const int64_t number_rows_B = kMatrixSize - (k + 1);
    const int64_t number_rows_r_stack_B = number_rows_B + 1;

    auto r = l_block.block(k, 0, 1, k);
    auto r_bar = grad_block.block(k, 0, 1, k);
    auto d = l_block(k, k);  // This needs to be a scalar rather than a view.
    auto d_bar = grad_block.block(k, k, 1, 1);
    // B is not included explicitly because it is not used on its own.
    auto B_bar = grad_block.block(k + 1, 0, number_rows_B, k);
    auto c = l_block.block(k + 1, k, number_rows_B, 1);
    auto c_bar = grad_block.block(k + 1, k, number_rows_B, 1);
    // Result of vertical stacking d_bar and c_bar.
    auto d_stack_c_bar = grad_block.block(k, k, number_rows_r_stack_B, 1);
    // Result of vertical stacking of r and B.
    auto r_stack_B = l_block.block(k, 0, number_rows_r_stack_B, k);
    d_bar -= (c.adjoint() * c_bar) / d;
    d_stack_c_bar /= d;
    r_bar -= d_stack_c_bar.adjoint() * r_stack_B;
    B_bar -= c_bar * r;
    d_bar /= 2.;
  }
}

REGISTER_CPU_KERNEL(CholeskyGrad, CholeskyGradCpuKernel);
}  // namespace aicpu
