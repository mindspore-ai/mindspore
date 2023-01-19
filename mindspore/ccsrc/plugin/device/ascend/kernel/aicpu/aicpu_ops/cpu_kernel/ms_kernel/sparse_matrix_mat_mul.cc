/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
#include "sparse_matrix_mat_mul.h"
#include <securec.h>
#include <complex>
#include <numeric>
#include <string>
#include <vector>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_log.h"
#include "status.h"
#include "utils/allocator_utils.h"
#include "utils/kernel_util.h"

using namespace std;

namespace aicpu {
const char *SparseMatrixMatMul = "SparseMatrixMatMul";
const int INPUT_PARAMS_NUM = 6;
const int OUTPUT_PARAMS_NUM = 1;
}  // namespace aicpu

namespace aicpu {
uint32_t SparseMatrixMatMulCpuKernel::Compute(CpuKernelContext &ctx) {
  if (ValidParam(ctx) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("valid sparse matrix mat mul param error.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  DataType indice_type = ctx.Input(0)->GetDataType();
  DataType value_type = ctx.Input(4)->GetDataType();
  uint32_t status;
  switch (indice_type) {
    case DT_INT32:
      switch (value_type) {
        case DT_FLOAT:
          status = DoCompute<int32_t, float_t>(ctx);
          break;
        case DT_DOUBLE:
          status = DoCompute<int32_t, double_t>(ctx);
          break;
        case DT_COMPLEX64:
          status = DoCompute<int32_t, complex<float_t> >(ctx);
          break;
        case DT_COMPLEX128:
          status = DoCompute<int32_t, complex<double_t> >(ctx);
          break;
        default:
          KERNEL_LOG_ERROR("data type of dense shape is not int32 or int64");
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    case DT_INT64:
      switch (value_type) {
        case DT_FLOAT:
          status = DoCompute<int64_t, float_t>(ctx);
          break;
        case DT_DOUBLE:
          status = DoCompute<int64_t, double_t>(ctx);
          break;
        case DT_COMPLEX64:
          status = DoCompute<int64_t, complex<float_t> >(ctx);
          break;
        case DT_COMPLEX128:
          status = DoCompute<int64_t, complex<double_t> >(ctx);
          break;
        default:
          KERNEL_LOG_ERROR("data type of dense shape is not int32 or int64");
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    default:
      KERNEL_LOG_ERROR("data type of dense shape is not int32 or int64");
      return KERNEL_STATUS_PARAM_INVALID;
  }

  if (status != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("error in do the actual compute!");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename indiceT, typename valueT>
Eigen::Ref<const Eigen::SparseMatrix<valueT, Eigen::RowMajor, indiceT> >
SparseMatrixMatMulCpuKernel::CreateEigenSparseMatrix(indiceT rows, indiceT cols, int64_t nnz, indiceT *row_pointers,
                                                     indiceT *col_indices, valueT *values, bool transpose,
                                                     bool adjoint) {
  Eigen::Map<const Eigen::SparseMatrix<valueT, Eigen::RowMajor, indiceT> > sparse_matrix(rows, cols, nnz, row_pointers,
                                                                                         col_indices, values);
  // The transpose/adjoint expressions are not actually evaluated until
  // necessary. Hence we don't create copies or modify the input matrix
  // inplace.
  if (transpose) {
    return sparse_matrix.transpose();
  }
  if (adjoint) {
    return sparse_matrix.adjoint();
  }
  return sparse_matrix;
}

uint32_t SparseMatrixMatMulCpuKernel::ValidParam(CpuKernelContext &ctx) {
  KERNEL_LOG_DEBUG("Start to execute ValidParam.");
  // valid input and output nullptr
  if (NormalCheck(ctx, INPUT_PARAMS_NUM, OUTPUT_PARAMS_NUM) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // check if the matrix can mul
  DataType dt = ctx.Input(0)->GetDataType();  // dense shape x1
  uint32_t checkStatus;
  switch (dt) {
    case DT_INT32:
      checkStatus = CheckMatMul<int32_t>(ctx);
      break;
    case DT_INT64:
      checkStatus = CheckMatMul<int64_t>(ctx);
      break;
    default:
      // KERNEL_LOG_ERROR("data type of dense shape is not int32 or int64");
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (checkStatus != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("the two input matrixs cannot mul cause their dim!");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SparseMatrixMatMulCpuKernel::CheckMatMul(CpuKernelContext &ctx) {
  KERNEL_LOG_DEBUG("check if the matrix can mul");

  const int rank = ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  const int row_dim = (rank == 2) ? 0 : 1;
  Tensor *dense_shape_x1 = ctx.Input(0);
  T *shape_x1 = static_cast<T *>(dense_shape_x1->GetData());
  std::vector<int64_t> shape_x2 = ctx.Input(5)->GetTensorShape()->GetDimSizes();

  bool transpose_a = false;
  bool transpose_b = false;
  bool adjoint_a = false;
  bool adjoint_b = false;

  if (ctx.GetAttr("transpose_x1") != nullptr) {
    transpose_a = ctx.GetAttr("transpose_x1")->GetBool();
  }
  if (ctx.GetAttr("transpose_x2") != nullptr) {
    transpose_b = ctx.GetAttr("transpose_x2")->GetBool();
  }
  if (ctx.GetAttr("adjoint_x1") != nullptr) {
    adjoint_a = ctx.GetAttr("adjoint_x1")->GetBool();
  }
  if (ctx.GetAttr("adjoint_x2") != nullptr) {
    adjoint_b = ctx.GetAttr("adjoint_x2")->GetBool();
  }

  T x1_col = (transpose_a || adjoint_a) ? shape_x1[row_dim] : shape_x1[row_dim + 1];
  T x2_row = (transpose_b || adjoint_b) ? shape_x2[row_dim + 1] : shape_x2[row_dim];
  if (x1_col != x2_row) {
    KERNEL_LOG_ERROR("x1's col is no equal x2's row, cannot do mat mul!");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename indiceT, typename valueT>
uint32_t SparseMatrixMatMulCpuKernel::DoCompute(CpuKernelContext &ctx) {
  using Matrix = Eigen::Matrix<valueT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  indiceT batch_size = ctx.Input(1)->NumElements() - 1;
  std::vector<Matrix> results(batch_size);
  int shift = (ctx.Input(0)->NumElements() == 2) ? 0 : 1;

  indiceT row_x1 = *(static_cast<indiceT *>(ctx.Input(0)->GetData()) + shift);
  indiceT col_x1 = *(static_cast<indiceT *>(ctx.Input(0)->GetData()) + shift + 1);
  indiceT *batch_pointers_x1 = static_cast<indiceT *>(ctx.Input(1)->GetData());
  indiceT *row_pointers_x1 = static_cast<indiceT *>(ctx.Input(2)->GetData());
  indiceT *col_indices_x1 = static_cast<indiceT *>(ctx.Input(3)->GetData());
  valueT *value_x1 = static_cast<valueT *>(ctx.Input(4)->GetData());

  std::vector<int64_t> shape_x2 = ctx.Input(5)->GetTensorShape()->GetDimSizes();
  const int rank = ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  const int row_dim = (rank == 2) ? 0 : 1;
  indiceT row_x2 = shape_x2[row_dim];
  indiceT col_x2 = shape_x2[row_dim + 1];
  valueT *value_x2 = static_cast<valueT *>(ctx.Input(5)->GetData());

  bool transpose_a = false;
  bool transpose_b = false;
  bool adjoint_a = false;
  bool adjoint_b = false;
  bool transpose_output = false;
  bool conjugate_output = false;
  if (ctx.GetAttr("transpose_x1") != nullptr) {
    transpose_a = ctx.GetAttr("transpose_x1")->GetBool();
  }
  if (ctx.GetAttr("transpose_x2") != nullptr) {
    transpose_b = ctx.GetAttr("transpose_x2")->GetBool();
  }
  if (ctx.GetAttr("adjoint_x1") != nullptr) {
    adjoint_a = ctx.GetAttr("adjoint_x1")->GetBool();
  }
  if (ctx.GetAttr("adjoint_x2") != nullptr) {
    adjoint_b = ctx.GetAttr("adjoint_x2")->GetBool();
  }
  if (ctx.GetAttr("transpose_output") != nullptr) {
    transpose_output = ctx.GetAttr("transpose_output")->GetBool();
  }
  if (ctx.GetAttr("conjugate_output") != nullptr) {
    conjugate_output = ctx.GetAttr("conjugate_output")->GetBool();
  }
  uint32_t min_core_num = 1;
  uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  max_core_num = std::min(max_core_num, (uint32_t)batch_size);
  if (max_core_num == 0) {
    KERNEL_LOG_ERROR("max core num cannot be zero");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  KERNEL_HANDLE_ERROR(
    CpuKernelUtils::ParallelFor(ctx, batch_size, batch_size / max_core_num,
                                [&](int64_t start, int64_t end) {
                                  for (int64_t i = start; i < end; i++) {
                                    int64_t nnz_x1 = batch_pointers_x1[i + 1] - batch_pointers_x1[i];
                                    indiceT *row_pointers_x1_batch_i = row_pointers_x1 + (row_x1 + 1) * i;
                                    indiceT *col_indices_x1_batch_i = col_indices_x1 + batch_pointers_x1[i];
                                    valueT *value_x1_batch_i = value_x1 + batch_pointers_x1[i];
                                    auto x1_sparse_matrix = CreateEigenSparseMatrix<indiceT, valueT>(
                                      row_x1, col_x1, nnz_x1, row_pointers_x1_batch_i, col_indices_x1_batch_i,
                                      value_x1_batch_i, transpose_a, adjoint_a);

                                    Eigen::Map<Matrix> x2_dense_matrix(value_x2 + col_x2 * row_x2 * i, row_x2, col_x2);
                                    Matrix temp;
                                    if (transpose_b) {
                                      temp = x1_sparse_matrix * x2_dense_matrix.transpose();
                                    } else if (adjoint_b) {
                                      temp = x1_sparse_matrix * x2_dense_matrix.adjoint();
                                    } else {
                                      temp = x1_sparse_matrix * x2_dense_matrix;
                                    }

                                    if (transpose_output) {
                                      results[i] = temp.transpose();
                                    } else if (conjugate_output) {
                                      results[i] = temp.conjugate();
                                    } else
                                      results[i] = temp;
                                  }
                                }),
    "SparseMatrixMatMul Compute failed.");

  // computer result_row_pointers|result_col_indices|result_values data
  indiceT row_output, col_output;
  row_output = results[0].rows();
  col_output = results[0].cols();
  for (int i = 0; i < batch_size; i++) {
    valueT *output_values_data = static_cast<valueT *>(ctx.Output(0)->GetData());
    std::copy(results[i].data(), results[i].data() + row_output * col_output,
              output_values_data + i * row_output * col_output);
  }

  KERNEL_LOG_DEBUG("DoCompute end!!");
  return KERNEL_STATUS_OK;
}

// register the opetaor
REGISTER_CPU_KERNEL(SparseMatrixMatMul, SparseMatrixMatMulCpuKernel);
}  // namespace aicpu