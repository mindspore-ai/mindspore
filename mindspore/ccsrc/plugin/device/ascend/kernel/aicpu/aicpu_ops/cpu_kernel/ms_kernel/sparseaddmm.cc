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
#include "sparseaddmm.h"
#include <securec.h>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_log.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 7;
const char *kSparseAddmm = "SparseAddmm";
constexpr int64_t kParallelDataNums = 16;

#define SPARSEADDMM_COMPUTE_CASE(DTYPE, TYPE, CTX)              \
  case (DTYPE): {                                               \
    if (indices_type == DT_INT64) {                             \
      uint32_t result = SparseAddmmCompute<TYPE, int64_t>(CTX); \
      if (result != KERNEL_STATUS_OK) {                         \
        KERNEL_LOG_ERROR("SparseAddmm kernel compute failed."); \
        return result;                                          \
      }                                                         \
      break;                                                    \
    } else {                                                    \
      uint32_t result = SparseAddmmCompute<TYPE, int32_t>(CTX); \
      if (result != KERNEL_STATUS_OK) {                         \
        KERNEL_LOG_ERROR("SparseAddmm kernel compute failed."); \
        return result;                                          \
      }                                                         \
      break;                                                    \
    }                                                           \
  }
}  // namespace

namespace aicpu {
uint32_t SparseAddmmCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kSparseAddmm);
  KERNEL_HANDLE_ERROR(SparseAddmmCheck(ctx), "[%s] check params failed.", kSparseAddmm);
  DataType data_type = ctx.Input(1)->GetDataType();
  DataType data_type1 = ctx.Input(3)->GetDataType();
  DataType indices_type = ctx.Input(0)->GetDataType();
  if (data_type != data_type1) {
    KERNEL_LOG_ERROR(
      "sparse data type is no equal dense data type, sparsetype [%d], "
      "densetype [%d].",
      data_type, data_type1);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  switch (data_type) {
    SPARSEADDMM_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    SPARSEADDMM_COMPUTE_CASE(DT_FLOAT, float, ctx)
    SPARSEADDMM_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    SPARSEADDMM_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    SPARSEADDMM_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    SPARSEADDMM_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    SPARSEADDMM_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    SPARSEADDMM_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    SPARSEADDMM_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    SPARSEADDMM_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    SPARSEADDMM_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    SPARSEADDMM_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    SPARSEADDMM_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("SparseAddmm kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t SparseAddmmCpuKernel::SparseAddmmCheck(CpuKernelContext &ctx) {
  Tensor *indices_tensor = ctx.Input(0);
  Tensor *values_tensor = ctx.Input(1);
  Tensor *shape_tensor = ctx.Input(2);
  Tensor *dense_tensor = ctx.Input(3);
  Tensor *alpha_tensor = ctx.Input(5);
  Tensor *beta_tensor = ctx.Input(6);

  if (alpha_tensor->GetTensorShape()->NumElements() != 1) {
    KERNEL_LOG_ERROR(
      "alpha_tensor should be a number,but got NumElements "
      "[%d].",
      alpha_tensor->GetTensorShape()->NumElements());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (beta_tensor->GetTensorShape()->NumElements() != 1) {
    KERNEL_LOG_ERROR(
      "beta_tensor should be a number,but got NumElements "
      "[%d].",
      beta_tensor->GetTensorShape()->NumElements());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // valid shape nullptr
  auto sparse_shape = shape_tensor->GetTensorShape();
  auto values_shape = values_tensor->GetTensorShape();
  auto dense_tensor_shape = dense_tensor->GetTensorShape();
  auto indices_shape = indices_tensor->GetTensorShape();
  // sparse_indices
  if (indices_shape->GetDims() > 2) {
    KERNEL_LOG_ERROR(
      "Sparse_indices should be a scalar, vector, or matrix, got dim "
      "size [%d].",
      indices_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  const int64_t elems_num = indices_shape->GetDims() > 0 ? indices_shape->GetDimSize(0) : 1;
  const int64_t dims_num = indices_shape->GetDims() > 1 ? indices_shape->GetDimSize(1) : 1;

  // output_shape
  if (sparse_shape->GetDims() != 1) {
    KERNEL_LOG_ERROR("Sparse_shape should be a vector, got dim size [%d].", sparse_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (shape_tensor->NumElements() != dims_num) {
    KERNEL_LOG_ERROR("Sparse_shape has incorrect number of elements [%lld], should be [%lld]",
                     shape_tensor->NumElements(), dims_num);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // valid data type
  int32_t IndiceType = indices_tensor->GetDataType();
  int32_t ShapeType = shape_tensor->GetDataType();
  bool validIndiceType = ((IndiceType != DT_INT32) && (IndiceType != DT_INT64));
  bool validShapeType = ((ShapeType != DT_INT32) && (ShapeType != DT_INT64));
  if (validShapeType || validIndiceType) {
    KERNEL_LOG_ERROR(
      "Valid indice or Sparse shape data type failed, indiceType [%d], "
      "shapeType [%d].",
      IndiceType, ShapeType);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // sparse_values
  int32_t values_dims_size = values_shape->GetDims();
  if ((values_dims_size != 0) && (values_dims_size != 1)) {
    KERNEL_LOG_ERROR("Values_shape should be a scalar or a vector, got dim size [%d].", values_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if ((values_dims_size == 1) && (values_tensor->NumElements() != elems_num)) {
    KERNEL_LOG_ERROR("Values_shape has incorrect number of elements [%lld], should be [%lld]",
                     values_tensor->NumElements(), elems_num);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename T1>
uint32_t SparseAddmmCpuKernel::SparseAddmmCompute(CpuKernelContext &ctx) {
  auto *indices_tensor = ctx.Input(0);
  auto *values_tensor = ctx.Input(1);
  auto *shape_tensor = ctx.Input(2);
  auto *dense_tensor = ctx.Input(3);
  auto *x3_dense_tensor = ctx.Input(4);
  auto *alpha_tensor = ctx.Input(5);
  auto *beta_tensor = ctx.Input(6);
  auto *output_tensor = ctx.Output(0);

  // auto indices = reinterpret_cast<int64_t *>(indices_tensor->GetData());
  auto values = reinterpret_cast<T *>(values_tensor->GetData());
  auto dense_data = reinterpret_cast<T *>(dense_tensor->GetData());
  auto x3_dense_data = reinterpret_cast<T *>(x3_dense_tensor->GetData());
  auto alpha = reinterpret_cast<T *>(alpha_tensor->GetData());
  auto beta = reinterpret_cast<T *>(beta_tensor->GetData());
  auto y = reinterpret_cast<T *>(output_tensor->GetData());

  std::vector<int64_t> temp_shape;
  for (int32_t index = 0; index < shape_tensor->GetTensorShape()->GetDimSize(0); ++index) {
    if (shape_tensor->GetDataType() == DT_INT32) {
      int32_t *temp_dim = reinterpret_cast<int32_t *>(shape_tensor->GetData());
      temp_shape.emplace_back(static_cast<int64_t>(temp_dim[index]));
    } else {
      int64_t *temp_dim = reinterpret_cast<int64_t *>(shape_tensor->GetData());
      temp_shape.emplace_back(temp_dim[index]);
    }
  }

  const int64_t row_x1 = temp_shape[0];
  const int64_t col_x1 = temp_shape[1];
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> sparse(row_x1, col_x1);
  sparse.setZero(row_x1, col_x1);
  std::vector<int64_t> temp_indices;
  auto indices_one = indices_tensor->GetTensorShape()->GetDimSize(0);
  auto indices_two = indices_tensor->GetTensorShape()->GetDimSize(1);
  for (int32_t index = 0; index < indices_one; ++index) {
    if (indices_tensor->GetDataType() == DT_INT32) {
      int32_t *temp_dim = reinterpret_cast<int32_t *>(indices_tensor->GetData());
      temp_indices.emplace_back(static_cast<int64_t>(temp_dim[index * indices_two + 0]));
      temp_indices.emplace_back(static_cast<int64_t>(temp_dim[index * indices_two + 1]));
    } else {
      int64_t *temp_dim = reinterpret_cast<int64_t *>(indices_tensor->GetData());
      temp_indices.emplace_back(temp_dim[index * indices_two + 0]);
      temp_indices.emplace_back(temp_dim[index * indices_two + 1]);
    }
  }

  if (indices_one <= kParallelDataNums) {
    for (int64_t i = 0; i < indices_one; i++) {
      int64_t row = temp_indices[i * indices_two + 0];
      int64_t col = temp_indices[i * indices_two + 1];
      sparse(row, col) += *(values + i);
    }
  } else {
    uint32_t minCoreNum = 1;
    int64_t maxCoreNum = std::max(minCoreNum, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    auto shardSparse = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        int64_t row = temp_indices[i * indices_two + 0];
        int64_t col = temp_indices[i * indices_two + 1];
        sparse(row, col) += *(values + i);
      }
    };
    CpuKernelUtils::ParallelFor(ctx, indices_one, indices_one / maxCoreNum, shardSparse);
  }

  std::vector<int64_t> shape_x2 = dense_tensor->GetTensorShape()->GetDimSizes();
  const int64_t row_x2 = shape_x2[0];
  const int64_t col_x2 = shape_x2[1];
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dense(row_x2, col_x2);

  std::vector<int64_t> shape_x3 = x3_dense_tensor->GetTensorShape()->GetDimSizes();
  const int64_t row_x3 = shape_x3[0];
  const int64_t col_x3 = shape_x3[1];

  if (row_x3 != row_x1) {
    KERNEL_LOG_ERROR("x1's row is no equal x3's row, cannot do add!");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (col_x3 != col_x2) {
    KERNEL_LOG_ERROR("x2's col is no equal x3's col, cannot do add!");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (row_x2 <= kParallelDataNums) {
    for (int64_t i = 0; i < row_x2; i++) {
      for (int64_t j = 0; j < col_x2; j++) {
        dense(i, j) = *(dense_data + i * col_x2 + j);
      }
    }
  } else {
    uint32_t minCoreNum = 1;
    int64_t maxCoreNum = std::max(minCoreNum, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    auto shardDense = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        for (int64_t j = 0; j < col_x2; j++) {
          dense(i, j) = *(dense_data + i * col_x2 + j);
        }
      }
    };
    CpuKernelUtils::ParallelFor(ctx, row_x2, row_x2 / maxCoreNum, shardDense);
  }

  if (col_x1 != row_x2) {
    KERNEL_LOG_ERROR("x1's col is no equal x2's row, cannot do mat mul!");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> temp;
  temp = sparse * dense;

  if (row_x1 <= kParallelDataNums) {
    for (int64_t i = 0; i < row_x1; i++) {
      for (int64_t j = 0; j < col_x2; j++) {
        *(y + i * col_x2 + j) = *(alpha + 0) * temp(i, j);
        *(y + i * col_x2 + j) += *(beta + 0) * (*(x3_dense_data + i * col_x2 + j));
      }
    }
  } else {
    uint32_t minCoreNum = 1;
    int64_t maxCoreNum = std::max(minCoreNum, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    auto shardMatMul = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        for (int64_t j = 0; j < col_x2; j++) {
          *(y + i * col_x2 + j) = *(alpha + 0) * temp(i, j);
          *(y + i * col_x2 + j) += *(beta + 0) * (*(x3_dense_data + i * col_x2 + j));
        }
      }
    };
    CpuKernelUtils::ParallelFor(ctx, row_x1, row_x1 / maxCoreNum, shardMatMul);
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSparseAddmm, SparseAddmmCpuKernel);
}  // namespace aicpu
