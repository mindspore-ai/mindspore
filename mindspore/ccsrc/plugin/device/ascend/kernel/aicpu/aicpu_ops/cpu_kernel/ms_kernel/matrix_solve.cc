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

#include "matrix_solve.h"

#include <complex>
#include "Eigen/Core"
#include "Eigen/LU"
#include "unsupported/Eigen/CXX11/Tensor"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kMatrixSolve = "MatrixSolve";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const int64_t kParallelDataNumSameShape = 8 * 1024;
const int64_t kParallelDataNumSameShapeMid = 128 * 1024;
}  // namespace

namespace aicpu {
uint32_t MatrixSolveCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "MatrixSolve check input and output number failed.");
  KERNEL_HANDLE_ERROR(MatrixSolveDataAndTypeCheck(ctx), "MatrixSolve check input and output params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      return MatrixSolveCompute<float>(ctx);
    case DT_DOUBLE:
      return MatrixSolveCompute<double>(ctx);
    case DT_COMPLEX64:
      return MatrixSolveCompute<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return MatrixSolveCompute<std::complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("MatrixSolve kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t MatrixSolveCpuKernel::MatrixSolveDataAndTypeCheck(CpuKernelContext &ctx) {
  DataType matrix_type = ctx.Input(0)->GetDataType();
  DataType rhs_type = ctx.Input(1)->GetDataType();
  KERNEL_CHECK_FALSE((matrix_type == rhs_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s] need be same with "
                     "input1 [%s].",
                     DTypeStr(matrix_type).c_str(), DTypeStr(rhs_type).c_str())

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MatrixSolveCpuKernel::MatrixSolveCompute(CpuKernelContext &ctx) {
  auto input0_tensor = ctx.Input(0);
  auto input0_tensor_shape = input0_tensor->GetTensorShape();
  auto input1_tensor = ctx.Input(1);
  auto input1_tensor_shape = input1_tensor->GetTensorShape();
  auto input0_data = reinterpret_cast<T *>(input0_tensor->GetData());
  auto input1_data = reinterpret_cast<T *>(input1_tensor->GetData());
  auto input0_shape = input0_tensor_shape->GetDimSizes();
  int32_t input0_dims = input0_tensor_shape->GetDims();
  int32_t input1_dims = input1_tensor_shape->GetDims();
  int64_t m = input0_shape[input0_dims - 1];
  int64_t size_mm = m * m;

  KERNEL_CHECK_FALSE((input0_shape[input0_dims - 1] == input0_shape[input0_dims - 2]), KERNEL_STATUS_PARAM_INVALID,
                     "Input[matrix] must be a square matrix")
  KERNEL_CHECK_FALSE((input1_dims >= 2), KERNEL_STATUS_PARAM_INVALID, "Input[rhs] must be a matrix")
  KERNEL_CHECK_FALSE(
    (input0_tensor_shape->GetDimSize(input0_dims - 1) == input1_tensor_shape->GetDimSize(input1_dims - 2)),
    KERNEL_STATUS_PARAM_INVALID, "Input matrix and rhs are incompatible")

  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MartixXd;
  auto adjoint = ctx.GetAttr("adjoint")->GetBool();
  auto input1_shape = input1_tensor_shape->GetDimSizes();
  int64_t k = input1_shape[input1_dims - 1];
  auto output_tensor = ctx.Output(0);
  auto output_data = reinterpret_cast<T *>(output_tensor->GetData());

  if (size_mm > 0) {
    size_t matrix_num = ctx.Input(0)->NumElements() / size_mm;
    int64_t data_size = ctx.Input(0)->NumElements() * sizeof(T);
    if (data_size >= kParallelDataNumSameShape) {
      uint32_t min_core_num = 1;
      uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
      if (data_size <= kParallelDataNumSameShapeMid) {
        max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
      }
      // 若AI CPU中核数大于矩阵个数，以矩阵个数作为max_core_num
      if (max_core_num > matrix_num) {
        max_core_num = matrix_num;
      }
      auto sharder_matrix_solve = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
          Eigen::Map<MartixXd> input0(input0_data + i * m * m, m, m);
          Eigen::Map<MartixXd> input1(input1_data + i * m * k, m, k);
          Eigen::Map<MartixXd> output(output_data + i * m * k, m, k);
          if (input0.rows() == 0 || input0.cols() == 0 || input1.cols() == 0) {
            return KERNEL_STATUS_PARAM_INVALID;
          }
          Eigen::PartialPivLU<MartixXd> lu_decomposition(input0.rows());
          if (adjoint) {
            lu_decomposition.compute(input0.adjoint());
          } else {
            lu_decomposition.compute(input0);
          }
          using RealScalar = typename Eigen::NumTraits<T>::Real;
          RealScalar pivot = lu_decomposition.matrixLU().diagonal().cwiseAbs().minCoeff();
          KERNEL_CHECK_FALSE((pivot > RealScalar(0)), KERNEL_STATUS_PARAM_INVALID, "Input matrix is not invertible");
          output.noalias() = lu_decomposition.solve(input1);
        }
        return KERNEL_STATUS_OK;
      };
      KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, matrix_num, matrix_num / max_core_num, sharder_matrix_solve),
                          "Matrix Solve Compute failed");

    } else {
      for (size_t i = 0; i < matrix_num; i++) {
        Eigen::Map<MartixXd> input0(input0_data + i * m * m, m, m);
        Eigen::Map<MartixXd> input1(input1_data + i * m * k, m, k);
        Eigen::Map<MartixXd> output(output_data + i * m * k, m, k);
        if (input0.rows() == 0 || input0.cols() == 0 || input1.cols() == 0) {
          return KERNEL_STATUS_PARAM_INVALID;
        }
        Eigen::PartialPivLU<MartixXd> lu_decomposition(input0.rows());
        if (adjoint) {
          lu_decomposition.compute(input0.adjoint());
        } else {
          lu_decomposition.compute(input0);
        }
        using RealScalar = typename Eigen::NumTraits<T>::Real;
        RealScalar pivot = lu_decomposition.matrixLU().diagonal().cwiseAbs().minCoeff();
        KERNEL_CHECK_FALSE((pivot > RealScalar(0)), KERNEL_STATUS_PARAM_INVALID, "Input matrix is not invertible");

        output.noalias() = lu_decomposition.solve(input1);
      }
    }
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMatrixSolve, MatrixSolveCpuKernel);
}  // namespace aicpu