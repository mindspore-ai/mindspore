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

#include "cpu_kernel/ms_kernel/matrix_logarithm.h"

#include <math.h>
#include <unsupported/Eigen/MatrixFunctions>
#include <algorithm>
#include <complex>
#include <vector>

#include "Eigen/Core"
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "iostream"
#include "common/kernel_log.h"
#include "cpu_kernel/common/status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"
namespace {
const uint32_t kMatrixLogarithmInputNum = 1;
const uint32_t kMatrixLogarithmOutputNum = 1;
const char *KMatrixLogarithm = "MatrixLogarithm";
constexpr int64_t kParallelDataNums = 7 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
#define MATRIX_LOGARITHM_COMPUTE_CASE(DTYPE, TYPE, CTX)           \
  case (DTYPE): {                                                 \
    uint32_t result = MatrixLogarithmCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                             \
      KERNEL_LOG_ERROR("MatrixLogarithm kernel compute failed."); \
      return result;                                              \
    }                                                             \
    break;                                                        \
  }
}  // namespace

namespace aicpu {
uint32_t MatrixLogarithmCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kMatrixLogarithmInputNum, kMatrixLogarithmOutputNum),
                      "[%s] check input and output failed.", KMatrixLogarithm);
  KERNEL_HANDLE_ERROR(MatrixLogarithmCheck(ctx), "[%s] check params failed.", KMatrixLogarithm);
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    MATRIX_LOGARITHM_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    MATRIX_LOGARITHM_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("MatrixLogarithm kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
uint32_t MatrixLogarithmCpuKernel::MatrixLogarithmCheck(const CpuKernelContext &ctx) {
  auto input_0 = ctx.Input(0);
  auto output_0 = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(input_0->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get input x tensor shape failed.")
  KERNEL_CHECK_NULLPTR(output_0->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get output y tensor shape failed.")
  std::vector<int64_t> shape_x = input_0->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_y = output_0->GetTensorShape()->GetDimSizes();
  size_t shape_size_x = shape_x.size();
  size_t shape_size_y = shape_y.size();
  int dim_x1 = shape_x[shape_size_x - 2];
  int dim_x2 = shape_x[shape_size_x - 1];
  int dim_y1 = shape_y[shape_size_y - 2];
  int dim_y2 = shape_y[shape_size_y - 1];
  KERNEL_CHECK_FALSE((shape_size_x > 1), KERNEL_STATUS_PARAM_INVALID, "Input x dimension must be at least 2.")
  KERNEL_CHECK_FALSE((dim_x1 == dim_x2), KERNEL_STATUS_PARAM_INVALID,
                     "Input x dimentsions must be equal, but are [%lld] and [%lld].", dim_x1, dim_x2)
  KERNEL_CHECK_FALSE((dim_y1 == dim_y2), KERNEL_STATUS_PARAM_INVALID,
                     "Output y dimentsions must be equal, but are [%lld] and [%lld].", dim_y1, dim_y2)
  KERNEL_CHECK_FALSE((input_0->GetTensorShape()->GetDimSize(0) == output_0->GetTensorShape()->GetDimSize(0)),
                     KERNEL_STATUS_PARAM_INVALID, "Input x dimentsions must be equal Output y")
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t MatrixLogarithmCpuKernel::MatrixLogarithmCompute(const CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  size_t shape_size = shape_x.size();
  int64_t m = shape_x[shape_size - 1];
  int64_t size_mm = m * m;
  if (size_mm > 0) {
    int64_t matrix_num = ctx.Input(0)->NumElements() / size_mm;
    int64_t data_size = ctx.Input(0)->NumElements() * sizeof(T);
    if (data_size <= kParallelDataNums) {
      for (int64_t i = 0; i < matrix_num; i++) {
        using MartixXd = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        MartixXd temp_out(m, m);
        MartixXd temp(m, m);
        for (int64_t j = 0; j < size_mm; j++) {
          auto identity = *(input_x + i * m * m + j);
          temp(j) = identity;
        }
        temp_out = temp.log();
        for (int64_t k = 0; k < size_mm; k++) {
          *(output_y + i * m * m + k) = temp_out(k);
        }
      }
    } else {
      uint32_t min_core_num = 1;
      uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
      if (data_size <= kParallelDataNumMid) {
        max_core_num = std::min(max_core_num, 4U);
      }
      if (max_core_num > matrix_num) {
        max_core_num = matrix_num;
      }
      auto shard_work = [&](size_t start, size_t end) {
        for (size_t l = start; l < end; l++) {
          using MartixXd = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
          MartixXd temp_out(m, m);
          MartixXd temp(m, m);
          for (int64_t j = 0; j < size_mm; j++) {
            temp(j) = *(input_x + l * m * m + j);
          }
          temp_out = temp.log();
          for (int64_t k = 0; k < size_mm; k++) {
            *(output_y + l * m * m + k) = temp_out(k);
          }
        }
      };
      if (max_core_num == 0) {
        KERNEL_LOG_ERROR("max_core_num could not be 0.");
      }
      CpuKernelUtils::ParallelFor(ctx, matrix_num, matrix_num / max_core_num, shard_work);
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(KMatrixLogarithm, MatrixLogarithmCpuKernel);
}  // namespace aicpu
