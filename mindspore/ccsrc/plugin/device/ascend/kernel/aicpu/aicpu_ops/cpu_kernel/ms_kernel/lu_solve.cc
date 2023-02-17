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
#include "lu_solve.h"
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "utils/eigen_tensor.h"
#include <Eigen/Dense>
#include <iostream>
namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;
const int64_t kParallelBatchNum1 = 50;
const int64_t kParallelBatchNum4 = 200;
const int64_t kParallelBatchNum8 = 500;
const int64_t kParallelBatchNumx = 1000;
const char *kLuSolve = "LuSolve";
}  // namespace
namespace aicpu {
uint32_t LuSolveCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check LuSolve params failed.");
  Tensor *input_0 = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(input_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input0 data failed.");
  Tensor *input_1 = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(input_1->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input1 data failed.");
  Tensor *input_2 = ctx.Input(2);
  KERNEL_CHECK_NULLPTR(input_2->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input2 data failed.");
  Tensor *output = ctx.Output(0);
  auto input_0_Shape = input_0->GetTensorShape();
  KERNEL_CHECK_NULLPTR(input_0_Shape, KERNEL_STATUS_PARAM_INVALID, "Get input_0_Shape failed.")
  auto input_1_Shape = input_1->GetTensorShape();
  KERNEL_CHECK_NULLPTR(input_1_Shape, KERNEL_STATUS_PARAM_INVALID, "Get input_1_Shape failed.")
  auto input_2_Shape = input_2->GetTensorShape();
  KERNEL_CHECK_NULLPTR(input_2_Shape, KERNEL_STATUS_PARAM_INVALID, "Get input_2_Shape failed.")
  int32_t b_dims = input_0_Shape->GetDims();
  int32_t lu_dims = input_1_Shape->GetDims();
  int32_t pivots_dims = input_2_Shape->GetDims();
  std::vector<int64_t> b_dims_vector = input_0_Shape->GetDimSizes();
  std::vector<int64_t> lu_dims_vector = input_1_Shape->GetDimSizes();
  std::vector<int64_t> pivots_dims_vector = input_2_Shape->GetDimSizes();
  if (b_dims == lu_dims) {
    for (int32_t i = 0; i <= b_dims - 2; i++) {
      if (b_dims_vector[i] != lu_dims_vector[i]) {
        KERNEL_LOG_ERROR("Incompatible matrix sizes for lu_solve!");
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }
  } else if (lu_dims > b_dims) {
    for (int32_t i = 0; i < b_dims - 2; i++) {
      if (b_dims_vector[i] != lu_dims_vector[lu_dims - b_dims + i]) {
        KERNEL_LOG_ERROR("Incompatible matrix sizes for lu_solve!");
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }
  } else {
    for (int32_t i = 0; i < lu_dims - 2; i++) {
      if (lu_dims_vector[i] != b_dims_vector[b_dims - lu_dims + i]) {
        KERNEL_LOG_ERROR("Incompatible matrix sizes for lu_solve!");
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }
  }
  for (int32_t i = 0; i < pivots_dims; i++) {
    if (lu_dims_vector[i] != pivots_dims_vector[i]) {
      KERNEL_LOG_ERROR("batch dimension of LU_pivots doesn't match batch dimension of LU_data!");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  auto data_type = ctx.Input(0)->GetDataType();
  KERNEL_LOG_DEBUG(
    "LuSolveCpuKernel[%s], input_0: size[%llu], input_1: size[%llu], input_2: size[%llu]"
    "output: size[%llu].",
    ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(), input_2->GetDataSize(),
    output->GetDataSize());
  switch (data_type) {
    case DT_FLOAT:
      return LuSolveCompute<float, float>(ctx);
    case DT_FLOAT16:
      return LuSolveCompute<float, Eigen::half>(ctx);
    default:
      KERNEL_LOG_ERROR("LuSolve kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename T2>
uint32_t LuSolveCpuKernel::LuSolve(CpuKernelContext &ctx, T *b_working_ptr, T *lu_working_ptr,
                                   int32_t *pivots_working_ptr, int64_t b_stride, int64_t a) {
  auto output_y = reinterpret_cast<T2 *>(ctx.Output(0)->GetData());
  auto input_0_Shape = ctx.Input(0)->GetTensorShape();
  auto input_1_Shape = ctx.Input(1)->GetTensorShape();
  int32_t lu_dims = input_1_Shape->GetDims();
  int64_t lu_maxtrix_sizes = input_1_Shape->GetDimSize(lu_dims - 2);
  int32_t b_dim = input_0_Shape->GetDims();
  int64_t b_m = input_0_Shape->GetDimSize(b_dim - 1);
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;
  MatrixXd matrix_b = Eigen::Map<MatrixXd>(b_working_ptr, lu_maxtrix_sizes, b_m);
  MatrixXd matrix_A = Eigen::Map<MatrixXd>(lu_working_ptr, lu_maxtrix_sizes, lu_maxtrix_sizes);
  for (int64_t i = 0; i < input_0_Shape->GetDimSize(b_dim - 2); i++) {
    matrix_b.row(i).swap(matrix_b.row(*(pivots_working_ptr + i) - 1));
  }
  MatrixXd result = matrix_A.template triangularView<Eigen::UnitLower>().solve(matrix_b);
  result.noalias() = matrix_A.template triangularView<Eigen::Upper>().solve(result);
  for (int64_t m = 0; m < b_stride; m++) {
    *(output_y + a * b_stride + m) = (T2) * (result.data() + m);
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename T2>
uint32_t LuSolveCpuKernel::LuSolveCompute(CpuKernelContext &ctx) {
  auto input_x0 = reinterpret_cast<T2 *>(ctx.Input(0)->GetData());
  auto input_x1 = reinterpret_cast<T2 *>(ctx.Input(1)->GetData());
  auto input_x2 = reinterpret_cast<int32_t *>(ctx.Input(2)->GetData());
  auto input_0_Shape = ctx.Input(0)->GetTensorShape();
  auto input_1_Shape = ctx.Input(1)->GetTensorShape();
  auto input_2_Shape = ctx.Input(2)->GetTensorShape();
  T *input_0 = new T[input_0_Shape->NumElements()];
  T *input_1 = new T[input_1_Shape->NumElements()];
  for (int64_t i = 0; i < input_0_Shape->NumElements(); i++) {
    *(input_0 + i) = (T) * (input_x0 + i);
  }
  for (int64_t i = 0; i < input_1_Shape->NumElements(); i++) {
    *(input_1 + i) = (T) * (input_x1 + i);
  }
  int32_t b_dims = input_0_Shape->GetDims();
  int32_t lu_dims = input_1_Shape->GetDims();
  std::vector<int64_t> b_dims_vector = input_0_Shape->GetDimSizes();
  std::vector<int64_t> lu_dims_vector = input_1_Shape->GetDimSizes();
  std::vector<int64_t> pivots_dims_vector = input_2_Shape->GetDimSizes();
  int64_t b_stride = input_0_Shape->GetDimSize(b_dims - 1) * input_0_Shape->GetDimSize(b_dims - 2);
  int64_t lu_stride = input_1_Shape->GetDimSize(lu_dims - 1) * input_1_Shape->GetDimSize(lu_dims - 2);
  int64_t pivots_stride = input_1_Shape->GetDimSize(lu_dims - 1);
  std::vector<int64_t> b_shape = b_dims_vector;
  std::vector<int64_t> lu_shape = lu_dims_vector;
  for (size_t i = 0; i < 2; i++) {
    b_shape.pop_back();
    lu_shape.pop_back();
  }
  Bcast bcast(b_shape, lu_shape);
  int64_t batch_num = ctx.Output(0)->NumElements() / b_stride;
  if (batch_num < kParallelBatchNum1) {
    for (int64_t i = 0; i < batch_num; i++) {
      T *b_working_ptr = &input_0[bcast.GetBroadcastXIndex(i) * b_stride];
      T *lu_working_ptr = &input_1[bcast.GetBroadcastYIndex(i) * lu_stride];
      int32_t *pivots_working_ptr = &input_x2[bcast.GetBroadcastYIndex(i) * pivots_stride];
      LuSolve<T, T2>(ctx, b_working_ptr, lu_working_ptr, pivots_working_ptr, b_stride, i);
    }
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (batch_num < kParallelBatchNumx) max_core_num = 8U;
    if (batch_num < kParallelBatchNum8) max_core_num = 4U;
    if (batch_num < kParallelBatchNum4) max_core_num = 2U;
    std::cout << max_core_num << std::endl;
    auto sharder = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        T *b_working_ptr = &input_0[bcast.GetBroadcastXIndex(i) * b_stride];
        T *lu_working_ptr = &input_1[bcast.GetBroadcastYIndex(i) * lu_stride];
        int32_t *pivots_working_ptr = &input_x2[bcast.GetBroadcastYIndex(i) * pivots_stride];
        LuSolve<T, T2>(ctx, b_working_ptr, lu_working_ptr, pivots_working_ptr, b_stride, i);
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, batch_num, batch_num / max_core_num, sharder),
                        "LuSolve Compute failed.");
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kLuSolve, LuSolveCpuKernel);
}  // namespace aicpu
