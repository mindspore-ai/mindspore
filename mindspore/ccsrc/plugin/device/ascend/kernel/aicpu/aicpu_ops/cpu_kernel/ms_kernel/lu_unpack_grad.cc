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
#include "lu_unpack_grad.h"
#include <iostream>
#include "Eigen/Core"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_log.h"
#include "securec.h"
#include "status.h"
#include "utils/broadcast_iterator.h"
#include "utils/kernel_util.h"

namespace {
const char *kLuUnpackGrad = "LuUnpackGrad";
const int64_t kParallelBatchNum = 30;
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 2;
const uint32_t kInputFirst = 0;
const uint32_t kInputSecond = 1;
const uint32_t kInputThird = 2;
}  // namespace

namespace aicpu {
uint32_t LuUnpackGradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Lu Unpack Grad check input and output number failed.");
  // choose compute function depend on dataType
  auto input_type = static_cast<DataType>(ctx.Input(kInputThird)->GetDataType());
  switch (input_type) {
    case DT_FLOAT16:
      return LuUnpackGradCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return LuUnpackGradCompute<float>(ctx);
    case DT_DOUBLE:
      return LuUnpackGradCompute<double>(ctx);
    case DT_INT8:
      return LuUnpackGradCompute<int8_t>(ctx);
    case DT_INT16:
      return LuUnpackGradCompute<int16_t>(ctx);
    case DT_INT32:
      return LuUnpackGradCompute<int32_t>(ctx);
    case DT_INT64:
      return LuUnpackGradCompute<int64_t>(ctx);
    case DT_UINT8:
      return LuUnpackGradCompute<uint8_t>(ctx);
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].", ctx.GetOpType().c_str(),
                       DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t LuUnpackGradCpuKernel::TriLU(CpuKernelContext &ctx, Tensor *L_grad_output, Tensor *U_grad_output, int64_t a) {
  Tensor *L_grad = NULL;
  Tensor *U_grad = NULL;
  Tensor *LU_data = NULL;
  L_grad = ctx.Input(kInputFirst);
  U_grad = ctx.Input(kInputSecond);
  LU_data = ctx.Input(kInputThird);
  auto LU_data_shape = LU_data->GetTensorShape();
  int32_t LU_data_dims = LU_data_shape->GetDims();
  int64_t LU_data_height = LU_data_shape->GetDimSize(LU_data_dims - 2);
  int64_t LU_data_width = LU_data_shape->GetDimSize(LU_data_dims - 1);
  auto LU_dim_min = std::min(LU_data_height, LU_data_width);
  auto input_U_shape = U_grad->GetTensorShape();
  auto input_U_dim_size = input_U_shape->GetDimSizes();
  auto input_U_dims = input_U_shape->GetDims();
  int64_t matrix_U_width = input_U_dim_size[input_U_dims - 2];
  int64_t matrix_U_height = input_U_dim_size[input_U_dims - 1];
  int64_t matrix_U_size = matrix_U_width * matrix_U_height;
  auto input_L_shape = L_grad->GetTensorShape();
  auto input_L_dim_size = input_L_shape->GetDimSizes();
  using MatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  auto input_L_dims = input_L_shape->GetDims();
  int64_t matrix_L_width = input_L_dim_size[input_L_dims - 2];
  int64_t matrix_L_height = input_L_dim_size[input_L_dims - 1];
  int64_t matrix_L_size = matrix_L_width * matrix_L_height;
  int64_t output_stride = LU_data_height * LU_data_width;

  MatrixMap input_L(reinterpret_cast<T *>(L_grad->GetData()) + a * matrix_L_size, matrix_L_width, matrix_L_height);
  MatrixMap input_U(reinterpret_cast<T *>(U_grad->GetData()) + a * matrix_U_size, matrix_U_width, matrix_U_height);
  if (LU_data_width > LU_data_height) {
    MatrixMap output_L(reinterpret_cast<T *>(L_grad_output->GetData()) + a * output_stride, LU_data_height,
                       LU_data_width);
    T *MiddlePtr = new T[matrix_L_size];
    MatrixMap MiddleData(MiddlePtr, matrix_L_width, matrix_L_height);
    MiddleData = input_L.template triangularView<Eigen::StrictlyLower>();
    for (auto i = 0; i < LU_data_height; i++) {
      for (auto j = 0; j < LU_dim_min; j++) {
        output_L(i, j) = MiddleData(i, j);
      }
    }
    delete[] MiddlePtr;
  } else {
    MatrixMap output_L(reinterpret_cast<T *>(L_grad_output->GetData()) + a * output_stride, LU_data_height,
                       LU_data_width);
    output_L = input_L.template triangularView<Eigen::StrictlyLower>();
  }
  if (LU_data_height > LU_data_width) {
    MatrixMap output_U(reinterpret_cast<T *>(U_grad_output->GetData()) + a * output_stride, LU_data_height,
                       LU_data_width);
    T *MiddlePtr = new T[matrix_U_size];
    MatrixMap MiddleData(MiddlePtr, matrix_U_width, matrix_U_height);
    MiddleData = input_U.template triangularView<Eigen::Upper>();
    for (auto i = 0; i < LU_dim_min; i++) {
      for (auto j = i; j < LU_data_width; j++) {
        output_U(i, j) = MiddleData(i, j);
      }
    }
    delete[] MiddlePtr;
  } else {
    MatrixMap output_U(reinterpret_cast<T *>(U_grad_output->GetData()) + a * output_stride, LU_data_height,
                       LU_data_width);
    output_U = input_U.template triangularView<Eigen::Upper>();
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t LuUnpackGradCpuKernel::LuUnpackGradCompute(CpuKernelContext &ctx) {
  Tensor *LU_data = NULL;
  Tensor *L_grad_output = NULL;
  Tensor *U_grad_output = NULL;
  LU_data = ctx.Input(kInputThird);
  L_grad_output = ctx.Output(0);
  U_grad_output = ctx.Output(1);

  auto LU_data_shape = LU_data->GetTensorShape();
  int32_t LU_data_dims = LU_data_shape->GetDims();
  int64_t LU_data_elem_num = LU_data->NumElements();

  int64_t LU_data_height = LU_data_shape->GetDimSize(LU_data_dims - 2);
  int64_t LU_data_width = LU_data_shape->GetDimSize(LU_data_dims - 1);
  int64_t LU_data_stride = LU_data_height * LU_data_width;
  int64_t matrix_num = LU_data_elem_num / LU_data_stride;

  auto L_grad_output_data = reinterpret_cast<T *>(L_grad_output->GetData());
  auto U_grad_output_data = reinterpret_cast<T *>(U_grad_output->GetData());
  for (auto i = 0; i < LU_data_elem_num; i++) {
    *(L_grad_output_data + i) = static_cast<T>(0);
    *(U_grad_output_data + i) = static_cast<T>(0);
  }
  if (matrix_num < kParallelBatchNum) {
    for (auto i = 0; i < matrix_num; i++) {
      TriLU<T>(ctx, L_grad_output, U_grad_output, i);
    }
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > matrix_num) {
      max_core_num = matrix_num;
    }
    auto sharder = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        TriLU<T>(ctx, L_grad_output, U_grad_output, i);
      }
    };
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, matrix_num, matrix_num / max_core_num, sharder),
                        "LuUnpackGrad Compute failed.");
  }

  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kLuUnpackGrad, LuUnpackGradCpuKernel);
}  // namespace aicpu
