/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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

#include "tril.h"

#include <securec.h>
#include "Eigen/Core"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
const char *kTril = "Tril";
constexpr int64_t kParallelDataNums = 1024 * 1024;
const int32_t minDims = 2;

#define TRIL_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                      \
    uint32_t result = DoCompute<TYPE>(CTX);            \
    if (result != KERNEL_STATUS_OK) {                  \
      KERNEL_LOG_ERROR("Tril kernel compute failed."); \
      return result;                                   \
    }                                                  \
    break;                                             \
  }
}  // namespace

namespace aicpu {
uint32_t TrilCpuKernel::ValidParam(CpuKernelContext &ctx) {
  auto input_shape = ctx.Input(0)->GetTensorShape();
  auto output_shape = ctx.Output(0)->GetTensorShape();
  auto input_dims = input_shape->GetDims();

  KERNEL_CHECK_FALSE(input_dims >= minDims, KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 2, but got rank [%d]", input_shape->GetDims());

  auto input_data_type = ctx.Input(0)->GetDataType();
  auto output_data_type = ctx.Output(0)->GetDataType();
  KERNEL_CHECK_FALSE(input_data_type == output_data_type, KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input [%s] need be same with output [%s].", DTypeStr(input_data_type).c_str(),
                     DTypeStr(output_data_type).c_str())

  KERNEL_CHECK_FALSE(input_shape->GetDimSizes() == output_shape->GetDimSizes(), KERNEL_STATUS_PARAM_INVALID,
                     "The output shape size should be same as the input shape size.");

  AttrValue *diagonal = ctx.GetAttr("diagonal");
  diagonal_ = (diagonal == nullptr) ? 0 : (diagonal->GetInt());
  KERNEL_LOG_DEBUG("%s Attr[diagonal] value[%d]", kTril, diagonal_);

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TrilCpuKernel::ComputeTril(CpuKernelContext &ctx, size_t k) {
  using MatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  MatrixMap input(reinterpret_cast<T *>(ctx.Input(0)->GetData()) + k * matrix_size_, matrix_width_, matrix_height_);
  MatrixMap output(reinterpret_cast<T *>(ctx.Output(0)->GetData()) + k * matrix_size_, matrix_width_, matrix_height_);
  output = input.template triangularView<Eigen::Lower>();
  if (diagonal_ > 0) {
    for (int i = 0; i < matrix_width_; i++) {
      for (int j = i + 1; j <= i + diagonal_ && j < matrix_height_; j++) {
        int offset = i * matrix_height_ + j;
        auto ret = memcpy_s(output.data() + offset, sizeof(T), input.data() + offset, sizeof(T));
        KERNEL_CHECK_FALSE((ret == EOK), KERNEL_STATUS_PARAM_INVALID, "memcpy_s error");
      }
    }
  } else {
    for (int j = 0; j < matrix_height_; j++) {
      for (int i = j; i < j - diagonal_ && i < matrix_width_; i++) {
        memset_s(output.data() + i * matrix_height_ + j, sizeof(T), 0, sizeof(T));
      }
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TrilCpuKernel::DoCompute(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  Tensor *output_tensor = ctx.Output(0);

  auto input_shape = input_tensor->GetTensorShape();
  auto output_shape = output_tensor->GetTensorShape();

  auto input_dim_size = input_shape->GetDimSizes();
  auto output_dim_size = input_shape->GetDimSizes();

  auto input_dims = input_shape->GetDims();
  const int32_t dim_width = 2;
  const int32_t dim_height = 1;
  matrix_width_ = input_dim_size[input_dims - dim_width];
  matrix_height_ = input_dim_size[input_dims - dim_height];
  matrix_size_ = matrix_width_ * matrix_height_;
  int64_t matrixs_num = input_tensor->NumElements() / matrix_size_;

  auto shard_tril = [&](size_t start, size_t end) {
    for (size_t k = start; k < end; k++) {
      ComputeTril<T>(ctx, k);
    }
  };

  if (input_tensor->GetDataSize() <= kParallelDataNums) {
    shard_tril(0, matrixs_num);
  } else {
    int64_t max_core_num = std::max(1, static_cast<int>(aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2));
    if (max_core_num > matrixs_num) {
      max_core_num = matrixs_num;
    }
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    uint32_t ret = CpuKernelUtils::ParallelFor(ctx, matrixs_num, matrixs_num / max_core_num, shard_tril);
    if (ret != KERNEL_STATUS_OK) {
      KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor failed.");
      return KERNEL_STATUS_INNER_ERROR;
    }
  }

  return KERNEL_STATUS_OK;
}

uint32_t TrilCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check Greater params failed.");
  KERNEL_HANDLE_ERROR(ValidParam(ctx), "[%s] check params failed.", kTril);

  auto data_type = ctx.Input(0)->GetDataType();

  switch (data_type) {
    TRIL_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    TRIL_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    TRIL_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    TRIL_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    TRIL_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    TRIL_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    TRIL_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    TRIL_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    TRIL_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    TRIL_COMPUTE_CASE(DT_FLOAT, float, ctx)
    TRIL_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    TRIL_COMPUTE_CASE(DT_BOOL, bool, ctx)
    default:
      KERNEL_LOG_ERROR("Tril kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kTril, TrilCpuKernel);
}  // namespace aicpu
