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

#include "cpu_kernel/ms_kernel/matrix_power.h"

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <map>
#include <functional>

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel/common/cpu_kernel_utils.h"

namespace {
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
const char *kMatrixPower = "MatrixPower";
const int64_t kParallelDataNum = 4 * 1024;
}  // namespace

namespace aicpu {
uint32_t MatrixPowerCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "MatrixPower normal check failed.");
  auto x_type = ctx.Input(0)->GetDataType();
  AttrValue *power = ctx.GetAttr("n");
  powervalue_ = power->GetInt();

  switch (x_type) {
    case DT_UINT8:
      return ComputeKernel<uint8_t>(ctx);
    case DT_INT8:
      return ComputeKernel<int8_t>(ctx);
    case DT_INT16:
      return ComputeKernel<int16_t>(ctx);
    case DT_INT32:
      return ComputeKernel<int32_t>(ctx);
    case DT_INT64:
      return ComputeKernel<int64_t>(ctx);
    case DT_FLOAT:
      return ComputeKernel<float>(ctx);
    case DT_DOUBLE:
      return ComputeKernel<double>(ctx);
    default:
      KERNEL_LOG_ERROR("For MatrixPower, input type is not supported: %s", DTypeStr(x_type).c_str());
      return KERNEL_STATUS_INNER_ERROR;
  }
}

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
uint32_t MatrixPowerCpuKernel::ComputeKernel(const CpuKernelContext &ctx) {
  Tensor *input_x = ctx.Input(0);
  Tensor *output_y = ctx.Output(0);
  auto x_shape = input_x->GetTensorShape()->GetDimSizes();
  size_t batch =
    static_cast<size_t>(std::accumulate(x_shape.begin(), x_shape.end() - 2, 1, std::multiplies<int64_t>()));
  size_t dim = static_cast<size_t>(x_shape.back());
  auto x_ptr = reinterpret_cast<T *>(input_x->GetData());
  auto y_ptr = reinterpret_cast<T *>(output_y->GetData());
  int64_t data_num = ctx.Input(0)->NumElements();

  size_t max_core_num = std::min(batch, (size_t)aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  max_core_num = max_core_num < 1 ? 1 : max_core_num;

  std::function<aicpu::KernelStatus(Matrix<T> &)> inv;
  if constexpr (std::is_integral_v<T>) {
    inv = [](auto &A) {
      KERNEL_LOG_ERROR("For MatrixPower, n < 0 is not supported for input of integer type.");
      return KERNEL_STATUS_INNER_ERROR;
    };
  } else {
    inv = [](auto &A) {
      Eigen::FullPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> LU(A);
      if (!(LU.isInvertible())) {
        KERNEL_LOG_ERROR("For MatrixPower, negative power can not apply to singular matrix.");
        return KERNEL_STATUS_INNER_ERROR;
      }
      A = LU.inverse();
      return KERNEL_STATUS_OK;
    };
  }

  auto status = KERNEL_STATUS_OK;
  auto shard_matrix_power = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      size_t offset = i * dim * dim;
      Matrix<T> A = Eigen::Map<Matrix<T>>(x_ptr + offset, dim, dim);
      auto n = powervalue_;
      if (n < 0) {
        n = -n;
        status = inv(A);
        if (status != KERNEL_STATUS_OK) return;
      }
      Eigen::Map<Matrix<T>> B(y_ptr + offset, dim, dim);
      B.setIdentity();
      while (n > 0) {
        if (n % 2 == 1) {
          B = B * A;
        }
        n = n / 2;
        A = A * A;
      }
    }
  };

  if (data_num >= kParallelDataNum) {
    CpuKernelUtils::ParallelFor(ctx, batch, batch / max_core_num, shard_matrix_power);
  } else {
    shard_matrix_power(0, batch);
  }
  return status;
}

REGISTER_CPU_KERNEL(kMatrixPower, MatrixPowerCpuKernel);
}  // namespace aicpu
