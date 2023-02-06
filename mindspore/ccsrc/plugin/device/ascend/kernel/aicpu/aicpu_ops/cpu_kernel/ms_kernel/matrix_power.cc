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

#include "matrix_power.h"

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <map>

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
  if (x_type == DT_FLOAT) {
    return ComputeKernel<float>(ctx);
  } else {
    return ComputeKernel<Eigen::half>(ctx);
  }
}

template <typename T>
uint32_t MatrixPowerCpuKernel::ComputeKernel(CpuKernelContext &ctx) {
  Tensor *input_x = ctx.Input(0);
  Tensor *output_y = ctx.Output(0);
  AttrValue *power = ctx.GetAttr("n");
  int64_t powervalue = power->GetInt();
  auto x_shape = input_x->GetTensorShape();
  size_t batch = x_shape->GetDimSize(0);
  size_t dim = x_shape->GetDimSize(1);
  auto x_ptr = reinterpret_cast<T *>(input_x->GetData());
  auto y_ptr = reinterpret_cast<T *>(output_y->GetData());
  int64_t data_num = ctx.Input(0)->NumElements() * sizeof(T);

  if (powervalue < 0) {
    powervalue = -powervalue;
    if (data_num >= kParallelDataNum) {
      uint32_t min_core_num = 1;
      uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
      if (max_core_num > batch) {
        max_core_num = batch;
      }
      if (max_core_num == 0) {
        max_core_num = 1;
      }
      int64_t NotInvertible = -1;
      auto shard_matrix_power = [&](size_t start, size_t end) {
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(dim, dim);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> B(dim, dim);
        for (size_t i = start; i < end; i++) {
          for (size_t p = 0; p < dim; p++) {
            for (size_t q = 0; q < dim; q++) {
              B(p, q) = (float)x_ptr[i * dim * dim + p * dim + q];
            }
          }
          Eigen::FullPivLU<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> LU(B);
          if (!(LU.isInvertible())) {
            NotInvertible = i;
          }
          A = LU.inverse();
          B.setIdentity();
          int64_t n = powervalue;
          while (n > 0) {
            if (n % 2 == 1) {
              B = B * A;
            }
            n = n / 2;
            A = A * A;
          }
          for (size_t p = 0; p < dim; p++) {
            for (size_t q = 0; q < dim; q++) {
              y_ptr[i * dim * dim + p * dim + q] = (T)B(p, q);
            }
          }
        }
      };
      CpuKernelUtils::ParallelFor(ctx, batch, batch / max_core_num, shard_matrix_power);
      KERNEL_CHECK_FALSE((NotInvertible < 0), KERNEL_STATUS_PARAM_INVALID,
                         "The %d-th matrix of input tensor is singular, but got n is negative.", NotInvertible)
    } else {
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(dim, dim);
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> B(dim, dim);
      for (size_t i = 0; i < batch; i++) {
        for (size_t p = 0; p < dim; p++) {
          for (size_t q = 0; q < dim; q++) {
            B(p, q) = (float)x_ptr[i * dim * dim + p * dim + q];
          }
        }
        Eigen::FullPivLU<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> LU(B);
        KERNEL_CHECK_FALSE((LU.isInvertible()), KERNEL_STATUS_PARAM_INVALID,
                           "The %d-th matrix of input tensor is singular, but got n is negative.", i)
        A = LU.inverse();
        B.setIdentity();
        int64_t n = powervalue;
        while (n > 0) {
          if (n % 2 == 1) {
            B = B * A;
          }
          n = n / 2;
          A = A * A;
        }
        for (size_t p = 0; p < dim; p++) {
          for (size_t q = 0; q < dim; q++) {
            y_ptr[i * dim * dim + p * dim + q] = (T)B(p, q);
          }
        }
      }
    }
  } else {
    if (data_num >= kParallelDataNum) {
      uint32_t min_core_num = 1;
      uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
      if (max_core_num > batch) {
        max_core_num = batch;
      }
      if (max_core_num == 0) {
        max_core_num = 1;
      }
      auto shard_matrix_power = [&](size_t start, size_t end) {
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(dim, dim);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> B(dim, dim);
        for (size_t i = start; i < end; i++) {
          for (size_t p = 0; p < dim; p++) {
            for (size_t q = 0; q < dim; q++) {
              A(p, q) = (float)x_ptr[i * dim * dim + p * dim + q];
            }
          }
          B.setIdentity();
          int64_t n = powervalue;
          while (n > 0) {
            if (n % 2 == 1) {
              B = B * A;
            }
            n = n / 2;
            A = A * A;
          }
          for (size_t p = 0; p < dim; p++) {
            for (size_t q = 0; q < dim; q++) {
              y_ptr[i * dim * dim + p * dim + q] = (T)B(p, q);
            }
          }
        }
      };
      CpuKernelUtils::ParallelFor(ctx, batch, batch / max_core_num, shard_matrix_power);
    } else {
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A(dim, dim);
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> B(dim, dim);
      for (size_t i = 0; i < batch; i++) {
        for (size_t p = 0; p < dim; p++) {
          for (size_t q = 0; q < dim; q++) {
            A(p, q) = (float)x_ptr[i * dim * dim + p * dim + q];
          }
        }
        B.setIdentity();
        int64_t n = powervalue;
        while (n > 0) {
          if (n % 2 == 1) {
            B = B * A;
          }
          n = n / 2;
          A = A * A;
        }
        for (size_t p = 0; p < dim; p++) {
          for (size_t q = 0; q < dim; q++) {
            y_ptr[i * dim * dim + p * dim + q] = (T)B(p, q);
          }
        }
      }
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMatrixPower, MatrixPowerCpuKernel);
}  // namespace aicpu
