/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

#include "glu_grad.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/SpecialFunctions>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kGluGradInputNum = 2;
const uint32_t kGluGradOutputNum = 1;
const char *kGluGrad = "GluGrad";

#define GLU_GRAD_COMPUTE_CASE(DTYPE, TYPE, CTX)           \
  case (DTYPE): {                                         \
    uint32_t result = GluGradCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                     \
      KERNEL_LOG_ERROR("GluGrad kernel compute failed."); \
      return result;                                      \
    }                                                     \
    break;                                                \
  }
}  // namespace

namespace aicpu {
uint32_t GluGradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kGluGradInputNum, kGluGradOutputNum), "GluGrad check params failed.");
  auto data_type_grads = ctx.Input(0)->GetDataType();
  auto data_type_x = ctx.Input(1)->GetDataType();
  auto data_type_y = ctx.Output(0)->GetDataType();
  if (data_type_grads != data_type_x) {
    KERNEL_LOG_ERROR("The data types of grads and x are not matched!");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (data_type_x != data_type_y) {
    KERNEL_LOG_ERROR("The data types of x and y are not matched!");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  switch (data_type_x) {
    GLU_GRAD_COMPUTE_CASE(DT_DOUBLE, double, ctx);
    GLU_GRAD_COMPUTE_CASE(DT_FLOAT, float, ctx);
    GLU_GRAD_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx);
    default:
      KERNEL_LOG_ERROR("GluGrad kernel data type [%u] not support.", data_type_x);
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GluGradCpuKernel::GluGradCompute(CpuKernelContext &ctx) {
  Tensor *input_grad = ctx.Input(0);
  Tensor *input_x = ctx.Input(1);
  Tensor *output_y = ctx.Output(0);

  auto axis = ctx.GetAttr("axis");
  int64_t dim = axis->GetInt();
  KERNEL_CHECK_NULLPTR(axis, KERNEL_STATUS_PARAM_INVALID, "Get attr[axis] failed");
  auto input_x_shape = input_x->GetTensorShape();
  int64_t rank = input_x_shape->GetDims();
  if (dim < -rank || dim >= rank) {
    KERNEL_LOG_ERROR("The axis value is out of range!");
  }
  if (dim < 0) {
    dim = dim + rank;
  }
  std::vector<int64_t> dim_sizes = input_x_shape->GetDimSizes();
  int64_t input_shape_dims = ctx.Input(1)->NumElements();

  int two = 2;
  int64_t size = input_shape_dims;
  for (int i = 0; i <= dim; i++) {
    if (i < dim) {
      size = size / (dim_sizes[i]);
    } else if (i == dim) {
      size = size / two;
    }
  }

  T *input_x_data = static_cast<T *>(input_x->GetData());
  T *input_grad_data = static_cast<T *>(input_grad->GetData());
  T *output_y_data = static_cast<T *>(output_y->GetData());

  if (dim_sizes[dim] % two != 0) {
    KERNEL_LOG_ERROR("The split dim should be even!");
  }
  int64_t n_m = 1;
  int64_t size_m = 0;
  int64_t grad_offset_b = 0;
  int64_t grad_offset_a = 0;

  for (int i = 0; i < input_shape_dims; i++) {
    if (n_m % two != 0) {
      *(output_y_data + i) =
        (T(1.0) / (T(1.0) + exp(-(*(input_x_data + (i + size)))))) * (*(input_grad_data + grad_offset_b));
      grad_offset_b += 1;
      size_m = size_m + 1;
      if (size_m == size) {
        n_m += 1;
        size_m = 0;
      }
    } else {
      *(output_y_data + i) = *(input_x_data + (i - size)) * (T(1.0) / (T(1.0) + exp(-(*(input_x_data + i))))) *
                             (T(1.0) - (T(1.0) / (T(1.0) + exp(-(*(input_x_data + i)))))) *
                             (*(input_grad_data + grad_offset_a));
      grad_offset_a += 1;
      size_m = size_m + 1;
      if (size_m == size) {
        n_m += 1;
        size_m = 0;
      }
    }
  }
  for (int i = 0; i < input_shape_dims; i++) {
    std::cout << *(output_y_data + i) << std::endl;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kGluGrad, GluGradCpuKernel);
}  // namespace aicpu
