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

#include "coalesce.h"

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <map>

namespace {
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 3;
const char *kCoalesce = "Coalesce";
}  // namespace

namespace aicpu {
uint32_t CoalesceCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Coalesce normal check failed.");
  auto x_values_type = ctx.Input(1)->GetDataType();
  if (x_values_type == DT_FLOAT) {
    return ComputeKernel<float>(ctx);
  } else {
    return ComputeKernel<Eigen::half>(ctx);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CoalesceCpuKernel::ComputeKernel(CpuKernelContext &ctx) {
  Tensor *x_indices = ctx.Input(0);
  Tensor *x_values = ctx.Input(1);
  Tensor *x_shape = ctx.Input(2);
  Tensor *y_indices = ctx.Output(0);
  Tensor *y_values = ctx.Output(1);
  Tensor *y_shape = ctx.Output(2);
  auto x_indices_ptr = reinterpret_cast<int64_t *>(x_indices->GetData());
  auto x_values_ptr = reinterpret_cast<T *>(x_values->GetData());
  auto x_shape_ptr = reinterpret_cast<int64_t *>(x_shape->GetData());
  auto y_indices_ptr = reinterpret_cast<int64_t *>(y_indices->GetData());
  auto y_values_ptr = reinterpret_cast<T *>(y_values->GetData());
  auto y_shape_ptr = reinterpret_cast<int64_t *>(y_shape->GetData());
  int64_t x_nnz = x_indices->GetTensorShape()->GetDimSize(1);
  int64_t num_dims = x_indices->GetTensorShape()->GetDimSize(0);

  for (int64_t i = 0; i < x_nnz; i++) {
    for (int64_t j = 0; j < num_dims; j++) {
      KERNEL_CHECK_FALSE(
        (x_indices_ptr[j * x_nnz + i] >= 0), KERNEL_STATUS_PARAM_INVALID,
        "For Coalesce, values of elements of x_indices should be non-negative, but got x_indices[%d][%d] = %d.", j, i,
        x_indices_ptr[j * x_nnz + i])
      KERNEL_CHECK_FALSE((x_indices_ptr[j * x_nnz + i] < x_shape_ptr[j]), KERNEL_STATUS_PARAM_INVALID,
                         "For Coalesce, values of elements of x_indices should not exceed the limit set by x_shape, "
                         "but got x_indices[%d][%d] = %d, got x_shape[%d] = %d.",
                         j, i, x_indices_ptr[j * x_nnz + i], j, x_shape_ptr[j])
    }
  }

  std::vector<int64_t> reorder(x_nnz);
  std::iota(reorder.begin(), reorder.end(), 0);

  auto sorter = [x_indices_ptr, num_dims, x_nnz](int64_t i, int64_t j) -> bool {
    for (int64_t n = 0; n < num_dims; n++) {
      if (x_indices_ptr[n * x_nnz + i] < x_indices_ptr[n * x_nnz + j]) {
        return true;
      }
      if (x_indices_ptr[n * x_nnz + i] > x_indices_ptr[n * x_nnz + j]) {
        return false;
      }
    }
    return true;
  };
  std::sort(reorder.begin(), reorder.end(), sorter);

  std::vector<bool> del(x_nnz);
  del[0] = false;
  int64_t jump = 0;
  y_values_ptr[0] = x_values_ptr[reorder[0]];
  for (int64_t i = 1; i < x_nnz; i++) {
    del[i] = true;
    for (int64_t j = 0; j < num_dims; j++) {
      if (x_indices_ptr[j * x_nnz + reorder[i]] != x_indices_ptr[j * x_nnz + reorder[i - 1]]) {
        del[i] = false;
        break;
      }
    }
    if (del[i]) {
      y_values_ptr[jump] += x_values_ptr[reorder[i]];
    } else {
      jump++;
      y_values_ptr[jump] = x_values_ptr[reorder[i]];
    }
  }

  int64_t up = 0;
  for (int64_t i = 0; i < x_nnz; i++) {
    if (!del[i]) {
      for (int64_t j = 0; j < num_dims; j++) {
        y_indices_ptr[j * (jump + 1) + up] = x_indices_ptr[j * x_nnz + reorder[i]];
      }
      up++;
    }
  }

  for (int64_t i = 0; i < num_dims; i++) {
    y_shape_ptr[i] = x_shape_ptr[i];
  }

  std::vector<int64_t> dims = {num_dims, jump + 1};
  auto y_indices_shape = y_indices->GetTensorShape();
  y_indices_shape->SetDimSizes(dims);
  dims = {jump + 1};
  auto y_values_shape = y_values->GetTensorShape();
  y_values_shape->SetDimSizes(dims);

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kCoalesce, CoalesceCpuKernel);
}  // namespace aicpu