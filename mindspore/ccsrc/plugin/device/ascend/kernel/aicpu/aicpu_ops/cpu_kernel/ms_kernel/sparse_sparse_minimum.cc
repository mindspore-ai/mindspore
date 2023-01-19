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

#include "sparse_sparse_minimum.h"

#include <algorithm>
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 6;
const char *kSparseSparseMinimum = "SparseSparseMinimum";

#define SPARSE_SPARSE_MINIMUM_COMPUTE_CASE(DTYPE, TYPE, CTX)          \
  case (DTYPE): {                                                     \
    uint32_t result = SparseSparseMinimumCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                                 \
      KERNEL_LOG_ERROR("SparseSparseMinimum kernel compute failed."); \
      return result;                                                  \
    }                                                                 \
    break;                                                            \
  }
}  // namespace

namespace aicpu {
uint32_t SparseSparseMinimumCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "SparseSparseMinimum normal check failed.");

  const Tensor *x1_indices = ctx.Input(0);
  const Tensor *x1_values_t = ctx.Input(1);
  const Tensor *x1_shape = ctx.Input(2);
  const Tensor *x2_indices = ctx.Input(3);
  const Tensor *x2_values_t = ctx.Input(4);
  const Tensor *x2_shape = ctx.Input(5);

  auto x1_indices_shape = x1_indices->GetTensorShape();
  auto x2_indices_shape = x2_indices->GetTensorShape();
  KERNEL_CHECK_FALSE(((x1_indices_shape->GetDims() == 2) && (x2_indices_shape->GetDims() == 2)),
                     KERNEL_STATUS_PARAM_INVALID, "Input indices should be matrices but received dims: %d and %d.",
                     x1_indices_shape->GetDims(), x2_indices_shape->GetDims())
  const int64_t x1_nnz = x1_indices_shape->GetDimSize(0);
  const int64_t x2_nnz = x2_indices_shape->GetDimSize(0);

  auto x1_values_shape = x1_values_t->GetTensorShape();
  auto x2_values_shape = x2_values_t->GetTensorShape();
  KERNEL_CHECK_FALSE(((x1_values_shape->GetDims() == 1) && (x2_values_shape->GetDims() == 1)),
                     KERNEL_STATUS_PARAM_INVALID, "Input values should be vectors but received dims: %d and %d.",
                     x1_values_shape->GetDims(), x2_values_shape->GetDims())
  KERNEL_CHECK_FALSE(((x1_values_t->NumElements() == x1_nnz) && (x2_values_t->NumElements() == x2_nnz)),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Expected %d and %d non-empty input values, but received : %d and %d.", x1_nnz, x2_nnz,
                     x1_values_t->NumElements(), x2_values_t->NumElements())
  KERNEL_CHECK_FALSE((x1_values_t->GetDataType() == x2_values_t->GetDataType()), KERNEL_STATUS_PARAM_INVALID,
                     "Data types of the input values should be the same, but "
                     "received %d-th and %d-th data type in the DataType enum.",
                     x1_values_t->GetDataType(), x2_values_t->GetDataType())

  auto x1_shape_shape = x1_shape->GetTensorShape();
  auto x2_shape_shape = x2_shape->GetTensorShape();
  KERNEL_CHECK_FALSE(((x1_shape_shape->GetDims() == 1) && (x2_shape_shape->GetDims() == 1)),
                     KERNEL_STATUS_PARAM_INVALID, "Input shapes should be vectors but received dims: %d and %d.",
                     x1_shape_shape->GetDims(), x2_shape_shape->GetDims())
  KERNEL_CHECK_FALSE((x1_shape_shape->GetDimSize(0) == x2_shape_shape->GetDimSize(0)), KERNEL_STATUS_PARAM_INVALID,
                     "Operands' should have the same ranks but received: %d and %d.", x1_shape_shape->GetDimSize(0),
                     x2_shape_shape->GetDimSize(0))
  auto shape_x1 = reinterpret_cast<int64_t *>(x1_shape->GetData());
  auto shape_x2 = reinterpret_cast<int64_t *>(x2_shape->GetData());
  for (int i = 0; i < x1_shape->NumElements(); ++i) {
    KERNEL_CHECK_FALSE(shape_x1[i] == shape_x2[i], KERNEL_STATUS_PARAM_INVALID,
                       "Operands' shapes do not match: got %d and %d for dimension %d", shape_x1[i], shape_x2[i], i)
  }

  auto data_type = ctx.Input(1)->GetDataType();
  switch (data_type) {
    SPARSE_SPARSE_MINIMUM_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    SPARSE_SPARSE_MINIMUM_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    SPARSE_SPARSE_MINIMUM_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    SPARSE_SPARSE_MINIMUM_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    SPARSE_SPARSE_MINIMUM_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    SPARSE_SPARSE_MINIMUM_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    SPARSE_SPARSE_MINIMUM_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    SPARSE_SPARSE_MINIMUM_COMPUTE_CASE(DT_FLOAT, float, ctx)
    SPARSE_SPARSE_MINIMUM_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("SparseSparseMinimum kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

int SparseSparseMinimumCpuKernel::cmp(const TTypes<int64_t>::ConstMatrix &x_idx, const int64_t x_row, const int dims,
                                      const TTypes<int64_t>::ConstMatrix &y_idx, const int64_t y_row) {
  for (int d = 0; d < dims; ++d) {
    const int64_t x = x_idx(x_row, d);
    const int64_t y = y_idx(y_row, d);
    if (x < y) {
      return -1;
    } else if (x > y) {
      return 1;
    }
  }
  return 0;
}

template <typename T>
uint32_t SparseSparseMinimumCpuKernel::SparseSparseMinimumCompute(CpuKernelContext &ctx) {
  const EigenTensor x1_indices_ET(ctx.Input(0), ctx.Input(0)->GetData());
  const EigenTensor x2_indices_ET(ctx.Input(3), ctx.Input(3)->GetData());
  auto x1_indices_mat = x1_indices_ET.matrix<int64_t>();
  auto x2_indices_mat = x2_indices_ET.matrix<int64_t>();

  const int64_t x1_nnz = x1_indices_mat.dimension(0);
  const int64_t x2_nnz = x2_indices_mat.dimension(0);
  std::vector<std::pair<bool, int64_t>> entries_to_copy;
  entries_to_copy.reserve(x1_nnz + x2_nnz);
  std::vector<T> out_values;
  const int num_dims = ctx.Input(2)->GetTensorShape()->GetDimSize(0);

  EigenTensor x1_values_ET(ctx.Input(1), ctx.Input(1)->GetData());
  EigenTensor x2_values_ET(ctx.Input(4), ctx.Input(4)->GetData());
  auto x1_values = x1_values_ET.vec<T>();
  auto x2_values = x2_values_ET.vec<T>();
  int64_t i = 0, j = 0;
  T s;
  while (i < x1_nnz && j < x2_nnz) {
    switch (cmp(x1_indices_mat, i, num_dims, x2_indices_mat, j)) {
      case -1:
        s = std::min(x1_values(i), T(0));
        entries_to_copy.emplace_back(true, i);
        out_values.push_back(s);
        ++i;
        break;
      case 0:
        s = std::min(x1_values(i), x2_values(j));
        entries_to_copy.emplace_back(true, i);
        out_values.push_back(s);
        ++i;
        ++j;
        break;
      case 1:
        s = std::min(T(0), x2_values(j));
        entries_to_copy.emplace_back(false, j);
        out_values.push_back(s);
        ++j;
        break;
      default:
        KERNEL_LOG_ERROR("Some inner error happens in the SparseSparseMinimum computation.");
        return KERNEL_STATUS_INNER_ERROR;
    }
  }

#define HANDLE_LEFTOVERS(X1_OR_X2, IDX, IS_A)       \
  while ((IDX) < X1_OR_X2##_nnz) {                  \
    entries_to_copy.emplace_back(IS_A, IDX);        \
    s = std::min((X1_OR_X2##_values)((IDX)), T(0)); \
    out_values.push_back(s);                        \
    ++(IDX);                                        \
  }

  HANDLE_LEFTOVERS(x1, i, true);
  HANDLE_LEFTOVERS(x2, j, false);
#undef HANDLE_LEFTOVERS

  const int64_t y_nnz = out_values.size();
  Tensor *out_indices_t = ctx.Output(0);
  EigenTensor out_indices_ET(out_indices_t, out_indices_t->GetData());
  auto out_indices_mat = out_indices_ET.matrix<int64_t>();
  for (int64_t i = 0; i < y_nnz; ++i) {
    const bool from_x1 = entries_to_copy[i].first;
    const int64_t idx = entries_to_copy[i].second;
    out_indices_mat.chip<0>(i) = from_x1 ? x1_indices_mat.chip<0>(idx) : x2_indices_mat.chip<0>(idx);
  }
  std::vector<int64_t> indices_dims = {y_nnz, num_dims};
  auto out_indices_shape = out_indices_t->GetTensorShape();
  out_indices_shape->SetDimSizes(indices_dims);
  out_indices_t->SetTensorShape(out_indices_shape.get());

  Tensor *out_values_t = ctx.Output(1);
  EigenTensor out_values_ET(out_values_t, out_values_t->GetData());
  auto out_values_flat = out_values_ET.vec<T>();
  if (y_nnz > 0) {
    std::copy_n(out_values.begin(), y_nnz, &out_values_flat(0));
  }
  std::vector<int64_t> values_dims = {y_nnz};
  auto out_values_shape = out_values_t->GetTensorShape();
  out_values_shape->SetDimSizes(values_dims);
  out_values_t->SetTensorShape(out_values_shape.get());

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSparseSparseMinimum, SparseSparseMinimumCpuKernel);
}  // namespace aicpu