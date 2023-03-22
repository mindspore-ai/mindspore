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
#include "sparse_softmax.h"

#include <securec.h>
#include <iostream>
#include <stack>
#include <memory>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_log.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "utils/sparse_tensor.h"

namespace {
const uint32_t kSparseSoftmaxInputNum = 3;
const uint32_t kSparseSoftmaxOutputNum = 1;
const uint32_t kIndex0 = 0;
const uint32_t kIndex1 = 1;
const uint32_t kIndex2 = 2;
const uint32_t kSize1 = 1;
const uint32_t kSize2 = 2;
const char *kSparseSoftmax = "SparseSoftmax";
#define SPARSESOFTMAX_COMPUTE_CASE(DTYPE, TYPE, CTX)         \
  case (DTYPE): {                                            \
    uint32_t result = SparseSoftmaxCompute<TYPE>(CTX);       \
    if (result != KERNEL_STATUS_OK) {                        \
      KERNEL_LOG_ERROR("SparseSoft kernel compute failed."); \
      return result;                                         \
    }                                                        \
    break;                                                   \
  }

inline bool CompareIndices(const int64_t *a, const int64_t *b, const size_t &len) {
  size_t i = 0;
  while (i < len) {
    if (a[i] != b[i]) {
      return a[i] > b[i];
    }
    ++i;
  }
  return true;
}

template <typename T>
inline void CopyIndicesAndValue(int64_t *dst_indices_addr, T *dst_values_addr, const int64_t *src_indices_addr,
                                const T *src_values_addr, const size_t &indices_size) {
  memcpy_s(dst_indices_addr, indices_size, src_indices_addr, indices_size);
  *dst_values_addr = *src_values_addr;
}

template <typename T>
inline int64_t Partition(int64_t *__restrict indices_addr, T *__restrict values_addr, int64_t *__restrict tmp_indices,
                         const size_t &indices_len, const int64_t &left, const int64_t &right) {
  int64_t i = left, j = right;
  T tmp_values = 0;
  const size_t indices_size = indices_len * sizeof(int64_t);
#define INDICES_OFFSET_ADDR(addr, index, len) addr + index *len

  CopyIndicesAndValue(tmp_indices, &tmp_values, INDICES_OFFSET_ADDR(indices_addr, left, indices_len),
                      values_addr + left, indices_size);
  while (i < j) {
    while (i < j && CompareIndices(INDICES_OFFSET_ADDR(indices_addr, j, indices_len), tmp_indices, indices_len)) {
      --j;
    }
    CopyIndicesAndValue(INDICES_OFFSET_ADDR(indices_addr, i, indices_len), values_addr + i,
                        INDICES_OFFSET_ADDR(indices_addr, j, indices_len), values_addr + j, indices_size);
    while (i < j && !CompareIndices(INDICES_OFFSET_ADDR(indices_addr, i, indices_len), tmp_indices, indices_len)) {
      ++i;
    }
    CopyIndicesAndValue(INDICES_OFFSET_ADDR(indices_addr, j, indices_len), values_addr + j,
                        INDICES_OFFSET_ADDR(indices_addr, i, indices_len), values_addr + i, indices_size);
  }
  CopyIndicesAndValue(INDICES_OFFSET_ADDR(indices_addr, i, indices_len), values_addr + i, tmp_indices, &tmp_values,
                      indices_size);
  return i;
}

template <typename T>
void QuickSortIndicesAndValues(int64_t *__restrict indices_addr, T *__restrict values_addr, const size_t indices_len,
                               const int64_t &left, const int64_t &right) {
  std::stack<int64_t> index_stk;
  index_stk.emplace(right);
  index_stk.emplace(left);
  int64_t *indices_buff = new int64_t[indices_len];

  while (!index_stk.empty()) {
    int64_t i = index_stk.top();
    index_stk.pop();
    int64_t j = index_stk.top();
    index_stk.pop();
    if (i < j) {
      int64_t k = Partition(indices_addr, values_addr, indices_buff, indices_len, i, j);
      if (k > i) {
        index_stk.emplace(k - 1);
        index_stk.emplace(i);
      }
      if (j > k) {
        index_stk.emplace(j);
        index_stk.emplace(k + 1);
      }
    }
  }
  delete[] indices_buff;
}
}  // namespace

namespace aicpu {
uint32_t SparseSoftmaxCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kSparseSoftmaxInputNum, kSparseSoftmaxOutputNum),
                      "[%s] check input and output failed.", kSparseSoftmax);
  // parse params
  KERNEL_HANDLE_ERROR(SparseSoftmaxCheck(ctx), "[%s] check params failed.", kSparseSoftmax);
  auto data_type = ctx.Input(1)->GetDataType();
  switch (data_type) {
    SPARSESOFTMAX_COMPUTE_CASE(DT_FLOAT, float, ctx)
    SPARSESOFTMAX_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("SparseSoftmax kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
uint32_t SparseSoftmaxCpuKernel::SparseSoftmaxCheck(CpuKernelContext &ctx) {
  std::vector<int64_t> shape_indices = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_values = ctx.Input(kIndex1)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_shape = ctx.Input(kIndex2)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_output = ctx.Output(kIndex0)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_indices.size() == kSize2), KERNEL_STATUS_PARAM_INVALID,
                     "Indices must be rank 2D, got [%zu].", shape_indices.size())
  KERNEL_CHECK_FALSE((shape_values.size() == kSize1), KERNEL_STATUS_PARAM_INVALID, "values must be rank 1D, got [%zu].",
                     shape_values.size())
  KERNEL_CHECK_FALSE((shape_shape.size() == kSize1), KERNEL_STATUS_PARAM_INVALID, "shape must be rank 1D, got [%zu].",
                     shape_shape.size())
  KERNEL_CHECK_FALSE((ctx.Input(kIndex2)->GetTensorShape()->NumElements() >= kSize2), KERNEL_STATUS_PARAM_INVALID,
                     "shape number must be more than 1, got [%zu].", shape_shape.size())
  KERNEL_CHECK_FALSE((shape_values.size() == shape_output.size()), KERNEL_STATUS_PARAM_INVALID,
                     "The input shape size should be same as the output shape size")
  const int64_t nnz = shape_indices[0];
  const int64_t data_num = ctx.Input(kIndex1)->NumElements();
  KERNEL_CHECK_FALSE((nnz == data_num), KERNEL_STATUS_PARAM_INVALID,
                     "The values number should be same as the indices_size(0)");
  auto data_type_indices = ctx.Input(kIndex0)->GetDataType();
  auto data_type_shape = ctx.Input(kIndex2)->GetDataType();
  KERNEL_CHECK_FALSE((data_type_indices == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     "data type of indices should be int64");
  KERNEL_CHECK_FALSE((data_type_shape == DT_INT64), KERNEL_STATUS_PARAM_INVALID, "data type of shape should be int64");
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t SparseSoftmaxCpuKernel::SparseSoftmaxCompute(CpuKernelContext &ctx) {
  int64_t data_num = ctx.Input(kIndex1)->NumElements();

  auto *indices_t = ctx.Input(kIndex0);
  auto *values_t = ctx.Input(kIndex1);
  auto *shape_t = ctx.Input(kIndex2);
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  std::vector<int64_t> shape_indices = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  const int64_t nnz = shape_indices[0];
  const int64_t rank = static_cast<int64_t>(shape_indices[1]);

  SparseTensor st;

  std::vector<int64_t> order;
  std::vector<int64_t> shape_flat;
  int64_t *temp_dim = reinterpret_cast<int64_t *>(shape_t->GetData());
  for (int32_t index = 0; index < shape_t->GetTensorShape()->GetDimSize(0); ++index) {
    shape_flat.emplace_back(temp_dim[index]);
    order.push_back(shape_flat[index]);
  }
  std::iota(order.begin(), order.end(), 0);

  QuickSortIndicesAndValues(reinterpret_cast<int64_t *>(indices_t->GetData()),
                            reinterpret_cast<T *>(values_t->GetData()), shape_t->NumElements(), 0, data_num - 1);

  if (st.CreateSparseTensor(indices_t, values_t, shape_flat, order) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Create sparse tensor failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  Eigen::Tensor<T, 1, Eigen::RowMajor> output_flat(nnz);

  // { 0, ..., rank-1 }.
  std::vector<int64_t> kReorderDims(rank);
  std::iota(kReorderDims.begin(), kReorderDims.end(), 0);
  // All but the last dim -- the class dimension to be max-reduced along.
  std::vector<int64_t> kGroupByDims(rank - 1);
  std::iota(kGroupByDims.begin(), kGroupByDims.end(), 0);
  st.Reorder<T>(kReorderDims);

  int64_t count = 0;

  for (const auto &g : st.group(kGroupByDims)) {
    const auto group_vals = g.values<T>();
    const int group_size = group_vals.size();
    Eigen::Tensor<T, 0, Eigen::RowMajor> tmp_scalar;
    tmp_scalar = group_vals.maximum();

    Eigen::Tensor<T, 1, Eigen::RowMajor> tmp(group_size);
    tmp = (group_vals - tmp.constant(tmp_scalar())).exp();
    tmp_scalar = tmp.sum().inverse();

    tmp = tmp * tmp.constant(tmp_scalar());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> output_part(output_flat.data() + count, group_size);
    output_part = tmp;

    count += group_size;
  }
  for (int64_t index = 0; index < data_num; ++index) {
    output_data[index] = static_cast<T>(output_flat[index]);
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kSparseSoftmax, SparseSoftmaxCpuKernel);
}  // namespace aicpu
