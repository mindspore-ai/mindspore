/**
 * Copyright 2021 Huawei Technologies Co., Ltd.
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
#include "list_diff.h"
#include <unordered_set>
#include "cpu_kernel_utils.h"
#include "kernel_log.h"
#include "securec.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/allocator_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kListDiff = "ListDiff";
constexpr uint32_t kListDiffInputNum = 2;
constexpr uint32_t kListDiffOutputNum = 2;

#define LIST_DIFF_COMPUTE_CASE(DTYPE, TYPE, OUT_IDX, CTX) \
  case (DTYPE): {                                         \
    uint32_t result = KERNEL_STATUS_INNER_ERROR;          \
    if ((OUT_IDX) == DT_INT32) {                          \
      result = DoCompute<TYPE, int32_t>(CTX);             \
    } else {                                              \
      result = DoCompute<TYPE, int64_t>(CTX);             \
    }                                                     \
    if (result != KERNEL_STATUS_OK) {                     \
      KERNEL_LOG_ERROR("Less kernel compute failed.");    \
      return result;                                      \
    }                                                     \
    break;                                                \
  }
}  // namespace

namespace aicpu {
uint32_t ListDiffCpuKernel::ParamCheck(CpuKernelContext &ctx) {
  // check input number and output number
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kListDiffInputNum, kListDiffOutputNum), "[%s] check params failed.", kListDiff);
  // get all input and output
  const Tensor *x = ctx.Input(0);
  const Tensor *y = ctx.Input(1);
  const Tensor *out = ctx.Output(0);
  const Tensor *idx = ctx.Output(1);

  // input tensor must be 1D vector
  KERNEL_CHECK_FALSE(IsVector(x->GetTensorShape()->GetDimSizes()), KERNEL_STATUS_PARAM_INVALID,
                     "Input Tensor x should be a 1D vector.");
  KERNEL_CHECK_FALSE(IsVector(y->GetTensorShape()->GetDimSizes()), KERNEL_STATUS_PARAM_INVALID,
                     "Input Tensor y should be a 1D vector.");
  // out_idx type check
  AttrValue *out_idx_att = ctx.GetAttr("out_idx");
  if (out_idx_att) {
    // private value out_idx store out_idx
    out_idx = out_idx_att->GetDataType();
    KERNEL_CHECK_FALSE((out_idx == DT_INT32 || out_idx == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                       "attr value 'out_idx_att' should in (DT_INT32, DT_INT64)");
  }
  // datype check for x, y, out, idx
  KERNEL_CHECK_FALSE(x->GetDataType() == y->GetDataType() && y->GetDataType() == out->GetDataType(),
                     KERNEL_STATUS_PARAM_INVALID, "The DataType of input x and y should be same");
  KERNEL_CHECK_FALSE(idx->GetDataType() == out_idx, KERNEL_STATUS_PARAM_INVALID,
                     "The DataType of idx should be out_idx");

  return KERNEL_STATUS_OK;
}

template <typename T, typename Tidx>
uint32_t ListDiffCpuKernel::DoCompute(CpuKernelContext &ctx) {
  Tensor *x = ctx.Input(0);
  Tensor *y = ctx.Input(1);
  Tensor *out = ctx.Output(0);
  Tensor *idx = ctx.Output(1);
  // construct EigenTensor
  EigenTensor x_et(x, x->GetData());
  EigenTensor y_et(y, y->GetData());

  const auto x_vec = x_et.vec<T>();
  const size_t x_size = x_vec.size();
  const auto y_vec = y_et.vec<T>();
  const size_t y_size = y_vec.size();

  std::unordered_set<T> y_set;
  y_set.reserve(y_size);
  for (size_t i = 0; i < y_size; ++i) {
    y_set.insert(y_vec(i));
  }

  // Compute the size of the output.
  uint64_t out_size = 0;
  for (size_t i = 0; i < x_size; ++i) {
    if (0 == y_set.count(x_vec(i))) {
      ++out_size;
    }
  }
  // allocate memory for out and idx
  DataType out_type = out->GetDataType();
  // this function just allocate memory. Must update TensorShape by hands
  uint32_t ret = CpuKernelAllocatorUtils::AllocateOutputTensorDataMemory({out_size}, out_type, out);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_INNER_ERROR, "Allocate memory for out tensor failed.");
  ret = CpuKernelAllocatorUtils::AllocateOutputTensorDataMemory({out_size}, out_idx, idx);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_INNER_ERROR, "Allocate memory for idx tensor failed.")
  // construct EigenTensor
  EigenTensor out_et(out, out->GetData());
  EigenTensor idx_et(idx, idx->GetData());
  auto out_vec = out_et.vec<T>();
  auto idx_vec = idx_et.vec<Tidx>();

  // calculate results
  for (Tidx i = 0, p = 0; i < static_cast<Tidx>(x_size); ++i) {
    if (0 == y_set.count(x_vec(i))) {
      KERNEL_CHECK_FALSE(p < static_cast<Tidx>(out_size), KERNEL_STATUS_INNER_ERROR,
                         "Tried to set output index failure for index out of out_size");
      out_vec(p) = x_vec(i);
      idx_vec(p) = i;
      p++;
    }
  }
  // update out tensor shape information required by mindspore
  std::vector<int64_t> shapes = {static_cast<int64_t>(out_size)};
  auto out_shape = out->GetTensorShape();
  out_shape->SetDimSizes(shapes);
  auto idx_shape = idx->GetTensorShape();
  idx_shape->SetDimSizes(shapes);
  return KERNEL_STATUS_OK;
}

uint32_t ListDiffCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(ParamCheck(ctx), "ListDiffCpuKernel check params failed");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    LIST_DIFF_COMPUTE_CASE(DT_INT8, int8_t, out_idx, ctx)
    LIST_DIFF_COMPUTE_CASE(DT_INT16, int16_t, out_idx, ctx)
    LIST_DIFF_COMPUTE_CASE(DT_INT32, int32_t, out_idx, ctx)
    LIST_DIFF_COMPUTE_CASE(DT_INT64, int64_t, out_idx, ctx)
    LIST_DIFF_COMPUTE_CASE(DT_UINT8, uint8_t, out_idx, ctx)
    LIST_DIFF_COMPUTE_CASE(DT_UINT16, uint16_t, out_idx, ctx)
    LIST_DIFF_COMPUTE_CASE(DT_FLOAT16, Eigen::half, out_idx, ctx)
    LIST_DIFF_COMPUTE_CASE(DT_FLOAT, float, out_idx, ctx)
    LIST_DIFF_COMPUTE_CASE(DT_DOUBLE, double, out_idx, ctx)
    default:
      KERNEL_LOG_ERROR("ListDiff kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kListDiff, ListDiffCpuKernel);
}  // namespace aicpu