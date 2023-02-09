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
#include "setsize.h"
#include <securec.h>
#include "cpu_kernel_utils.h"
#include "status.h"
#include "utils/kernel_util.h"
using namespace std;

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;
const char *kSetSize = "SetSize";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNumMid = 16 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;

#define SETSIZE_COMPUTE_CASE(DTYPE, TYPE, CTX, ST)        \
  case (DTYPE): {                                         \
    uint32_t result;                                      \
    result = SetSizeCompute<TYPE>(CTX, ST);               \
    if (result != KERNEL_STATUS_OK) {                     \
      KERNEL_LOG_ERROR("SetSize kernel compute failed."); \
      return result;                                      \
    }                                                     \
    break;                                                \
  }
}  // namespace

namespace aicpu {
uint32_t SetSizeCpuKernel::Compute(CpuKernelContext &ctx) {
  set_indices_ = ctx.Input(0);
  set_values_ = ctx.Input(1);
  int64_t input_index = 2;
  set_shape_ = ctx.Input(input_index);
  output_ = ctx.Output(0);
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "SetSize check input and output number failed.");
  // check dim
  KERNEL_CHECK_FALSE((set_indices_->GetTensorShape()->GetDims() == input_index), KERNEL_STATUS_PARAM_INVALID,
                     "Indices tensor dim size equal to 2, got size [%d].", set_indices_->GetTensorShape()->GetDims())
  KERNEL_CHECK_FALSE((set_values_->GetTensorShape()->GetDims() == 1), KERNEL_STATUS_PARAM_INVALID,
                     "Values tensor dim size equal to 1, got size [%d].", set_values_->GetTensorShape()->GetDims())
  KERNEL_CHECK_FALSE((set_shape_->GetTensorShape()->GetDims() == 1), KERNEL_STATUS_PARAM_INVALID,
                     "Shape tensor dim size equal to 1, got size [%d].", set_shape_->GetTensorShape()->GetDims())
  auto data_type_0 = set_indices_->GetDataType();
  auto data_type_1 = set_values_->GetDataType();
  auto data_type_2 = set_shape_->GetDataType();
  KERNEL_CHECK_FALSE((data_type_0 == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 requested dtype int64 for Tensor "
                     "with dtype [%s]",
                     DTypeStr(data_type_0).c_str())
  KERNEL_CHECK_FALSE((data_type_2 == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input2 requested dtype int64 for Tensor "
                     "with dtype [%s]",
                     DTypeStr(data_type_2).c_str())
  dims_ = set_indices_->GetTensorShape()->GetDimSize(1);
  AttrValue *validate_indices = ctx.GetAttr("validate_indices");
  validate_indices_ = (validate_indices == nullptr) ? true : (validate_indices->GetBool());
  SparseTensor st;
  if (SparseTensorFromContext(ctx, validate_indices->GetBool(), st) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Create sparse tensor failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  switch (data_type_1) {
    SETSIZE_COMPUTE_CASE(DT_INT8, int8_t, ctx, st)
    SETSIZE_COMPUTE_CASE(DT_INT16, int16_t, ctx, st)
    SETSIZE_COMPUTE_CASE(DT_INT32, int32_t, ctx, st)
    SETSIZE_COMPUTE_CASE(DT_INT64, int64_t, ctx, st)
    SETSIZE_COMPUTE_CASE(DT_UINT8, uint8_t, ctx, st)
    SETSIZE_COMPUTE_CASE(DT_UINT16, uint16_t, ctx, st)
    case DT_STRING:
      uint32_t result;
      result = SetSizeCompute_string(ctx, st);
      if (result != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("SetSize kernel compute failed.");
        return result;
      }
      break;
    default:
      KERNEL_LOG_ERROR(
        "Value passed to parameter 'set_values_' has DataType [%s] not in "
        "list of allowed values: int8, int16, int32, int64, uint8, uint16, "
        "string.",
        DTypeStr(data_type_1).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t SetSizeCpuKernel::SparseTensorFromContext(CpuKernelContext &ctx, const bool validate_indices,
                                                   SparseTensor &st) {
  auto sparse_shape = set_shape_->GetTensorShape();
  std::vector<int64_t> dense_shape;
  std::vector<int64_t> order;
  for (int32_t index = 0; index < sparse_shape->GetDimSize(0); ++index) {
    int64_t *temp_dim = reinterpret_cast<int64_t *>(set_shape_->GetData());
    dense_shape.emplace_back(temp_dim[index]);
    order.push_back(dense_shape[index]);
  }
  shape_.assign(dense_shape.begin(), dense_shape.end());
  order_.assign(order.begin(), order.end());
  std::iota(order.begin(), order.end(), 0);
  uint32_t result = st.CreateSparseTensor(set_indices_, set_values_, dense_shape, order);
  if (!validate_indices || result != KERNEL_STATUS_OK) {
    return result;
  }
  return IndicesValid(ctx, st);
}

uint32_t SetSizeCpuKernel::IndicesValid(CpuKernelContext &ctx, SparseTensor &st) {
  int64_t dim_size =
    (set_indices_->GetTensorShape()->GetDims() == 0) ? 1 : set_indices_->GetTensorShape()->GetDimSize(0);
  if (dim_size >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (dim_size <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);
    }
    if (max_core_num > dim_size) {
      max_core_num = dim_size;
    }
    auto invalid_setsize = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        if (st.EigenTensorIndicesValid<int64_t>(ctx) != KERNEL_STATUS_OK) {
          KERNEL_LOG_ERROR("Indices valid failed.");
          return KERNEL_STATUS_PARAM_INVALID;
        }
      }
      return KERNEL_STATUS_OK;
    };
    if (max_core_num == 0) {
      return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, dim_size, dim_size / max_core_num, invalid_setsize),
                        "SetSize Compute failed.");
  } else {
    for (int64_t n = 0; n < dim_size; ++n) {
      if (st.EigenTensorIndicesValid<int64_t>(ctx) != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Indices valid failed.");
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SetSizeCpuKernel::CheckGroup(CpuKernelContext &ctx, const Group &group,
                                      const std::vector<int64_t> &sparse_tensor_shape) {
  const int64_t num_values = ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  const auto indices_t = reinterpret_cast<int64_t *>(ctx.Input(0)->GetData());
  for (int32_t j = 0; j < dims_; ++j) {
    const auto dim_size = sparse_tensor_shape[j];
    KERNEL_CHECK_FALSE((dim_size > 0), KERNEL_STATUS_PARAM_INVALID, "Invalid dim_size [%d] = [%d].", j, dim_size)
    for (int64_t i = 0; i < num_values; ++i) {
      const auto index = *(indices_t + (i * dims_) + j);
      KERNEL_CHECK_FALSE((dim_size > 0), KERNEL_STATUS_PARAM_INVALID, "indices[%d, %d] expected < %d, got %d.", i, j,
                         dim_size, index)
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SetSizeCpuKernel::PopulateFromSparseGroup(CpuKernelContext &ctx, const Group &group,
                                                   const std::vector<int64_t> &sparse_tensor_shape,
                                                   std::unordered_set<T> *result) {
  if (validate_indices_ == false) CheckGroup<T>(ctx, group, sparse_tensor_shape);
  result->clear();
  const auto &group_values = group.values<T>();
  int64_t dim_size = group_values.size();
  if (dim_size >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (dim_size <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);
    }
    if (max_core_num > dim_size) {
      max_core_num = dim_size;
    }
    auto group_value_setsize = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        result->insert(group_values(i));
      }
    };
    if (max_core_num == 0) {
      return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, dim_size, dim_size / max_core_num, group_value_setsize),
                        "SetSize Compute failed.");
  } else {
    for (int64_t i = 0; i < group_values.size(); ++i) {
      result->insert(group_values(i));
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SetSizeCpuKernel::SetSizeCompute(CpuKernelContext &ctx, SparseTensor &st) {
  auto output_t = reinterpret_cast<int32_t *>(ctx.Output(0)->GetData());
  std::vector<int64_t> group_ix(dims_ - 1);
  std::iota(group_ix.begin(), group_ix.end(), 0);
  std::vector<int64_t> strides(dims_);
  int64_t num2 = 2;
  auto shape_t = reinterpret_cast<int64_t *>(ctx.Input(num2)->GetData());
  if (dims_ > 1) {
    strides[dims_ - num2] = 1;
  }
  for (int32_t d = dims_ - 3; d >= 0; --d) {
    strides[d] = strides[d + 1] * shape_t[d + 1];
  }
  int64_t output_size = 1;
  for (int32_t d = 0; d < dims_ - 1; ++d) {
    output_size = output_size * shape_t[d];
  }
  memset_s(output_t, sizeof(int32_t) * output_size, 0, sizeof(int32_t) * output_size);
  std::unordered_set<T> group_set;
  for (const auto &group : st.group(group_ix)) {
    uint32_t result = PopulateFromSparseGroup<T>(ctx, group, shape_, &group_set);
    if (result != KERNEL_STATUS_OK) {
      KERNEL_LOG_ERROR("SetSize kernel compute failed.");
      return result;
    }
    const auto group_key = group.group();
    const auto output_index = std::inner_product(group_key.begin(), group_key.end(), strides.begin(), 0LL);
    *(output_t + output_index) = (int32_t)group_set.size();
  }
  return KERNEL_STATUS_OK;
}

uint32_t SetSizeCpuKernel::SetSizeCompute_string(CpuKernelContext &ctx, SparseTensor &st) {
  auto output_t = reinterpret_cast<int32_t *>(ctx.Output(0)->GetData());
  std::vector<int64_t> group_ix(dims_ - 1);
  std::iota(group_ix.begin(), group_ix.end(), 0);
  std::vector<int64_t> strides(dims_);
  auto shape_t = reinterpret_cast<int64_t *>(ctx.Input(2)->GetData());
  int64_t num2 = 2;
  if (dims_ > 1) {
    strides[dims_ - num2] = 1;
  }
  for (int32_t d = dims_ - 3; d >= 0; --d) {
    strides[d] = strides[d + 1] * shape_t[d + 1];
  }
  int32_t output_size = 1;
  for (int32_t d = 0; d < dims_ - 1; ++d) {
    output_size = output_size * shape_t[d];
  }
  memset_s(output_t, sizeof(int32_t) * output_size, 0, sizeof(int32_t) * output_size);
  std::unordered_set<std::string> group_set;
  for (const auto &group : st.group(group_ix)) {
    PopulateFromSparseGroup<std::string>(ctx, group, shape_, &group_set);
    const auto group_key = group.group();
    const auto output_index = std::inner_product(group_key.begin(), group_key.end(), strides.begin(), 0LL);
    *(output_t + output_index) = (int32_t)group_set.size();
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kSetSize, SetSizeCpuKernel);
}  // namespace aicpu
