/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "sparsefillemptyrows.h"

#include <securec.h>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include "cpu_kernel_utils.h"
#include "utils/allocator_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "kernel_log.h"
#include "status.h"

namespace {
const char *kSparseFillEmptyRows = "SparseFillEmptyRows";
const uint32_t kOutputNum = 4;
const uint32_t kInputNum = 4;
const uint32_t kIndexZero = 0;
const uint32_t kIndexOne = 1;
const uint32_t kIndexTwo = 2;
const uint32_t kIndexThree = 3;
const uint32_t kIndexFour = 4;
const uint32_t kIndexFive = 5;
bool isMatrix(const std::shared_ptr<aicpu::TensorShape> shape) { return shape->GetDims() == 2; }
bool isVector(const std::shared_ptr<aicpu::TensorShape> shape) { return shape->GetDims() == 1; }
}  // namespace

namespace aicpu {
template <typename T>
uint32_t SparseFillEmptyRowsCpuKernel::ComputeSparseFillEmptyRows(DataBank &databank) {
  EigenTensor indices_e(databank.indices, databank.indices->GetData());
  EigenTensor values_e(databank.values, databank.values->GetData());
  EigenTensor dense_shape_e(databank.dense_shape, databank.dense_shape->GetData());
  EigenTensor empty_row_indicator_e(databank.empty_row_indicator, databank.empty_row_indicator->GetData());
  EigenTensor reverse_index_map_e(databank.reverse_index_map, databank.reverse_index_map->GetData());
  EigenTensor y_indices_e(databank.y_indices, databank.y_indices->GetData());
  EigenTensor y_values_e(databank.y_values, databank.y_values->GetData());

  T *default_value = reinterpret_cast<T *>(databank.default_value->GetData());
  auto indices = indices_e.matrix<int64_t>();
  auto values = values_e.vec<T>();
  auto dense_shape = dense_shape_e.vec<int64_t>();
  auto empty_row_indicator = empty_row_indicator_e.vec<bool>();
  auto reverse_index_map = reverse_index_map_e.vec<int64_t>();
  auto y_indices = y_indices_e.matrix<int64_t>();
  auto y_values = y_values_e.vec<T>();

  const int64_t N = databank.indices->GetTensorShape()->GetDimSize(0);
  const int64_t dense_rows = dense_shape(0);
  int64_t rank = databank.indices->GetTensorShape()->GetDimSize(1);

  databank.empty_row_indicator->GetTensorShape()->SetDimSizes({dense_rows});
  databank.reverse_index_map->GetTensorShape()->SetDimSizes({N});

  if (dense_rows == 0) {
    KERNEL_CHECK_FALSE(N == 0, KERNEL_STATUS_PARAM_INVALID,
                       "Received SparseTensor with dense_shape[0] is 0 and not "
                       "equal to indices.shape[0]")
    // Exit early, nothing more to do.
    databank.y_indices->GetTensorShape()->SetDimSizes({0, rank});
    databank.y_values->GetTensorShape()->SetDimSizes({0});
    return KERNEL_STATUS_OK;
  }

  std::vector<int64_t> scratch(dense_rows, 0);
  std::vector<int64_t> filled_count(dense_rows, 0);
  for (int64_t i = 0; i < N; ++i) {
    const int64_t row = indices(i, 0);
    KERNEL_CHECK_FALSE(row >= 0 && row < dense_rows, KERNEL_STATUS_PARAM_INVALID, "indices is invalid")
    ++scratch[indices(i, 0)];
  }
  for (int64_t row = 0; row < dense_rows; ++row) {
    // Scratch here describes the number of elements in this dense row
    empty_row_indicator(row) = (scratch[row] == 0);
    // In filled version, each row has at least one element.
    scratch[row] = std::max(scratch[row], int64_t{1});
    if (row > 0) {
      scratch[row] += scratch[row - 1];
    }
  }
  auto ret = memset_s(databank.y_indices->GetData(), scratch[dense_rows - 1] * rank * sizeof(int64_t), 0,
                      scratch[dense_rows - 1] * rank * sizeof(int64_t));
  if (ret != 0) {
    KERNEL_LOG_ERROR("Memst failed, ret is [%d]", ret);
    return KERNEL_STATUS_INNER_ERROR;
  }
  for (int64_t i = 0; i < scratch[dense_rows - 1]; ++i) {
    y_values(i) = (*default_value);
  }

  // Fill in values for rows that are not missing
  for (int64_t i = 0; i < N; ++i) {
    const int64_t row = indices(i, 0);
    int64_t &offset = filled_count[row];
    const int64_t output_i = ((row == 0) ? 0 : scratch[row - 1]) + offset;
    offset++;  // Increment the filled count for this row.
    std::copy_n(&indices(i, 0), rank, &y_indices(output_i, 0));
    y_values(output_i) = values(i);
    // We'll need this reverse index map to backprop correctly.
    reverse_index_map(i) = output_i;
  }

  // Fill in values for rows that are missing
  for (int64_t row = 0; row < dense_rows; ++row) {
    const int64_t row_count = filled_count[row];
    if (row_count == 0) {  // We haven't filled this row
      const int64_t starting_index = (row == 0) ? 0 : scratch[row - 1];
      // Remaining index values were set to zero already.
      // The value at this index was set to default_value already.
      // Just need to set the row index in the right location.
      y_indices(starting_index, 0) = row;
    }
  }
  databank.y_indices->GetTensorShape()->SetDimSizes({scratch[dense_rows - 1], rank});
  databank.y_values->GetTensorShape()->SetDimSizes({scratch[dense_rows - 1]});
  return KERNEL_STATUS_OK;
}

uint32_t SparseFillEmptyRowsCpuKernel::NullptrAndMatVecCheck(CpuKernelContext &ctx, DataBank &databank) {
  databank.indices = ctx.Input(kIndexZero);
  KERNEL_CHECK_NULLPTR(databank.indices, KERNEL_STATUS_PARAM_INVALID, "Get input indices failed.")
  databank.values = ctx.Input(kIndexOne);
  KERNEL_CHECK_NULLPTR(databank.values, KERNEL_STATUS_PARAM_INVALID, "Get input values failed.")
  databank.dense_shape = ctx.Input(kIndexTwo);
  KERNEL_CHECK_NULLPTR(databank.dense_shape, KERNEL_STATUS_PARAM_INVALID, "Get input dense_shape failed.")
  databank.default_value = ctx.Input(kIndexThree);
  KERNEL_CHECK_NULLPTR(databank.default_value, KERNEL_STATUS_PARAM_INVALID, "Get input default_value failed.")
  databank.y_indices = ctx.Output(kIndexZero);
  KERNEL_CHECK_NULLPTR(databank.y_indices, KERNEL_STATUS_PARAM_INVALID, "Get output y_indices failed.")
  databank.y_values = ctx.Output(kIndexOne);
  KERNEL_CHECK_NULLPTR(databank.y_values, KERNEL_STATUS_PARAM_INVALID, "Get output y_values failed.")
  databank.empty_row_indicator = ctx.Output(kIndexTwo);
  KERNEL_CHECK_NULLPTR(databank.empty_row_indicator, KERNEL_STATUS_PARAM_INVALID,
                       "Get output empty_row_indicator failed.")
  databank.reverse_index_map = ctx.Output(kIndexThree);
  KERNEL_CHECK_NULLPTR(databank.reverse_index_map, KERNEL_STATUS_PARAM_INVALID, "Get output reverse_index_map failed.")
  KERNEL_CHECK_FALSE(isMatrix(databank.indices->GetTensorShape()), KERNEL_STATUS_PARAM_INVALID,
                     "Inputs indices should be matrix")
  KERNEL_CHECK_FALSE(isVector(databank.dense_shape->GetTensorShape()), KERNEL_STATUS_PARAM_INVALID,
                     "Inputs dense_shape should be vectors")
  KERNEL_CHECK_FALSE(isVector(databank.values->GetTensorShape()), KERNEL_STATUS_PARAM_INVALID,
                     "Inputs values should be vectors")
  KERNEL_CHECK_FALSE(databank.default_value->NumElements() == 1, KERNEL_STATUS_PARAM_INVALID,
                     "Input dafault_value should be scalar")
  return KERNEL_STATUS_OK;
}

uint32_t SparseFillEmptyRowsCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "SparseFillEmptyRows check input and output number failed.");
  DataBank databank;
  KERNEL_HANDLE_ERROR(NullptrAndMatVecCheck(ctx, databank), "SparseFillEmptyRows check params failed.");
  DataType dt = static_cast<DataType>(databank.values->GetDataType());

  uint32_t KERNEL_STATUS;
  switch (dt) {
    case DT_INT8:
      KERNEL_STATUS = ComputeSparseFillEmptyRows<int8_t>(databank);
      break;
    case DT_UINT8:
      KERNEL_STATUS = ComputeSparseFillEmptyRows<uint8_t>(databank);
      break;
    case DT_INT16:
      KERNEL_STATUS = ComputeSparseFillEmptyRows<int16_t>(databank);
      break;
    case DT_UINT16:
      KERNEL_STATUS = ComputeSparseFillEmptyRows<uint16_t>(databank);
      break;
    case DT_INT32:
      KERNEL_STATUS = ComputeSparseFillEmptyRows<int32_t>(databank);
      break;
    case DT_UINT32:
      KERNEL_STATUS = ComputeSparseFillEmptyRows<uint32_t>(databank);
      break;
    case DT_INT64:
      KERNEL_STATUS = ComputeSparseFillEmptyRows<int64_t>(databank);
      break;
    case DT_UINT64:
      KERNEL_STATUS = ComputeSparseFillEmptyRows<uint64_t>(databank);
      break;
    case DT_STRING:
      KERNEL_STATUS = ComputeSparseFillEmptyRows<std::string>(databank);
      break;
    case DT_FLOAT16:
      KERNEL_STATUS = ComputeSparseFillEmptyRows<Eigen::half>(databank);
      break;
    case DT_FLOAT:
      KERNEL_STATUS = ComputeSparseFillEmptyRows<float>(databank);
      break;
    case DT_DOUBLE:
      KERNEL_STATUS = ComputeSparseFillEmptyRows<double>(databank);
      break;
    case DT_BOOL:
      KERNEL_STATUS = ComputeSparseFillEmptyRows<bool>(databank);
      break;
    case DT_COMPLEX64:
      KERNEL_STATUS = ComputeSparseFillEmptyRows<std::complex<float>>(databank);
      break;
    case DT_COMPLEX128:
      KERNEL_STATUS = ComputeSparseFillEmptyRows<std::complex<double>>(databank);
      break;
    default:
      KERNEL_LOG_ERROR("SparseFillEmptyRows can't support this data type [%s].", DTypeStr(dt).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (KERNEL_STATUS != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("SparseFillEmptyRows failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSparseFillEmptyRows, SparseFillEmptyRowsCpuKernel);
}  // namespace aicpu
