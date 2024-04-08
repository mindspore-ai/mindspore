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
#include "cpu_kernel/ms_kernel/bincount.h"
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;
const char *kBincount = "Bincount";
const int64_t kParallelDataNum = 64 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;

#define BINCOUNT_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                    \
    uint32_t result = BincountCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                                \
      CUST_KERNEL_LOG_ERROR(ctx, "Bincount kernel compute failed."); \
      return result;                                                 \
    }                                                                \
    break;                                                           \
  }

int64_t get_tensor_length(aicpu::Tensor *t) {
  std::vector<int64_t> dim_sizes = t->GetTensorShape()->GetDimSizes();
  int64_t length = 1;
  length = std::accumulate(dim_sizes.begin(), dim_sizes.end(), 1, std::multiplies<int64_t>());
  return length;
}
}  // namespace

namespace aicpu {
template <typename T_in, typename T_out>
void BincountTask(Tensor *input_arr, int32_t num_bins, Tensor *input_weights, Tensor *output, CpuKernelContext &ctx) {
  auto bin_array = reinterpret_cast<T_in *>(input_arr->GetData());
  T_out *bin_weights = nullptr;
  if (input_weights != nullptr) {
    bin_weights = reinterpret_cast<T_out *>(input_weights->GetData());
  }
  auto output_data = reinterpret_cast<T_out *>(output->GetData());

  const int64_t data_num = get_tensor_length(input_arr);
  for (int64_t i = 0; i < num_bins; i++) {
    output_data[i] = 0;
  }

  if (input_weights == nullptr) {
    for (int64_t i = 0; i < data_num; i++) {
      T_in value = bin_array[i];
      if (value < num_bins) {
        output_data[value] += T_out(1);
      }
    }
  } else {
    for (int64_t i = 0; i < data_num; i++) {
      T_in value = bin_array[i];
      if (value < num_bins) {
        output_data[value] += bin_weights[i];
      }
    }
  }
}

void BincountCpuKernel::SetMap() {
  calls_[DT_INT32][DT_FLOAT] = BincountTask<int32_t, float>;
  calls_[DT_INT32][DT_INT32] = BincountTask<int32_t, int32_t>;
  calls_[DT_INT32][DT_INT64] = BincountTask<int32_t, int64_t>;
  calls_[DT_INT32][DT_DOUBLE] = BincountTask<int32_t, double>;
}

uint32_t BincountCpuKernel::Compute(CpuKernelContext &ctx) {
  // normal check
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "Bincount check input and output number failed.");

  Tensor *input_arr = ctx.Input(kFirstInputIndex);
  Tensor *input_size = ctx.Input(kSecondInputIndex);
  Tensor *input_weights = ctx.Input(kThirdInputIndex);

  bool has_weight = (input_weights->GetDataSize() != 0);
  if (has_weight) {
    auto input_arr_sizes = input_arr->GetTensorShape()->NumElements();
    auto input_weights_sizes = input_weights->GetTensorShape()->NumElements();
    CUST_KERNEL_CHECK_FALSE(ctx, (input_arr_sizes == input_weights_sizes), KERNEL_STATUS_PARAM_INVALID,
                            "The shape size of input_arr [%d] need be same with"
                            "input_weights [%d].",
                            input_arr_sizes, input_weights_sizes);
  }

  // check input datatype
  DataType dt_arr = input_arr->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, (dt_arr == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                          "Input_arr data type must DT_INT32, but got data type[%s].", DTypeStr(dt_arr).c_str());

  DataType dt_size = input_size->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, (dt_size == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                          "Input_size data type must DT_INT32, but got data type[%s].", DTypeStr(dt_size).c_str());

  DataType dt_weights = input_weights->GetDataType();
  CUST_KERNEL_CHECK_FALSE(
    ctx, (dt_weights == DT_FLOAT || dt_weights == DT_INT32 || dt_weights == DT_INT64 || dt_weights == DT_DOUBLE),
    KERNEL_STATUS_PARAM_INVALID,
    "Input_weights data type must DT_FLOAT or DT_INT32 or DT_INT64 or DT_DOUBLE,"
    "but got data type[%s].",
    DTypeStr(dt_weights).c_str());

  // check input dimension
  CUST_KERNEL_CHECK_FALSE(ctx,
                          (input_size->GetTensorShape()->GetDims() == 0 ||
                           (input_size->GetTensorShape()->GetDims() == 1 && get_tensor_length(input_size) == 1)),
                          KERNEL_STATUS_PARAM_INVALID, "Input_size should be a scalar");

  // check num_bins nonnegative
  auto num_bins = reinterpret_cast<int32_t *>(input_size->GetData());  // int32_t
  CUST_KERNEL_CHECK_FALSE(ctx, (*num_bins >= 0), KERNEL_STATUS_PARAM_INVALID,
                          "num_size should be nonnegative, but got [%d].", *num_bins);

  // check input_arr nonnegative
  auto bin_array = reinterpret_cast<int32_t *>(input_arr->GetData());
  const int64_t array_num = get_tensor_length(input_arr);
  for (int64_t i = 0; i < array_num; i++) {
    CUST_KERNEL_CHECK_FALSE(ctx, (bin_array[i] >= 0), KERNEL_STATUS_PARAM_INVALID,
                            "array should be nonnegative, but got [%d].", bin_array[i]);
  }

  // check output datatype
  Tensor *output = ctx.Output(kFirstOutputIndex);
  DataType dt_output = output->GetDataType();
  CUST_KERNEL_CHECK_FALSE(
    ctx, (dt_output == DT_FLOAT || dt_output == DT_INT32 || dt_output == DT_INT64 || dt_output == DT_DOUBLE),
    KERNEL_STATUS_PARAM_INVALID,
    "Output data type must DT_FLOAT or DT_INT32 or DT_INT64 or DT_DOUBLE,"
    "but got data type[%s].",
    DTypeStr(dt_output).c_str());

  // check that input weights and output have the same datatype
  CUST_KERNEL_CHECK_FALSE(ctx, (dt_weights == dt_output), KERNEL_STATUS_PARAM_INVALID,
                          "The data type of input_weights [%s] need be same with "
                          "output [%s].",
                          DTypeStr(dt_weights).c_str(), DTypeStr(dt_output).c_str());

  SetMap();
  if (!has_weight) {
    input_weights = nullptr;
  }

  calls_[dt_arr][dt_weights](input_arr, *num_bins, input_weights, output, ctx);
  calls_.clear();

  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kBincount, BincountCpuKernel);
}  // namespace aicpu
