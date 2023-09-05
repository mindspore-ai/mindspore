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
#include "cpu_kernel/ms_kernel/bucketize.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kBucketize = "Bucketize";
const int64_t kParallelDataNumSameShape = 64 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;

#define BUCKETIZE_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                           \
    uint32_t result = BucketizeCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                       \
      KERNEL_LOG_ERROR("Bucketize kernel compute failed."); \
      return result;                                        \
    }                                                       \
    break;                                                  \
  }

int64_t get_tensor_length(aicpu::Tensor *t) {
  std::vector<int64_t> dim_sizes = t->GetTensorShape()->GetDimSizes();
  int64_t length = 1;
  length = std::accumulate(dim_sizes.begin(), dim_sizes.end(), 1, std::multiplies<int64_t>());
  return length;
}
}  // namespace

namespace aicpu {
uint32_t BucketizeCpuKernel::Compute(CpuKernelContext &ctx) {
  // normal check
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Bucketize check input and output number failed.");
  auto data_type = ctx.Input(0)->GetDataType();

  KERNEL_CHECK_NULLPTR(ctx.GetAttr("boundaries"), KERNEL_STATUS_PARAM_INVALID, "Get boundaries failed")

  // check input datatype
  Tensor *input = ctx.Input(kFirstInputIndex);
  DataType dt_input = input->GetDataType();
  KERNEL_CHECK_FALSE((dt_input == DT_FLOAT || dt_input == DT_INT32 || dt_input == DT_INT64 || dt_input == DT_DOUBLE),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Input data type must DT_FLOAT or DT_INT32 or DT_INT64 or DT_DOUBLE,"
                     "but got data type[%s].",
                     DTypeStr(dt_input).c_str());

  // check output datatype
  Tensor *output = ctx.Output(kFirstOutputIndex);
  DataType dt_output = output->GetDataType();
  KERNEL_CHECK_FALSE((dt_output == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                     "Output data type must DT_INT32, but got data type[%s].", DTypeStr(dt_output).c_str());

  auto input_sizes = input->GetTensorShape()->GetDimSizes();
  auto output_sizes = output->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((input_sizes == output_sizes), KERNEL_STATUS_PARAM_INVALID,
                     "The tensor shape of input [%s] need be same with "
                     "output [%s].",
                     VectorToString(input_sizes).c_str(), VectorToString(output_sizes).c_str());

  switch (data_type) {
    BUCKETIZE_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    BUCKETIZE_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    BUCKETIZE_COMPUTE_CASE(DT_FLOAT, float, ctx)
    BUCKETIZE_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Bucketize kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t BucketizeCpuKernel::BucketizeCompute(const CpuKernelContext &ctx) {
  const int64_t data_num = get_tensor_length(ctx.Input(0));
  auto boundaries = ctx.GetAttr("boundaries");
  std::vector<float> boundaries_data = boundaries->GetListFloat();
  std::sort(boundaries_data.begin(), boundaries_data.end());
  auto input_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_data = reinterpret_cast<int32_t *>(ctx.Output(0)->GetData());

  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto sharder_bucketize = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        auto first_bigger_it = std::upper_bound(boundaries_data.begin(), boundaries_data.end(), input_data[i]);
        output_data[i] = first_bigger_it - boundaries_data.begin();
      }
    };
    CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_bucketize);
  } else {
    for (int64_t i = 0; i < data_num; i++) {
      auto first_bigger_it = std::upper_bound(boundaries_data.begin(), boundaries_data.end(), input_data[i]);
      output_data[i] = first_bigger_it - boundaries_data.begin();
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kBucketize, BucketizeCpuKernel);
}  // namespace aicpu
