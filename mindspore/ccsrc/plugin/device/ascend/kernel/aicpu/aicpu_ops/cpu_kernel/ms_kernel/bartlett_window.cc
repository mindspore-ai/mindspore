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

#include "cpu_kernel/ms_kernel/bartlett_window.h"
#include <vector>
#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kBartlettWindow = "BartlettWindow";

}  // namespace

namespace aicpu {
uint32_t BartlettWindowCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "BartlettWindow check input and output number failed.");
  CUST_KERNEL_HANDLE_ERROR(ctx, BartlettWindowCheck(ctx), "BartlettWindow check params failed.");
  auto input_dtype = ctx.Input(0)->GetDataType();
  auto output_dtype = ctx.Output(0)->GetDataType();
  if (output_dtype == DT_FLOAT16) {
    if (input_dtype == DT_INT32) {
      return BartlettWindowCompute<int32_t, Eigen::half>(ctx);
    } else if (input_dtype == DT_INT64) {
      return BartlettWindowCompute<int64_t, Eigen::half>(ctx);
    }
  } else if (output_dtype == DT_FLOAT) {
    if (input_dtype == DT_INT32) {
      return BartlettWindowCompute<int32_t, float>(ctx);
    } else if (input_dtype == DT_INT64) {
      return BartlettWindowCompute<int64_t, float>(ctx);
    }
  } else if (output_dtype == DT_DOUBLE) {
    if (input_dtype == DT_INT32) {
      return BartlettWindowCompute<int32_t, double>(ctx);
    } else if (input_dtype == DT_INT64) {
      return BartlettWindowCompute<int64_t, double>(ctx);
    }
  } else {
    CUST_KERNEL_LOG_ERROR(ctx, "BartlettWindow kernel data type [%s] not support.", DTypeStr(input_dtype).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t BartlettWindowCpuKernel::BartlettWindowCheck(CpuKernelContext &ctx) {
  auto input_info = ctx.Input(0);
  auto output_info = ctx.Output(0);
  DataType input_type = input_info->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, (input_type == DT_INT32) || (input_type == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                          "The data type of input:[%s] should be an integertype ", DTypeStr(input_type).c_str())
  DataType output_type = output_info->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, (output_type == DT_FLOAT16) || (output_type == DT_FLOAT) || (output_type == DT_DOUBLE),
                          KERNEL_STATUS_PARAM_INVALID, "The data type of output:[%s] should be half, float or double ",
                          DTypeStr(output_type).c_str());
  auto input_data = reinterpret_cast<int64_t *>(input_info->GetData());
  CUST_KERNEL_CHECK_FALSE(ctx, (int)(*input_data) >= 0, KERNEL_STATUS_PARAM_INVALID,
                          "The value of input:[%d] must be a non-negative integer", *input_data);
  std::vector<int64_t> dim_vec = input_info->GetTensorShape()->GetDimSizes();
  int64_t dimsize = dim_vec.size();
  CUST_KERNEL_CHECK_FALSE(ctx, dimsize <= 1, KERNEL_STATUS_PARAM_INVALID,
                          "The dim of input:[%d] should not more than 1", dimsize)
  CUST_KERNEL_LOG_DEBUG(ctx,
                        "BartlettWindowCpuKernel[%s], input: size[%llu];"
                        "output: size[%llu].",
                        ctx.GetOpType().c_str(), input_info->GetDataSize(), output_info->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename T, typename DT_VAL>
uint32_t BartlettWindowCpuKernel::BartlettWindowCompute(CpuKernelContext &ctx) {
  auto input_info = ctx.Input(0);
  auto output_info = ctx.Output(0);
  auto input_x = reinterpret_cast<T *>(input_info->GetData());
  auto output_y = reinterpret_cast<DT_VAL *>(output_info->GetData());
  AttrValue *attr_per = ctx.GetAttr("periodic");
  bool attr_per_value = (attr_per == nullptr) ? true : attr_per->GetBool();
  const int64_t window_length = static_cast<int64_t>(*input_x);

  if (*input_x == 1) {
    *output_y = static_cast<DT_VAL>(1.);
    return KERNEL_STATUS_OK;
  }
  if (attr_per_value) {
    *input_x += 1;
  }

  const int64_t first_half_size = static_cast<int64_t>((*input_x - 1) / 2);
  const double x = static_cast<double>(*input_x);

  for (int i = 0; i <= first_half_size; i++) {
    auto value = static_cast<DT_VAL>((2. * i) / (x - 1.));
    *(output_y + i) = value;
  }
  for (int i = first_half_size + 1; i < window_length; i++) {
    auto value = static_cast<DT_VAL>(2. - (2. * i) / (x - 1.));
    *(output_y + i) = value;
  }

  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kBartlettWindow, BartlettWindowCpuKernel);
}  // namespace aicpu
