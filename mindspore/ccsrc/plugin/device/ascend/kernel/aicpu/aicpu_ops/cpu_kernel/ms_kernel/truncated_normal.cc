/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "truncated_normal.h"

#include <cmath>
#include <ctime>
#include <iostream>

#include "random/utils.h"
#include "context/inc/cpu_kernel_utils.h"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "utils/eigen_tensor.h"

namespace {
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
const uint32_t kInputDims = 1;
const uint32_t kInputSizes = 2;
const uint32_t kCountsIndex = 1;
const uint32_t kStatesIndex = 2;
const char *kTruncatedNormal = "TruncatedNormal";
}  // namespace

namespace aicpu {
template <typename T>
uint32_t TruncatedNormalCpuKernel::DoCompute(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  Tensor *output = ctx.Output(0);
  auto input_data_nums = input->NumElements();
  auto input_data = reinterpret_cast<T *>(input->GetData());
  auto output_type = output->GetDataType();
  auto output_nums = output->NumElements();

  std::vector<int64_t> out_put_dims;
  for (auto i = 0; i < input_data_nums; ++i) {
    if (*(input_data + i) <= 0) {
      CUST_KERNEL_LOG_ERROR(ctx, "Shape elements must be > 0.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    out_put_dims.push_back(input_data[i]);
  }

  // get random generator seed
  uint32_t kernel_ret = 0;
  uint64_t rng_seed =
    random::GetCpuKernelRandomStates(ctx, kCountsIndex, kStatesIndex, seed_, seed2_, "TruncatedNormal", &kernel_ret);
  if (kernel_ret != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  rng_.seed(rng_seed);

  if (output_type == DT_FLOAT16) {
    auto output_data = reinterpret_cast<Eigen::half *>(output->GetData());
    std::normal_distribution<float> dis(0, 1);
    for (int j = 0; j < output_nums;) {
      auto data = dis(rng_);
      if (data >= -2 && data <= 2) {
        *(output_data + j) = static_cast<Eigen::half>(data);
        ++j;
      }
    }
  } else if (output_type == DT_FLOAT) {
    auto output_data = reinterpret_cast<float_t *>(output->GetData());
    std::normal_distribution<float> dis(0, 1);
    for (int j = 0; j < output_nums;) {
      auto data = dis(rng_);
      if (data >= -2 && data <= 2) {
        *(output_data + j) = data;
        ++j;
      }
    }
  } else {
    auto output_data = reinterpret_cast<double_t *>(output->GetData());
    std::normal_distribution<double> dis(0, 1);
    for (int j = 0; j < output_nums;) {
      auto data = dis(rng_);
      if (data >= -2 && data <= 2) {
        *(output_data + j) = data;
        ++j;
      }
    }
  }
  output->GetTensorShape()->SetDimSizes(out_put_dims);
  return KERNEL_STATUS_OK;
}

uint32_t TruncatedNormalCpuKernel::DataAndTypeCheck(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  Tensor *output = ctx.Output(0);
  auto input_data_nums = input->NumElements();
  CUST_KERNEL_CHECK_FALSE(ctx, (input_data_nums >= kInputSizes), KERNEL_STATUS_PARAM_INVALID,
                          "Input data elements must >= 2.");
  CUST_KERNEL_CHECK_FALSE(ctx, (input->GetTensorShape()->GetDimSizes().size() == kInputDims),
                          KERNEL_STATUS_PARAM_INVALID, "Input tensor must be a 1-D tensor.");
  auto input_datatype = input->GetDataType();
  auto output_datatype = output->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, (input_datatype == DT_INT32 || input_datatype == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                          "Input type must be int64 or int32.");
  CUST_KERNEL_CHECK_FALSE(
    ctx, (output_datatype == DT_FLOAT16 || output_datatype == DT_FLOAT || output_datatype == DT_DOUBLE),
    KERNEL_STATUS_PARAM_INVALID, "Out put type must be one of float16, float32 or double.");
  return KERNEL_STATUS_OK;
}

uint32_t TruncatedNormalCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check params failed.", kTruncatedNormal);
  CUST_KERNEL_HANDLE_ERROR(ctx, DataAndTypeCheck(ctx), " TruncatedNormal input elements value  check failed.");
  auto input_datatype = ctx.Input(0)->GetDataType();

  // get attrs
  AttrValue *seed_ptr = ctx.GetAttr("seed");
  AttrValue *seed2_ptr = ctx.GetAttr("seed2");
  seed_ = (seed_ptr == nullptr) ? static_cast<uint64_t>(0) : static_cast<uint64_t>(seed_ptr->GetInt());
  seed2_ = (seed2_ptr == nullptr) ? static_cast<uint64_t>(0) : static_cast<uint64_t>(seed2_ptr->GetInt());

  uint32_t ret;
  switch (input_datatype) {
    case DT_INT32:
      ret = DoCompute<int32_t>(ctx);
      break;
    case DT_INT64:
      ret = DoCompute<int64_t>(ctx);
      break;
    default: {
      CUST_KERNEL_LOG_WARN(ctx, "TruncatedNormal kernel data type [%s] not support.", DTypeStr(input_datatype).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == KERNEL_STATUS_OK), ret, "Compute failed.");
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kTruncatedNormal, TruncatedNormalCpuKernel);
}  // namespace aicpu
