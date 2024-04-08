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

#include "cpu_kernel/ms_kernel/identityn.h"
#include <securec.h>
#include <algorithm>
#include <vector>

#include "cpu_types.h"
#include "inc/kernel_log.h"
#include "context/common/status.h"
#include "utils/kernel_util.h"

namespace {
const char *kIdentityN = "IdentityN";
}  // namespace

namespace aicpu {
uint32_t IdentityNCpuKernel::IdentityNParamCheck(CpuKernelContext &ctx) {
  // input size and output size check
  uint32_t input_size = ctx.GetInputsSize();
  uint32_t output_size = ctx.GetOutputsSize();
  CUST_KERNEL_CHECK_FALSE(ctx, (input_size == output_size), KERNEL_STATUS_PARAM_INVALID,
                          "Input size should equal to Output size.");
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, input_size, output_size), "[%s] check params failed.", kIdentityN);
  for (uint32_t idx = 0; idx < input_size; ++idx) {
    Tensor *in_tensor = ctx.Input(idx);
    Tensor *out_tensor = ctx.Output(idx);
    // TensorShape check
    auto in_shape = in_tensor->GetTensorShape();
    auto out_shape = out_tensor->GetTensorShape();
    CUST_KERNEL_CHECK_FALSE(ctx, (in_shape->GetDimSizes() == out_shape->GetDimSizes()), KERNEL_STATUS_PARAM_INVALID,
                            "In tensor shape should equal to out tensor shape.");
    // DataType Check
    DataType in_type = in_tensor->GetDataType();
    DataType out_type = out_tensor->GetDataType();
    CUST_KERNEL_CHECK_FALSE(ctx, (in_type == out_type), KERNEL_STATUS_PARAM_INVALID,
                            "In tensor data type should equal to out tensor data type.");
    bool type_support =
      std::find(support_data_type.begin(), support_data_type.end(), in_type) != support_data_type.end();
    CUST_KERNEL_CHECK_FALSE(ctx, type_support, KERNEL_STATUS_PARAM_INVALID,
                            "IdentityN kernel data type [%s] not support.", DTypeStr(in_type).c_str());
  }
  return KERNEL_STATUS_OK;
}

uint32_t IdentityNCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, IdentityNParamCheck(ctx), "IdentityNCpuKernel check params failed");
  uint32_t input_size = ctx.GetInputsSize();
  for (uint32_t idx = 0; idx < input_size; ++idx) {
    Tensor *in_tensor = ctx.Input(idx);
    Tensor *out_tensor = ctx.Output(idx);
    auto in_data = in_tensor->GetData();
    auto out_data = out_tensor->GetData();
    uint64_t in_size = in_tensor->GetDataSize();
    uint64_t out_size = out_tensor->GetDataSize();

    // memory copy
    if (out_data != in_data) {
      // Don't memory copy when out_data is empty tensor.
      if (out_size == 0) {
        continue;
      }
      int cpret = memcpy_s(out_data, out_size, in_data, in_size);
      CUST_KERNEL_CHECK_FALSE(ctx, (cpret == EOK), KERNEL_STATUS_INNER_ERROR,
                              "[%s] memcpy_s to output failed, destMax [%ld], count [%ld].", kIdentityN, out_size,
                              in_size);
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kIdentityN, IdentityNCpuKernel);
}  // namespace aicpu
