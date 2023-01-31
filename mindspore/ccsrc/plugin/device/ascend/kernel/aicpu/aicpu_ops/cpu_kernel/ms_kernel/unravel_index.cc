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
#include "unravel_index.h"
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
const char *KUnravelIndex = "UnravelIndex";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const int64_t kParallelDataNumSameShape = 1000;
}  // namespace

namespace aicpu {
uint32_t UnravelIndexCpuKernel::Compute(CpuKernelContext &ctx) {
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    case DT_INT32: {
      KERNEL_HANDLE_ERROR(DataAndTypeCheck<int32_t>(ctx), " data or type check failed.");
      UnravelCompute<int32_t>(ctx);
      break;
    }
    case DT_INT64: {
      KERNEL_HANDLE_ERROR(DataAndTypeCheck<int64_t>(ctx), " data or type check failed.");
      UnravelCompute<int64_t>(ctx);
      break;
    }
    default: {
      KERNEL_LOG_ERROR("UnravelIndex kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t UnravelIndexCpuKernel::DataAndTypeCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Unravel_Index check input and output number failed.");
  Tensor *indices = ctx.Input(0);
  Tensor *dims = ctx.Input(1);
  auto dims_number = ctx.Input(1)->NumElements();
  auto indices_number = ctx.Input(0)->NumElements();
  auto dims_data = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto indices_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto indices_type = indices->GetDataType();
  auto dims_type = dims->GetDataType();
  T dims_multi = 1;
  KERNEL_CHECK_FALSE((indices_type == dims_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s] need be same with "
                     "input1 [%s].",
                     DTypeStr(indices_type).c_str(), DTypeStr(dims_type).c_str())

  for (auto i = 0; i < dims_number; i++) {
    KERNEL_CHECK_FALSE((*(dims_data + i) > 0), KERNEL_STATUS_PARAM_INVALID, "Dimension number must be more than 0.")
    dims_multi = dims_multi * (*(dims_data + i));
  }
  for (auto i = 0; i < indices_number; i++) {
    KERNEL_CHECK_FALSE((*(indices_data + i) >= 0), KERNEL_STATUS_PARAM_INVALID, "Indice number must be more than 0.")
    KERNEL_CHECK_FALSE((*(indices_data + i) < dims_multi), KERNEL_STATUS_PARAM_INVALID,
                       "Index is out of bound as with dims");
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t UnravelIndexCpuKernel ::UnravelCompute(CpuKernelContext &ctx) {
  auto indices_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto dims_data = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto dims_number = ctx.Input(1)->NumElements();
  auto indices_number = ctx.Input(0)->NumElements();
  auto data_num = indices_number;

  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_unravel_index = [&](size_t start, size_t end) {
      for (auto j = start; j < end; j++) {
        T Quotient = *(indices_data + j);
        for (auto i = (dims_number - 1); i >= 0; i--) {
          *(output_data + i * indices_number + j) = Quotient % *(dims_data + i);
          Quotient = Quotient / *(dims_data + i);
        }
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_unravel_index),
                        "Unravel Index Compute failed.");
  } else {
    for (auto j = 0; j < indices_number; j++) {
      T Quotient = *(indices_data + j);
      for (auto i = (dims_number - 1); i >= 0; i--) {
        *(output_data + i * indices_number + j) = Quotient % *(dims_data + i);
        Quotient = Quotient / *(dims_data + i);
      }
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(KUnravelIndex, UnravelIndexCpuKernel);
}  // namespace aicpu
