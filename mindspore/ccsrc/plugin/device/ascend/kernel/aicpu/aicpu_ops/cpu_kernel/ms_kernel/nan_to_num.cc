/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "nan_to_num.h"

#include <iostream>
#include <limits>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kNanToNumInputNum = 1;
constexpr uint32_t kNanToNumOutputNum = 1;
const int64_t paralled_data_size = 64 * 1024;
const char *kNanToNum = "NanToNum";
}  // namespace

namespace aicpu {
template <typename T>
uint32_t NanToNumCpuKernel::NanToNumCompute(CpuKernelContext &ctx) {
  T *x_addr = static_cast<T *>(ctx.Input(0)->GetData());
  auto y_addr = static_cast<T *>(ctx.Output(0)->GetData());
  auto attr_value_nan = ctx.GetAttr("nan");
  auto attr_nan = 0.0;
  if (attr_value_nan != nullptr) {
    attr_nan = attr_value_nan->GetFloat();
  }
  auto attr_value_posinf = ctx.GetAttr("posinf");
  auto attr_posinf = static_cast<float>(std::numeric_limits<T>::max());
  if (attr_value_posinf != nullptr) {
    attr_posinf = attr_value_posinf->GetFloat();
  }
  auto attr_value_neginf = ctx.GetAttr("neginf");
  auto attr_neginf = static_cast<float>(std::numeric_limits<T>::lowest());
  if (attr_value_neginf != nullptr) {
    attr_neginf = attr_value_neginf->GetFloat();
  }
  size_t x_size = ctx.Input(0)->NumElements();
  size_t date_size = x_size * sizeof(T);
  if (date_size <= paralled_data_size) {
    for (size_t i = 0; i < x_size; i++) {
      if (x_addr[i] > static_cast<T>(0) && std::isinf(static_cast<double>(x_addr[i]))) {
        y_addr[i] = static_cast<T>(attr_posinf);
      } else if (x_addr[i] < static_cast<T>(0) && std::isinf(static_cast<double>(x_addr[i]))) {
        y_addr[i] = static_cast<T>(attr_neginf);
      } else if (std::isnan(static_cast<double>(x_addr[i]))) {
        y_addr[i] = static_cast<T>(attr_nan);
      } else {
        y_addr[i] = x_addr[i];
      }
    }
  } else {
    auto shard_nan_to_num = [&](size_t start, size_t end) {
      for (size_t i = 0; i < x_size; i++) {
        if (x_addr[i] > static_cast<T>(0) && std::isinf(static_cast<double>(x_addr[i]))) {
          y_addr[i] = static_cast<T>(attr_posinf);
        } else if (x_addr[i] < static_cast<T>(0) && std::isinf(static_cast<double>(x_addr[i]))) {
          y_addr[i] = static_cast<T>(attr_neginf);
        } else if (std::isnan(static_cast<double>(x_addr[i]))) {
          y_addr[i] = static_cast<T>(attr_nan);
        } else {
          y_addr[i] = x_addr[i];
        }
      }
    };
    uint32_t min_core_num = 1;
    size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num == 0) {
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (max_core_num > date_size) {
      max_core_num = date_size;
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, date_size, date_size / max_core_num, shard_nan_to_num),
                        "NanoNum Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

uint32_t NanToNumCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kNanToNumInputNum, kNanToNumOutputNum), "[%s] check params failed.", kNanToNum);
  auto data_type = ctx.Input(0)->GetDataType();
  uint32_t res = KERNEL_STATUS_OK;
  switch (data_type) {
    case DT_FLOAT16:
      res = NanToNumCompute<Eigen::half>(ctx);
      break;
    case DT_FLOAT:
      res = NanToNumCompute<float>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("NanToNum invalid input type [%s]", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (res != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kNanToNum, NanToNumCpuKernel);
}  // namespace aicpu