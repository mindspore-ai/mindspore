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

#include "lcm.h"

#include <cmath>
#include <set>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kLcmOutputNum = 1;
const uint32_t kLcmInputNum = 2;
const char *kLcm = "Lcm";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
const int32_t kInput_32_32 = 3;
const int32_t kInput_32_64 = 2;
const int32_t kInput_64_32 = 1;
const int32_t kInput_64_64 = 0;
}  // namespace

namespace aicpu {
// Simple recursive gcd.
template <class T>
T elewise_gcd(T a, T b) {
  if (b == 0) {
    return a;
  }
  return elewise_gcd(b, a % b);
}
// Simple lcm.
template <typename T>
T elewise_lcm(T a, T b) {
  T gcd_tmp = elewise_gcd<T>(a, b);
  if (gcd_tmp == 0) {
    return static_cast<T>(0);
  }
  return std::abs(a / gcd_tmp * b);
}

uint32_t LcmIOTypeCheck(CpuKernelContext &ctx, int32_t &dual_types) {
  Tensor *x1 = ctx.Input(kFirstInputIndex);
  Tensor *x2 = ctx.Input(kSecondInputIndex);
  Tensor *y = ctx.Output(kFirstOutputIndex);
  const std::set<DataType> supported_types{DT_INT32, DT_INT64};
  auto x1_type = x1->GetDataType();
  auto x2_type = x2->GetDataType();
  auto y_type = y->GetDataType();
  KERNEL_CHECK_FALSE(supported_types.count(x1_type) != 0, KERNEL_STATUS_PARAM_INVALID,
                     "[Lcm] input x1 data type [%s] is not supported.", DTypeStr(x1_type).c_str());
  KERNEL_CHECK_FALSE(supported_types.count(x2_type) != 0, KERNEL_STATUS_PARAM_INVALID,
                     "[Lcm] input x2 data type [%s] is not supported.", DTypeStr(x2_type).c_str());
  int32_t x1_is_i32 = static_cast<int32_t>(x1_type == DT_INT32) << 1;
  int32_t x2_is_i32 = static_cast<int32_t>(x2_type == DT_INT32);
  int32_t _dual_types = x1_is_i32 | x2_is_i32;
  switch (_dual_types) {
    case kInput_64_64:
    case kInput_64_32:
    case kInput_32_64:
      KERNEL_CHECK_FALSE(y_type == DT_INT64, KERNEL_STATUS_PARAM_INVALID,
                         "[Lcm] output y data type [%s] is not supported.", DTypeStr(y_type).c_str());
      dual_types = _dual_types;
      break;
    case kInput_32_32:
      KERNEL_CHECK_FALSE(y_type == DT_INT32, KERNEL_STATUS_PARAM_INVALID,
                         "[Lcm] output y data type [%s] is not supported.", DTypeStr(y_type).c_str());
      dual_types = _dual_types;
      break;
    default:
      KERNEL_LOG_ERROR("[Lcm] input data type tuple is not supported.");
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <class T1, class T2, class T3>
uint32_t LcmElewiseCompute(CpuKernelContext &ctx, const T1 *x1_ptr, const T2 *x2_ptr, T3 *y_ptr, Bcast &bcast) {
  int64_t data_num = ctx.Output(kFirstOutputIndex)->NumElements();
  auto lcm_shard = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      T3 x1_ele_abs = std::abs(static_cast<T3>(x1_ptr[bcast.GetBroadcastXIndex(i)]));
      T3 x2_ele_abs = std::abs(static_cast<T3>(x2_ptr[bcast.GetBroadcastYIndex(i)]));
      y_ptr[i] = elewise_lcm(x1_ele_abs, x2_ele_abs);
    }
  };
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("[Lcm] max_core_num is 0, please check the cpu num.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    uint32_t ret = CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, lcm_shard);
    if (ret != KERNEL_STATUS_OK) {
      KERNEL_LOG_ERROR("[Lcm] Lcm Compute failed.");
      return ret;
    }
  } else {
    lcm_shard(0, data_num);
  }

  return KERNEL_STATUS_OK;
}

template <class T1, class T2, class T3>
uint32_t LcmCompute(CpuKernelContext &ctx) {
  Tensor *x1 = ctx.Input(kFirstInputIndex);
  Tensor *x2 = ctx.Input(kSecondInputIndex);
  Tensor *y = ctx.Output(kFirstOutputIndex);
  const T1 *x1_ptr = reinterpret_cast<const T1 *>(x1->GetData());
  const T2 *x2_ptr = reinterpret_cast<const T2 *>(x2->GetData());
  T3 *y_ptr = reinterpret_cast<T3 *>(y->GetData());
  auto x1_shape = x1->GetTensorShape()->GetDimSizes();
  auto x2_shape = x2->GetTensorShape()->GetDimSizes();
  Bcast bcast(x1_shape, x2_shape);
  if (bcast.IsValid()) {
    return LcmElewiseCompute<T1, T2, T3>(ctx, x1_ptr, x2_ptr, y_ptr, bcast);
  } else {
    KERNEL_LOG_ERROR("[Lcm] broadcast failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t LcmCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kLcmInputNum, kLcmOutputNum), "[Lcm] check input and output number failed.");
  int32_t dual_types = static_cast<int32_t>(-1);
  KERNEL_HANDLE_ERROR(LcmIOTypeCheck(ctx, dual_types), "[Lcm] check data type failed.");
  switch (dual_types) {
    case kInput_64_64:
      return LcmCompute<int64_t, int64_t, int64_t>(ctx);
      break;
    case kInput_64_32:
      return LcmCompute<int64_t, int32_t, int64_t>(ctx);
      break;
    case kInput_32_64:
      return LcmCompute<int32_t, int64_t, int64_t>(ctx);
      break;
    case kInput_32_32:
      return LcmCompute<int32_t, int32_t, int32_t>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("[Lcm] input data type tuple is not supported.");
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kLcm, LcmCpuKernel);
}  // namespace aicpu
