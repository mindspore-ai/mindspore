/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2024.All rights reserved.
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
#include "cpu_kernel/ms_kernel/bessel_i0.h"

#include <cstdint>
#include <algorithm>

#include "Eigen/Dense"
#include "context/inc/cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"
#include "inc/kernel_log.h"
#include "securec.h"
#include "context/common/status.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *const kBesselI0 = "BesselI0";

const int64_t MAX_CPU_CORE = 4;
const double NUMBER_HALF = 0.5;
const double NUMBER_2 = 2.0;
const double NUMBER_8 = 8.0;
const double NUMBER_32 = 32.0;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;

const int LEN_A = 30;
const int LEN_B = 25;
}  // namespace

namespace aicpu {
uint32_t BesselI0CpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.",
                           kBesselI0);
  auto data_type = ctx.Input(0)->GetDataType();
  uint32_t ret;
  switch (data_type) {
    case (DT_FLOAT16):
      ret = ParallelForCompute(ctx);
      break;
    case (DT_FLOAT):
      ret = ParallelForCompute(ctx);
      break;
    case (DT_DOUBLE):
      ret = ParallelForCompute(ctx);
      break;
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "BesselI0 kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ret != KERNEL_STATUS_OK) {
    CUST_KERNEL_LOG_ERROR(ctx, "BesselI0 kernel compute failed.");
  }
  return ret;
}

uint32_t BesselI0CpuKernel::ParallelForCompute(CpuKernelContext &ctx) {
  int64_t data_num = ctx.Output(0)->NumElements();
  auto data_type = ctx.Input(0)->GetDataType();

  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num,
                              static_cast<int64_t>(MAX_CPU_CORE));  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    if (data_type == DT_FLOAT16) {
      auto sharder_bessel_i0 = [&](int64_t start, int64_t end) { BesselI0ComputeFloat16(start, end, ctx); };
      CUST_KERNEL_HANDLE_ERROR(ctx,
                               CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_bessel_i0),
                               "BesselI0 Compute failed.");
    } else if (data_type == DT_FLOAT) {
      auto sharder_bessel_i0 = [&](int64_t start, int64_t end) { BesselI0Compute<float>(start, end, ctx); };
      CUST_KERNEL_HANDLE_ERROR(ctx,
                               CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_bessel_i0),
                               "BesselI0 Compute failed.");
    } else if (data_type == DT_DOUBLE) {
      auto sharder_bessel_i0 = [&](int64_t start, int64_t end) { BesselI0Compute<double>(start, end, ctx); };
      CUST_KERNEL_HANDLE_ERROR(ctx,
                               CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_bessel_i0),
                               "BesselI0 Compute failed.");
    } else {
      CUST_KERNEL_LOG_ERROR(ctx, "BesselI0 kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  } else {
    if (data_type == DT_FLOAT16) {
      BesselI0ComputeFloat16(0, data_num, ctx);
    } else if (data_type == DT_FLOAT) {
      BesselI0Compute<float>(0, data_num, ctx);
    } else if (data_type == DT_DOUBLE) {
      BesselI0Compute<double>(0, data_num, ctx);
    } else {
      CUST_KERNEL_LOG_ERROR(ctx, "BesselI0 kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void BesselI0CpuKernel::BesselI0Compute(int64_t start, int64_t end, CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  static const T A[] = {
    -4.41534164647933937950e-18, +3.33079451882223809783e-17, -2.43127984654795469359e-16, +1.71539128555513303061e-15,
    -1.16853328779934516808e-14, +7.67618549860493561688e-14, -4.85644678311192946090e-13, +2.95505266312963983461e-12,
    -1.72682629144155570723e-11, +9.67580903537323691224e-11, -5.18979560163526290666e-10, +2.65982372468238665035e-09,
    -1.30002500998624804212e-08, +6.04699502254191894932e-08, -2.67079385394061173391e-07, +1.11738753912010371815e-06,
    -4.41673835845875056359e-06, +1.64484480707288970893e-05, -5.75419501008210370398e-05, +1.88502885095841655729e-04,
    -5.76375574538582365885e-04, +1.63947561694133579842e-03, -4.32430999505057594430e-03, +1.05464603945949983183e-02,
    -2.37374148058994688156e-02, +4.93052842396707084878e-02, -9.49010970480476444210e-02, +1.71620901522208775349e-01,
    -3.04682672343198398683e-01, +6.76795274409476084995e-01,
  };

  static const T B[] = {
    -7.23318048787475395456e-18, -4.83050448594418207126e-18, +4.46562142029675999901e-17, +3.46122286769746109310e-17,
    -2.82762398051658348494e-16, -3.42548561967721913462e-16, +1.77256013305652638360e-15, +3.81168066935262242075e-15,
    -9.55484669882830764870e-15, -4.15056934728722208663e-14, +1.54008621752140982691e-14, +3.85277838274214270114e-13,
    +7.18012445138366623367e-13, -1.79417853150680611778e-12, -1.32158118404477131188e-11, -3.14991652796324136454e-11,
    +1.18891471078464383424e-11, +4.94060238822496958910e-10, +3.39623202570838634515e-09, +2.26666899049817806459e-08,
    +2.04891858946906374183e-07, +2.89137052083475648297e-06, +6.88975834691682398426e-05, +3.36911647825569408990e-03,
    +8.04490411014108831608e-01,
  };

  for (int64_t i = start; i < end; i++) {
    T p;
    T q = 0.0;
    T x = *(input_x + i);
    if (std::abs(x) <= T(NUMBER_8)) {
      T a = A[0];
      for (uint8_t index = 1; index < LEN_A; index++) {
        p = q;
        q = a;
        a = ((std::abs(x) / T(NUMBER_2)) - T(NUMBER_2)) * q - p + A[index];
      }
      *(output_y + i) = std::exp(std::abs(x)) * (T(NUMBER_HALF) * (a - p));
    } else {
      T b = B[0];
      for (uint8_t index = 1; index < LEN_B; index++) {
        p = q;
        q = b;
        b = (T(NUMBER_32) / std::abs(x) - T(NUMBER_2)) * q - p + B[index];
      }
      *(output_y + i) = std::exp(std::abs(x)) * (T(NUMBER_HALF) * (b - p)) / std::sqrt(std::abs(x));
    }
  }
}

void BesselI0CpuKernel::BesselI0ComputeFloat16(int64_t start, int64_t end, CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<Eigen::half *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<Eigen::half *>(ctx.Output(0)->GetData());

  static const float A[] = {
    -4.41534164647933937950e-18, +3.33079451882223809783e-17, -2.43127984654795469359e-16, +1.71539128555513303061e-15,
    -1.16853328779934516808e-14, +7.67618549860493561688e-14, -4.85644678311192946090e-13, +2.95505266312963983461e-12,
    -1.72682629144155570723e-11, +9.67580903537323691224e-11, -5.18979560163526290666e-10, +2.65982372468238665035e-09,
    -1.30002500998624804212e-08, +6.04699502254191894932e-08, -2.67079385394061173391e-07, +1.11738753912010371815e-06,
    -4.41673835845875056359e-06, +1.64484480707288970893e-05, -5.75419501008210370398e-05, +1.88502885095841655729e-04,
    -5.76375574538582365885e-04, +1.63947561694133579842e-03, -4.32430999505057594430e-03, +1.05464603945949983183e-02,
    -2.37374148058994688156e-02, +4.93052842396707084878e-02, -9.49010970480476444210e-02, +1.71620901522208775349e-01,
    -3.04682672343198398683e-01, +6.76795274409476084995e-01,
  };

  static const float B[] = {
    -7.23318048787475395456e-18, -4.83050448594418207126e-18, +4.46562142029675999901e-17, +3.46122286769746109310e-17,
    -2.82762398051658348494e-16, -3.42548561967721913462e-16, +1.77256013305652638360e-15, +3.81168066935262242075e-15,
    -9.55484669882830764870e-15, -4.15056934728722208663e-14, +1.54008621752140982691e-14, +3.85277838274214270114e-13,
    +7.18012445138366623367e-13, -1.79417853150680611778e-12, -1.32158118404477131188e-11, -3.14991652796324136454e-11,
    +1.18891471078464383424e-11, +4.94060238822496958910e-10, +3.39623202570838634515e-09, +2.26666899049817806459e-08,
    +2.04891858946906374183e-07, +2.89137052083475648297e-06, +6.88975834691682398426e-05, +3.36911647825569408990e-03,
    +8.04490411014108831608e-01,
  };

  for (int64_t i = start; i < end; i++) {
    float p;
    float q = (float)0.0;
    float x = (float)*(input_x + i);
    if (std::abs(x) <= float(NUMBER_8)) {
      float a = A[0];
      for (uint8_t index = 1; index < LEN_A; index++) {
        p = q;
        q = a;
        a = ((std::abs(x) / float(NUMBER_2)) - float(NUMBER_2)) * q - p + A[index];
      }
      *(output_y + i) = static_cast<Eigen::half>(std::exp(std::abs(x)) * (float(NUMBER_HALF) * (a - p)));
    } else {
      float b = B[0];
      for (uint8_t index = 1; index < LEN_B; index++) {
        p = q;
        q = b;
        b = (float(NUMBER_32) / std::abs(x) - float(NUMBER_2)) * q - p + B[index];
      }
      *(output_y + i) =
        static_cast<Eigen::half>(std::exp(std::abs(x)) * (float(NUMBER_HALF) * (b - p)) / std::sqrt(std::abs(x)));
    }
  }
}

REGISTER_MS_CPU_KERNEL(kBesselI0, BesselI0CpuKernel);
}  // namespace aicpu
