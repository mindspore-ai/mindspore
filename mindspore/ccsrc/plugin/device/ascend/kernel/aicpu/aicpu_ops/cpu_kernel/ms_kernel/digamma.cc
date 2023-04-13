/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

#include "digamma.h"

#include <unsupported/Eigen/CXX11/Tensor>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kDigammaInputNum{1};
const std::uint32_t kDigammaOutputNum{1};
const char *kDigamma{"Digamma"};
const std::int64_t kDigammaParallelNum{64 * 1024};
}  // namespace

namespace aicpu {
namespace detail {
static inline double calc_digamma(double x);

template <typename T>
inline T ScalarDigamma(T x) {
  return calc_digamma(x);
}

template <>
inline Eigen::half ScalarDigamma(Eigen::half x) {
  const Eigen::half val{static_cast<Eigen::half>(calc_digamma(static_cast<std::float_t>(x)))};
  return val;
}

template <typename T>
static inline T polevl(const T x, const T A[], size_t len) {
  T result = 0;
  for (size_t i = 0; i <= len; i++) {
    result = result * x + A[i];
  }
  return result;
}

static inline double calc_digamma(double x) {
  static double PSI_10 = 2.25175258906672110764;
  static double PI = 3.141592653589793238462;
  static int64_t TEN = 10;
  static double HALF = 0.5;
  static int64_t SIX = 6;
  if (x == 0) {
    return std::copysign(INFINITY, -x);
  }

  bool x_is_integer = x == trunc(x);
  if (x < 0) {
    if (x_is_integer) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    double q = 0;
    double r = std::modf(x, &q);
    return calc_digamma(1 - x) - PI / tan(PI * r);
  }

  double result = 0;
  while (x < TEN) {
    result -= 1 / x;
    x += 1;
  }
  if (x == TEN) {
    return result + PSI_10;
  }

  static const double A[] = {
    8.33333333333333333333E-2, -2.10927960927960927961E-2, 7.57575757575757575758E-3, -4.16666666666666666667E-3,
    3.96825396825396825397E-3, -8.33333333333333333333E-3, 8.33333333333333333333E-2,
  };

  double y = 0;
  if (x < 1.0e17) {
    double z = 1.0 / (x * x);
    y = z * polevl(z, A, SIX);
  }
  return result + log(x) - (HALF / x) - y;
}

inline std::uint32_t ParallelForDigamma(const CpuKernelContext &ctx, std::int64_t total, std::int64_t per_unit_size,
                                        const std::function<void(std::int64_t, std::int64_t)> &work) {
  if (total > kDigammaParallelNum)
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  else
    work(0, total);
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeDigammaKernel(const CpuKernelContext &ctx) {
  T *input0{static_cast<T *>(ctx.Input(0)->GetData())};
  T *output{static_cast<T *>(ctx.Output(0)->GetData())};
  std::int64_t total{ctx.Input(0)->NumElements()};
  std::uint32_t cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  return ParallelForDigamma(ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
    std::transform(input0 + begin, input0 + end, output + begin, ScalarDigamma<T>);
  });
}

template <typename T>
inline std::uint32_t ComputeDigamma(const CpuKernelContext &ctx) {
  std::uint32_t result{ComputeDigammaKernel<T>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Digamma compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckDigamma(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR("The data type of the input [%s] need be the same as the output [%s].",
                     DTypeStr(ctx.Input(0)->GetDataType()).c_str(), DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
    KERNEL_LOG_ERROR(
      "The data size of the input [%llu] need be the same as the output "
      "[%llu].",
      ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

inline std::uint32_t CheckDigamma(CpuKernelContext &ctx, std::uint32_t inputs_num, std::uint32_t outputs_num) {
  return NormalCheck(ctx, kDigammaInputNum, kDigammaOutputNum) ? KERNEL_STATUS_PARAM_INVALID : ExtraCheckDigamma(ctx);
}

inline std::uint32_t ComputeDigamma(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeDigamma<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeDigamma<std::float_t>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t DigammaCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::CheckDigamma(ctx, kDigammaInputNum, kDigammaOutputNum) ? KERNEL_STATUS_PARAM_INVALID
                                                                        : detail::ComputeDigamma(ctx);
}

REGISTER_CPU_KERNEL(kDigamma, DigammaCpuKernel);
}  // namespace aicpu