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

#include "polygamma.h"

#include <unsupported/Eigen/CXX11/Tensor>

#include <Eigen/Dense>
#include <limits>
#include <cmath>
#include "context/inc/cpu_kernel_utils.h"
#include "cpu_types.h"
#include "inc/kernel_log.h"
#include "context/common/status.h"
#include "utils/kernel_util.h"
#include "utils/igamma_utils.h"

namespace {
const std::uint32_t kPolygammaInputNum{2};
const std::uint32_t kPolygammaOutputNum{1};
const char *kPolygamma{"Polygamma"};
const std::int64_t kPolygammaParallelNum{64 * 1024};
}  // namespace

namespace aicpu {
namespace detail {
template <typename scalar_t>
static inline scalar_t zeta(scalar_t x, scalar_t q) {
  const scalar_t MACHEP = scalar_t{1.11022302462515654042E-16};
  constexpr scalar_t zero = scalar_t{0.0};
  constexpr scalar_t half = scalar_t{0.5};
  constexpr scalar_t one = scalar_t{1.0};
  constexpr scalar_t nine = scalar_t{9.0};
  constexpr int64_t NINE = 9;
  constexpr int64_t TWELVE = 12;
  static const scalar_t A[] = {12.0,
                               -720.0,
                               30240.0,
                               -1209600.0,
                               47900160.0,
                               -1.8924375803183791606e9,
                               7.47242496e10,
                               -2.950130727918164224e12,
                               1.1646782814350067249e14,
                               -4.5979787224074726105e15,
                               1.8152105401943546773e17,
                               -7.1661652561756670113e18};

  int i = 0;
  scalar_t a;
  scalar_t b;
  scalar_t k;
  scalar_t s;
  scalar_t t;
  scalar_t w;
  if (x == one) {
    return std::numeric_limits<scalar_t>::infinity();
  }

  if (x < one) {
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }

  if (q <= zero) {
    if (q == ::floor(q)) {
      return std::numeric_limits<scalar_t>::infinity();
    }
    if (x != ::floor(x)) {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
  }

  s = ::pow(q, -x);
  a = q;
  i = 0;
  b = zero;
  while ((i < NINE) || (a <= nine)) {
    i += 1;
    a += one;
    b = ::pow(a, -x);
    s += b;
    if ((-MACHEP * s < b) && (b < MACHEP * s)) {
      return static_cast<scalar_t>(s);
    }
  }

  w = a;
  s += b * w / (x - one);
  s -= half * b;
  a = one;
  k = zero;
  for (int i = 0; i < TWELVE; i++) {
    a *= x + k;
    b /= w;
    t = a * b / A[i];
    s = s + t;
    // div by zero is allow
    t = ::fabs(t / s);
    if (t < MACHEP) {
      return static_cast<scalar_t>(s);
    }
    k += one;
    a *= x + k;
    b /= w;
    k += one;
  }
  return static_cast<scalar_t>(s);
}

template <typename T1, typename T2>
static inline T2 calc_polygamma(T1 a, T2 x) {
  if (a == static_cast<T1>(0)) {
    return Digamma<T2>(x);
  }
  const auto one = T1{1};
  const auto two = T1{2};
  return ((a % two) ? one : -one) * ::exp(::lgamma(static_cast<T1>(a) + one)) * zeta<T2>(static_cast<T2>(a + 1), x);
}

template <typename T1, typename T2>
inline T2 ScalarPolygamma(T1 a, T2 x) {
  return calc_polygamma(a, x);
}

template <>
inline Eigen::half ScalarPolygamma(int32_t a, Eigen::half x) {
  const Eigen::half val{static_cast<Eigen::half>(calc_polygamma(a, static_cast<std::float_t>(x)))};
  return val;
}

template <>
inline Eigen::half ScalarPolygamma(int64_t a, Eigen::half x) {
  const Eigen::half val{static_cast<Eigen::half>(calc_polygamma(a, static_cast<std::float_t>(x)))};
  return val;
}

inline std::uint32_t ParallelForPolygamma(CpuKernelContext &ctx, std::int64_t total, std::int64_t per_unit_size,
                                          const std::function<void(std::int64_t, std::int64_t)> &work) {
  if (total > kPolygammaParallelNum)
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  else
    work(0, total);
  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
inline std::uint32_t ComputePolygammaKernel(CpuKernelContext &ctx) {
  auto input0 = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto input1 = reinterpret_cast<T2 *>(ctx.Input(1)->GetData());
  auto output = reinterpret_cast<T2 *>(ctx.Output(0)->GetData());
  std::int64_t total{ctx.Output(0)->NumElements()};
  std::uint32_t cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  auto Cal_Poly = [&](int64_t start, int64_t end) {
    for (std::int64_t i = start; i < end; ++i) {
      *(output + i) = ScalarPolygamma<T1, T2>(*input0, *(input1 + i));
    }
  };
  return ParallelForPolygamma(ctx, total, per_unit_size, Cal_Poly);
}

template <typename T1, typename T2>
inline std::uint32_t ComputePolygamma(CpuKernelContext &ctx) {
  std::uint32_t result{ComputePolygammaKernel<T1, T2>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    CUST_KERNEL_LOG_ERROR(ctx, "Polygamma compute failed.");
  }
  return result;
  return KERNEL_STATUS_OK;
}

inline std::uint32_t CheckPolygamma(CpuKernelContext &ctx, std::uint32_t inputs_num, std::uint32_t outputs_num) {
  return NormalCheck(ctx, kPolygammaInputNum, kPolygammaOutputNum) ? KERNEL_STATUS_PARAM_INVALID : KERNEL_STATUS_OK;
}

inline std::uint32_t ComputePolygamma(CpuKernelContext &ctx) {
  DataType input0_type{ctx.Input(0)->GetDataType()};
  DataType input1_type{ctx.Input(1)->GetDataType()};
  switch (input0_type) {
    case DT_INT32:
      switch (input1_type) {
        case DT_FLOAT16:
          return ComputePolygamma<std::int32_t, Eigen::half>(ctx);
        case DT_FLOAT:
          return ComputePolygamma<std::int32_t, std::float_t>(ctx);
        case DT_DOUBLE:
          return ComputePolygamma<std::int32_t, std::double_t>(ctx);
        default:
          CUST_KERNEL_LOG_ERROR(ctx, "Unsupported input1 data type [%s].", DTypeStr(input1_type).c_str());
      }
    case DT_INT64:
      switch (input1_type) {
        case DT_FLOAT16:
          return ComputePolygamma<std::int64_t, Eigen::half>(ctx);
        case DT_FLOAT:
          return ComputePolygamma<std::int64_t, std::float_t>(ctx);
        case DT_DOUBLE:
          return ComputePolygamma<std::int64_t, std::double_t>(ctx);
        default:
          CUST_KERNEL_LOG_ERROR(ctx, "Unsupported input1 data type [%s].", DTypeStr(input1_type).c_str());
      }
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Unsupported input0 data type [%s].", DTypeStr(input0_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t PolygammaCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::CheckPolygamma(ctx, kPolygammaInputNum, kPolygammaOutputNum) ? KERNEL_STATUS_PARAM_INVALID
                                                                              : detail::ComputePolygamma(ctx);
}

REGISTER_MS_CPU_KERNEL(kPolygamma, PolygammaCpuKernel);
}  // namespace aicpu