/**
 * Copyright 2021 Jilin University
 * Copyright 2020 Huawei Technologies Co., Ltd.
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

#include "cpu_kernel/ms_kernel/adjust_saturation.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include "context/inc/cpu_kernel_utils.h"
#include "cpu_types.h"
#include "inc/kernel_log.h"
#include "context/common/status.h"
#include "utils/kernel_util.h"
#include "cpu_context.h"

namespace {
const std::uint32_t kAdjustSaturationInputNum{2u};
const std::uint32_t kAdjustSaturationOutputNum{1u};
const char *kAdjustSaturation{"AdjustSaturation"};
const std::int64_t kAdjustSaturationParallelNum{64 * 1024};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
struct Rgb {
  T r;
  T g;
  T b;
} __attribute__((packed));

template <>
struct Rgb<Eigen::half> {
  Eigen::half r;
  Eigen::half g;
  Eigen::half b;
};

template <typename T>
struct Hsv {
  T h;
  T s;
  T v;
} __attribute__((packed));

template <typename T>
inline Hsv<T> RgbToHsv(Rgb<T> in) {
  T min{in.r < in.g ? in.r : in.g};
  min = min < in.b ? min : in.b;

  T max{in.r > in.g ? in.r : in.g};
  max = max > in.b ? max : in.b;

  T delta{max - min};
  if (delta < static_cast<T>(0.00001)) {
    return Hsv<T>{static_cast<T>(0.0), static_cast<T>(0.0), max};
  }
  if (max < static_cast<T>(0.0)) {
    return Hsv<T>{static_cast<T>(0.0), static_cast<T>(NAN), max};
  }

  Hsv<T> out;

  if (in.r >= max) {
    out.h = (in.g - in.b) / delta;
  } else if (in.g >= max) {
    out.h = static_cast<T>(2.0) + (in.b - in.r) / delta;
  } else {
    out.h = static_cast<T>(4.0) + (in.r - in.g) / delta;
  }

  out.h /= static_cast<T>(6.0);

  if (out.h < static_cast<T>(0.0)) {
    out.h += static_cast<T>(1.0);
  }

  out.v = max;
  out.s = (delta / max);

  return out;
}

template <typename T>
inline Rgb<T> Hsv2Rgb(Hsv<T> in) {
  if (in.s <= static_cast<T>(0.0)) {
    return Rgb<T>{in.v, in.v, in.v};
  }
  T h{in.h};
  if (h >= static_cast<T>(1.0)) {
    h = static_cast<T>(0.0);
  }
  h *= static_cast<T>(6.0);
  auto i{static_cast<int64_t>(h)};
  auto f{static_cast<T>(h - static_cast<T>(i))};
  T p{in.v * (static_cast<T>(1.0) - in.s)};
  T q{in.v * (static_cast<T>(1.0) - (in.s * f))};
  T t{in.v * (static_cast<T>(1.0) - (in.s * (static_cast<T>(1.0) - f)))};

  switch (i) {
    case 0:
      return Rgb<T>{in.v, t, p};
    case 1:
      return Rgb<T>{q, in.v, p};
    case 2:
      return Rgb<T>{p, in.v, t};
    case 3:
      return Rgb<T>{p, q, in.v};
    case 4:
      return Rgb<T>{t, p, in.v};
    default:
      return Rgb<T>{in.v, p, q};
  }
}

template <typename T>
inline Rgb<T> ScalarAdjustSaturation(Rgb<T> image, std::float_t saturation_factor) {
  auto Hsv{RgbToHsv(image)};
  Hsv.s *= static_cast<T>(saturation_factor);
  if (Hsv.s > static_cast<T>(1.0)) {
    Hsv.s = static_cast<T>(1.0);
  }
  return Hsv2Rgb(Hsv);
}

template <>
inline Rgb<Eigen::half> ScalarAdjustSaturation(Rgb<Eigen::half> image, std::float_t saturation_factor) {
  auto Hsv{RgbToHsv(Rgb<std::float_t>{static_cast<std::float_t>(image.r), static_cast<std::float_t>(image.g),
                                      static_cast<std::float_t>(image.b)})};
  Hsv.s *= static_cast<std::float_t>(saturation_factor);
  if (Hsv.s > static_cast<std::float_t>(1.0)) Hsv.s = static_cast<std::float_t>(1.0);
  auto out{Hsv2Rgb(Hsv)};
  return Rgb<Eigen::half>{static_cast<Eigen::half>(out.r), static_cast<Eigen::half>(out.g),
                          static_cast<Eigen::half>(out.b)};
}

inline std::uint32_t ParallelForAdjustSaturation(CpuKernelContext &ctx, std::int64_t total, std::int64_t per_unit_size,
                                                 const std::function<void(std::int64_t, std::int64_t)> &work) {
  if (total > kAdjustSaturationParallelNum)
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  else
    work(0, total);
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeAdjustSaturationKernel(CpuKernelContext &ctx) {
  auto input{static_cast<Rgb<T> *>(ctx.Input(0)->GetData())};
  auto saturation_factor{static_cast<std::float_t *>(ctx.Input(1)->GetData())};
  auto output{static_cast<Rgb<T> *>(ctx.Output(0)->GetData())};
  auto ScalarAdjustSaturation1 = [&](Rgb<T> image) { return ScalarAdjustSaturation(image, saturation_factor[0]); };
  std::int64_t total{ctx.Input(0)->NumElements() / 3};
  auto cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  return ParallelForAdjustSaturation(ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
    std::transform(input + begin, input + end, output + begin, ScalarAdjustSaturation1);
  });
}

template <typename T>
inline std::uint32_t ComputeAdjustSaturation(CpuKernelContext &ctx) {
  std::uint32_t result{ComputeAdjustSaturationKernel<T>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    CUST_KERNEL_LOG_ERROR(ctx, "AdjustSaturation compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckAdjustSaturation(CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    CUST_KERNEL_LOG_ERROR(ctx, "The data type of the input [%s] need be the same as the output [%s].",
                          DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
                          DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "The data size of the input [%llu] need be the same as the output "
                          "[%llu].",
                          ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(1)->GetDataType() != aicpu::DataType::DT_FLOAT) {
    CUST_KERNEL_LOG_ERROR(ctx, "The data type of the input [%s] need be [%s].",
                          DTypeStr(ctx.Input(1)->GetDataType()).c_str(), DTypeStr(aicpu::DataType::DT_FLOAT).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(1)->GetDataSize() != 4) {
    CUST_KERNEL_LOG_ERROR(ctx, "The data size of the input [%llu] need be [%llu].", ctx.Input(1)->GetDataSize(), 4);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

inline std::uint32_t CheckAdjustSaturation(CpuKernelContext &ctx, std::uint32_t inputs_num, std::uint32_t outputs_num) {
  return NormalCheck(const_cast<CpuKernelContext &>(ctx), kAdjustSaturationInputNum, kAdjustSaturationOutputNum)
           ? KERNEL_STATUS_PARAM_INVALID
           : ExtraCheckAdjustSaturation(ctx);
}

inline std::uint32_t ComputeAdjustSaturation(CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeAdjustSaturation<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeAdjustSaturation<std::float_t>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t AdjustSaturationCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::CheckAdjustSaturation(ctx, kAdjustSaturationInputNum, kAdjustSaturationOutputNum)
           ? KERNEL_STATUS_PARAM_INVALID
           : detail::ComputeAdjustSaturation(ctx);
}

REGISTER_MS_CPU_KERNEL(kAdjustSaturation, AdjustSaturationCpuKernel);
}  // namespace aicpu
