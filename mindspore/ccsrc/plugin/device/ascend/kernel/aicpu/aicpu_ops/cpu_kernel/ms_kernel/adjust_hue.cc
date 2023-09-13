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

#include "cpu_kernel/ms_kernel/adjust_hue.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <memory>
#include <iostream>
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "cpu_kernel/inc/cpu_types.h"
#include "common/kernel_log.h"
#include "cpu_kernel/common/status.h"
#include "utils/kernel_util.h"
#include "cpu_kernel/inc/cpu_context.h"

namespace {
const char *kAdjustHue = "AdjustHue";
const std::int64_t kAdjustHueParallelNum = 8 * 1024;
const std::int64_t kAdjustHueZero = 0;
const std::int64_t kAdjustHueOne = 1;
const std::int64_t kAdjustHueTwo = 2;
const std::int64_t kAdjustHueThree = 3;
const std::int64_t kAdjustHueFour = 4;
const std::int64_t kAdjustHueFive = 5;
}  // namespace

namespace aicpu {
namespace detail {
inline std::uint32_t ExtraCheckAdjustHue(const CpuKernelContext &ctx) {
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
  if (ctx.Input(1)->GetDataType() != aicpu::DataType::DT_FLOAT) {
    KERNEL_LOG_ERROR("The data type of the input [%s] need be [%s].", DTypeStr(ctx.Input(1)->GetDataType()).c_str(),
                     DTypeStr(aicpu::DataType::DT_FLOAT).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(1)->GetDataSize() != kAdjustHueFour) {
    KERNEL_LOG_ERROR("The data size of the input [%llu] need be [%llu].", ctx.Input(1)->GetDataSize(), kAdjustHueFour);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  std::shared_ptr<TensorShape> images_shape = ctx.Input(0)->GetTensorShape();
  int32_t dims = images_shape->GetDims();
  int64_t last_dim_size = images_shape->GetDimSize(dims - 1);
  if (last_dim_size != kAdjustHueThree) {
    KERNEL_LOG_ERROR("input must have 3 channels but instead has %llu channels.", last_dim_size);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

inline std::uint32_t CheckAdjustHue(const CpuKernelContext &ctx, std::uint32_t inputs_num, std::uint32_t outputs_num) {
  return NormalCheck(const_cast<CpuKernelContext &>(ctx), kAdjustHueTwo, kAdjustHueOne) ? KERNEL_STATUS_PARAM_INVALID
                                                                                        : ExtraCheckAdjustHue(ctx);
}
}  // namespace detail

static void rgb_to_hv_range(float r, float g, float b, float *h, float *v_min, float *v_max) {
  float v_mid;
  int h_category;
  // According to the figures in:
  // https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma
  // For the conditions, we don't care about the case where two components are
  // equal. It is okay to count it in either side in that case.
  if (r < g) {
    if (b < r) {
      // b < r < g
      *v_max = g;
      v_mid = r;
      *v_min = b;
      h_category = kAdjustHueOne;
    } else if (b > g) {
      // r < g < b
      *v_max = b;
      v_mid = g;
      *v_min = r;
      h_category = kAdjustHueThree;
    } else {
      // r < b < g
      *v_max = g;
      v_mid = b;
      *v_min = r;
      h_category = kAdjustHueTwo;
    }
  } else {
    // g < r
    if (b < g) {
      // b < g < r
      *v_max = r;
      v_mid = g;
      *v_min = b;
      h_category = kAdjustHueZero;
    } else if (b > r) {
      // g < r < b
      *v_max = b;
      v_mid = r;
      *v_min = g;
      h_category = kAdjustHueFour;
    } else {
      // g < b < r
      *v_max = r;
      v_mid = b;
      *v_min = g;
      h_category = kAdjustHueFive;
    }
  }
  if (*v_max == *v_min) {
    *h = 0;
    return;
  }
  auto ratio = (v_mid - *v_min) / (*v_max - *v_min);
  bool increase = ((h_category & 0x1) == 0);
  *h = h_category + (increase ? ratio : (1 - ratio));
}

// Helper function to convert from H-and-V-range to RGB.
template <typename T>
static void hv_range_to_rgb(float h, float v_min, float v_max, T *r, T *g, T *b) {
  int h_category = static_cast<int>(h);
  float ratio = h - h_category;
  bool increase = ((h_category & 0x1) == 0);
  if (!increase) {
    ratio = 1 - ratio;
  }
  float v_mid = v_min + ratio * (v_max - v_min);
  // According to the figures in:
  // https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma
  switch (h_category) {
    case kAdjustHueZero:
      *r = static_cast<T>(v_max);
      *g = static_cast<T>(v_mid);
      *b = static_cast<T>(v_min);
      break;
    case kAdjustHueOne:
      *r = static_cast<T>(v_mid);
      *g = static_cast<T>(v_max);
      *b = static_cast<T>(v_min);
      break;
    case kAdjustHueTwo:
      *r = static_cast<T>(v_min);
      *g = static_cast<T>(v_max);
      *b = static_cast<T>(v_mid);
      break;
    case kAdjustHueThree:
      *r = static_cast<T>(v_min);
      *g = static_cast<T>(v_mid);
      *b = static_cast<T>(v_max);
      break;
    case kAdjustHueFour:
      *r = static_cast<T>(v_mid);
      *g = static_cast<T>(v_min);
      *b = static_cast<T>(v_max);
      break;
    case kAdjustHueFive:
    default:
      *r = static_cast<T>(v_max);
      *g = static_cast<T>(v_min);
      *b = static_cast<T>(v_mid);
  }
}

HsvTuple rgb2hsv(const float r, const float g, const float b) {
  HsvTuple tuple;
  const float M = fmaxf(r, fmaxf(g, b));
  const float m = fminf(r, fminf(g, b));
  const float chroma = M - m;
  float h = 0.0f;
  float s = 0.0f;
  // hue
  if (chroma > 0.0f) {
    if (M == r) {
      const float num = (g - b) / chroma;
      const float sign = copysignf(1.0f, num);
      h = ((sign < 0.0f) * 6.0f + sign * fmodf(sign * num, 6.0f)) / 6.0f;
    } else if (M == g) {
      h = ((b - r) / chroma + 2.0f) / 6.0f;
    } else {
      h = ((r - g) / chroma + 4.0f) / 6.0f;
    }
  } else {
    h = 0.0f;
  }
  // saturation
  if (M > 0) {
    s = chroma / M;
  } else {
    s = 0.0f;
  }
  tuple.h = h;
  tuple.s = s;
  tuple.v = M;
  return tuple;
}

RgbTuple hsv2rgb(const float h, const float s, const float v) {
  RgbTuple tuple;
  const float new_h = h * 6.0f;
  const float chroma = v * s;
  const float x = chroma * (1.0f - fabsf(fmodf(new_h, 2.0f) - 1.0f));
  const float new_m = v - chroma;
  const bool between_0_and_1 = new_h >= 0.0f && new_h < 1.0f;
  const bool between_1_and_2 = new_h >= 1.0f && new_h < 2.0f;
  const bool between_2_and_3 = new_h >= 2.0f && new_h < 3.0f;
  const bool between_3_and_4 = new_h >= 3.0f && new_h < 4.0f;
  const bool between_4_and_5 = new_h >= 4.0f && new_h < 5.0f;
  const bool between_5_and_6 = new_h >= 5.0f && new_h < 6.0f;
  tuple.r = chroma * (between_0_and_1 || between_5_and_6) + x * (between_1_and_2 || between_4_and_5) + new_m;
  tuple.g = chroma * (between_1_and_2 || between_2_and_3) + x * (between_0_and_1 || between_3_and_4) + new_m;
  tuple.b = chroma * (between_3_and_4 || between_4_and_5) + x * (between_2_and_3 || between_5_and_6) + new_m;
  return tuple;
}

template <typename T>
uint32_t AdjustHueCpuKernel::DoCompute(const CpuKernelContext &ctx, const ComputeOptions &options) {
  const Tensor *input = options.input;
  const Tensor *delta = options.delta;
  Tensor *output = options.output;
  static const int64_t kChannelSize = 3;
  int64_t num_elements = input->NumElements();
  auto input_data = static_cast<T *>(input->GetData());
  auto output_data = static_cast<T *>(output->GetData());
  auto delta_h = static_cast<float *>(delta->GetData())[0];
  auto sharder_adjusthue = [&](int64_t start, int64_t end) {
    for (int64_t i = start * kChannelSize; i < end * kChannelSize; i = i + kChannelSize) {
      // CPU compute
      float h;
      float v_min;
      float v_max;
      rgb_to_hv_range(static_cast<float>(*(input_data + i)), static_cast<float>(*(input_data + i + 1)),
                      static_cast<float>(*(input_data + i + 2)), &h, &v_min, &v_max);

      static const int kChannelRange = 6;
      // Adjust the hue value. And adjust the hue back into the valid
      // range of [0, 6). It is faster than a fmod by avoiding
      // a float-point division since h is often very close to this
      // range.
      h += delta_h * kChannelRange;
      while (h < 0) {
        h += kChannelRange;
      }
      while (h >= kChannelRange) {
        h -= kChannelRange;
      }

      hv_range_to_rgb<T>(h, v_min, v_max, &output_data[i], &output_data[i + 1], &output_data[i + 2]);
    }
  };

  std::int64_t total = num_elements / kChannelSize;
  if (total > kAdjustHueParallelNum) {
    auto cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
    std::int64_t per_unit_size = total / std::min(std::max(1L, cores - 2L), total);
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, sharder_adjusthue),
                        "AdjustHue Compute failed.");
  } else {
    sharder_adjusthue(0, total);
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t AdjustHueCpuKernel::DoComputeHalf(const CpuKernelContext &ctx, const ComputeOptions &options) {
  const Tensor *input = options.input;
  const Tensor *delta = options.delta;
  Tensor *output = options.output;
  static const int64_t kChannelSize = 3;
  int64_t num_elements = input->NumElements();
  auto input_data = static_cast<T *>(input->GetData());
  auto output_data = static_cast<T *>(output->GetData());
  auto delta_h = static_cast<float *>(delta->GetData())[0];
  auto sharder_adjusthue = [&](int64_t start, int64_t end) {
    for (int64_t i = start * kChannelSize; i < end * kChannelSize; i = i + kChannelSize) {
      const HsvTuple hsv = rgb2hsv(static_cast<float>(*(input_data + i)), static_cast<float>(*(input_data + i + 1)),
                                   static_cast<float>(*(input_data + i + 2)));
      float new_h = hsv.h;
      float new_s = hsv.s;
      float new_v = hsv.v;
      // hue adjustment
      new_h = fmodf(hsv.h + delta_h, 1.0f);
      if (new_h < 0.0f) {
        new_h = fmodf(1.0f + new_h, 1.0f);
      }
      const RgbTuple rgb = hsv2rgb(new_h, new_s, new_v);
      output_data[i] = static_cast<T>(rgb.r);
      output_data[i + 1] = static_cast<T>(rgb.g);
      output_data[i + 2] = static_cast<T>(rgb.b);
    }
  };

  std::int64_t total = num_elements / kChannelSize;
  if (total > kAdjustHueParallelNum) {
    auto cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
    std::int64_t per_unit_size = total / std::min(std::max(1L, cores - 2L), total);
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, sharder_adjusthue),
                        "AdjustHue Compute failed.");
  } else {
    sharder_adjusthue(0, total);
  }

  return KERNEL_STATUS_OK;
}
uint32_t AdjustHueCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(detail::CheckAdjustHue(ctx, kAdjustHueTwo, kAdjustHueOne),
                      "AdjustHue check input and output number failed.");
  Tensor *input = ctx.Input(0);
  Tensor *delta = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  int64_t num_elements = input->NumElements();
  int32_t dims = input->GetTensorShape()->GetDims();

  auto channels = input->GetTensorShape()->GetDimSize(dims - 1);
  if (channels == 0) {
    KERNEL_LOG_ERROR("input must have 3 channels but instead has 0 channels.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto data_type = ctx.Input(0)->GetDataType();

  switch (data_type) {
    case DT_FLOAT16:
      if (num_elements > 0) {
        const int64_t channel_count = num_elements / channels;
        ComputeOptions options;
        options.input = input;
        options.delta = delta;
        options.output = output;
        options.channel_count = channel_count;

        DoComputeHalf<Eigen::half>(ctx, options);
      }
      break;
    case DT_FLOAT:
      if (num_elements > 0) {
        const int64_t channel_count = num_elements / channels;
        ComputeOptions options;
        options.input = input;
        options.delta = delta;
        options.output = output;
        options.channel_count = channel_count;

        DoCompute<float>(ctx, options);
      }
      break;
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kAdjustHue, AdjustHueCpuKernel);
}  // namespace aicpu
