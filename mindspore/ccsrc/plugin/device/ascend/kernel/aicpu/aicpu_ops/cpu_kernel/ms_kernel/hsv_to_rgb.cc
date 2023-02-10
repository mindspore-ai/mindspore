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
#include "hsv_to_rgb.h"
#include <cmath>
#include <vector>

#include "Eigen/Core"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kInputNum = 1;
constexpr uint32_t kOutputNum = 1;
const char *kHSVToRGB = "HSVToRGB";
constexpr int64_t kParallelDataNums = 64 * 64;
}  // namespace

namespace aicpu {
uint32_t HSVToRGBCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(HSVToRGBCheck(ctx), "HSVToRGB check params failed.", kHSVToRGB);
  auto data_type = ctx.Input(kFirstInputIndex)->GetDataType();
  switch (data_type) {
    case (DT_FLOAT16): {
      uint32_t result = HSVToRGBComputeHalf(ctx);
      if (result != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("HSVToRGB kernel compute failed.");
        return result;
      }
      break;
    }
    case (DT_FLOAT): {
      uint32_t result = HSVToRGBCompute<float>(ctx);
      if (result != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("HSVToRGB kernel compute failed.");
        return result;
      }
      break;
    }
    case (DT_DOUBLE): {
      uint32_t result = HSVToRGBCompute<double>(ctx);
      if (result != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("HSVToRGB kernel compute failed.");
        return result;
      }
      break;
    }
    default:
      KERNEL_LOG_ERROR("HSVToRGB kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void HSVToRGBCpuKernel::ConvertOnePixel(T h, T s, T v, T *r, T *g, T *b) {
  T c = s * v;
  T m = v - c;
  T dh = h * 6;
  T rr, gg, bb;
  const int32_t h_category = static_cast<int32_t>(std::floor(dh));
  T fmodu = dh;
  if (fmodu <= 0 || fmodu >= 2) {
    const int32_t tmp = static_cast<int32_t>(fmodu);
    fmodu -= static_cast<T>((tmp / 2) * 2);
    if (fmodu <= 0) {
      fmodu += 2;
    } else if (fmodu >= 2) {
      fmodu -= 2;
    }
  }
  T x = c * (1 - std::abs(fmodu - 1));
  switch (h_category) {
    case 0:
      rr = c;
      gg = x;
      bb = 0;
      break;
    case 1:
      rr = x;
      gg = c;
      bb = 0;
      break;
    case 2:
      rr = 0;
      gg = c;
      bb = x;
      break;
    case 3:
      rr = 0;
      gg = x;
      bb = c;
      break;
    case 4:
      rr = x;
      gg = 0;
      bb = c;
      break;
    case 5:
      rr = c;
      gg = 0;
      bb = x;
      break;
    default:
      rr = c;
      gg = 0;
      bb = 0;
  }
  *r = rr + m;
  *g = gg + m;
  *b = bb + m;
}

uint32_t HSVToRGBCpuKernel::HSVToRGBCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check HSVToRGB check failed.");
  Tensor *input = ctx.Input(kFirstInputIndex);
  Tensor *output = ctx.Output(kFirstOutputIndex);
  KERNEL_LOG_INFO(
    "HSVToRGBCpuKernel[%s]"
    "input: addr[%p], size[%llu]; "
    "output: addr[%p], size[%llu]. "
    "data type: [%s]",
    ctx.GetOpType().c_str(), input->GetData(), input->GetDataSize(), output->GetData(), output->GetDataSize(),
    DTypeStr(input->GetDataType()).c_str());
  const std::vector<int64_t> dims = input->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE(dims.cend()[-1] == 3, KERNEL_STATUS_PARAM_INVALID, "Last dimension must be size 3.", kHSVToRGB);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t HSVToRGBCpuKernel::HSVToRGBCompute(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(kFirstInputIndex);
  Tensor *output = ctx.Output(kFirstOutputIndex);
  T *input_ptr = reinterpret_cast<T *>(input->GetData());
  T *output_ptr = reinterpret_cast<T *>(output->GetData());
  const int64_t data_num = input->GetTensorShape()->NumElements() / 3;
  const int64_t core_num =
    std::max(static_cast<int64_t>(1), static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx)));
  const int64_t per_uint_size = data_num / std::min(data_num, core_num);

  auto shard_hsv_to_rgb = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      T *h = input_ptr + 3 * i + 0;
      T *s = input_ptr + 3 * i + 1;
      T *v = input_ptr + 3 * i + 2;
      T *r = output_ptr + 3 * i + 0;
      T *g = output_ptr + 3 * i + 1;
      T *b = output_ptr + 3 * i + 2;
      ConvertOnePixel<T>(*h, *s, *v, r, g, b);
    }
  };
  if (data_num > kParallelDataNums) {
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, per_uint_size, shard_hsv_to_rgb),
                        "HSVtoRGB compute failed.");
  } else {
    shard_hsv_to_rgb(0, data_num);
  }
  return KERNEL_STATUS_OK;
}

uint32_t HSVToRGBCpuKernel::HSVToRGBComputeHalf(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(kFirstInputIndex);
  Tensor *output = ctx.Output(kFirstOutputIndex);
  Eigen::half *input_ptr = reinterpret_cast<Eigen::half *>(input->GetData());
  Eigen::half *output_ptr = reinterpret_cast<Eigen::half *>(output->GetData());
  const int64_t data_num = input->GetTensorShape()->NumElements() / 3;
  const int64_t core_num =
    std::max(static_cast<int64_t>(1), static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx)));
  const int64_t per_uint_size = data_num / std::min(data_num, core_num);

  auto shard_hsv_to_rgb = [&](size_t start, size_t end) {
    float tmp[3];
    for (size_t i = start; i < end; ++i) {
      const float h = static_cast<float>(*(input_ptr + 3 * i + 0));
      const float s = static_cast<float>(*(input_ptr + 3 * i + 1));
      const float v = static_cast<float>(*(input_ptr + 3 * i + 2));
      Eigen::half *r = output_ptr + 3 * i + 0;
      Eigen::half *g = output_ptr + 3 * i + 1;
      Eigen::half *b = output_ptr + 3 * i + 2;
      ConvertOnePixel<float>(h, s, v, tmp, tmp + 1, tmp + 2);
      *r = Eigen::half(tmp[0]);
      *g = Eigen::half(tmp[1]);
      *b = Eigen::half(tmp[2]);
    }
  };
  if (data_num > kParallelDataNums) {
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, per_uint_size, shard_hsv_to_rgb),
                        "HSVtoRGB compute failed.");
  } else {
    shard_hsv_to_rgb(0, data_num);
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kHSVToRGB, HSVToRGBCpuKernel);
}  // namespace aicpu
