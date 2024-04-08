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
#include "ms_kernel/rgb_to_hsv.h"

#include <iostream>
#include <algorithm>

#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr size_t kInputShapeRank = 3;
constexpr size_t kOutputShapeRank = 3;
constexpr int64_t kImageChannels = 3;
const char *kInputStr = "input";
const char *kOutputStr = "output";
const char *kRGBToHSV = "RGBToHSV";
// when input data size is more than kParallelDataNum, use Parallel func
}  // namespace

namespace aicpu {

const std::map<std::string, RGBToHSVCpuKernel::KernelFunction> RGBToHSVCpuKernel::kernels_ = {
  {"(DT_FLOAT16,DT_FLOAT16)", &RGBToHSVCpuKernel::DoCompute<Eigen::half, Eigen::half>},
  {"(DT_FLOAT,DT_FLOAT)", &RGBToHSVCpuKernel::DoCompute<float, float>},
  {"(DT_DOUBLE,DT_DOUBLE)", &RGBToHSVCpuKernel::DoCompute<double, double>}};

const std::vector<std::string> RGBToHSVCpuKernel::kernels_name_ = {"(DT_FLOAT16,DT_FLOAT16)", "(DT_FLOAT,DT_FLOAT)",
                                                                   "(DT_DOUBLE,DT_DOUBLE)"};

template <typename T1, typename T2>
uint32_t RGBToHSVCpuKernel::DoCompute(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  int64_t input0_elements_nums = input_tensor->NumElements();
  auto input_data = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto out = reinterpret_cast<T2 *>(ctx.Output(0)->GetData());

  for (int64_t i = 0; i < input0_elements_nums; i = i + 3) {
    auto t_red = *(input_data + i);
    auto t_green = *(input_data + i + 1);
    auto t_blue = *(input_data + i + 2);
    auto t_value = std::max(std::max(t_red, t_blue), t_green);
    auto t_minimum = std::min(std::min(t_red, t_blue), t_green);
    auto range = t_value - t_minimum;
    auto t_saturation = t_value > static_cast<T1>(0) ? (range / t_value) : static_cast<T1>(0);
    auto norm = static_cast<T1>(1.0) / static_cast<T1>(6.0) / range;
    auto t_hue = t_green == t_value ? (norm * (t_blue - t_red) + static_cast<T1>(2.0) / static_cast<T1>(6.0))
                                    : (norm * (t_red - t_green) + static_cast<T1>(4.0) / static_cast<T1>(6.0));
    t_hue = t_red == t_value ? (norm * (t_green - t_blue)) : t_hue;
    t_hue = range > static_cast<T1>(0) ? t_hue : static_cast<T1>(0);
    t_hue = t_hue < static_cast<T1>(0) ? (t_hue + static_cast<T1>(1)) : t_hue;
    *(out + i) = t_hue;
    *(out + i + 1) = t_saturation;
    *(out + i + 2) = t_value;
  }

  return KERNEL_STATUS_OK;
}

uint32_t RGBToHSVCpuKernel::CheckParam(CpuKernelContext &ctx, const std::string &in_or_out, uint32_t index,
                                       size_t rank) {
  Tensor *param = nullptr;
  if (in_or_out == kInputStr) {
    param = ctx.Input(index);
  } else if (in_or_out == kOutputStr) {
    param = ctx.Output(index);
  }
  std::string err_header = ConcatString(kRGBToHSV, " op ", in_or_out, "[", index, "]");

  CUST_KERNEL_CHECK_NULLPTR(ctx, param, KERNEL_STATUS_PARAM_INVALID, "%s tensor is nullptr.", err_header.c_str());

  auto param_shape = param->GetTensorShape();
  CUST_KERNEL_CHECK_NULLPTR(ctx, param_shape, KERNEL_STATUS_PARAM_INVALID, "%s tensor shape is nullptr.",
                            err_header.c_str());
  auto param_dim_sizes = param_shape->GetDimSizes();
  if (param_dim_sizes.size() < 1) {
    CUST_KERNEL_LOG_ERROR(ctx, "%s shape rank must be at least 1, but got shape[%zu].", err_header.c_str(),
                          VectorToString(param_dim_sizes).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (param->GetData() == nullptr) {
    CUST_KERNEL_CHECK_NULLPTR(ctx, param, KERNEL_STATUS_PARAM_INVALID, "%s tensor data is nullptr.",
                              err_header.c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t RGBToHSVCpuKernel::CheckShapes(CpuKernelContext &ctx) {
  auto input0_shape = ctx.Input(kFirstInputIndex)->GetTensorShape()->GetDimSizes();
  if (input0_shape.back() != kImageChannels) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "%s op input[0] shape last dim should be [%d], but got "
                          "shape[%s].",
                          kRGBToHSV, kImageChannels, VectorToString(input0_shape).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t RGBToHSVCpuKernel::CheckParams(CpuKernelContext &ctx) {
  auto ret = CheckParam(ctx, kInputStr, kFirstInputIndex, kInputShapeRank);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  ret = CheckShapes(ctx);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  return KERNEL_STATUS_OK;
}

uint32_t RGBToHSVCpuKernel::Compute(CpuKernelContext &ctx) {
  auto input0 = ctx.Input(kFirstInputIndex);
  CUST_KERNEL_CHECK_NULLPTR(ctx, input0, KERNEL_STATUS_PARAM_INVALID, "%s input[0] tensor is nullptr.", kRGBToHSV);
  DataType input0_data_type = input0->GetDataType();
  CUST_KERNEL_LOG_DEBUG(ctx, "%s op input[0] data type is [%s].", kRGBToHSV, DTypeStr(input0_data_type).c_str());

  auto output = ctx.Output(kFirstOutputIndex);
  CUST_KERNEL_CHECK_NULLPTR(ctx, output, KERNEL_STATUS_PARAM_INVALID, "%s output[0] tensor is nullptr.", kRGBToHSV);
  DataType output_data_type = output->GetDataType();
  CUST_KERNEL_LOG_DEBUG(ctx, "%s op output[0] data type is [%s].", kRGBToHSV, DTypeStr(output_data_type).c_str());

  std::string kernel_name = ConcatString("(", DTypeStr(input0_data_type), ",", DTypeStr(output_data_type), ")");

  auto it = kernels_.find(kernel_name);
  if (it != kernels_.end()) {
    auto ret = CheckParams(ctx);
    if (ret != KERNEL_STATUS_OK) {
      return ret;
    }
    auto kernel = it->second;
    ret = kernel(ctx);
    CUST_KERNEL_LOG_DEBUG(ctx, "%s op end.", kRGBToHSV);
    return ret;
  }

  CUST_KERNEL_LOG_ERROR(ctx, "%s op only support data type [%s], but got [%s].", kRGBToHSV,
                        VectorToString(kernels_name_).c_str(), kernel_name.c_str());
  return KERNEL_STATUS_PARAM_INVALID;
}

REGISTER_MS_CPU_KERNEL(kRGBToHSV, RGBToHSVCpuKernel);
}  // namespace aicpu
