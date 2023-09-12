/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "fractional_max_pool_3d_grad_with_fixed_ksize.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "cpu_kernel_utils.h"
#include "securec.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
const uint32_t Num2 = 2;
const uint32_t Num3 = 3;
const uint32_t Num4 = 4;
const uint32_t Num5 = 5;
const char *kFractionalMaxPool3DGradWithFixedKsize = "FractionalMaxPool3DGradWithFixedKsize";
}  // namespace

namespace aicpu {
uint32_t FractionalMaxPool3DGradWithFixedKsizeCpuKernel::GetInputAndCheck(const CpuKernelContext &ctx) {
  Tensor *origin_input = ctx.Input(0);
  Tensor *out_backprop = ctx.Input(1);
  Tensor *argmax = ctx.Input(2);
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "FractionalMaxPool3DGradWithFixedKsize check params failed.");
  AttrValue *data_format = ctx.GetAttr("data_format");
  KERNEL_CHECK_NULLPTR(data_format, KERNEL_STATUS_PARAM_INVALID, "[%s] get attr:data_format failed.",
                       kFractionalMaxPool3DGradWithFixedKsize);
  input_shape = origin_input->GetTensorShape()->GetDimSizes();
  out_backprop_shape = out_backprop->GetTensorShape()->GetDimSizes();
  argmax_shape = argmax->GetTensorShape()->GetDimSizes();
  int64_t input_dims = input_shape.size();
  int64_t out_backprop_dims = out_backprop_shape.size();
  int64_t argmax_dims = argmax_shape.size();

  for (int64_t i = 0; i < input_dims; i++) {
    KERNEL_CHECK_FALSE((origin_input->GetTensorShape()->GetDimSize(i) > 0), KERNEL_STATUS_PARAM_INVALID,
                       "FractionalMaxPool3DGradWithFixedKsize: expected input to have non-empty spatial dimensions, "
                       "but input has sizes [%d] with dimension [%d] being empty.",
                       input_dims, i);
  }
  KERNEL_CHECK_FALSE((input_dims == Num4 || input_dims == Num5), KERNEL_STATUS_PARAM_INVALID,
                     "Non-empty [4D] or [5D] (batch mode) tensor expected for input.");

  for (int64_t i = 0; i < out_backprop_dims; i++) {
    KERNEL_CHECK_FALSE(
      (out_backprop->GetTensorShape()->GetDimSize(i) > 0), KERNEL_STATUS_PARAM_INVALID,
      "FractionalMaxPool3DGradWithFixedKsize: expected out_backprop to have non-empty spatial dimensions, "
      "but out_backprop has sizes [%d] with dimension [%d] being empty.",
      out_backprop_dims, i);
  }
  KERNEL_CHECK_FALSE((out_backprop_dims == Num4 || out_backprop_dims == Num5), KERNEL_STATUS_PARAM_INVALID,
                     "Non-empty [4D] or [5D] (batch mode) tensor expected for out_backprop.");

  for (int64_t i = 0; i < argmax_dims; i++) {
    KERNEL_CHECK_FALSE((argmax->GetTensorShape()->GetDimSize(i) > 0), KERNEL_STATUS_PARAM_INVALID,
                       "FractionalMaxPool3DGradWithFixedKsize: expected argmax to have non-empty spatial dimensions, "
                       "but argmax has sizes [%d] with dimension [%d] being empty.",
                       argmax_dims, i);
  }
  KERNEL_CHECK_FALSE((argmax_dims == Num4 || argmax_dims == Num5), KERNEL_STATUS_PARAM_INVALID,
                     "Non-empty [4D] or [5D] (batch mode) tensor expected for argmax.");
  return KERNEL_STATUS_OK;
}

template <typename backprop_t, typename argmax_t>
uint32_t FractionalMaxPool3DGradWithFixedKsizeCpuKernel::Process4DCase(const CpuKernelContext &ctx, TmpVar *tmp_var) {
  auto out_backprop_data = reinterpret_cast<backprop_t *>(ctx.Input(1)->GetData());
  auto argmax_data = reinterpret_cast<argmax_t *>(ctx.Input(2)->GetData());
  auto output_data = reinterpret_cast<backprop_t *>(ctx.Output(0)->GetData());
  uint32_t min_core_num = 1;
  uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
  if (max_core_num > tmp_var->inputC) {
    max_core_num = tmp_var->inputC;
  }
  if (max_core_num == 0) {
    KERNEL_LOG_ERROR("max_core_num should not be 0.");
  }
  CpuKernelUtils::ParallelFor(ctx, tmp_var->inputC, tmp_var->inputC / max_core_num, [&](int64_t start, int64_t end) {
    for (auto plane = start; plane < end; ++plane) {
      backprop_t *outputForPlane = output_data + plane * tmp_var->inputT * tmp_var->inputH * tmp_var->inputW;
      backprop_t *outbackpropForPlane =
        out_backprop_data + plane * tmp_var->outputT * tmp_var->outputH * tmp_var->outputW;
      argmax_t *argmaxForPlane = argmax_data + plane * tmp_var->outputT * tmp_var->outputH * tmp_var->outputW;
      int64_t h, w, t;
      for (t = 0; t < tmp_var->outputT; ++t) {
        for (h = 0; h < tmp_var->outputH; ++h) {
          for (w = 0; w < tmp_var->outputW; ++w) {
            argmax_t outputIndex = t * tmp_var->outputH * tmp_var->outputW + h * tmp_var->outputW + w;
            argmax_t index = argmaxForPlane[outputIndex];
            KERNEL_CHECK_FALSE(index >= 0 && index < tmp_var->inputT * tmp_var->inputH * tmp_var->inputW,
                               KERNEL_STATUS_PARAM_INVALID,
                               "FractionalMaxPool3DGradWithFixedKsize index value is illegal.");
            outputForPlane[index] += outbackpropForPlane[outputIndex];
          }
        }
      }
    }
    return KERNEL_STATUS_OK;
  });
  return KERNEL_STATUS_OK;
}

template <typename backprop_t, typename argmax_t>
uint32_t FractionalMaxPool3DGradWithFixedKsizeCpuKernel::Process5DCase(const CpuKernelContext &ctx, TmpVar *tmp_var) {
  auto out_backprop_data = reinterpret_cast<backprop_t *>(ctx.Input(1)->GetData());
  auto argmax_data = reinterpret_cast<argmax_t *>(ctx.Input(2)->GetData());
  auto output_data = reinterpret_cast<backprop_t *>(ctx.Output(0)->GetData());
  uint32_t min_core_num = 1;
  uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
  if (max_core_num > tmp_var->inputN) {
    max_core_num = tmp_var->inputN;
  }
  if (max_core_num == 0) {
    KERNEL_LOG_ERROR("max_core_num should not be 0.");
  }
  CpuKernelUtils::ParallelFor(ctx, tmp_var->inputN, tmp_var->inputN / max_core_num, [&](int64_t start, int64_t end) {
    for (auto batch = start; batch < end; ++batch) {
      for (auto plane = 0; plane < tmp_var->inputC; ++plane) {
        auto output_data_n =
          output_data + batch * tmp_var->inputC * tmp_var->inputW * tmp_var->inputH * tmp_var->inputT;
        auto out_backprop_data_n =
          out_backprop_data + batch * tmp_var->inputC * tmp_var->outputW * tmp_var->outputH * tmp_var->outputT;
        auto argmax_data_n =
          argmax_data + batch * tmp_var->inputC * tmp_var->outputW * tmp_var->outputH * tmp_var->outputT;
        backprop_t *outputForPlane = output_data_n + plane * tmp_var->inputT * tmp_var->inputH * tmp_var->inputW;
        backprop_t *outbackpropForPlane =
          out_backprop_data_n + plane * tmp_var->outputT * tmp_var->outputH * tmp_var->outputW;
        argmax_t *argmaxForPlane = argmax_data_n + plane * tmp_var->outputT * tmp_var->outputH * tmp_var->outputW;
        int64_t h;
        int64_t w;
        int64_t t;
        for (t = 0; t < tmp_var->outputT; ++t) {
          for (h = 0; h < tmp_var->outputH; ++h) {
            for (w = 0; w < tmp_var->outputW; ++w) {
              argmax_t outputIndex = t * tmp_var->outputH * tmp_var->outputW + h * tmp_var->outputW + w;
              argmax_t index = argmaxForPlane[outputIndex];
              KERNEL_CHECK_FALSE(index >= 0 && index < tmp_var->inputT * tmp_var->inputH * tmp_var->inputW,
                                 KERNEL_STATUS_PARAM_INVALID,
                                 "FractionalMaxPool3DGradWithFixedKsize index value is illegal.");
              outputForPlane[index] += outbackpropForPlane[outputIndex];
            }
          }
        }
      }
    }
    return KERNEL_STATUS_OK;
  });
  return KERNEL_STATUS_OK;
}

template <typename backprop_t, typename argmax_t>
uint32_t FractionalMaxPool3DGradWithFixedKsizeCpuKernel::FractionalMaxPool3DGradWithFixedKsizeOutCpuTemplate(
  const CpuKernelContext &ctx) {
  int64_t input_dims = input_shape.size();
  std::string format = "NCDHW";
  AttrValue *data_format = ctx.GetAttr("data_format");
  if (data_format != nullptr) {
    format = data_format->GetString();
  }
  TmpVar tmp_var;
  if (format == "NCDHW") {
    tmp_var.c_dim = 0;
    tmp_var.d_dim = 1;
    tmp_var.h_dim = Num2;
    tmp_var.w_dim = Num3;
    if (input_dims == Num5) {
      tmp_var.inputN = input_shape[0];
      tmp_var.c_dim++;
      tmp_var.d_dim++;
      tmp_var.h_dim++;
      tmp_var.w_dim++;
    }
    tmp_var.inputC = input_shape[tmp_var.c_dim];
    tmp_var.inputT = input_shape[tmp_var.d_dim];
    tmp_var.inputH = input_shape[tmp_var.h_dim];
    tmp_var.inputW = input_shape[tmp_var.w_dim];
    tmp_var.outputT = out_backprop_shape[tmp_var.d_dim];
    tmp_var.outputH = out_backprop_shape[tmp_var.h_dim];
    tmp_var.outputW = out_backprop_shape[tmp_var.w_dim];
  } else {
    tmp_var.c_dim = Num3;
    tmp_var.d_dim = 0;
    tmp_var.h_dim = 1;
    tmp_var.w_dim = Num2;
    if (input_dims == Num5) {
      tmp_var.inputN = input_shape[0];
      tmp_var.c_dim++;
      tmp_var.d_dim++;
      tmp_var.h_dim++;
      tmp_var.w_dim++;
    }
    tmp_var.inputC = input_shape[tmp_var.c_dim];
    tmp_var.inputT = input_shape[tmp_var.d_dim];
    tmp_var.inputH = input_shape[tmp_var.h_dim];
    tmp_var.inputW = input_shape[tmp_var.w_dim];
    tmp_var.outputT = out_backprop_shape[tmp_var.d_dim];
    tmp_var.outputH = out_backprop_shape[tmp_var.h_dim];
    tmp_var.outputW = out_backprop_shape[tmp_var.w_dim];
  }

  auto output_size = ctx.Output(0)->GetDataSize();
  auto output_addr = reinterpret_cast<char *>(ctx.Output(0)->GetData());
  while (output_size > 0) {
    auto copy_size = std::min(output_size, static_cast<uint64_t>(INT32_MAX));
    auto ret = memset_s(output_addr, output_size, 0, copy_size);
    if (ret != EOK) {
      KERNEL_LOG_ERROR("For 'FractionalMaxPool3DGradWithFixedKsize', memset_s failed, ret=%d.", ret);
      return KERNEL_STATUS_INNER_ERROR;
    }
    output_size -= copy_size;
    output_addr += copy_size;
  }

  if (input_dims == Num4) {
    return Process4DCase<backprop_t, int32_t>(ctx, &tmp_var);
  }
  return Process5DCase<backprop_t, int32_t>(ctx, &tmp_var);
}

template <typename backprop_t>
uint32_t FractionalMaxPool3DGradWithFixedKsizeCpuKernel::DoComputeWithArgmaxType(const CpuKernelContext &ctx,
                                                                                 DataType argmax_type) {
  switch (argmax_type) {
    case DT_INT32:
      return FractionalMaxPool3DGradWithFixedKsizeOutCpuTemplate<backprop_t, int32_t>(ctx);
    case DT_INT64:
      return FractionalMaxPool3DGradWithFixedKsizeOutCpuTemplate<backprop_t, int64_t>(ctx);
    default:
      KERNEL_LOG_ERROR("argmax_type [%s] must be in [{DT_INT32, DT_INT64}].", DTypeStr(argmax_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t FractionalMaxPool3DGradWithFixedKsizeCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(GetInputAndCheck(ctx), "kFractionalMaxPool3DGradWithFixedKsize check params failed.");
  auto origin_input_type = ctx.Input(0)->GetDataType();
  auto out_backprop_type = ctx.Input(1)->GetDataType();
  auto argmax_type = ctx.Input(2)->GetDataType();
  KERNEL_CHECK_FALSE((origin_input_type == out_backprop_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of origin_input [%s] need be same with "
                     "out_backprop [%s].",
                     DTypeStr(origin_input_type).c_str(), DTypeStr(out_backprop_type).c_str());
  switch (out_backprop_type) {
    case DT_FLOAT16:
      return DoComputeWithArgmaxType<Eigen::half>(ctx, argmax_type);
    case DT_FLOAT:
      return DoComputeWithArgmaxType<float>(ctx, argmax_type);
    case DT_DOUBLE:
      return DoComputeWithArgmaxType<double>(ctx, argmax_type);
    case DT_INT32:
      return DoComputeWithArgmaxType<int32_t>(ctx, argmax_type);
    case DT_INT64:
      return DoComputeWithArgmaxType<int64_t>(ctx, argmax_type);
    default:
      KERNEL_LOG_ERROR("kFractionalMaxPool3DGradWithFixedKsize kernel out_backprop_type type [%s] not support.",
                       DTypeStr(out_backprop_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kFractionalMaxPool3DGradWithFixedKsize, FractionalMaxPool3DGradWithFixedKsizeCpuKernel);
}  // namespace aicpu
