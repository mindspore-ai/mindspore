/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "cpu_kernel/ms_kernel/adaptive_max_pool_2d_grad.h"

#include <cmath>
#include <algorithm>
#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kAdaptiveMaxPool2dGrad = "AdaptiveMaxPool2dGrad";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;
const int64_t kParallelDataNumCHW = 2 * 1024;
const int64_t kParallelDataNumMidCHW = 16 * 1024;
const int64_t kParallelDataNumNCHW = 16 * 1024;
const int64_t kParallelDataNumMidNCHW = 64 * 1024;
const int64_t four = 4;
const int64_t three = 3;
template <typename SCALAR_T, typename INDICES_T>
struct AdaptiveCalcArgs {
  SCALAR_T *input_grad_data = nullptr;
  SCALAR_T *input_data = nullptr;
  SCALAR_T *output_grad_data = nullptr;
  INDICES_T *indices_data = nullptr;

  int64_t in_size_b = 0;
  int64_t in_size_d = 0;
  int64_t in_size_h = 0;
  int64_t in_size_w = 0;
  int64_t out_size_h = 0;
  int64_t out_size_w = 0;
};
}  // namespace

namespace aicpu {
template <typename SCALAR_T, typename INDICES_T>
void ComputeSingleThread(int64_t start, int64_t end, AdaptiveCalcArgs<SCALAR_T, INDICES_T> args) {
  for (auto d = start; d < end; d++) {
    SCALAR_T *grad_input_p_d = args.input_grad_data + d * args.in_size_h * args.in_size_w;
    SCALAR_T *grad_output_p_d = args.output_grad_data + d * args.out_size_h * args.out_size_w;
    INDICES_T *ind_p_d = args.indices_data + d * args.out_size_h * args.out_size_w;
    /* calculate max points */
    int64_t oh;
    int64_t ow;
    for (oh = 0; oh < args.out_size_h; oh++) {
      for (ow = 0; ow < args.out_size_w; ow++) {
        /* retrieve position of max */
        INDICES_T maxp = ind_p_d[oh * args.out_size_w + ow];

        grad_input_p_d[maxp] += grad_output_p_d[oh * args.out_size_w + ow];
      }
    }
  }
}

template <typename SCALAR_T, typename INDICES_T>
void AdaptiveMaxPool2dGradSingleOutFrame(CpuKernelContext &ctx, AdaptiveCalcArgs<SCALAR_T, INDICES_T> args) {
  auto data_num = ctx.Input(0)->GetTensorShape()->NumElements();
  if (data_num >= kParallelDataNumCHW) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (data_num <= kParallelDataNumMidCHW) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }
    auto sharder_adaptive_max_pool_2d_grad = [&](int64_t start, int64_t end) { ComputeSingleThread(start, end, args); };
    if (max_core_num != 0) {
      CpuKernelUtils::ParallelFor(ctx, args.in_size_d, std::ceil(static_cast<float>(args.in_size_d) / max_core_num),
                                  sharder_adaptive_max_pool_2d_grad);
    }
  } else {
    ComputeSingleThread(0, args.in_size_d, args);
  }
}

template <typename SCALAR_T, typename INDICES_T>
void AdaptiveMaxPool2dGradOutFrame(CpuKernelContext &ctx, AdaptiveCalcArgs<SCALAR_T, INDICES_T> args) {
  auto data_num = ctx.Input(0)->GetTensorShape()->NumElements();
  if (data_num >= kParallelDataNumNCHW) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumMidNCHW) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto sharder_adaptive_max_pool_2d_grad = [&](int64_t start, int64_t end) {
      for (auto b = start; b < end; b++) {
        AdaptiveCalcArgs<SCALAR_T, INDICES_T> sub_args = args;
        sub_args.input_grad_data = args.input_grad_data + b * args.in_size_d * args.in_size_h * args.in_size_w;
        sub_args.output_grad_data = args.output_grad_data + b * args.in_size_d * args.out_size_h * args.out_size_w;
        sub_args.indices_data = args.indices_data + b * args.in_size_d * args.out_size_h * args.out_size_w;

        AdaptiveMaxPool2dGradSingleOutFrame<SCALAR_T, INDICES_T>(ctx, sub_args);
      }
    };
    if (max_core_num != 0) {
      CpuKernelUtils::ParallelFor(ctx, args.in_size_b, std::ceil(static_cast<float>(args.in_size_b) / max_core_num),
                                  sharder_adaptive_max_pool_2d_grad);
    }
  } else {
    for (auto b = 0; b < args.in_size_b; b++) {
      AdaptiveCalcArgs<SCALAR_T, INDICES_T> sub_args = args;
      sub_args.input_grad_data = args.input_grad_data + b * args.in_size_d * args.in_size_h * args.in_size_w;
      sub_args.output_grad_data = args.output_grad_data + b * args.in_size_d * args.out_size_h * args.out_size_w;
      sub_args.indices_data = args.indices_data + b * args.in_size_d * args.out_size_h * args.out_size_w;

      AdaptiveMaxPool2dGradSingleOutFrame<SCALAR_T, INDICES_T>(ctx, sub_args);
    }
  }
}

template <typename SCALAR_T, typename INDICES_T>
uint32_t AdaptiveMaxPool2dGradOutCpuTemplate(CpuKernelContext &ctx) {
  int64_t dim_w = 2;
  int64_t dim_h = 1;
  int64_t size_b = 1;
  Tensor &input = *(ctx.Input(kSecondInputIndex));
  auto input_shape_ptr = input.GetTensorShape();
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_shape_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input x0 shape failed.");
  int32_t input_dims = input_shape_ptr->GetDims();
  if (input_dims == four) {
    size_b = input_shape_ptr->GetDimSize(0);
    dim_w++;
    dim_h++;
  }
  AdaptiveCalcArgs<SCALAR_T, INDICES_T> args;
  args.in_size_b = size_b;
  args.in_size_d = input_shape_ptr->GetDimSize(dim_h - 1);
  args.in_size_h = input_shape_ptr->GetDimSize(dim_h);
  args.in_size_w = input_shape_ptr->GetDimSize(dim_w);
  args.out_size_h = ctx.Input(0)->GetTensorShape()->GetDimSize(dim_h);
  args.out_size_w = ctx.Input(0)->GetTensorShape()->GetDimSize(dim_w);
  // indices will contain i,j locations for each output point
  args.input_data = static_cast<SCALAR_T *>(input.GetData());
  args.output_grad_data = static_cast<SCALAR_T *>(ctx.Input(kFirstInputIndex)->GetData());
  args.indices_data = static_cast<INDICES_T *>(ctx.Input(kThirdInputIndex)->GetData());
  args.input_grad_data = static_cast<SCALAR_T *>(ctx.Output(kFirstOutputIndex)->GetData());
  for (auto i = 0; i < ctx.Input(1)->GetTensorShape()->NumElements(); i++) {
    args.input_grad_data[i] = static_cast<SCALAR_T>(0);
  }
  // resize output
  if (input_dims == three) {
    AdaptiveMaxPool2dGradSingleOutFrame<SCALAR_T, INDICES_T>(ctx, args);
  } else {
    AdaptiveMaxPool2dGradOutFrame<SCALAR_T, INDICES_T>(ctx, args);
  }
  return KERNEL_STATUS_OK;
}

template <typename SCALAR_T>
uint32_t AdaptiveMaxPool2dGrad::DoCompute(CpuKernelContext &ctx, DataType indices_type) {
  // Compute by indices_type
  switch (indices_type) {
    case DT_INT32:
      return AdaptiveMaxPool2dGradOutCpuTemplate<SCALAR_T, int32_t>(ctx);
    case DT_INT64:
      return AdaptiveMaxPool2dGradOutCpuTemplate<SCALAR_T, int64_t>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Output data_type [%s] must be in [{DT_INT32, DT_INT64}].",
                            DTypeStr(indices_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t AdaptiveMaxPool2dGrad::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "AdaptiveMaxPool2dGrad check input and output number failed.");

  auto data_type = static_cast<DataType>(ctx.Input(1)->GetDataType());
  // Compute by data_type
  auto indices_type = ctx.Input(2)->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      return DoCompute<float>(ctx, indices_type);
    case DT_DOUBLE:
      return DoCompute<double>(ctx, indices_type);
    case DT_FLOAT16:
      return DoCompute<Eigen::half>(ctx, indices_type);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "AdptetiveMaxPool2dGrad kernel data type [%s] not support.",
                            DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kAdaptiveMaxPool2dGrad, AdaptiveMaxPool2dGrad);
}  // namespace aicpu
