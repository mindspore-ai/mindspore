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

#include "expand.h"

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "context/inc/cpu_kernel_utils.h"

#include <iostream>
using namespace std;

namespace {
const char *kExpand = "Expand";
const size_t ExpandOutputDescNum = 1;
const size_t ExpandInputNum = 2;
const uint32_t Num2 = 2;
const uint32_t Num3 = 3;
const uint32_t Num4 = 4;
const uint32_t Num5 = 5;
const uint32_t Num6 = 6;
const uint32_t Num7 = 7;
const uint32_t Num8 = 8;
}  // namespace

namespace aicpu {
uint32_t ExpandCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, ExpandInputNum, ExpandOutputDescNum),
                           "Expand check input and output number failed.");
  auto data_type = static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  switch (data_type) {
    case DT_FLOAT16:
      return ExpandCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return ExpandCompute<float>(ctx);
    case DT_INT8:
      return ExpandCompute<int8_t>(ctx);
    case DT_INT32:
      return ExpandCompute<int32_t>(ctx);
    case DT_UINT8:
      return ExpandCompute<uint8_t>(ctx);
    case DT_INT64:
      return ExpandCompute<int64_t>(ctx);
    case DT_DOUBLE:
      return ExpandCompute<double>(ctx);
    case DT_BOOL:
      return ExpandCompute<bool>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "[%s] Data type of input is not support, input data type is [%s].",
                            ctx.GetOpType().c_str(), DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <int32_t RANK, typename T, int32_t OPTION>
uint32_t ExpandCpuKernel::BroadcastCompute(CpuKernelContext &ctx, BCalcInfo &calc_info) {
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input0(static_cast<T *>(calc_info.input_0->GetData()),
                                                       calc_info.input_0->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input1(static_cast<T *>(calc_info.input_1->GetData()),
                                                       calc_info.input_1->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> output(static_cast<T *>(calc_info.output->GetData()),
                                                       calc_info.output->GetTensorShape()->NumElements());

  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape_0;
  Eigen::DSizes<Eigen::DenseIndex, RANK> shape_out;
  Eigen::array<Eigen::DenseIndex, RANK> bcast_0;

  for (int32_t i = 0; i < RANK; i++) {
    reshape_0[RANK - i - 1] = calc_info.reshape_0[i];
    shape_out[RANK - i - 1] = calc_info.shape_out[i];
    bcast_0[RANK - i - 1] = calc_info.bcast_0[i];
  }
  output.reshape(shape_out) = input0.reshape(reshape_0).broadcast(bcast_0);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ExpandCpuKernel::ExpandCompute(CpuKernelContext &ctx) {
  BCalcInfo calc_info;
  calc_info.input_0 = ctx.Input(kFirstInputIndex);
  calc_info.input_1 = ctx.Output(kFirstOutputIndex);
  calc_info.output = ctx.Output(kFirstOutputIndex);

  Bcast bcast(ctx);
  if (bcast.GenerateBcastInfo(calc_info) != KERNEL_STATUS_OK) {
    CUST_KERNEL_LOG_ERROR(ctx, "[%s] Generate broadcast info failed.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  (void)bcast.GetBcastVec(calc_info);
  int32_t rank = static_cast<int32_t>(calc_info.shape_out.size());

  switch (rank) {
    case 0: {
      T v0 = *(reinterpret_cast<const T *>(calc_info.input_0->GetData()));
      T *value_out = reinterpret_cast<T *>(calc_info.output->GetData());
      *(value_out) = v0;
      return KERNEL_STATUS_OK;
    }
    case 1:
      return ExpandCalculateWithAlignedCheck<1, T>(ctx, calc_info);
    case Num2:
      return ExpandCalculateWithAlignedCheck<Num2, T>(ctx, calc_info);
    case Num3:
      return ExpandCalculateWithAlignedCheck<Num3, T>(ctx, calc_info);
    case Num4:
      return ExpandCalculateWithAlignedCheck<Num4, T>(ctx, calc_info);
    case Num5:
      return ExpandCalculateWithAlignedCheck<Num5, T>(ctx, calc_info);
    case Num6:
      return ExpandCalculateWithAlignedCheck<Num6, T>(ctx, calc_info);
    case Num7:
      return ExpandCalculateWithAlignedCheck<Num7, T>(ctx, calc_info);
    case Num8:
      return ExpandCalculateWithAlignedCheck<Num8, T>(ctx, calc_info);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "[%s] Rank of output should expand than 8 but get [%zu].", ctx.GetOpType().c_str(),
                            calc_info.shape_out.size());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <int32_t RANK, typename T>
uint32_t ExpandCpuKernel::ExpandCalculateWithAlignedCheck(CpuKernelContext &ctx, BCalcInfo &calc_info) {
  if (AlignedCheck(calc_info)) {
    return BroadcastCompute<RANK, T, Eigen::Aligned>(ctx, calc_info);
  }
  return BroadcastCompute<RANK, T, Eigen::Unaligned>(ctx, calc_info);
}

bool ExpandCpuKernel::AlignedCheck(const BCalcInfo &calc_info) {
  return AddrAlignedCheck(calc_info.input_0->GetData()) && AddrAlignedCheck(calc_info.input_1->GetData()) &&
         AddrAlignedCheck(calc_info.output->GetData());
}

REGISTER_MS_CPU_KERNEL(kExpand, ExpandCpuKernel);
}  // namespace aicpu
