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

#include "mul.h"
#include <algorithm>
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "context/inc/cpu_kernel_utils.h"

namespace {
const int32_t kMaxDim = 8;
const int32_t kMaxIndex = 7;
const char *const kMul = "Mul";
}  // namespace

namespace aicpu {
uint32_t MulCpuKernel::Compute(CpuKernelContext &ctx) {
  if (NormalMathCheck(ctx) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *input0 = ctx.Input(kFirstInputIndex);
  Tensor *input1 = ctx.Input(kSecondInputIndex);
  if ((input0->GetDataSize() == 0) || (input1->GetDataSize() == 0)) {
    CUST_KERNEL_LOG_INFO(ctx, "[%s] Input is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // choose compute function depend on dataType
  auto data_type = static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  switch (data_type) {
    case DT_FLOAT16:
      return MulCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return MulCompute<float>(ctx);
    case DT_DOUBLE:
      return MulCompute<double>(ctx);
    case DT_INT8:
      return MulCompute<int8_t>(ctx);
    case DT_INT16:
      return MulCompute<int16_t>(ctx);
    case DT_INT32:
      return MulCompute<int32_t>(ctx);
    case DT_INT64:
      return MulCompute<int64_t>(ctx);
    case DT_UINT8:
      return MulCompute<uint8_t>(ctx);
    case DT_UINT16:
      return MulCompute<uint16_t>(ctx);
    case DT_UINT32:
      return MulCompute<uint32_t>(ctx);
    case DT_UINT64:
      return MulCompute<uint64_t>(ctx);
    case DT_COMPLEX64:
      return MulCompute<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return MulCompute<std::complex<double>>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "[%s] Data type of input is not support, input data type is [%s].",
                            ctx.GetOpType().c_str(), DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t MulCpuKernel::MulCompute(CpuKernelContext &ctx) {
  BCalcInfo calcInfo;
  calcInfo.input_0 = ctx.Input(kFirstInputIndex);
  calcInfo.input_1 = ctx.Input(kSecondInputIndex);
  calcInfo.output = ctx.Output(kFirstOutputIndex);
  CUST_KERNEL_CHECK_NULLPTR(ctx, calcInfo.input_0->GetData(), KERNEL_STATUS_PARAM_INVALID,
                            "[%s] Get input 0 data failed", ctx.GetOpType().c_str())
  CUST_KERNEL_CHECK_NULLPTR(ctx, calcInfo.input_1->GetData(), KERNEL_STATUS_PARAM_INVALID,
                            "[%s] Get input 1 data failed", ctx.GetOpType().c_str())
  CUST_KERNEL_CHECK_NULLPTR(ctx, calcInfo.output->GetData(), KERNEL_STATUS_PARAM_INVALID, "[%s] Get output data failed",
                            ctx.GetOpType().c_str())
  CUST_KERNEL_LOG_INFO(ctx,
                       "[%s] Input[0] data size is [%llu], input[1] data size is [%llu], "
                       "output data size is [%llu].",
                       ctx.GetOpType().c_str(), calcInfo.input_0->GetDataSize(), calcInfo.input_1->GetDataSize(),
                       calcInfo.output->GetDataSize());
  // broadcast input
  Bcast bcast(ctx);
  if (bcast.GenerateBcastInfo(calcInfo) != KERNEL_STATUS_OK) {
    CUST_KERNEL_LOG_ERROR(ctx, "[%s] Generate broadcast info failed.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  bcast.GetBcastVec(calcInfo);
  int32_t rank = static_cast<int32_t>(calcInfo.shape_out.size());
  if (rank == 0) {
    T v0 = *(reinterpret_cast<const T *>(calcInfo.input_0->GetData()));
    T v1 = *(reinterpret_cast<const T *>(calcInfo.input_1->GetData()));
    T *value_out = reinterpret_cast<T *>(calcInfo.output->GetData());
    *(value_out) = v0 * v1;
    return KERNEL_STATUS_OK;
  } else if (rank < 0 || rank > kMaxDim) {
    CUST_KERNEL_LOG_ERROR(ctx, "[%s] Rank of output should less than 9 but get [%zu].", ctx.GetOpType().c_str(),
                          calcInfo.shape_out.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return MulCalculateWithAlignedCheck<T>(calcInfo, rank);
}

template <typename T>
uint32_t MulCpuKernel::MulCalculateWithAlignedCheck(BCalcInfo &calcInfo, const int32_t &rank) {
  if (AlignedCheck(calcInfo)) {
    return MulCalculate<T, Eigen::Aligned>(calcInfo, rank);
  }
  return MulCalculate<T, Eigen::Unaligned>(calcInfo, rank);
}

bool MulCpuKernel::AlignedCheck(const BCalcInfo &calcInfo) const {
  return AddrAlignedCheck(calcInfo.input_0->GetData()) && AddrAlignedCheck(calcInfo.input_1->GetData()) &&
         AddrAlignedCheck(calcInfo.output->GetData());
}

template <typename T, int32_t OPTION>
uint32_t MulCpuKernel::MulCalculate(BCalcInfo &calcInfo, const int32_t &rank) {
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input0(static_cast<T *>(calcInfo.input_0->GetData()),
                                                       calcInfo.input_0->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input1(static_cast<T *>(calcInfo.input_1->GetData()),
                                                       calcInfo.input_1->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> output(static_cast<T *>(calcInfo.output->GetData()),
                                                       calcInfo.output->GetTensorShape()->NumElements());
  auto input_shape_0 = calcInfo.input_0->GetTensorShape()->GetDimSizes();
  auto input_shape_1 = calcInfo.input_1->GetTensorShape()->GetDimSizes();
  if (input_shape_0.empty()) {
    T v0 = *(reinterpret_cast<const T *>(calcInfo.input_0->GetData()));
    output = v0 * input1;
    return KERNEL_STATUS_OK;
  }

  if (input_shape_1.empty()) {
    T v1 = *(reinterpret_cast<const T *>(calcInfo.input_1->GetData()));
    output = input0 * v1;
    return KERNEL_STATUS_OK;
  }

  Eigen::DSizes<Eigen::DenseIndex, kMaxDim> reshape_0;
  Eigen::DSizes<Eigen::DenseIndex, kMaxDim> reshape_1;
  Eigen::DSizes<Eigen::DenseIndex, kMaxDim> shape_out;
  Eigen::array<Eigen::DenseIndex, kMaxDim> bcast_0;
  Eigen::array<Eigen::DenseIndex, kMaxDim> bcast_1;

  for (int32_t i = 0; i < kMaxDim; i++) {
    if (i < rank) {
      reshape_0[kMaxIndex - i] = calcInfo.reshape_0[i];
      reshape_1[kMaxIndex - i] = calcInfo.reshape_1[i];
      shape_out[kMaxIndex - i] = calcInfo.shape_out[i];
      bcast_0[kMaxIndex - i] = calcInfo.bcast_0[i];
      bcast_1[kMaxIndex - i] = calcInfo.bcast_1[i];
    } else {
      reshape_0[kMaxIndex - i] = 1;
      reshape_1[kMaxIndex - i] = 1;
      shape_out[kMaxIndex - i] = 1;
      bcast_0[kMaxIndex - i] = 1;
      bcast_1[kMaxIndex - i] = 1;
    }
  }
  output.reshape(shape_out) =
    input0.reshape(reshape_0).broadcast(bcast_0) * input1.reshape(reshape_1).broadcast(bcast_1);
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kMul, MulCpuKernel);
}  // namespace aicpu
