/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "masked_select_grad.h"

#include "Eigen/Core"
#include "securec.h"
#include "cpu_types.h"
#include "kernel_log.h"
#include "status.h"
#include "utils/broadcast_iterator.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kMaskedSelectGradInputNum = 3;
constexpr uint32_t kMaskedSelectGradOutputNum = 1;
const char *const kMaskedSelectGrad = "MaskedSelectGrad";
}  // namespace

namespace aicpu {
uint32_t MaskedSelectGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kMaskedSelectGradInputNum, kMaskedSelectGradOutputNum),
                      "[%s] check params failed.", kMaskedSelectGrad);

  // choose compute function depend on dataType
  auto data_type0 = static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  auto data_type1 = static_cast<DataType>(ctx.Input(kSecondInputIndex)->GetDataType());
  auto data_type2 = static_cast<DataType>(ctx.Input(2)->GetDataType());
  if (data_type1 != DT_BOOL) {
    KERNEL_LOG_ERROR("[%s] Data type of mask requires bool, but got data type [%s].", ctx.GetOpType().c_str(),
                     DTypeStr(data_type1).c_str());
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  if (data_type0 != data_type2) {
    KERNEL_LOG_ERROR("[%s] Data type of x and y requires same, but got data type [%s] and [%s].",
                     ctx.GetOpType().c_str(), DTypeStr(data_type0).c_str(), DTypeStr(data_type2).c_str());
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  switch (data_type0) {
    case DT_FLOAT16:
      return MaskedSelectGradCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return MaskedSelectGradCompute<float>(ctx);
    case DT_DOUBLE:
      return MaskedSelectGradCompute<double>(ctx);
    case DT_INT8:
      return MaskedSelectGradCompute<int8_t>(ctx);
    case DT_INT16:
      return MaskedSelectGradCompute<int16_t>(ctx);
    case DT_INT32:
      return MaskedSelectGradCompute<int32_t>(ctx);
    case DT_INT64:
      return MaskedSelectGradCompute<int64_t>(ctx);
    case DT_UINT8:
      return MaskedSelectGradCompute<uint8_t>(ctx);
    case DT_UINT16:
      return MaskedSelectGradCompute<uint16_t>(ctx);
    case DT_UINT32:
      return MaskedSelectGradCompute<uint32_t>(ctx);
    case DT_UINT64:
      return MaskedSelectGradCompute<uint64_t>(ctx);
    case DT_BOOL:
      return MaskedSelectGradCompute<bool>(ctx);
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].", ctx.GetOpType().c_str(),
                       DTypeStr(data_type0).c_str());
      return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
}

template <typename T>
uint32_t MaskedSelectGradCpuKernel::MaskedSelectGradCompute(CpuKernelContext &ctx) {
  bool *mask = reinterpret_cast<bool *>(ctx.Input(1)->GetData());
  KERNEL_CHECK_NULLPTR(mask, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "[%s] get input_data[1] failed.",
                       kMaskedSelectGrad);
  T *grad = reinterpret_cast<T *>(ctx.Input(2)->GetData());
  KERNEL_CHECK_NULLPTR(grad, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "[%s] get input_data[2] failed.",
                       kMaskedSelectGrad);
  T *dx = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  KERNEL_CHECK_NULLPTR(dx, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "[%s] get output_data[0] failed.",
                       kMaskedSelectGrad);

  auto input_shape_a = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  auto input_shape_b = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> output_shape;
  auto ret = GetBroadcastShape(input_shape_a, input_shape_b, output_shape);
  KERNEL_CHECK_FALSE(ret == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID, "Shape of x and mask can't be broadcast.");
  uint64_t tensor_size = 1;
  for (const int64_t &d : output_shape) {
    tensor_size *= static_cast<uint64_t>(d);
  }
  const T NUM_ZERO = static_cast<T>(0);
  for (uint64_t k = 0; k < tensor_size; ++k) {
    dx[k] = NUM_ZERO;
  }

  uint64_t j = 0;
  if (input_shape_a == input_shape_b) {
    for (uint64_t l = 0; l < tensor_size; ++l) {
      if (mask[l]) {
        dx[l] += grad[j++];
      }
    }
  } else {
    BroadcastIterator iter(input_shape_a, input_shape_b, output_shape);
    iter.SetPos(0);
    for (uint64_t i = 0; i < tensor_size; ++i) {
      if (mask[iter.GetInputPosB()]) {
        dx[iter.GetInputPosA()] += grad[j++];
      }
      iter.GenNextPos();
    }
  }

  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}
REGISTER_CPU_KERNEL(kMaskedSelectGrad, MaskedSelectGradCpuKernel);
}  // namespace aicpu
