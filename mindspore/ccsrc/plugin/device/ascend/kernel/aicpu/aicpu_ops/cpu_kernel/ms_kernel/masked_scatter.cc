/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "cpu_kernel/ms_kernel/masked_scatter.h"

#include <vector>

#include "Eigen/Core"
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "cpu_kernel/inc/cpu_types.h"
#include "cpu_kernel/ms_kernel/log.h"
#include "securec/include/securec.h"
#include "cpu_kernel/common/status.h"
#include "utils/broadcast_iterator.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kMaskedScatterInputNum = 3;
constexpr uint32_t kMaskedScatterOutputNum = 1;
const char *kMaskedScatter = "MaskedScatter";
}  // namespace

namespace aicpu {
uint32_t MaskedScatterCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kMaskedScatterInputNum, kMaskedScatterOutputNum), "[%s] check params failed.",
                      kMaskedScatter);
  KERNEL_HANDLE_ERROR(InputCheck(ctx), "[%s] check input failed.", kMaskedScatter);

  // choose compute function depend on dataType
  auto data_type0 = static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  auto data_type1 = static_cast<DataType>(ctx.Input(kSecondInputIndex)->GetDataType());
  auto data_type2 = static_cast<DataType>(ctx.Input(kThirdInputIndex)->GetDataType());
  auto data_type3 = static_cast<DataType>(ctx.Output(kFirstOutputIndex)->GetDataType());
  if (data_type1 != DT_BOOL) {
    KERNEL_LOG_ERROR("[%s] Data type of mask requires bool, but got data type [%s].", ctx.GetOpType().c_str(),
                     DTypeStr(data_type1).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (data_type0 != data_type2) {
    KERNEL_LOG_ERROR(
      "[%s] Data type of x and updates requires same, but got data type [%s] "
      "and [%s].",
      ctx.GetOpType().c_str(), DTypeStr(data_type0).c_str(), DTypeStr(data_type2).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (data_type0 != data_type3) {
    KERNEL_LOG_ERROR(
      "[%s] Data type of x and y requires same, but got data type [%s] and "
      "[%s].",
      ctx.GetOpType().c_str(), DTypeStr(data_type0).c_str(), DTypeStr(data_type3).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  switch (data_type0) {
    case DT_FLOAT16:
      return MaskedScatterCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return MaskedScatterCompute<float>(ctx);
    case DT_DOUBLE:
      return MaskedScatterCompute<double>(ctx);
    case DT_INT8:
      return MaskedScatterCompute<int8_t>(ctx);
    case DT_INT16:
      return MaskedScatterCompute<int16_t>(ctx);
    case DT_INT32:
      return MaskedScatterCompute<int32_t>(ctx);
    case DT_INT64:
      return MaskedScatterCompute<int64_t>(ctx);
    case DT_UINT8:
      return MaskedScatterCompute<uint8_t>(ctx);
    case DT_BOOL:
      return MaskedScatterCompute<bool>(ctx);
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].", ctx.GetOpType().c_str(),
                       DTypeStr(data_type0).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t MaskedScatterCpuKernel::InputCheck(const CpuKernelContext &ctx) {
  auto input_shape_a = ctx.Input(0)->GetTensorShape();
  auto input_shape_b = ctx.Input(1)->GetTensorShape();

  std::vector<int64_t> input_dims_a = input_shape_a->GetDimSizes();
  std::vector<int64_t> input_dims_b = input_shape_b->GetDimSizes();

  std::vector<int64_t> input_dims_a_inverse = input_dims_a;
  std::vector<int64_t> input_dims_b_inverse = input_dims_b;

  int64_t input_Totaldim_a = input_shape_a->GetDims();
  int64_t input_Totaldim_b = input_shape_b->GetDims();

  for (int64_t i = 0; i < input_Totaldim_a; ++i) {
    input_dims_a_inverse[i] = input_dims_a[input_Totaldim_a - i - 1];
  }

  for (int64_t i = 0; i < input_Totaldim_b; ++i) {
    input_dims_b_inverse[i] = input_dims_b[input_Totaldim_b - i - 1];
  }

  int64_t mask_numElements = input_shape_b->NumElements();

  KERNEL_CHECK_FALSE(mask_numElements > 0, KERNEL_STATUS_PARAM_INVALID, "Number of elements of mask must > 0.");
  KERNEL_CHECK_FALSE((input_Totaldim_a >= input_Totaldim_b), KERNEL_STATUS_PARAM_INVALID,
                     "expand: the number of sizes provided (%d) must be greater"
                     " or equal to the number of dimensions in the tensor (%d).",
                     input_Totaldim_a, input_Totaldim_b);

  for (int64_t i = 0; i < input_Totaldim_b; ++i) {
    KERNEL_CHECK_FALSE((input_dims_b_inverse[i] == 1 || input_dims_b_inverse[i] == input_dims_a_inverse[i]),
                       KERNEL_STATUS_PARAM_INVALID,
                       "The expanded size of the tensor (%d) must match the "
                       "existing  size (%d) at non-singleton dimension 1.",
                       input_dims_a_inverse[i], input_dims_b_inverse[i]);
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MaskedScatterCpuKernel::MaskedScatterCompute(const CpuKernelContext &ctx) {
  T *x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  bool *mask = reinterpret_cast<bool *>(ctx.Input(1)->GetData());
  T *updates = reinterpret_cast<T *>(ctx.Input(2)->GetData());
  T *y = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  auto x_shape_ptr = ctx.Input(0)->GetTensorShape();
  auto mask_shape_ptr = ctx.Input(1)->GetTensorShape();
  auto update_shape_ptr = ctx.Input(2)->GetTensorShape();

  std::vector<int64_t> x_shape = x_shape_ptr->GetDimSizes();
  std::vector<int64_t> mask_shape = mask_shape_ptr->GetDimSizes();

  int64_t x_numElements = x_shape_ptr->NumElements();
  int64_t updates_numElements = update_shape_ptr->NumElements();

  if (x_shape == mask_shape) {
    int64_t j = 0;
    for (int64_t i = 0; i < x_numElements; i++) {
      if (mask[i]) {
        KERNEL_CHECK_FALSE(j < updates_numElements, KERNEL_STATUS_PARAM_INVALID,
                           "Number of elements of updates < number of ones in mask.");
        y[i] = updates[j], j += 1;
      } else {
        y[i] = x[i];
      }
    }
  } else {
    std::vector<int64_t> output_dims = x_shape;
    BroadcastIterator iter(x_shape, mask_shape, output_dims);
    iter.SetPos(0);
    int64_t j = 0;
    for (int64_t i = 0; i < x_numElements; i++, iter.GenNextPos()) {
      if (mask[iter.GetInputPosB()]) {
        KERNEL_CHECK_FALSE(j < updates_numElements, KERNEL_STATUS_PARAM_INVALID,
                           "Number of elements of updates < number of ones in mask.");
        y[iter.GetInputPosA()] = updates[j], j += 1;
      } else {
        y[i] = x[i];
      }
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kMaskedScatter, MaskedScatterCpuKernel);
}  // namespace aicpu
