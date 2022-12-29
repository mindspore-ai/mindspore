/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
#ifndef AICPU_UTILS_EQUAL_UTIL_H
#define AICPU_UTILS_EQUAL_UTIL_H

#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
/**
 * @brief Parameter verification
 * @param flag equal or not equal
 * @return status code
 */
template <typename T>
uint32_t EqualCalculate(const CpuKernelContext &ctx, BCalcInfo &calcInfo, bool flag) {
  auto input_x1 = reinterpret_cast<T *>(calcInfo.input_0->GetData());
  auto input_x2 = reinterpret_cast<T *>(calcInfo.input_1->GetData());
  auto output_y = reinterpret_cast<bool *>(calcInfo.output->GetData());
  KERNEL_CHECK_NULLPTR(input_x1, KERNEL_STATUS_PARAM_INVALID, "Get input x1 data failed.")
  KERNEL_CHECK_NULLPTR(input_x2, KERNEL_STATUS_PARAM_INVALID, "Get input x2 data failed.")
  KERNEL_CHECK_NULLPTR(output_y, KERNEL_STATUS_PARAM_INVALID, "Get output data failed.")
  size_t data_num = calcInfo.x_indexes.size();
  auto shard_equal = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto x_index = input_x1 + calcInfo.x_indexes[i];
      auto y_index = input_x2 + calcInfo.y_indexes[i];
      output_y[i] = (flag == true) ? (*x_index == *y_index) : (*x_index != *y_index);
    }
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, 1, shard_equal), "Equal calculate failed.");
  return KERNEL_STATUS_OK;
}
/**
 * @brief Parameter verification
 * @param ctx op context
 * @param flag equal or not equal
 * @return status code
 */
template <typename T>
uint32_t EqualCompute(const CpuKernelContext &ctx, bool flag) {
  BCalcInfo calcInfo;
  calcInfo.input_0 = ctx.Input(0);
  calcInfo.input_1 = ctx.Input(1);
  calcInfo.output = ctx.Output(0);
  DataType input0_type = calcInfo.input_0->GetDataType();
  DataType input1_type = calcInfo.input_1->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "DataType of x1 [%d] should be same as x2 [%d].", static_cast<int32_t>(input0_type),
                     static_cast<int32_t>(input1_type))
  KERNEL_LOG_INFO(
    "CpuKernel[%s], input x1 : addr[%p], size[%llu];"
    "input x2: addr[%p], size[%llu];"
    "output: addr[%p], size[%llu].",
    ctx.GetOpType().c_str(), calcInfo.input_0->GetData(), calcInfo.input_0->GetDataSize(), calcInfo.input_1->GetData(),
    calcInfo.input_1->GetDataSize(), calcInfo.output->GetData(), calcInfo.output->GetDataSize());

  Bcast bcast;
  KERNEL_HANDLE_ERROR(bcast.GenerateBcastInfo(calcInfo), "Generate broadcast info failed.");
  bcast.BCastIndexes(calcInfo.x_indexes, calcInfo.y_indexes);
  bcast.GetBcastVec(calcInfo);

  return EqualCalculate<T>(ctx, calcInfo, flag);
}
}  // namespace aicpu
#endif
