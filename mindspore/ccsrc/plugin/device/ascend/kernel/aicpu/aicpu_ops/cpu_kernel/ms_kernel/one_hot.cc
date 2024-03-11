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

/*!
 * \file one_hot.cc
 * \brief
 */
#include "one_hot.h"
#include <string>
#include "context/inc/cpu_kernel_utils.h"
#include "inc/kernel_log.h"
#include "context/common/status.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"
#include "utils/eigen_tensor.h"
#include "utils/sparse_tensor.h"

namespace {
const uint32_t kInputNum = 4;
const uint32_t kOutputNum = 1;
const char *kOneHot = "OneHot";
const int64_t kParallelDataNumSameShape = 100 * 1024;
#define ONE_HOT_INPUT_COMPUTE_CASE(DTYPE, TYPE, ODTYPE, CTX)                             \
  case (DTYPE): {                                                                        \
    switch (ODTYPE) {                                                                    \
      ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, DT_COMPLEX64, std::complex<float>, CTX)   \
      ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, DT_COMPLEX128, std::complex<double>, CTX) \
      ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, DT_DOUBLE, double, CTX)                   \
      ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, DT_FLOAT, float_t, CTX);                  \
      ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, DT_FLOAT16, Eigen::half, CTX)             \
      ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, DT_INT8, int8_t, CTX)                     \
      ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, DT_INT16, int16_t, CTX)                   \
      ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, DT_INT32, int32_t, CTX)                   \
      ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, DT_INT64, int64_t, CTX)                   \
      ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, DT_UINT8, uint8_t, CTX)                   \
      ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, DT_UINT16, uint16_t, CTX)                 \
      ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, DT_UINT32, uint32_t, CTX)                 \
      ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, DT_UINT64, uint64_t, CTX)                 \
      ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, DT_BOOL, bool, CTX)                       \
      ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, DT_STRING, std::string, CTX)              \
      default:                                                                           \
        CUST_KERNEL_LOG_ERROR(ctx, "OneHot kernel output data type [%s] not support.",   \
                              DTypeStr(output_data_type).c_str());                       \
        return KERNEL_STATUS_PARAM_INVALID;                                              \
    }                                                                                    \
    break;                                                                               \
  }

#define ONE_HOT_OUTPUT_COMPUTE_CASE(DTYPE, TYPE, ODTYPE, OTYPE, CTX) \
  case (ODTYPE): {                                                   \
    uint32_t result = OneHotCompute<OTYPE, TYPE>(CTX);               \
    if (result != KERNEL_STATUS_OK) {                                \
      CUST_KERNEL_LOG_ERROR(ctx, "OneHot kernel compute failed.");   \
      return result;                                                 \
    }                                                                \
    break;                                                           \
  }
}  // namespace

namespace aicpu {
uint32_t OneHotCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "OneHot check input and output number failed.");
  auto input_data_type = ctx.Input(0)->GetDataType();
  auto output_data_type = ctx.Output(0)->GetDataType();
  switch (input_data_type) {
    ONE_HOT_INPUT_COMPUTE_CASE(DT_UINT8, uint8_t, output_data_type, ctx);
    ONE_HOT_INPUT_COMPUTE_CASE(DT_INT32, int32_t, output_data_type, ctx);
    ONE_HOT_INPUT_COMPUTE_CASE(DT_INT64, int64_t, output_data_type, ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "OneHot kernel input data type [%s] not support.", DTypeStr(input_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename TI>
uint32_t OneHotCpuKernel::OneHotCompute(CpuKernelContext &ctx) {
  // 输入张量
  Tensor *indices = ctx.Input(0);
  // 输出张量
  Tensor *output = ctx.Output(0);
  // 输入张量数据
  auto indices_data = reinterpret_cast<TI *>(indices->GetData());
  // 输出张量数据
  auto output_data = reinterpret_cast<T *>(output->GetData());
  // depth值
  auto depth = reinterpret_cast<int32_t *>(ctx.Input(1)->GetData());
  // on_value值
  auto on_value = reinterpret_cast<T *>(ctx.Input(2)->GetData());
  // off_value值
  auto off_value = reinterpret_cast<T *>(ctx.Input(3)->GetData());
  // 输入张量形状
  auto indices_shape = indices->GetTensorShape();
  // axis值
  int64_t axis = ctx.GetAttr("axis") == nullptr ? -1 : ctx.GetAttr("axis")->GetInt();
  if (axis == -1) {
    axis = indices_shape->GetDims();
  }
  // 输出张量形状
  auto output_shape = output->GetTensorShape();
  // 对输出张量用off_value进行初始化匿名函数
  auto init_output_func = [&](int64_t start, int64_t end) -> void {
    for (int i = start; i < end; ++i) {
      *(output_data + i) = *(off_value);
    }
  };
  // 计算axis前维度大小
  int64_t prefix_dim_size = 1;
  for (int i = 0; i < axis; ++i) {
    prefix_dim_size *= indices_shape->GetDimSize(i);
  }
  // 计算计算axis后维度大小
  int64_t suffix_dim_size = indices_shape->NumElements() / prefix_dim_size;
  // 输入张量元素总个数
  int64_t data_num = indices_shape->NumElements();
  // depth_value为depth的具体值
  int32_t depth_value = *(depth);
  // 将输出张量的维度看做{prefix_dim_size，depth, suffix_dim_size}
  // 通过offset = suffix_dim_size == 1?(d0 * depth_value + d1):(d0 * prefix_dim_size * depth_value + d1 *
  // suffix_dim_size  + d2)来计算出独热张量有效值的位置 然后对输出张量的该位置赋值为on_value
  const auto get_output_func = [&](int64_t start, int64_t end) -> void {
    for (int64_t i = start; i < end; ++i) {
      int64_t d0 = i / suffix_dim_size;
      int64_t d1 = i - (d0 * suffix_dim_size);
      int64_t depth_v = SubtleMustCopy<int64_t>(*(indices_data + d0 * suffix_dim_size + d1));
      if (depth_v < static_cast<int64_t>(depth_value) && depth_v >= 0) {
        int64_t offset = suffix_dim_size == 1 ? i * depth_value + depth_v
                                              : d0 * depth_value * suffix_dim_size + depth_v * suffix_dim_size + d1;
        *(output_data + offset) = *(on_value);
      }
    }
  };
  // 使用CpuKernelUtils::GetCPUNum接口获取AI CPU的核数
  uint32_t max_core_num = std::max(1U, aicpu::CpuKernelUtils::GetCPUNum(ctx));
  // 多线程执行状态
  bool run_state = true;
  // 对于数据量小于100K的场景则只单核运行，否则使用实际的AI CPU总核数进行计算
  if (data_num >= kParallelDataNumSameShape) {
    max_core_num = (max_core_num > data_num) ? data_num : max_core_num;
    max_core_num = max_core_num == 0 ? 1 : max_core_num;
    uint32_t ret1 = CpuKernelUtils::ParallelFor(ctx, output_shape->NumElements(),
                                                (output_shape->NumElements() / max_core_num), init_output_func);
    uint32_t ret2 = CpuKernelUtils::ParallelFor(ctx, data_num, (data_num / max_core_num), get_output_func);
    run_state = (ret1 == KERNEL_STATUS_OK) && (ret2 == KERNEL_STATUS_OK);
  } else {
    // 输入数据大小没有100k，单核调用
    init_output_func(0, output_shape->NumElements());
    get_output_func(0, data_num);
  }
  return run_state ? KERNEL_STATUS_OK : KERNEL_STATUS_INNER_ERROR;
}

REGISTER_MS_CPU_KERNEL(kOneHot, OneHotCpuKernel);
}  // namespace aicpu
