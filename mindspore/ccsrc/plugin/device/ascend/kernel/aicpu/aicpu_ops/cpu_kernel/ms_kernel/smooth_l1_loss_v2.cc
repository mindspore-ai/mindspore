/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

#include "cpu_kernel/ms_kernel/smooth_l1_loss_v2.h"

#include <mutex>
#include <algorithm>

#include "Eigen/Core"
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "common/kernel_log.h"
#include "utils/kernel_util.h"

namespace {
const char *SmoothL1LossV2 = "SmoothL1LossV2";
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
constexpr int64_t kParallelDataNums = 16 * 1024;
const float opHalf = 0.5;
float sigma = 1.0;
std::mutex mtx;

#define COMPUTE_CASE(DTYPE, REDUCTION, TYPE, CTX)                  \
  case (DTYPE): {                                                  \
    KERNEL_LOG_DEBUG("Compute [%s]", DTypeStr(data_type).c_str()); \
    uint32_t result = KERNEL_STATUS_PARAM_INVALID;                 \
    if ((REDUCTION) == "mean") {                                   \
      result = ComputeMean<TYPE>(CTX);                             \
    } else if ((REDUCTION) == "sum") {                             \
      result = ComputeSum<TYPE>(CTX);                              \
    } else if ((REDUCTION) == "none") {                            \
      result = ComputeNone<TYPE>(CTX);                             \
    }                                                              \
    if (result != KERNEL_STATUS_OK) {                              \
      KERNEL_LOG_ERROR("SmoothL1LossV2 compute failed.");          \
      return result;                                               \
    }                                                              \
    break;                                                         \
  }
}  // namespace

namespace aicpu {
uint32_t SmoothL1LossV2CpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check SmoothL1LossV2 params failed.");
  KERNEL_HANDLE_ERROR(ParamCheck(ctx), "Check SmoothL1LossV2 params failed.");

  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    COMPUTE_CASE(DT_FLOAT16, reduction, Eigen::half, ctx)
    COMPUTE_CASE(DT_FLOAT, reduction, float, ctx)
    COMPUTE_CASE(DT_DOUBLE, reduction, double, ctx)
    default:
      KERNEL_LOG_ERROR("SmoothL1LossV2 data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t SmoothL1LossV2CpuKernel::ParamCheck(const CpuKernelContext &ctx) {
  Tensor *input_0 = ctx.Input(0);
  Tensor *input_1 = ctx.Input(1);
  Tensor *output_0 = ctx.Output(0);
  DataType input0_type = input_0->GetDataType();
  DataType input1_type = input_1->GetDataType();
  DataType output0_type = output_0->GetDataType();

  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s] need be same with "
                     "input1 [%s].",
                     DTypeStr(input0_type).c_str(), DTypeStr(input1_type).c_str());
  KERNEL_CHECK_FALSE((input0_type == output0_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s] need be same with "
                     "output0 [%s].",
                     DTypeStr(input0_type).c_str(), DTypeStr(output0_type).c_str());
  auto input0_shape = input_0->GetTensorShape();
  auto input1_shape = input_1->GetTensorShape();
  int32_t input0_dims = input0_shape->GetDims();
  int32_t input1_dims = input1_shape->GetDims();
  KERNEL_CHECK_FALSE((input0_dims == input1_dims), KERNEL_STATUS_PARAM_INVALID,
                     "the input shape dim of input0 [%d] need be same with "
                     "input1 [%d].",
                     input0_dims, input1_dims);
  for (int32_t i = 0; i < input0_dims; i++) {
    KERNEL_CHECK_FALSE((input0_shape->GetDimSize(i) == input1_shape->GetDimSize(i)), KERNEL_STATUS_PARAM_INVALID,
                       "the every input shape dim of input0 [%d] need be same with "
                       "input1 [%d] where dim in [%d].",
                       input0_shape->GetDimSize(i), input1_shape->GetDimSize(i), i);
  }
  KERNEL_LOG_DEBUG(
    "SmoothL1LossV2CpuKernel[%s], input0: size[%llu];"
    "input1: size[%llu], output: size[%llu].",
    ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(), output_0->GetDataSize());

  return AttributeCheck(ctx);
}

uint32_t SmoothL1LossV2CpuKernel::AttributeCheck(const CpuKernelContext &ctx) {
  Tensor *input_0 = ctx.Input(0);
  Tensor *output_0 = ctx.Output(0);
  auto input0_shape = input_0->GetTensorShape();
  auto output0_shape = output_0->GetTensorShape();
  int32_t input0_dims = input0_shape->GetDims();
  int32_t output0_dims = output0_shape->GetDims();

  auto sigma_attr = ctx.GetAttr("sigma");
  auto reduction_attr = ctx.GetAttr("reduction");
  sigma = sigma_attr == nullptr ? 1.0 : sigma_attr->GetFloat();
  reduction = reduction_attr == nullptr ? "mean" : reduction_attr->GetString();
  KERNEL_CHECK_FALSE(sigma >= 0, KERNEL_STATUS_PARAM_INVALID,
                     "the sigma value need to greater than or equal to 0 "
                     "when input sigma value is [%f].",
                     sigma);
  KERNEL_CHECK_FALSE((reduction == "none" || reduction == "mean" || reduction == "sum"), KERNEL_STATUS_PARAM_INVALID,
                     "the reduction value need be the member of ['none','mean','sum'] "
                     "when input reduction value is [%s].",
                     reduction);
  if (reduction == "none") {
    KERNEL_CHECK_FALSE((input0_dims == output0_dims), KERNEL_STATUS_PARAM_INVALID,
                       "the input shape dim of input0 [%d] need be same with "
                       "output0 [%d].",
                       input0_dims, output0_dims);
    for (int32_t i = 0; i < input0_dims; i++) {
      KERNEL_CHECK_FALSE((input0_shape->GetDimSize(i) == output0_shape->GetDimSize(i)), KERNEL_STATUS_PARAM_INVALID,
                         "the every input shape dim of input0 [%d] need be same with "
                         "output0 [%d] where dim in [%d].",
                         input0_shape->GetDimSize(i), output0_shape->GetDimSize(i), i);
    }
  } else if (reduction == "sum" || reduction == "mean") {
    KERNEL_CHECK_FALSE((output0_dims == 0) || ((output0_dims == 1) && (output_0->NumElements() == 1)),
                       KERNEL_STATUS_PARAM_INVALID, "the output shape dim of output0 [%d] need be [1] or a scalar.",
                       output0_dims);
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SmoothL1LossV2CpuKernel::ComputeMean(const CpuKernelContext &ctx) {
  uint32_t compute_sum_res = ComputeSum<T>(ctx);
  if (compute_sum_res != KERNEL_STATUS_OK) {
    return compute_sum_res;
  }
  Tensor *predict_tensor = ctx.Input(0);
  int64_t data_num = predict_tensor->NumElements();
  Tensor *loss_tensor = ctx.Output(0);
  T *loss_val = reinterpret_cast<T *>(loss_tensor->GetData());
  T *res = loss_val;
  if (data_num == 0) {
    *(res) = T(0);
    return KERNEL_STATUS_OK;
  }
  *(res) = *(res) / data_num;

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SmoothL1LossV2CpuKernel::ComputeSum(const CpuKernelContext &ctx) {
  Tensor *predict_tensor = ctx.Input(0);
  Tensor *label_tensor = ctx.Input(1);
  Tensor *loss_tensor = ctx.Output(0);
  T *predict_val = reinterpret_cast<T *>(predict_tensor->GetData());
  T *label_val = reinterpret_cast<T *>(label_tensor->GetData());
  T *loss_val = reinterpret_cast<T *>(loss_tensor->GetData());
  int64_t data_num = predict_tensor->NumElements();
  int64_t data_size = data_num * sizeof(T);

  double res = 0;
  if (data_size <= kParallelDataNums) {
    for (int64_t i = 0; i < data_num; i++) {
      T predict = *(predict_val + i);
      T label = *(label_val + i);
      T z = predict - label > T(0) ? predict - label : label - predict;
      if (sigma == 0) {
        res += static_cast<double>(z);
      } else {
        res += static_cast<double>(z < T(sigma) ? T(opHalf) * z * z / T(sigma) : z - T(opHalf) * T(sigma));
      }
    }
    *(loss_val) = static_cast<T>(res);
    return KERNEL_STATUS_OK;
  } else {
    auto shared_smoothl1lossv2 = [&](size_t start, size_t end) -> double {
      double sum = 0;
      for (size_t i = start; i < end; i++) {
        T predict = *(predict_val + i);
        T label = *(label_val + i);
        T z = predict - label > T(0) ? predict - label : label - predict;
        if (sigma == 0) {
          res += static_cast<double>(z);
        } else {
          sum += static_cast<double>(z < T(sigma) ? T(opHalf) * z * z / T(sigma) : z - T(opHalf) * T(sigma));
        }
      }
      mtx.lock();
      res = res + sum;
      mtx.unlock();
      return KERNEL_STATUS_OK;
    };

    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    auto result = CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shared_smoothl1lossv2);
    *(loss_val) = static_cast<T>(res);
    return result;
  }
}

template <typename T>
uint32_t SmoothL1LossV2CpuKernel::ComputeNone(const CpuKernelContext &ctx) {
  Tensor *predict_tensor = ctx.Input(0);
  Tensor *label_tensor = ctx.Input(1);
  Tensor *loss_tensor = ctx.Output(0);
  T *predict_val = reinterpret_cast<T *>(predict_tensor->GetData());
  T *label_val = reinterpret_cast<T *>(label_tensor->GetData());
  T *loss_val = reinterpret_cast<T *>(loss_tensor->GetData());
  int64_t data_num = predict_tensor->NumElements();

  T *res = loss_val;
  int64_t data_size = data_num * sizeof(T);
  if (data_size <= kParallelDataNums) {
    for (int64_t i = 0; i < data_num; i++) {
      T predict = *(predict_val + i);
      T label = *(label_val + i);
      T z = predict - label > T(0) ? predict - label : label - predict;
      if (sigma == 0) {
        *(res + i) = z;
      } else {
        *(res + i) = z < T(sigma) ? T(opHalf) * z * z / T(sigma) : z - T(opHalf) * T(sigma);
      }
    }
    return KERNEL_STATUS_OK;
  } else {
    auto shared_smoothl1lossv2 = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        T predict = *(predict_val + i);
        T label = *(label_val + i);
        T z = predict - label > T(0) ? predict - label : label - predict;
        if (sigma == 0) {
          *(res + i) = z;
        } else {
          *(res + i) = z < T(sigma) ? T(opHalf) * z * z / T(sigma) : z - T(opHalf) * T(sigma);
        }
      }
    };
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    return CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shared_smoothl1lossv2);
  }
}

REGISTER_CPU_KERNEL(SmoothL1LossV2, SmoothL1LossV2CpuKernel);
}  // namespace aicpu
