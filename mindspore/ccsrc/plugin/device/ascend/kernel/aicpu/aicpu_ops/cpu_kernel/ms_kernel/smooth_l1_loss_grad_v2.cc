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

#include "cpu_kernel/ms_kernel/smooth_l1_loss_grad_v2.h"

#include <algorithm>
#include <mutex>

#include "Eigen/Core"
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "common/kernel_log.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
const char *kSmoothL1LossGradV2 = "SmoothL1LossGradV2";
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;
float sigma = 1.0;
std::mutex mtx;

#define SmoothL1LossGradV2_COMPUTE_CASE(DTYPE, REDUCTION, TYPE, CTX) \
  case (DTYPE): {                                                    \
    KERNEL_LOG_INFO("Compute [%s]", DTypeStr(data_type).c_str());    \
    uint32_t result = KERNEL_STATUS_PARAM_INVALID;                   \
    if ((REDUCTION) == "mean") {                                     \
      result = ComputeMean<TYPE>(CTX);                               \
    } else if ((REDUCTION) == "sum") {                               \
      result = ComputeSum<TYPE>(CTX);                                \
    } else if ((REDUCTION) == "none") {                              \
      result = ComputeNone<TYPE>(CTX);                               \
    }                                                                \
    if (result != KERNEL_STATUS_OK) {                                \
      KERNEL_LOG_ERROR("SmoothL1LossGradV2 kernel compute failed."); \
      return result;                                                 \
    }                                                                \
    break;                                                           \
  }
}  // namespace

namespace aicpu {
uint32_t SmoothL1LossGradV2CpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "SmoothL1LossGradV2 check input and output number failed.");
  KERNEL_HANDLE_ERROR(ParamCheck(ctx), "SmoothL1LossGradV2 check params failed.");

  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    SmoothL1LossGradV2_COMPUTE_CASE(DT_FLOAT16, reduction, Eigen::half, ctx)
      SmoothL1LossGradV2_COMPUTE_CASE(DT_FLOAT, reduction, float, ctx)
        SmoothL1LossGradV2_COMPUTE_CASE(DT_DOUBLE, reduction, double, ctx) default
        : KERNEL_LOG_ERROR("SmoothL1LossGradV2 kernel data type [%s] not support.", DTypeStr(data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t SmoothL1LossGradV2CpuKernel::ParamCheck(const CpuKernelContext &ctx) {
  Tensor *predict_tensor = ctx.Input(0);
  Tensor *label_tensor = ctx.Input(1);
  Tensor *dout_tensor = ctx.Input(2);
  Tensor *gradient_tensor = ctx.Output(0);
  DataType predict_type = predict_tensor->GetDataType();
  DataType label_type = label_tensor->GetDataType();
  DataType dout_type = dout_tensor->GetDataType();
  DataType gradient_type = gradient_tensor->GetDataType();
  KERNEL_CHECK_FALSE((predict_type == label_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of predict [%s] need be same with "
                     "label [%s].",
                     DTypeStr(predict_type).c_str(), DTypeStr(label_type).c_str());
  KERNEL_CHECK_FALSE((predict_type == dout_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of predict [%s] need be same with "
                     "dout [%s].",
                     DTypeStr(predict_type).c_str(), DTypeStr(dout_type).c_str());
  KERNEL_CHECK_FALSE((predict_type == gradient_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of predict [%s] need be same with "
                     "gradient [%s].",
                     DTypeStr(predict_type).c_str(), DTypeStr(gradient_type).c_str());
  auto predict_shape = predict_tensor->GetTensorShape();
  auto label_shape = label_tensor->GetTensorShape();
  auto gradient_shape = gradient_tensor->GetTensorShape();
  int32_t predict_dims = predict_shape->GetDims();
  int32_t label_dims = label_shape->GetDims();
  int32_t gradient_dims = gradient_shape->GetDims();
  KERNEL_CHECK_FALSE((predict_dims == label_dims), KERNEL_STATUS_PARAM_INVALID,
                     "the input shape dim of predict [%d] need be same with "
                     "label [%d].",
                     predict_dims, label_dims);
  KERNEL_CHECK_FALSE((predict_dims == gradient_dims), KERNEL_STATUS_PARAM_INVALID,
                     "the input shape dim of predict [%d] need be same with "
                     "gradient [%d].",
                     predict_dims, gradient_dims);
  for (int32_t i = 0; i < predict_dims; i++) {
    KERNEL_CHECK_FALSE((predict_shape->GetDimSize(i) == label_shape->GetDimSize(i)), KERNEL_STATUS_PARAM_INVALID,
                       "the every input shape dim of predict [%d] need be same with "
                       "label [%d] where dim in [%d].",
                       predict_shape->GetDimSize(i), label_shape->GetDimSize(i), i);
    KERNEL_CHECK_FALSE((predict_shape->GetDimSize(i) == gradient_shape->GetDimSize(i)), KERNEL_STATUS_PARAM_INVALID,
                       "the every input shape dim of predict [%d] need be same with "
                       "gradient [%d] where dim in [%d].",
                       predict_shape->GetDimSize(i), gradient_shape->GetDimSize(i), i);
  }
  KERNEL_LOG_DEBUG(
    "SmoothL1LossGradV2CpuKernel[%s], predict: size[%llu];"
    "label: size[%llu], dout: size[%llu], gradient: size[%llu].",
    ctx.GetOpType().c_str(), predict_tensor->GetDataSize(), label_tensor->GetDataSize(), dout_tensor->GetDataSize(),
    gradient_tensor->GetDataSize());
  return AttributesCheck(ctx);
}

uint32_t SmoothL1LossGradV2CpuKernel::AttributesCheck(const CpuKernelContext &ctx) {
  Tensor *predict_tensor = ctx.Input(0);
  Tensor *dout_tensor = ctx.Input(2);
  Tensor *gradient_tensor = ctx.Output(0);
  auto predict_shape = predict_tensor->GetTensorShape();
  auto dout_shape = dout_tensor->GetTensorShape();
  auto gradient_shape = gradient_tensor->GetTensorShape();
  int32_t predict_dims = predict_shape->GetDims();
  int32_t dout_dims = dout_shape->GetDims();
  int32_t gradient_dims = gradient_shape->GetDims();
  auto sigma_attr = ctx.GetAttr("sigma");
  auto reduction_attr = ctx.GetAttr("reduction");
  sigma = sigma_attr == nullptr ? 1.0 : sigma_attr->GetFloat();
  reduction = reduction_attr == nullptr ? "mean" : reduction_attr->GetString();
  KERNEL_CHECK_FALSE(sigma >= 0, KERNEL_STATUS_PARAM_INVALID,
                     "the sigma value must greater than or equal to 0 "
                     "when value of input sigma is [%f].",
                     sigma);
  KERNEL_CHECK_FALSE((reduction == "none" || reduction == "mean" || reduction == "sum"), KERNEL_STATUS_PARAM_INVALID,
                     "the reduction value must be a value in a range of ['none','mean','sum'].", reduction);
  if (reduction == "none" || reduction == "mean" || reduction == "sum") {
    KERNEL_CHECK_FALSE((predict_dims == gradient_dims), KERNEL_STATUS_PARAM_INVALID,
                       "the input shape dim of predict [%d] need be same with "
                       "gradient [%d].",
                       predict_dims, gradient_dims);
    for (int32_t i = 0; i < predict_dims; i++) {
      KERNEL_CHECK_FALSE((predict_shape->GetDimSize(i) == gradient_shape->GetDimSize(i)), KERNEL_STATUS_PARAM_INVALID,
                         "the input shape dim of predict [%d] must be same with "
                         "gradient [%d] where dim in [%d].",
                         predict_shape->GetDimSize(i), gradient_shape->GetDimSize(i), i);
    }
  }
  if (reduction == "none") {
    KERNEL_CHECK_FALSE((predict_dims == dout_dims), KERNEL_STATUS_PARAM_INVALID,
                       "the input shape dim of predict [%d] need be same with "
                       "dout [%d].",
                       predict_dims, dout_dims);
    for (int32_t i = 0; i < predict_dims; i++) {
      KERNEL_CHECK_FALSE((predict_shape->GetDimSize(i) == dout_shape->GetDimSize(i)), KERNEL_STATUS_PARAM_INVALID,
                         "the every input shape dim of predict [%d] need be same with "
                         "dout [%d] where dim in [%d].",
                         predict_shape->GetDimSize(i), dout_shape->GetDimSize(i), i);
    }
  } else if (reduction == "sum" || reduction == "mean") {
    KERNEL_CHECK_FALSE((dout_dims == 0) || ((dout_dims == 1) && (dout_tensor->NumElements() == 1)),
                       KERNEL_STATUS_PARAM_INVALID, "the dout shape dim of dout [%d] need be a scalar.", dout_dims);
  }
  return KERNEL_STATUS_OK;
}

// 1 * dout         if  x  >= sigma
// -1 * dout        if  x  <= -sigma
// x / sigma * dout if |x| <  sigma
template <typename T>
uint32_t SmoothL1LossGradV2CpuKernel::ComputeSum(const CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("SmoothL1LossGradV2CpuKernel::ComputeSum start");
  Tensor *predict_tensor = ctx.Input(0);
  Tensor *label_tensor = ctx.Input(1);
  Tensor *dout_tensor = ctx.Input(2);
  Tensor *gradient_tensor = ctx.Output(0);
  T *predict_val = static_cast<T *>(predict_tensor->GetData());
  T *label_val = static_cast<T *>(label_tensor->GetData());
  T *dout_val = static_cast<T *>(dout_tensor->GetData());
  T *gradient_val = static_cast<T *>(gradient_tensor->GetData());
  int64_t data_num = predict_tensor->NumElements();
  int64_t data_size = data_num * sizeof(T);
  T *result = gradient_val;
  if (data_size <= kParallelDataNum) {
    for (int64_t i = 0; i < data_num; i++) {
      T predict = *(predict_val + i);
      T label = *(label_val + i);
      T dout = *dout_val;
      T x = predict - label;
      if (x == T(0)) {
        *(result + i) = T(0) * dout;
      } else if (x <= -T(sigma)) {
        *(result + i) = T(-1) * dout;
      } else if (x >= T(sigma)) {
        *(result + i) = T(1) * dout;
      } else if (sigma == 0) {
        KERNEL_LOG_ERROR("attribute sigma could not be 0.");
      } else {
        *(result + i) = x / T(sigma) * dout;
      }
    }
    return KERNEL_STATUS_OK;
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num cannot be 0.");
    }
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shared_smoothl1lossgradv2 = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        T predict = *(predict_val + i);
        T label = *(label_val + i);
        T dout = *dout_val;
        T x = predict - label;
        if (x == T(0)) {
          *(result + i) = T(0) * dout;
        } else if (x <= -T(sigma)) {
          *(result + i) = T(-1) * dout;
        } else if (x >= T(sigma)) {
          *(result + i) = T(1) * dout;
        } else if (sigma == 0) {
          KERNEL_LOG_ERROR("attribute sigma could not be 0.");
        } else {
          *(result + i) = x / T(sigma) * dout;
        }
      }
    };
    return CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shared_smoothl1lossgradv2);
  }
  KERNEL_LOG_INFO("SmoothL1LossGradV2CpuKernel::ComputeSum end");
}

// Mean's result is Sum's result divided by the total number of elements per
// element
template <typename T>
uint32_t SmoothL1LossGradV2CpuKernel::ComputeMean(const CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("SmoothL1LossGradV2CpuKernel::ComputeMean start");
  Tensor *predict_tensor = ctx.Input(0);
  Tensor *label_tensor = ctx.Input(1);
  Tensor *dout_tensor = ctx.Input(2);
  Tensor *gradient_tensor = ctx.Output(0);
  T *predict_val = static_cast<T *>(predict_tensor->GetData());
  T *label_val = static_cast<T *>(label_tensor->GetData());
  T *dout_val = static_cast<T *>(dout_tensor->GetData());
  T *gradient_val = static_cast<T *>(gradient_tensor->GetData());
  int64_t data_num = predict_tensor->NumElements();
  if (data_num == 0) {
    KERNEL_LOG_ERROR("data_num cannot be 0.");
  }
  int64_t data_size = data_num * sizeof(T);
  T *result = gradient_val;
  if (data_size <= kParallelDataNum) {
    for (int64_t i = 0; i < data_num; i++) {
      T predict = *(predict_val + i);
      T label = *(label_val + i);
      T dout = *dout_val;
      T x = predict - label;
      if (x == T(0)) {
        *(result + i) = T(0) * dout;
      } else if (x <= -T(sigma)) {
        *(result + i) = T(-1) / data_num * dout;
      } else if (x >= T(sigma)) {
        *(result + i) = T(1) / data_num * dout;
      } else if (sigma == 0) {
        KERNEL_LOG_ERROR("attribute sigma could not be 0.");
      } else {
        *(result + i) = x / T(sigma) / data_num * dout;
      }
    }
    return KERNEL_STATUS_OK;
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num cannot be 0.");
    }
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shared_smoothl1lossgradv2 = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        T predict = *(predict_val + i);
        T label = *(label_val + i);
        T dout = *dout_val;
        T x = predict - label;
        if (x == T(0)) {
          *(result + i) = T(0) * dout;
        } else if (x <= -T(sigma)) {
          *(result + i) = T(-1) / data_num * dout;
        } else if (x >= T(sigma)) {
          *(result + i) = T(1) / data_num * dout;
        } else if (sigma == 0) {
          KERNEL_LOG_ERROR("attribute sigma could not be 0.");
        } else {
          *(result + i) = x / T(sigma) / data_num * dout;
        }
      }
    };
    return CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shared_smoothl1lossgradv2);
  }
  KERNEL_LOG_INFO("SmoothL1LossGradV2CpuKernel::ComputeMean end");
}

// "None" takes grad_output as a parameter,
// and the end result is that result of "Sum" is multiplied by the grad_output
// one by one, that is, the weight is increased
template <typename T>
uint32_t SmoothL1LossGradV2CpuKernel::ComputeNone(const CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("SmoothL1LossGradV2CpuKernel::ComputeNone start");
  Tensor *predict_tensor = ctx.Input(0);
  Tensor *label_tensor = ctx.Input(1);
  Tensor *dout_tensor = ctx.Input(2);
  Tensor *gradient_tensor = ctx.Output(0);
  T *predict_val = static_cast<T *>(predict_tensor->GetData());
  T *label_val = static_cast<T *>(label_tensor->GetData());
  T *dout_val = static_cast<T *>(dout_tensor->GetData());
  T *gradient_val = static_cast<T *>(gradient_tensor->GetData());
  int64_t data_num = predict_tensor->NumElements();
  int64_t data_size = data_num * sizeof(T);
  T *result = gradient_val;
  if (data_size <= kParallelDataNum) {
    for (int64_t i = 0; i < data_num; i++) {
      T predict = *(predict_val + i);
      T label = *(label_val + i);
      T x = predict - label;
      T dout = *(dout_val + i);
      if (x == T(0)) {
        *(result + i) = T(0) * dout;
      } else if (x <= -T(sigma)) {
        *(result + i) = T(-1) * dout;
      } else if (x >= T(sigma)) {
        *(result + i) = T(1) * dout;
      } else if (sigma == 0) {
        KERNEL_LOG_ERROR("attribute sigma could not be 0.");
      } else {
        *(result + i) = dout * x / T(sigma);
      }
    }
    return KERNEL_STATUS_OK;
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num cannot be 0.");
    }
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shared_smoothl1lossgradv2 = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        T predict = *(predict_val + i);
        T label = *(label_val + i);
        T x = predict - label;
        T dout = *(dout_val + i);
        if (x == T(0)) {
          *(result + i) = T(0) * dout;
        } else if (x <= -T(sigma)) {
          *(result + i) = T(-1) * dout;
        } else if (x >= T(sigma)) {
          *(result + i) = T(1) * dout;
        } else if (sigma == 0) {
          KERNEL_LOG_ERROR("attribute sigma could not be 0.");
        } else {
          *(result + i) = dout * x / T(sigma);
        }
      }
    };
    return CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shared_smoothl1lossgradv2);
  }
  KERNEL_LOG_INFO("SmoothL1LossGradV2CpuKernel::ComputeNone end");
}

REGISTER_CPU_KERNEL(kSmoothL1LossGradV2, SmoothL1LossGradV2CpuKernel);
}  // namespace aicpu
