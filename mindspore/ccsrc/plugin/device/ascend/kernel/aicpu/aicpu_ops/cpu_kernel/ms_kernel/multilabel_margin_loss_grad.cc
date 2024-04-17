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

#include "multilabel_margin_loss_grad.h"

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const char *kMultilabelMarginLossGrad = "MultilabelMarginLossGrad";
}  // namespace

namespace aicpu {
uint32_t MultilabelMarginLossGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  uint32_t kInputNum = 4;
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "MultilabelMarginLossGrad check input and output number failed.");
  CUST_KERNEL_HANDLE_ERROR(ctx, MultilabelMarginLossGradCheck(ctx), "MultilabelMarginLossGrad check params failed.");
  auto data_type = ctx.Input(1)->GetDataType();
  switch (data_type) {
    case DT_FLOAT16:
      return MultilabelMarginLossGradComputeFP16<Eigen::half>(ctx);
    case DT_FLOAT:
      return MultilabelMarginLossGradCompute<float>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "MultilabelMarginLossGrad kernel data type [%s] not support.",
                            DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t MultilabelMarginLossGradCpuKernel::MultilabelMarginLossGradCheck(CpuKernelContext &ctx) {
  auto target = reinterpret_cast<int32_t *>(ctx.Input(2)->GetData());
  size_t dims = ctx.Input(1)->GetTensorShape()->GetDims();
  int64_t batch_size =
    (dims == 2) ? ctx.Input(1)->GetTensorShape()->GetDimSize(1) : ctx.Input(1)->GetTensorShape()->GetDimSize(0);
  size_t data_num = ctx.Input(1)->GetTensorShape()->NumElements();
  AttrValue *Attr_red = ctx.GetAttr("reduction");
  std::string reduction = (Attr_red == nullptr) ? "mean" : Attr_red->GetString();
  for (size_t i = 0; i < data_num; i++) {
    CUST_KERNEL_CHECK_FALSE(ctx, *(target + i) >= -1 && (*(target + i) < batch_size), KERNEL_STATUS_PARAM_INVALID,
                            "[%s]'s target out of range.", ctx.GetOpType().c_str());
  }
  if (reduction == "none") {
    if (dims == 1) {
      CUST_KERNEL_CHECK_FALSE(ctx, ctx.Input(0)->GetTensorShape()->GetDims() <= 1, KERNEL_STATUS_PARAM_INVALID,
                              "[%s]'s y_grad should be a scalar "
                              "when rank of x is 1.",
                              ctx.GetOpType().c_str())
    } else {
      CUST_KERNEL_CHECK_FALSE(
        ctx,
        ctx.Input(0)->GetTensorShape()->GetDims() == 1 &&
          ctx.Input(0)->GetTensorShape()->GetDimSize(0) == ctx.Input(1)->GetTensorShape()->GetDimSize(0),
        KERNEL_STATUS_PARAM_INVALID,
        "[%s]'s y_grad's shape should be the same as "
        "{x_shape[0]} when the rank of x is 2 and reduction is none.",
        ctx.GetOpType().c_str())
    }
  } else {
    // change condition "== 0" to "<= 0" as a hotfix that 0-dim tensor has a rank of 1 when dynamic shape
    CUST_KERNEL_CHECK_FALSE(ctx, ctx.Input(0)->GetTensorShape()->GetDims() <= 1, KERNEL_STATUS_PARAM_INVALID,
                            "[%s]'s y_grad should be a scalar "
                            "when reduction is mean or sum.",
                            ctx.GetOpType().c_str())
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MultilabelMarginLossGradCpuKernel::MultilabelMarginLossGradCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto input_target = reinterpret_cast<int32_t *>(ctx.Input(2)->GetData());
  auto input_istarget = reinterpret_cast<int32_t *>(ctx.Input(3)->GetData());
  auto output_x_grad = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  AttrValue *Attr_red = ctx.GetAttr("reduction");
  std::string reduction = (Attr_red == nullptr) ? "mean" : Attr_red->GetString();
  size_t dims = ctx.Input(1)->GetTensorShape()->GetDims();
  size_t batch_size =
    (dims == 2) ? ctx.Input(1)->GetTensorShape()->GetDimSize(1) : ctx.Input(1)->GetTensorShape()->GetDimSize(0);
  size_t data_num = ctx.Input(1)->GetTensorShape()->NumElements();
  size_t nframe = data_num / batch_size;
  auto g = static_cast<T>(reduction == "mean" ? 1. / data_num : 1. / batch_size);
  std::vector<T> output_vector(data_num, 0);
  for (size_t t = 0; t < nframe; t++) {
    for (size_t m = 0; m < batch_size; m++) {
      int32_t target_idx = input_target[m];
      if (target_idx < 0) {
        break;
      }
      auto calc_target = input_x[target_idx];
      for (size_t n = 0; n < batch_size; n++) {
        if (input_istarget[n] == 0) {
          float z = 1 - calc_target + input_x[n];
          if (z > 0) {
            output_vector[t * batch_size + target_idx] -= g;
            output_vector[t * batch_size + n] += g;
          }
        }
      }
    }
    input_x += batch_size;
    input_target += batch_size;
    input_istarget += batch_size;
  }
  auto y_grad = ctx.Input(0);
  auto y_grad_data = reinterpret_cast<T *>(y_grad->GetData());
  size_t y_grad_dims = y_grad->GetTensorShape()->GetDims();
  if (reduction != "none" || y_grad_dims == 0) {
    for (size_t i = 0; i < data_num; i++) {
      *(output_x_grad + i) = output_vector[i] * (*(y_grad_data));
    }
  } else {
    for (size_t i = 0; i < nframe; i++) {
      for (size_t j = 0; j < batch_size; j++) {
        *(output_x_grad + i * batch_size + j) = output_vector[i * batch_size + j] * (*(y_grad_data + i));
      }
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MultilabelMarginLossGradCpuKernel::MultilabelMarginLossGradComputeFP16(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto input_target = reinterpret_cast<int32_t *>(ctx.Input(2)->GetData());
  auto input_istarget = reinterpret_cast<int32_t *>(ctx.Input(3)->GetData());
  auto output_x_grad = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  AttrValue *Attr_red = ctx.GetAttr("reduction");
  std::string reduction = (Attr_red == nullptr) ? "mean" : Attr_red->GetString();
  size_t dims = ctx.Input(1)->GetTensorShape()->GetDims();
  size_t batch_size =
    (dims == 2) ? ctx.Input(1)->GetTensorShape()->GetDimSize(1) : ctx.Input(1)->GetTensorShape()->GetDimSize(0);
  size_t data_num = ctx.Input(1)->GetTensorShape()->NumElements();
  size_t nframe = data_num / batch_size;
  float g = static_cast<float>(reduction == "mean" ? 1. / data_num : 1. / batch_size);
  std::vector<float> output_vector(data_num, 0);
  for (size_t t = 0; t < nframe; t++) {
    for (size_t m = 0; m < batch_size; m++) {
      int32_t target_idx = input_target[m];
      if (target_idx < 0) {
        break;
      }
      float calc_target = static_cast<float>(input_x[target_idx]);
      for (size_t n = 0; n < batch_size; n++) {
        if (input_istarget[n] == 0) {
          float z = 1 - calc_target + static_cast<float>(input_x[n]);
          if (z > 0) {
            output_vector[t * batch_size + target_idx] -= g;
            output_vector[t * batch_size + n] += g;
          }
        }
      }
    }
    input_x += batch_size;
    input_target += batch_size;
    input_istarget += batch_size;
  }
  auto y_grad = ctx.Input(0);
  auto y_grad_data = reinterpret_cast<T *>(y_grad->GetData());
  size_t y_grad_dims = y_grad->GetTensorShape()->GetDims();
  if (reduction != "none" || y_grad_dims == 0) {
    for (size_t i = 0; i < data_num; i++) {
      *(output_x_grad + i) = static_cast<T>(output_vector[i] * static_cast<float>(*(y_grad_data)));
    }
  } else {
    for (size_t i = 0; i < nframe; i++) {
      for (size_t j = 0; j < batch_size; j++) {
        *(output_x_grad + i * batch_size + j) =
          static_cast<T>(output_vector[i * batch_size + j] * static_cast<float>(*(y_grad_data + i)));
      }
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kMultilabelMarginLossGrad, MultilabelMarginLossGradCpuKernel);
}  // namespace aicpu
