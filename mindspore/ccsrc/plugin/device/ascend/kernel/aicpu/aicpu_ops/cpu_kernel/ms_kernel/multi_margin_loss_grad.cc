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

#include "multi_margin_loss_grad.h"

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const char *kMultiMarginLossGrad = "MultiMarginLossGrad";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 28 * 1024;
}  // namespace

namespace aicpu {
uint32_t MultiMarginLossGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  uint32_t kInputNum = 4;
  constexpr int SERV_TYPE_QUERY = 3;
  if (ctx.GetInputsSize() == SERV_TYPE_QUERY) {
    kInputNum = SERV_TYPE_QUERY;
  }
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "MultiMarginLossGrad check input and output number failed.");
  KERNEL_HANDLE_ERROR(MultiMarginLossGradCheck(ctx), "MultiMarginLossGrad check params failed.");
  auto data_type = ctx.Input(1)->GetDataType();
  switch (data_type) {
    case DT_FLOAT16:
      return MultiMarginLossGradComputeFP16<Eigen::half>(ctx);
    case DT_FLOAT:
      return MultiMarginLossGradCompute<float>(ctx);
    case DT_DOUBLE:
      return MultiMarginLossGradCompute<double>(ctx);
    default:
      KERNEL_LOG_ERROR("MultiMarginLossGrad kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t MultiMarginLossGradCpuKernel::MultiMarginLossGradCheck(CpuKernelContext &ctx) {
  auto input_x = ctx.Input(1);
  auto input_target = ctx.Input(2);
  constexpr int SERV_TYPE_BRWD = 1;
  constexpr int SERV_TYPE_SET = 2;
  constexpr int ADULT_AGE = 4;
  DataType input_x_type = input_x->GetDataType();
  DataType input_target_type = input_target->GetDataType();
  KERNEL_CHECK_FALSE((input_target_type == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of target [%s] should be int64.", DTypeStr(input_target_type).c_str())
  auto target = reinterpret_cast<int64_t *>(ctx.Input(2)->GetData());
  int64_t dims = ctx.Input(1)->GetTensorShape()->GetDimSize(1);
  int64_t batch_size = ctx.Input(1)->GetTensorShape()->GetDimSize(0);
  if (ctx.GetInputsSize() == ADULT_AGE) {
    auto input_weight = ctx.Input(3);
    DataType input_weight_type = input_weight->GetDataType();
    KERNEL_CHECK_FALSE((input_weight_type == input_x_type), KERNEL_STATUS_PARAM_INVALID,
                       "weight should have the same dtype with x, but get [%s].", DTypeStr(input_weight_type).c_str())
  }
  KERNEL_CHECK_FALSE((ctx.Input(1)->GetTensorShape()->GetDims() == SERV_TYPE_SET), KERNEL_STATUS_PARAM_INVALID,
                     "Rank of x should be 2.")
  KERNEL_CHECK_FALSE((ctx.Input(2)->GetTensorShape()->GetDims() == SERV_TYPE_BRWD), KERNEL_STATUS_PARAM_INVALID,
                     "Rank of target should be 1.")
  KERNEL_CHECK_FALSE((batch_size == ctx.Input(2)->GetTensorShape()->GetDimSize(0)), KERNEL_STATUS_PARAM_INVALID,
                     "[%s] 's x's dimension[0] should be the same as target's "
                     "dimension[0].",
                     ctx.GetOpType().c_str())
  for (int64_t i = 0; i < batch_size; i++) {
    KERNEL_CHECK_FALSE(*(target + i) >= 0 && (*(target + i) < dims), KERNEL_STATUS_PARAM_INVALID,
                       "[%s]'s target out of range.", ctx.GetOpType().c_str());
  }
  AttrValue *Attr_red = ctx.GetAttr("reduction");
  std::string reduction = (Attr_red == nullptr) ? "mean" : Attr_red->GetString();
  if (reduction == "none") {
    KERNEL_CHECK_FALSE(ctx.Input(0)->GetTensorShape()->GetDimSize(0) == ctx.Input(1)->GetTensorShape()->GetDimSize(0),
                       KERNEL_STATUS_PARAM_INVALID,
                       "[%s] 's y_grad's shape should be the same as "
                       "target when reduction is none.",
                       ctx.GetOpType().c_str())
  } else {
    KERNEL_CHECK_FALSE(ctx.Input(0)->GetTensorShape()->GetDims() <= 1, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] 's y_grad should be a scalar "
                       "when reduction is mean or sum.",
                       ctx.GetOpType().c_str())
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MultiMarginLossGradCpuKernel::MultiMarginLossGradCompute(CpuKernelContext &ctx) {
  constexpr int SERV_TYPE_BRWD = 1;
  constexpr int SERV_TYPE_SET = 2;
  constexpr int ADULT_AGE = 4;
  std::vector<int64_t> shape_x = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_target = ctx.Input(2)->GetTensorShape()->GetDimSizes();
  auto input_x = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto input_target = reinterpret_cast<int64_t *>(ctx.Input(2)->GetData());
  T *input_weight = nullptr;
  bool weight_defined_ = (ctx.GetInputsSize() == 4);
  if (weight_defined_) {
    input_weight = reinterpret_cast<T *>(ctx.Input(3)->GetData());
    int64_t weight_length = ctx.Input(3)->NumElements();
    int64_t x_length = ctx.Input(1)->GetTensorShape()->GetDimSize(1);
    if (weight_length < x_length) {
      for (int64_t i = 0; i < x_length - weight_length; i++) {
        input_weight[i + weight_length] = static_cast<T>(0);
      }
    }
  }
  auto output_x_grad = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  AttrValue *Attr_p = ctx.GetAttr("p");
  int p = (Attr_p == nullptr) ? 1 : Attr_p->GetInt();
  if (p != SERV_TYPE_BRWD && p != SERV_TYPE_SET) {
    KERNEL_LOG_ERROR("MultiMarginLossGrad kernel attr p should be 1 or 2.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  AttrValue *Attr_margin = ctx.GetAttr("margin");
  T margin = static_cast<T>((Attr_margin == nullptr) ? 1 : Attr_margin->GetFloat());
  AttrValue *Attr_red = ctx.GetAttr("reduction");
  std::string reduction = (Attr_red == nullptr) ? "mean" : Attr_red->GetString();
  int64_t batch_size = ctx.Input(1)->GetTensorShape()->GetDimSize(0);
  int64_t dims = ctx.Input(1)->GetTensorShape()->GetDimSize(1);
  T weights = static_cast<T>(0);
  weights = reduction == "mean" ? (static_cast<T>(1) / (static_cast<T>(dims) * static_cast<T>(batch_size)))
                                : (static_cast<T>(1) / static_cast<T>(dims));
  int64_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, (int64_t)aicpu::CpuKernelUtils::GetCPUNum(ctx));
  auto shard_multi_margin_loss_grad = [&](size_t start, size_t end) {
    Eigen::Array<T, Eigen::Dynamic, 1> calculate(dims, 1);
    auto calculate_data = calculate.data();
    int64_t once_compute_thread_size = end - start;
    if (dims == 0) {
      KERNEL_LOG_ERROR("dims could not be 0.");
    }
    for (int64_t i = 0; i < once_compute_thread_size / dims; i++) {
      int64_t m = start / dims;
      int64_t target_idx = input_target[m];
      T input_target_value = input_x[start + target_idx];
      T grad_input_target = static_cast<T>(0);
      calculate.setZero();
      for (int64_t d = 0; d < dims; d++) {
        calculate_data[d] = margin + input_x[start + d] - input_target_value;
        if (d == target_idx) {
          continue;
        }
        if (calculate_data[d] > static_cast<T>(0)) {
          calculate_data[d] = (p == 1) ? weights : static_cast<T>(2) * weights * calculate_data[d];
          if (weight_defined_) {
            calculate_data[d] *= (input_weight[target_idx]);
          }
          grad_input_target -= calculate_data[d];
          *(output_x_grad + start + d) = calculate_data[d];
        } else {
          *(output_x_grad + start + d) = static_cast<T>(0);
        }
      }
      *(output_x_grad + start + target_idx) = grad_input_target;
      start += dims;
    }
  };
  if ((ctx.Input(1)->NumElements()) * sizeof(T) <= kParallelDataNum) {
    Eigen::Array<T, Eigen::Dynamic, 1> calculate(dims, 1);
    auto calculate_data = calculate.data();
    for (int64_t i = 0; i < batch_size; i++) {
      int64_t target_idx = input_target[i];
      T input_target_value = input_x[i * dims + target_idx];
      T grad_input_target = static_cast<T>(0);
      calculate.setZero();
      for (int64_t d = 0; d < dims; d++) {
        calculate_data[d] = margin + input_x[i * dims + d] - input_target_value;
        if (d == target_idx) {
          continue;
        }
        if (calculate_data[d] > static_cast<T>(0)) {
          calculate_data[d] = (p == 1) ? weights : static_cast<T>(SERV_TYPE_SET) * weights * calculate_data[d];
          if (weight_defined_) {
            calculate_data[d] *= static_cast<T>(input_weight[target_idx]);
          }
          grad_input_target -= calculate_data[d];
          *(output_x_grad + i * dims + d) = calculate_data[d];
        } else {
          *(output_x_grad + i * dims + d) = static_cast<T>(0);
        }
      }
      *(output_x_grad + i * dims + target_idx) = grad_input_target;
    }
  } else {
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    CpuKernelUtils::ParallelFor(ctx, ctx.Input(1)->NumElements(), dims * ADULT_AGE * (batch_size / max_core_num + 1),
                                shard_multi_margin_loss_grad);
  }
  auto y_grad = ctx.Input(0);
  T y_grad_value = static_cast<T>(1);
  auto y_grad_data =
    (reinterpret_cast<T *>(y_grad->GetData()) == nullptr) ? &y_grad_value : reinterpret_cast<T *>(y_grad->GetData());
  int64_t y_grad_dims = y_grad->GetTensorShape()->GetDims();
  if (reduction != "none" || y_grad_dims == 0) {
    for (int64_t i = 0; i < batch_size * dims; i++) {
      *(output_x_grad + i) *= *(y_grad_data);
    }
  } else {
    for (int64_t i = 0; i < batch_size; i++) {
      for (int64_t j = 0; j < dims; j++) {
        *(output_x_grad + i * dims + j) *= *(y_grad_data + i);
      }
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MultiMarginLossGradCpuKernel::MultiMarginLossGradComputeFP16(CpuKernelContext &ctx) {
  constexpr int SERV_TYPE_BRWD = 1;
  constexpr int SERV_TYPE_SET = 2;
  constexpr int SERV_TYPE_QUERY = 3;
  constexpr int ADULT_AGE = 4;
  std::vector<int64_t> shape_x = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_target = ctx.Input(SERV_TYPE_SET)->GetTensorShape()->GetDimSizes();
  auto input_x = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto input_target = reinterpret_cast<int64_t *>(ctx.Input(2)->GetData());
  T *input_weight = nullptr;
  bool weight_defined_ = (ctx.GetInputsSize() == 4);
  if (weight_defined_) {
    input_weight = reinterpret_cast<T *>(ctx.Input(SERV_TYPE_QUERY)->GetData());
    int64_t weight_length = ctx.Input(3)->NumElements();
    int64_t x_length = ctx.Input(1)->GetTensorShape()->GetDimSize(1);
    if (weight_length < x_length) {
      for (int64_t i = 0; i < x_length - weight_length; i++) {
        input_weight[i + weight_length] = static_cast<T>(0);
      }
    }
  }
  auto output_x_grad = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  AttrValue *Attr_p = ctx.GetAttr("p");
  int p = (Attr_p == nullptr) ? 1 : Attr_p->GetInt();
  if (p != SERV_TYPE_BRWD && p != SERV_TYPE_SET) {
    KERNEL_LOG_ERROR("MultiMarginLossGrad kernel attr p should be 1 or 2.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  AttrValue *Attr_margin = ctx.GetAttr("margin");
  float margin = static_cast<float>((Attr_margin == nullptr) ? 1 : Attr_margin->GetFloat());
  AttrValue *Attr_red = ctx.GetAttr("reduction");
  std::string reduction = (Attr_red == nullptr) ? "mean" : Attr_red->GetString();
  int64_t batch_size = ctx.Input(1)->GetTensorShape()->GetDimSize(0);
  int64_t dims = ctx.Input(1)->GetTensorShape()->GetDimSize(1);
  float weights = 0;
  weights = reduction == "mean" ? (static_cast<float>(1) / (static_cast<float>(dims) * static_cast<float>(batch_size)))
                                : (static_cast<float>(1) / static_cast<float>(dims));
  int64_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, (int64_t)aicpu::CpuKernelUtils::GetCPUNum(ctx));
  auto shard_multi_margin_loss_grad = [&](size_t start, size_t end) {
    Eigen::Array<float, Eigen::Dynamic, 1> calculate(dims, 1);
    auto calculate_data = calculate.data();
    int64_t once_compute_thread_size = end - start;
    if (dims == 0) {
      KERNEL_LOG_ERROR("dims could not be 0.");
    }
    for (int64_t i = 0; i < once_compute_thread_size / dims; i++) {
      int64_t m = start / dims;
      int64_t target_idx = input_target[m];
      float input_target_value = static_cast<float>(input_x[start + target_idx]);
      float grad_input_target = static_cast<float>(0);
      calculate.setZero();
      for (int64_t d = 0; d < dims; d++) {
        calculate_data[d] = margin + static_cast<float>(input_x[start + d]) - input_target_value;
        if (d == target_idx) {
          continue;
        }
        if (calculate_data[d] > 0) {
          calculate_data[d] = (p == 1) ? weights : static_cast<float>(2) * weights * calculate_data[d];
          if (weight_defined_) {
            calculate_data[d] *= static_cast<float>(input_weight[target_idx]);
          }
          grad_input_target -= calculate_data[d];
          *(output_x_grad + start + d) = static_cast<T>(calculate_data[d]);
        } else {
          *(output_x_grad + start + d) = static_cast<T>(0);
        }
      }
      *(output_x_grad + start + target_idx) = static_cast<T>(grad_input_target);
      start += dims;
    }
  };
  if ((ctx.Input(1)->NumElements()) * sizeof(T) <= kParallelDataNum) {
    Eigen::Array<float, Eigen::Dynamic, 1> calculate(dims, 1);
    auto calculate_data = calculate.data();
    for (int64_t i = 0; i < batch_size; i++) {
      int64_t target_idx = input_target[i];
      float input_target_value = static_cast<float>(input_x[i * dims + target_idx]);
      float grad_input_target = 0;
      calculate.setZero();
      for (int64_t d = 0; d < dims; d++) {
        calculate_data[d] = margin + static_cast<float>(input_x[i * dims + d]) - input_target_value;
        if (d == target_idx) {
          continue;
        }
        if (calculate_data[d] > 0) {
          calculate_data[d] = (p == 1) ? weights : static_cast<float>(SERV_TYPE_SET) * weights * calculate_data[d];
          if (weight_defined_) {
            calculate_data[d] *= static_cast<float>(input_weight[target_idx]);
          }
          grad_input_target -= calculate_data[d];
          *(output_x_grad + i * dims + d) = static_cast<T>(calculate_data[d]);
        } else {
          *(output_x_grad + i * dims + d) = static_cast<T>(0);
        }
      }
      *(output_x_grad + i * dims + target_idx) = static_cast<T>(grad_input_target);
    }
  } else {
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    CpuKernelUtils::ParallelFor(ctx, ctx.Input(1)->NumElements(), dims * ADULT_AGE * (batch_size / max_core_num + 1),
                                shard_multi_margin_loss_grad);
  }
  auto y_grad = ctx.Input(0);
  T y_grad_value = static_cast<T>(1);
  auto y_grad_data =
    (reinterpret_cast<T *>(y_grad->GetData()) == nullptr) ? &y_grad_value : reinterpret_cast<T *>(y_grad->GetData());
  int64_t y_grad_dims = y_grad->GetTensorShape()->GetDims();
  if (reduction != "none" || y_grad_dims == 0) {
    for (int64_t i = 0; i < batch_size * dims; i++) {
      *(output_x_grad + i) =
        static_cast<T>(static_cast<float>(*(output_x_grad + i)) * static_cast<float>(*(y_grad_data)));
    }
  } else {
    for (int64_t i = 0; i < batch_size; i++) {
      for (int64_t j = 0; j < dims; j++) {
        *(output_x_grad + i * dims + j) =
          static_cast<T>(static_cast<float>(*(output_x_grad + i * dims + j)) * static_cast<float>(*(y_grad_data + i)));
      }
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMultiMarginLossGrad, MultiMarginLossGradCpuKernel);
}  // namespace aicpu
