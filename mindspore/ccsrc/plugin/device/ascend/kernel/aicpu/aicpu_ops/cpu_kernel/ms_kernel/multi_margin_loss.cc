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

#include "multi_margin_loss.h"

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const char *kMultiMarginLoss = "MultiMarginLoss";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 28 * 1024;
}  // namespace

namespace aicpu {
uint32_t MultiMarginLossCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  uint32_t kInputNum = 3;
  constexpr int SERV_TYPE_SET = 2;
  if (ctx.GetInputsSize() == SERV_TYPE_SET) {
    kInputNum = SERV_TYPE_SET;
  }
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "MultiMarginLoss check input and output number failed.");
  KERNEL_HANDLE_ERROR(MultiMarginLossCheck(ctx), "MultiMarginLoss check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    case DT_FLOAT16:
      return MultiMarginLossComputeFP16<Eigen::half>(ctx);
    case DT_FLOAT:
      return MultiMarginLossCompute<float>(ctx);
    case DT_DOUBLE:
      return MultiMarginLossCompute<double>(ctx);
    default:
      KERNEL_LOG_ERROR("MultiMarginLoss kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t MultiMarginLossCpuKernel::MultiMarginLossCheck(CpuKernelContext &ctx) {
  auto input_0 = ctx.Input(0);
  auto input_1 = ctx.Input(1);

  constexpr int SERV_TYPE_SET = 2;
  constexpr int SERV_TYPE_QUERY = 3;

  DataType input0_type = input_0->GetDataType();
  DataType input1_type = input_1->GetDataType();
  KERNEL_CHECK_FALSE((input1_type == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of target [%s] should be int64.", DTypeStr(input1_type).c_str())
  auto target = reinterpret_cast<int64_t *>(ctx.Input(1)->GetData());
  int64_t target_num = ctx.Input(0)->GetTensorShape()->GetDimSize(1);
  int64_t batch_size = ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  if (ctx.GetInputsSize() == SERV_TYPE_QUERY) {
    auto input_weight = ctx.Input(2);
    DataType input2_type = input_weight->GetDataType();
    KERNEL_CHECK_FALSE((input2_type == input0_type), KERNEL_STATUS_PARAM_INVALID,
                       "weight should have the same dtype with x, but get [%s].", DTypeStr(input2_type).c_str())
  }
  KERNEL_CHECK_FALSE((ctx.Input(0)->GetTensorShape()->GetDims() == SERV_TYPE_SET), KERNEL_STATUS_PARAM_INVALID,
                     "Rank of x should be 2.")
  KERNEL_CHECK_FALSE((ctx.Input(1)->GetTensorShape()->GetDims() == 1), KERNEL_STATUS_PARAM_INVALID,
                     "Rank of target should be 1.")
  KERNEL_CHECK_FALSE((batch_size == ctx.Input(1)->GetTensorShape()->GetDimSize(0)), KERNEL_STATUS_PARAM_INVALID,
                     "[%s] 's x's shape[0] should be the same as target's "
                     "shape[0].",
                     ctx.GetOpType().c_str())
  for (int64_t i = 0; i < batch_size; i++) {
    KERNEL_CHECK_FALSE(*(target + i) >= 0 && (*(target + i) < target_num), KERNEL_STATUS_PARAM_INVALID,
                       "[%s]'s target out of range", ctx.GetOpType().c_str());
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MultiMarginLossCpuKernel::MultiMarginLossCompute(CpuKernelContext &ctx) {
  constexpr int SERV_TYPE_BRWD = 1;
  constexpr int SERV_TYPE_SET = 2;
  constexpr int ADULT_AGE = 4;
  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_target = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input_target = reinterpret_cast<int64_t *>(ctx.Input(1)->GetData());
  T *input_weight = nullptr;
  bool weight_defined_ = (ctx.GetInputsSize() == 3);
  if (weight_defined_) {
    input_weight = reinterpret_cast<T *>(ctx.Input(2)->GetData());
    int64_t weight_length = ctx.Input(2)->NumElements();
    int64_t x_length = ctx.Input(0)->GetTensorShape()->GetDimSize(1);
    if (weight_length < x_length) {
      for (int64_t i = 0; i < x_length - weight_length; i++) {
        input_weight[i + weight_length] = static_cast<T>(0);
      }
    }
  }
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  AttrValue *Attr_p = ctx.GetAttr("p");
  int p = (Attr_p == nullptr) ? 1 : Attr_p->GetInt();
  if (p != SERV_TYPE_BRWD && p != SERV_TYPE_SET) {
    KERNEL_LOG_ERROR("MultiMarginLoss kernel attr p should be 1 or 2.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  AttrValue *Attr_margin = ctx.GetAttr("margin");
  T margin = static_cast<T>((Attr_margin == nullptr) ? 1 : Attr_margin->GetFloat());
  AttrValue *Attr_red = ctx.GetAttr("reduction");
  std::string reduction = (Attr_red == nullptr) ? "mean" : Attr_red->GetString();
  int64_t batch_size = ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  int64_t dims = ctx.Input(0)->GetTensorShape()->GetDimSize(1);
  Eigen::Array<T, Eigen::Dynamic, 1> output(batch_size, 1);
  output.setZero();
  auto output_data = output.data();
  int64_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, (int64_t)aicpu::CpuKernelUtils::GetCPUNum(ctx));
  auto shard_multi_margin_loss = [&](size_t start, size_t end) {
    int64_t once_compute_thread_size = end - start;
    Eigen::Array<T, Eigen::Dynamic, 1> cacl(dims, 1);
    auto cacl_data = cacl.data();
    cacl.setZero();
    if (dims == 0) {
      KERNEL_LOG_ERROR("dims could not be 0.");
    }
    for (int64_t m = 0; m < (once_compute_thread_size) / dims; m++) {
      int64_t i = start / dims;
      for (int64_t d = 0; d < dims; d++) {
        if (d == input_target[i]) {
          continue;
        }
        cacl_data[d] = margin + input_x[start + d] - input_x[start + input_target[i]];
        if (cacl_data[d] > T(0)) {
          cacl_data[d] = (p == 1) ? cacl_data[d] : cacl_data[d] * cacl_data[d];
          if (weight_defined_) {
            cacl_data[d] *= (input_weight[input_target[i]]);
          }
          output_data[i] += cacl_data[d];
        }
      }
      output_data[i] = output_data[i] / static_cast<T>(dims);
      start += dims;
    }
  };
  if ((ctx.Input(0)->NumElements()) * sizeof(T) <= kParallelDataNum) {
    Eigen::Array<T, Eigen::Dynamic, 1> cacl(dims, 1);
    auto cacl_data = cacl.data();
    cacl.setZero();
    T sum = static_cast<T>(0);
    for (int64_t i = 0; i < batch_size; i++) {
      int64_t target_idx = input_target[i];
      sum = static_cast<T>(0);
      cacl.setZero();
      for (int64_t d = 0; d < dims; d++) {
        if (d == target_idx) {
          continue;
        }
        cacl_data[d] = margin + input_x[i * dims + d] - input_x[i * dims + target_idx];
        if (cacl_data[d] > T(0)) {
          cacl_data[d] = (p == 1) ? cacl_data[d] : cacl_data[d] * cacl_data[d];
          if (weight_defined_) {
            cacl_data[d] *= static_cast<T>(input_weight[target_idx]);
          }
          sum += cacl_data[d];
        }
      }
      sum = sum / static_cast<T>(dims);
      output_data[i] = sum;
    }
  } else {
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    CpuKernelUtils::ParallelFor(ctx, ctx.Input(0)->NumElements(), dims * ADULT_AGE * (batch_size / max_core_num + 1),
                                shard_multi_margin_loss);
  }
  if (reduction == "mean") {
    *output_y = output.mean();
  }
  if (reduction == "sum") {
    *output_y = output.sum();
  }
  if (reduction == "none") {
    for (int64_t t = 0; t < batch_size; t++) {
      *(output_y + t) = output_data[t];
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MultiMarginLossCpuKernel::MultiMarginLossComputeFP16(CpuKernelContext &ctx) {
  constexpr int SERV_TYPE_BRWD = 1;
  constexpr int SERV_TYPE_SET = 2;
  constexpr int ADULT_AGE = 4;
  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_target = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input_target = reinterpret_cast<int64_t *>(ctx.Input(1)->GetData());
  T *input_weight = nullptr;
  bool weight_defined_ = (ctx.GetInputsSize() == 3);
  if (weight_defined_) {
    input_weight = reinterpret_cast<T *>(ctx.Input(SERV_TYPE_SET)->GetData());
    int64_t weight_length = ctx.Input(2)->NumElements();
    int64_t x_length = ctx.Input(0)->GetTensorShape()->GetDimSize(1);
    if (weight_length < x_length) {
      for (int64_t i = 0; i < x_length - weight_length; i++) {
        input_weight[i + weight_length] = static_cast<T>(0);
      }
    }
  }
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  AttrValue *Attr_p = ctx.GetAttr("p");
  int p = (Attr_p == nullptr) ? 1 : Attr_p->GetInt();
  if (p != SERV_TYPE_BRWD && p != SERV_TYPE_SET) {
    KERNEL_LOG_ERROR("MultiMarginLoss kernel attr p should be 1 or 2.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  AttrValue *Attr_margin = ctx.GetAttr("margin");
  float margin = static_cast<float>((Attr_margin == nullptr) ? 1 : Attr_margin->GetFloat());
  AttrValue *Attr_red = ctx.GetAttr("reduction");
  std::string reduction = (Attr_red == nullptr) ? "mean" : Attr_red->GetString();
  int64_t batch_size = ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  int64_t dims = ctx.Input(0)->GetTensorShape()->GetDimSize(1);
  Eigen::Array<float, Eigen::Dynamic, 1> output(batch_size, 1);
  output.setZero();
  auto output_data = output.data();
  int64_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, (int64_t)aicpu::CpuKernelUtils::GetCPUNum(ctx));
  auto shard_multi_margin_loss = [&](size_t start, size_t end) {
    int64_t once_compute_thread_size = end - start;
    Eigen::Array<float, Eigen::Dynamic, 1> cacl(dims, 1);
    auto cacl_data = cacl.data();
    cacl.setZero();
    if (dims == 0) {
      KERNEL_LOG_ERROR("dims could not be 0.");
    }
    for (int64_t m = 0; m < (once_compute_thread_size) / dims; m++) {
      int64_t i = start / dims;
      for (int64_t d = 0; d < dims; d++) {
        if (d == input_target[i]) {
          continue;
        }
        cacl_data[d] =
          margin + static_cast<float>(input_x[start + d]) - static_cast<float>(input_x[start + input_target[i]]);
        if (cacl_data[d] > 0) {
          cacl_data[d] = (p == 1) ? cacl_data[d] : cacl_data[d] * cacl_data[d];
          if (weight_defined_) {
            cacl_data[d] *= static_cast<float>(input_weight[input_target[i]]);
          }
          output_data[i] += cacl_data[d];
        }
      }
      output_data[i] = output_data[i] / static_cast<float>(dims);
      start += dims;
    }
  };
  if ((ctx.Input(0)->NumElements()) * sizeof(T) <= kParallelDataNum) {
    Eigen::Array<float, Eigen::Dynamic, 1> cacl(dims, 1);
    auto cacl_data = cacl.data();
    cacl.setZero();
    float sum = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      int64_t target_idx = input_target[i];
      sum = 0;
      cacl.setZero();
      for (int64_t d = 0; d < dims; d++) {
        if (d == target_idx) {
          continue;
        }
        cacl_data[d] =
          margin + static_cast<float>(input_x[i * dims + d]) - static_cast<float>(input_x[i * dims + target_idx]);
        if (cacl_data[d] > 0) {
          cacl_data[d] = (p == 1) ? cacl_data[d] : cacl_data[d] * cacl_data[d];
          if (weight_defined_) {
            cacl_data[d] *= static_cast<float>(input_weight[target_idx]);
          }
          sum += cacl_data[d];
        }
      }
      sum = sum / static_cast<float>(dims);
      output_data[i] = sum;
    }
  } else {
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    CpuKernelUtils::ParallelFor(ctx, ctx.Input(0)->NumElements(), dims * ADULT_AGE * (batch_size / max_core_num + 1),
                                shard_multi_margin_loss);
  }
  if (reduction == "mean") {
    *output_y = static_cast<T>(output.mean());
  }
  if (reduction == "sum") {
    *output_y = static_cast<T>(output.sum());
  }
  if (reduction == "none") {
    for (int64_t t = 0; t < batch_size; t++) {
      *(output_y + t) = static_cast<T>(output_data[t]);
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMultiMarginLoss, MultiMarginLossCpuKernel);
}  // namespace aicpu
