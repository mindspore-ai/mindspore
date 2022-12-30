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
#include "kldivlossgrad.h"

#include <complex>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kKlDivLossGrad = "KlDivLossGrad";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;
const uint32_t kGradIndex = 0;
const uint32_t kInputIndex = 1;
const uint32_t kTargetIndex = 2;
const std::string AttrReduction = "reduction";
const std::string AttrLog = "log_target";
const int64_t DataDefaultParallelNum = 16384;
}  // namespace

namespace aicpu {
template <typename T>
void KlDivLossGradOp(Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > &target,
                     Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > &grad,
                     Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > &output, std::int64_t &len, bool &log_target,
                     std::string &reduction) {
  T constant_zero{0};
  if (log_target) {
    output = -Eigen::exp(target) * grad;
    return;
  }
  if (reduction == "none") {
    for (uint32_t idx = 0; idx < len; ++idx) {
      if (target(idx) > constant_zero) {
        output(idx) = -target(idx) * grad(idx);
      }
    }
  } else {
    for (uint32_t idx = 0; idx < len; ++idx) {
      if (target(idx) > constant_zero) {
        output(idx) = -target(idx) * grad(0);
      }
    }
  }
  return;
}

std::uint32_t KlDivLossGradExtraCheck(CpuKernelContext &ctx) {
  Tensor *grad = ctx.Input(0);
  Tensor *input = ctx.Input(1);
  Tensor *target = ctx.Input(2);
  Tensor *output = ctx.Output(0);
  if (grad->GetDataSize() == 0) {
    KERNEL_LOG_ERROR("[%s] grad is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (input->GetDataSize() == 0) {
    KERNEL_LOG_ERROR("[%s] input is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (target->GetDataSize() == 0) {
    KERNEL_LOG_ERROR("[%s] target is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (output->GetDataSize() == 0) {
    KERNEL_LOG_ERROR("[%s] output is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if ((input->GetDataType() != grad->GetDataType()) || (target->GetDataType() != grad->GetDataType()) ||
      (output->GetDataType() != grad->GetDataType())) {
    KERNEL_LOG_ERROR(
      "The data type of the grad [%s], input [%s], target [%s], output y "
      "[%s] must be the same type.",
      DTypeStr(grad->GetDataType()).c_str(), DTypeStr(input->GetDataType()).c_str(),
      DTypeStr(target->GetDataType()).c_str(), DTypeStr(output->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  std::vector<int64_t> grad_dims = ctx.Input(kGradIndex)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> input_dims = ctx.Input(kInputIndex)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> target_dims = ctx.Input(kTargetIndex)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> output_dims = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  std::string reduction = ctx.GetAttr(AttrReduction)->GetString();
  if (output_dims != input_dims) {
    KERNEL_LOG_ERROR(
      "The data shape of the output need be the same as the input. output "
      "shape [%s], input shape [%s]",
      VectorToString(output_dims).c_str(), VectorToString(input_dims).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (target_dims != input_dims) {
    KERNEL_LOG_ERROR(
      "The data shape of the target need be the same as the input. target "
      "shape [%s], input shape [%s]",
      VectorToString(target_dims).c_str(), VectorToString(input_dims).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (reduction == "mean" || reduction == "sum" || reduction == "batchmean") {
    if (ctx.Input(0)->NumElements() != 1) {
      KERNEL_LOG_ERROR("The data num of the grad [%llu] must be 1", ctx.Input(0)->NumElements());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  } else if (reduction == "none") {
    if (input_dims != grad_dims) {
      KERNEL_LOG_ERROR(
        "The data shape of the grad need be the same as the input. grad "
        "shape "
        "[%s], input shape [%s]",
        VectorToString(grad_dims).c_str(), VectorToString(input_dims).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t KlDivLossGradCpuKernel::Compute(CpuKernelContext &ctx) {
  if (NormalCheck(ctx, kInputNum, kOutputNum) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (KlDivLossGradExtraCheck(ctx) == KERNEL_STATUS_PARAM_INVALID) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // choose compute function depend on dataType
  auto data_type = static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  switch (data_type) {
    case DT_FLOAT16:
      return KlDivLossGradCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return KlDivLossGradCompute<float>(ctx);
    case DT_DOUBLE:
      return KlDivLossGradCompute<double>(ctx);
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].", ctx.GetOpType().c_str(),
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t KlDivLossGradCpuKernel::KlDivLossGradCompute(CpuKernelContext &ctx) {
  int64_t grad_total = ctx.Input(0)->NumElements();
  int64_t input_total = ctx.Input(1)->NumElements();
  int64_t target_total = ctx.Input(2)->NumElements();
  int64_t output_y_total = ctx.Output(0)->NumElements();
  int64_t total = input_total;
  uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  T *grad = (T *)(ctx.Input(0)->GetData());
  T *input = (T *)(ctx.Input(1)->GetData());
  T *target = (T *)(ctx.Input(2)->GetData());
  T *output = (T *)(ctx.Output(0)->GetData());
  bool parallel_flag = false;
  uint64_t data_size = ctx.Input(1)->GetDataSize();
  // Determine whether to enable multi-core parallel computing
  if (data_size > DataDefaultParallelNum * sizeof(T)) {
    parallel_flag = true;
  }
  // Eigen::Array
  bool log_target{false};
  if (ctx.GetAttr(AttrLog) != nullptr) {
    log_target = ctx.GetAttr(AttrLog)->GetBool();
  }
  std::string reduction{"mean"};
  if (ctx.GetAttr(AttrReduction) != nullptr) {
    reduction = ctx.GetAttr(AttrReduction)->GetString();
  }
  if (cores == 0) {
    KERNEL_LOG_ERROR("KlDivLossGrad compute failed.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (parallel_flag) {
    const auto ParallelFor = aicpu::CpuKernelUtils::ParallelFor;
    std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
    auto shard_kldivlossgrad = [&](std::int64_t begin, std::int64_t end) {
      std::int64_t length = end - begin;
      std::int64_t grad_begin{0}, grad_length{grad_total};
      if (reduction == "none") {
        grad_begin = begin;
        grad_length = length;
      }
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_grad(grad + grad_begin, grad_length, 1);
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_input(input + begin, length, 1);
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_target(target + begin, length, 1);
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_output(output + begin, length, 1);
      T constant_zero{0};
      array_output = constant_zero;
      KlDivLossGradOp<T>(array_target, array_grad, array_output, length, log_target, reduction);
      if (reduction == "mean") {
        array_output = array_output / T(output_y_total);
      } else if (reduction == "batchmean") {
        std::vector<int64_t> input_dims = ctx.Input(1)->GetTensorShape()->GetDimSizes();
        array_output = array_output / T(input_dims[0]);
      }
    };
    KERNEL_HANDLE_ERROR(ParallelFor(ctx, total, per_unit_size, shard_kldivlossgrad), "KlDivLossGrad Compute failed.");
  } else {
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_grad(grad, grad_total, 1);
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_input(input, input_total, 1);
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_target(target, target_total, 1);
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_output(output, output_y_total, 1);
    T constant_zero{0};
    array_output = constant_zero;
    KlDivLossGradOp<T>(array_target, array_grad, array_output, output_y_total, log_target, reduction);
    if (reduction == "mean") {
      array_output = array_output / T(output_y_total);
    } else if (reduction == "batchmean") {
      std::vector<int64_t> input_dims = ctx.Input(1)->GetTensorShape()->GetDimSizes();
      array_output = array_output / T(input_dims[0]);
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kKlDivLossGrad, KlDivLossGradCpuKernel);
}  // namespace aicpu
