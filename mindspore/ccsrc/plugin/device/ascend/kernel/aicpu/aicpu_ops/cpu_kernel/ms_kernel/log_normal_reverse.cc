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

#include "log_normal_reverse.h"
#include <random>
#include <set>
#include "cpu_kernel_utils.h"
#include "cpu_ops_kernel.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include <ctime>
#include <iostream>

#include "Eigen/Core"
using namespace std;
using namespace Eigen;

namespace {
const uint32_t kNumInput = 1;
const uint32_t kNumOutput = 1;

const char *kLogNormalReverse = "LogNormalReverse";
const int64_t kParallelDataNumSameShape = 16 * 1024;
const int64_t kParallelDataNumMid = 128 * 1024;
}  // namespace
namespace aicpu {
uint32_t LogNormalReverseCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kNumInput, kNumOutput), "LogNormalReverse check input and output failed.");
  // get and check input
  Tensor *input = ctx.Input(0);
  inputs_.push_back(input);

  // get output Tensors
  Tensor *output = ctx.Output(0);
  outputs_.push_back(output);

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t LogNormalReverseCpuKernel::DoCompute(CpuKernelContext &ctx) {
  float input_mean = 1.0;
  float input_std = 2.0;

  auto mean_value = ctx.GetAttr("mean");
  auto std_value = ctx.GetAttr("std");

  if (mean_value != nullptr) {
    input_mean = mean_value->GetFloat();
  }
  if (std_value != nullptr) {
    input_std = std_value->GetFloat();
  }

  T *output_y = reinterpret_cast<T *>(outputs_[0]->GetData());

  static default_random_engine random_engine(time(0));
  static std::normal_distribution<float> normal_value(input_mean, input_std);

  int64_t Nums = inputs_[0]->GetTensorShape()->NumElements();

  int64_t data_num = Nums;
  if (data_num >= kParallelDataNumSameShape) {
    uint32_t max_core_num = std::max(1U, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);
    }
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto shared_lognormalreverse = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        output_y[i] = static_cast<T>(std::exp(normal_value(random_engine)));
      }
    };

    if (max_core_num == 0) {
      max_core_num = 1;
    }
    CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shared_lognormalreverse);
  } else {
    for (int64_t i = 0; i < Nums; i++) {
      output_y[i] = static_cast<T>(std::exp(normal_value(random_engine)));
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t LogNormalReverseCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case (DT_FLOAT16): {
      DoCompute<Eigen::half>(ctx);
      break;
    }
    case (DT_FLOAT): {
      DoCompute<float>(ctx);
      break;
    }
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].", ctx.GetOpType().c_str(),
                       DTypeStr(input_type).c_str());
      res = KERNEL_STATUS_PARAM_INVALID;
  }
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("log normal reverse failed");
    return res;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kLogNormalReverse, LogNormalReverseCpuKernel);
}  // namespace aicpu
