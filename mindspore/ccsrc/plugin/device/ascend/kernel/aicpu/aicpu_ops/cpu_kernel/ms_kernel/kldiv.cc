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

#include "kldiv.h"

#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kKLDivInputNum{2};
const std::uint32_t kKLDivOutputNum{1};
const std::int64_t ParallelNum{4096};
const char *kKLDiv{"KLDiv"};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline std::uint32_t ComputeKLDivKernel(const CpuKernelContext &ctx) {
  const auto ParallelFor = aicpu::CpuKernelUtils::ParallelFor;
  auto input = static_cast<T *>(ctx.Input(0)->GetData());
  auto target = static_cast<T *>(ctx.Input(1)->GetData());
  auto output = static_cast<T *>(ctx.Output(0)->GetData());
  std::int64_t total = ctx.Input(0)->NumElements();
  std::size_t data_size = ctx.Input(0)->GetDataSize();
  uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  std::string reduction = ctx.GetAttr("reduction")->GetString();
  if (reduction != "sum" && reduction != "batchmean" && reduction != "none" && reduction != "mean") {
    KERNEL_LOG_ERROR("%s is not a valid value for reduction", reduction.c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  bool parallel_flag = false;
  if (data_size > ParallelNum * sizeof(T)) {
    parallel_flag = true;
  }
  if (cores == 0) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  T *tmp_array = nullptr;
  if (reduction == "none") {
    tmp_array = output;
  } else {
    tmp_array = new T[total];
  }
  if (parallel_flag) {
    std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
    ParallelFor(ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
      std::int64_t length = end - begin;
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_input(input + begin, length, 1);
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_target(target + begin, length, 1);
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_reduce(tmp_array + begin, length, 1);
      T constant_zero{0};
      array_reduce = array_target * (Eigen::log(array_target) - array_input);
      for (std::int64_t idx = 0; idx < length; ++idx) {
        if (!(target[begin + idx] > constant_zero)) {
          array_reduce(idx) = constant_zero;
        }
      }
    });
  } else {
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_input(input, total, 1);
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_target(target, total, 1);
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > array_reduce(tmp_array, total, 1);
    array_reduce = array_target * (Eigen::log(array_target) - array_input);
    T constant_zero{0};
    for (uint32_t idx = 0; idx < total; ++idx) {
      if (!(target[idx] > constant_zero)) {
        array_reduce(idx) = constant_zero;
      }
    }
  }
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> > reduce(tmp_array, total, 1);
  if (reduction == "sum") {
    output[0] = reduce.sum();
  } else if (reduction == "batchmean") {
    std::vector<int64_t> input_dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
    output[0] = reduce.sum() / T(input_dims[0]);
  } else if (reduction == "mean") {
    output[0] = reduce.mean();
  }
  if (reduction != "none") {
    delete[] tmp_array;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeKLDiv(const CpuKernelContext &ctx) {
  uint32_t result = ComputeKLDivKernel<T>(ctx);
  if (result != 0) {
    KERNEL_LOG_ERROR("KLDiv compute failed.");
  }
  return result;
}

inline std::uint32_t KLDivExtraCheck(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get input x data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(1)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get input target data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Output(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get output y data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR("The data type of the input [%s] need be the same as the output [%s].",
                     DTypeStr(ctx.Input(0)->GetDataType()).c_str(), DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataSize() != ctx.Input(1)->GetDataSize()) {
    KERNEL_LOG_ERROR(
      "The data size of the input [%llu] need be the same as the target "
      "[%llu].",
      ctx.Input(0)->GetDataSize(), ctx.Input(1)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  std::vector<int64_t> input_dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> target_dims = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  if (input_dims.size() != target_dims.size()) {
    KERNEL_LOG_ERROR(
      "The data dim size of the input x [%llu] need be the same as the "
      "target "
      "[%llu].",
      input_dims.size(), target_dims.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t index = 0; index < input_dims.size(); index++) {
    if (input_dims[index] != target_dims[index]) {
      KERNEL_LOG_ERROR("The data dim of the input x need be the same as the target.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

std::uint32_t KLDivCheck(CpuKernelContext &ctx, uint32_t inputs_num, uint32_t outputs_num) {
  return NormalCheck(ctx, kKLDivInputNum, kKLDivOutputNum, {"reduction"}) ? KERNEL_STATUS_PARAM_INVALID
                                                                          : KLDivExtraCheck(ctx);
}
// DT_FLOAT16, DT_FLOAT, DT_DOUBLE
std::uint32_t KLDivCompute(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeKLDiv<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeKLDiv<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeKLDiv<std::double_t>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t KLDivCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::KLDivCheck(ctx, kKLDivInputNum, kKLDivOutputNum) ? KERNEL_STATUS_PARAM_INVALID
                                                                  : detail::KLDivCompute(ctx);
}

REGISTER_CPU_KERNEL(kKLDiv, KLDivCpuKernel);
}  // namespace aicpu
