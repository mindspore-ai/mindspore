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
#include "rsqrt.h"

#include <cfloat>
#include <complex>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *const kRsqrt = "Rsqrt";
const size_t kRsqrtInputNum = 1;
const size_t kRsqrtOutputNum = 1;
constexpr int64_t kParallelDataNums = 8 * 1024;
constexpr int64_t kParallelComplexDataNums = 4 * 1024;
}  // namespace

namespace aicpu {
uint32_t RsqrtCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kRsqrtOutputNum, kRsqrtInputNum), "Check Rsqrt params failed.");
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR("The data type of the input [%s] need be the same as the output [%s]",
                     DTypeStr(ctx.Input(0)->GetDataType()).c_str(), DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
    KERNEL_LOG_ERROR(
      "The data size of the input [%llu] need be the same as the output "
      "[%llu]",
      ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  const Tensor *x = ctx.Input(0);
  const Tensor *y = ctx.Output(0);
  int64_t datanum = x->NumElements();
  DataType datatype = x->GetDataType();
  uint32_t res = static_cast<uint32_t>(KERNEL_STATUS_OK);

  switch (datatype) {
    case DT_FLOAT16:
      res = RsqrtCompute<Eigen::half>(x, y, datanum, ctx);
      break;
    case DT_FLOAT:
      res = RsqrtCompute<float>(x, y, datanum, ctx);
      break;
    case DT_DOUBLE:
      res = RsqrtCompute<double>(x, y, datanum, ctx);
      break;
    case DT_COMPLEX64:
      res = RsqrtComputeComplex<std::complex<float>>(x, y, datanum, ctx);
      break;
    case DT_COMPLEX128:
      res = RsqrtComputeComplex<std::complex<double>>(x, y, datanum, ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Rsqrt invalid input type [%s]", DTypeStr(datatype).c_str());
      return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  if (res != static_cast<uint32_t>(KERNEL_STATUS_OK)) {
    return static_cast<uint32_t>(KERNEL_STATUS_INNER_ERROR);
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

template <typename T>
uint32_t RsqrtCpuKernel::RsqrtCompute(const Tensor *x, const Tensor *y, int64_t datanum,
                                      const CpuKernelContext &ctx) const {
  auto inputx = reinterpret_cast<T *>(x->GetData());
  KERNEL_CHECK_NULLPTR(inputx, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Get input data failed")
  auto outputy = reinterpret_cast<T *>(y->GetData());
  KERNEL_CHECK_NULLPTR(outputy, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Get output data failed")
  if (datanum <= kParallelDataNums) {
    for (int64_t i = 0; i < datanum; i++) {
      if (x->GetDataType() == DT_FLOAT16) {
        if ((Eigen::half)inputx[i] == Eigen::half{0.0f}) {
          KERNEL_LOG_ERROR("Rsqrt kernel input[%ld] cannot be 0", i);
          return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
        }
      } else if (x->GetDataType() == DT_FLOAT) {
        if ((std::fabs(static_cast<float>(inputx[i])) < FLT_EPSILON)) {
          KERNEL_LOG_ERROR("Rsqrt kernel input[%ld] cannot be 0", i);
          return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
        }
      } else if (x->GetDataType() == DT_DOUBLE) {
        if ((std::fabs(static_cast<double>(inputx[i])) < DBL_EPSILON)) {
          KERNEL_LOG_ERROR("Rsqrt kernel input[%ld] cannot be 0", i);
          return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
        }
      }
      outputy[i] = static_cast<T>(1) / (sqrt(inputx[i]));
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > datanum) {
      max_core_num = datanum;
    }
    auto shard_rsqrt = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        if (x->GetDataType() == DT_FLOAT16) {
          if ((Eigen::half)inputx[i] == Eigen::half{0.0f}) {
            KERNEL_LOG_ERROR("Rsqrt kernel input[%zu] cannot be 0", i);
          }
        } else if (x->GetDataType() == DT_FLOAT) {
          if ((std::fabs(static_cast<float>(inputx[i])) < FLT_EPSILON)) {
            KERNEL_LOG_ERROR("Rsqrt kernel input[%zu] cannot be 0", i);
          }
        } else if (x->GetDataType() == DT_DOUBLE) {
          if ((std::fabs(static_cast<double>(inputx[i])) < DBL_EPSILON)) {
            KERNEL_LOG_ERROR("Rsqrt kernel input[%zu] cannot be 0", i);
          }
        }
        outputy[i] = static_cast<T>(1) / (sqrt(inputx[i]));
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, datanum, datanum / max_core_num, shard_rsqrt),
                        "Rsqrt Compute failed.");
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

template <typename T>
uint32_t RsqrtCpuKernel::RsqrtComputeComplex(const Tensor *x, const Tensor *y, int64_t datanum,
                                             const CpuKernelContext &ctx) const {
  auto inputx = reinterpret_cast<T *>(x->GetData());
  KERNEL_CHECK_NULLPTR(inputx, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Get input data failed")
  auto outputy = reinterpret_cast<T *>(y->GetData());
  KERNEL_CHECK_NULLPTR(outputy, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "Get output data failed")
  if (datanum <= kParallelComplexDataNums) {
    for (int64_t i = 0; i < datanum; i++) {
      outputy[i] =
        sqrt(conj(inputx[i])) / sqrt(inputx[i].real() * inputx[i].real() + inputx[i].imag() * inputx[i].imag());
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > datanum) {
      max_core_num = datanum;
    }

    auto shard_rsqrt = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        outputy[i] =
          sqrt(conj(inputx[i])) / sqrt(inputx[i].real() * inputx[i].real() + inputx[i].imag() * inputx[i].imag());
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, datanum, datanum / max_core_num, shard_rsqrt),
                        "Rsqrt Compute failed.");
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}
REGISTER_CPU_KERNEL(kRsqrt, RsqrtCpuKernel);
}  // namespace aicpu
