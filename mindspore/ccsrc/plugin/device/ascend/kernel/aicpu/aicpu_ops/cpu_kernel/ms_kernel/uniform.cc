/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
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

#include "uniform.h"
#include <algorithm>
#include "random/philox_random_dist.h"
#include "random/random_distributions.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "random/utils.h"
#include "utils/philox_random.h"

namespace {
const char *const kUniform = "Uniform";
const int32_t kInputIndex = 0;
const uint32_t kCountsIndex = 1;
const uint32_t kStatesIndex = 2;
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
}  // namespace

namespace aicpu {
uint32_t UniformCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *inputTensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(inputTensor, KERNEL_STATUS_PARAM_INVALID, "Get input failed")
  Tensor *outputTensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(outputTensor, KERNEL_STATUS_PARAM_INVALID, "Get output failed")

  auto inputDataType = inputTensor->GetDataType();
  uint32_t status = KERNEL_STATUS_OK;

  // choose random data generate function depend on dataType
  switch (inputDataType) {
    case DT_FLOAT16:
      status = DoCompute<Eigen::half>(ctx, inputTensor, outputTensor);
      break;
    case DT_FLOAT:
      status = DoCompute<float>(ctx, inputTensor, outputTensor);
      break;
    case DT_DOUBLE:
      status = DoCompute<double>(ctx, inputTensor, outputTensor);
      break;
    default:
      KERNEL_LOG_ERROR("Uniform kernel data type [%u] not support.", inputDataType);
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return status;
}

template <typename T>
uint32_t UniformCpuKernel::DoCompute(const CpuKernelContext &ctx, Tensor *input, Tensor *output) {
  T *inputData = reinterpret_cast<T *>(input->GetData());
  T *outputData = reinterpret_cast<T *>(output->GetData());

  // get attrs
  AttrValue *from_ptr = ctx.GetAttr("from");
  AttrValue *to_ptr = ctx.GetAttr("to");
  AttrValue *seed_ptr = ctx.GetAttr("seed");
  AttrValue *offset_ptr = ctx.GetAttr("offset");
  float from = (from_ptr == nullptr) ? 0.0 : from_ptr->GetFloat();
  float to = (to_ptr == nullptr) ? 1.0 : to_ptr->GetFloat();
  KERNEL_CHECK_FALSE((from <= to), KERNEL_STATUS_PARAM_INVALID,
                     "the value of from[%f] must less or equal to the value of to[%f]", from, to);
  uint64_t seed = (seed_ptr == nullptr) ? static_cast<uint64_t>(0) : static_cast<uint64_t>(seed_ptr->GetInt());
  uint64_t offset = (offset_ptr == nullptr) ? static_cast<uint64_t>(0) : static_cast<uint64_t>(offset_ptr->GetInt());

  // get random generator seed
  uint32_t kernel_ret = 0;
  uint64_t rng_seed =
    random::GetCpuKernelRandomStates(ctx, kCountsIndex, kStatesIndex, seed, offset, "Uniform", &kernel_ret);
  if (kernel_ret != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_INNER_ERROR;
  }

  // start compute
  const int64_t parallelLimit = 7 * 1024;
  int64_t input_size = input->NumElements();
  int64_t output_size = output->NumElements();
  random::PhiloxRandomDist<random::UniformDistribution<random::PhiloxRandom, T>> philoxRandomDist(rng_seed, offset,
                                                                                                  parallelLimit);
  philoxRandomDist.DistCompute(ctx, inputData, outputData, input_size, output_size);
  ParaCompute<T>(ctx, input_size, outputData, from, to);

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t UniformCpuKernel::ParaCompute(const CpuKernelContext &ctx, int64_t input_size, T *outputData, float from,
                                       float to) {
  if (input_size >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (input_size <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, static_cast<int64_t>(4));  // up to 4 cpu cores
    }

    if (max_core_num > input_size) {
      max_core_num = input_size;
    }

    auto sharder_uniform = [&](int64_t start, int64_t end) { UniformCompute<T>(from, to, start, end, outputData); };
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }

    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, input_size, input_size / max_core_num, sharder_uniform),
                        "Uniform Compute failed.");
  } else {
    UniformCompute<T>(from, to, 0, input_size, outputData);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void UniformCpuKernel::UniformCompute(float from, float to, int64_t start, int64_t end, T *outputData) {
  for (int i = start; i < end; i++) {
    double random = static_cast<double>(outputData[i]);
    double temp = from + (to - from) * random;
    outputData[i] = static_cast<T>(temp);
  }
}

REGISTER_CPU_KERNEL(kUniform, UniformCpuKernel);
}  // namespace aicpu