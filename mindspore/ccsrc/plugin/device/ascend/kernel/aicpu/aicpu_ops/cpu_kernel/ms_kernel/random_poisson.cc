/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "ms_kernel/random_poisson.h"
#include <random>
#include "random/utils.h"

namespace {
const uint32_t kCountsIndex = 2;
const uint32_t kStatesIndex = 3;
const char *kRandomPoisson = "RandomPoisson";

#define RANDOM_POISSON_GENERATE_CASE(DTYPE, TYPE) \
  case (DTYPE): {                                 \
    Generate<TYPE>(ctx, output);                  \
    break;                                        \
  }

#define RANDOM_POISSON_EIGEN_TENSOR_ASSIGN_CASE(ALIGNMENT_TYPE)                                                \
  Eigen::TensorMap<Eigen::Tensor<T, 1>, ALIGNMENT_TYPE> eigen_output(static_cast<T *>(output->GetData()),      \
                                                                     output->GetTensorShape()->NumElements()); \
  PoissonRandomGenerator<T> m_generator(static_cast<double>(rate_flat[0]));                                    \
  for (int i = 0; i < num_of_rate; i++) {                                                                      \
    m_generator.setRate(static_cast<double>(rate_flat[i]));                                                    \
    for (int j = i; j < num_of_output; j += num_of_rate) {                                                     \
      eigen_output(j) = m_generator.gen(rng_);                                                                 \
    }                                                                                                          \
  }
}  // namespace

namespace aicpu {
uint32_t RandomPoissonCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *input_0 = ctx.Input(kFirstInputIndex);
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_0, KERNEL_STATUS_PARAM_INVALID, "Get input_0 shape failed")

  Tensor *input_1 = ctx.Input(kSecondInputIndex);
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_1, KERNEL_STATUS_PARAM_INVALID, "Get input_1 rate failed")

  Tensor *output = ctx.Output(kFirstOutputIndex);
  CUST_KERNEL_CHECK_NULLPTR(ctx, output, KERNEL_STATUS_PARAM_INVALID, "Get output failed")
  auto data_type = static_cast<DataType>(output->GetDataType());

  AttrValue *seed_ptr = ctx.GetAttr("seed");
  AttrValue *seed2_ptr = ctx.GetAttr("seed2");
  seed_ = (seed_ptr == nullptr) ? static_cast<uint64_t>(0) : static_cast<uint64_t>(seed_ptr->GetInt());
  seed2_ = (seed2_ptr == nullptr) ? static_cast<uint64_t>(0) : static_cast<uint64_t>(seed2_ptr->GetInt());

  switch (data_type) {
    RANDOM_POISSON_GENERATE_CASE(DT_FLOAT16, Eigen::half)
    RANDOM_POISSON_GENERATE_CASE(DT_FLOAT, float)
    RANDOM_POISSON_GENERATE_CASE(DT_DOUBLE, double)
    RANDOM_POISSON_GENERATE_CASE(DT_INT32, int)
    RANDOM_POISSON_GENERATE_CASE(DT_INT64, int64_t)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "RandomPoisson kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t RandomPoissonCpuKernel::Generate(CpuKernelContext &ctx, Tensor *output) {
  // reset the state of ops
  /*
  uint32_t kernel_ret = 0;
  uint64_t rng_seed =
    random::GetCpuKernelRandomStates(ctx, kCountsIndex, kStatesIndex, seed_, seed2_, "RandomPoisson", &kernel_ret);
  if (kernel_ret != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  // use the origin method to get seed
  */
  uint64_t rng_seed = std::random_device()();
  rng_seed = PCG_XSH_RS_state(rng_seed, rng_seed);
  rng_.seed(rng_seed);

  Tensor *input_1 = ctx.Input(kSecondInputIndex);
  T *rate_flat = static_cast<T *>(input_1->GetData());
  int64_t num_of_rate = input_1->NumElements();
  int64_t num_of_output = output->GetTensorShape()->NumElements();
  if (AddrAlignedCheck(output->GetData())) {
    RANDOM_POISSON_EIGEN_TENSOR_ASSIGN_CASE(Eigen::Aligned);
  } else {
    RANDOM_POISSON_EIGEN_TENSOR_ASSIGN_CASE(Eigen::Unaligned);
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kRandomPoisson, RandomPoissonCpuKernel);
}  // namespace aicpu
