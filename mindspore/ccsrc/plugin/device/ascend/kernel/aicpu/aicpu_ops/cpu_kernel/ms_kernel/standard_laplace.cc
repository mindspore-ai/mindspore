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
#include "ms_kernel/standard_laplace.h"
#include <random>
#include <complex>
#include <algorithm>
#include "context/inc/cpu_kernel_utils.h"
#include "random/utils.h"

namespace {
const char *kStandardLaplace = "StandardLaplace";
constexpr size_t kStandardLaplaceInputNum = 1;
constexpr size_t kStandardLaplaceOutputNum = 1;
}  // namespace

namespace aicpu {
uint32_t StandardLaplaceCpuKernel::Compute(CpuKernelContext &ctx) {
  NormalCheck(ctx, kStandardLaplaceInputNum, kStandardLaplaceOutputNum);
  auto output_dtype = ctx.Output(0)->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, output_dtype == DT_FLOAT, KERNEL_STATUS_INNER_ERROR,
                          "Output dtype should be float, but got %s.", DTypeStr(output_dtype).c_str());
  auto output_data = static_cast<float *>(ctx.Output(0)->GetData());
  auto output_size = ctx.Output(0)->GetDataSize() / sizeof(float);
  std::mt19937 rng;
  uint64_t rngseed = std::random_device()();
  rng.seed(rngseed);

  std::exponential_distribution<float> expo;
  std::uniform_real_distribution<float> uni;
  for (uint64_t i = 0; i < output_size; ++i) {
    float expo_random = expo(rng);
    float uni_random = uni(rng);
    // Laplace probability is 0.5.
    if (uni_random < 0.5) expo_random = -expo_random;
    output_data[i] = expo_random;
  }

  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kStandardLaplace, StandardLaplaceCpuKernel);
}  // namespace aicpu
