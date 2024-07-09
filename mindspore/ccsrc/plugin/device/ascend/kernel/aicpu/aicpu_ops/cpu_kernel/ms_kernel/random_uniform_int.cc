/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "ms_kernel/random_uniform_int.h"
#include <random>
#include "random/utils.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kRandomUniformIntInputNum = 3;
constexpr uint32_t kRandomUniformIntOutputNum = 1;
constexpr uint32_t kMinIdx = 1;
constexpr uint32_t kMaxIdx = 2;
const char *kRandomUniformInt = "RandomUniformInt";
}  // namespace

namespace aicpu {
uint32_t RandomUniformIntCpuKernel::Compute(CpuKernelContext &ctx) {
  NormalCheck(ctx, kRandomUniformIntInputNum, kRandomUniformIntOutputNum);
  auto output = ctx.Output(0);
  auto output_data = reinterpret_cast<int32_t *>(output->GetData());
  auto output_dtype = output->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, output_dtype == DT_INT32, KERNEL_STATUS_INNER_ERROR,
                          "Output only support data type int32_t, but got [%s].", DTypeStr(output_dtype).c_str());
  size_t output_size = output->GetDataSize() / sizeof(int32_t);

  auto input_min = ctx.Input(kMinIdx);
  auto input_max = ctx.Input(kMaxIdx);
  CUST_KERNEL_CHECK_FALSE(ctx, (input_min->GetDataType() == DT_INT32 && input_max->GetDataType() == DT_INT32),
                          KERNEL_STATUS_INNER_ERROR, "Input 'min' and 'max' only support dtype int32_t.");
  int min = *reinterpret_cast<int32_t *>(input_min->GetData());
  int max = *reinterpret_cast<int32_t *>(input_max->GetData());

  uint64_t rng_seed = std::random_device()();
  std::mt19937 rng;
  rng.seed(rng_seed);

  for (uint64_t i = 0; i < output_size; ++i) {
    std::uniform_int_distribution<int32_t> uni_int(min, max - 1);
    output_data[i] = uni_int(rng);
  }
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kRandomUniformInt, RandomUniformIntCpuKernel);
}  // namespace aicpu
