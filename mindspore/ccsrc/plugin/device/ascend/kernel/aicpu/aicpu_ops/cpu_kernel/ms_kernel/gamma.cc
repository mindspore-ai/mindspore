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
#include "ms_kernel/gamma.h"
#include <random>
#include "random/utils.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kGammaInputNum = 5;
constexpr uint32_t kGammaOutputNum = 1;
constexpr uint32_t kAlphaIdx = 1;
constexpr uint32_t kBetaIdx = 2;
const char *kGamma = "Gamma";
}  // namespace

namespace aicpu {
uint32_t GammaCpuKernel::Compute(CpuKernelContext &ctx) {
  NormalCheck(ctx, kGammaInputNum, kGammaOutputNum);
  auto output = ctx.Output(0);
  auto output_dtype = output->GetDataType();
  auto output_shape = output->GetTensorShape()->GetDimSizes();
  auto output_rank = output_shape.size();
  CUST_KERNEL_CHECK_FALSE(ctx, (output_dtype == DT_FLOAT), KERNEL_STATUS_INNER_ERROR,
                          "Output only support data type float, but got [%s].", DTypeStr(output_dtype).c_str());
  auto output_data = static_cast<float *>(output->GetData());
  size_t output_size = output->GetDataSize() / sizeof(output_dtype);

  auto input_alpha = ctx.Input(kAlphaIdx);
  auto input_beta = ctx.Input(kBetaIdx);
  CUST_KERNEL_CHECK_FALSE(ctx, (input_alpha->GetDataType() == DT_FLOAT && input_beta->GetDataType() == DT_FLOAT),
                          KERNEL_STATUS_INNER_ERROR,
                          "Input 'alpha' and 'beta' only support data type float, but got [%s].");
  auto alpha_shape = input_alpha->GetTensorShape()->GetDimSizes();
  auto beta_shape = input_beta->GetTensorShape()->GetDimSizes();
  size_t alpha_size = input_alpha->GetDataSize() / sizeof(float);
  size_t beta_size = input_beta->GetDataSize() / sizeof(float);
  auto alpha_data = static_cast<float *>(input_alpha->GetData());
  auto beta_data = static_cast<float *>(input_beta->GetData());

  for (size_t i = 0; i < (output_rank - alpha_shape.size()); ++i) {
    alpha_shape.insert(alpha_shape.begin(), 1);
  }
  for (size_t i = 0; i < (output_rank - beta_shape.size()); ++i) {
    beta_shape.insert(beta_shape.begin(), 1);
  }

  int64_t remainder = output_size;
  int64_t alpha_remainder = alpha_size;
  int64_t beta_remainder = beta_size;
  std::vector<int64_t> a_idx(output_size);
  std::vector<int64_t> b_idx(output_size);
  for (size_t i = 0; i < output_shape.size(); ++i) {
    int64_t alpha_pos = 0;
    int64_t beta_pos = 0;
    remainder = remainder / output_shape[i];
    alpha_remainder = alpha_remainder / alpha_shape[i];
    beta_remainder = beta_remainder / beta_shape[i];
    for (size_t j = 0; j < output_size; ++j) {
      a_idx[j] += alpha_pos * alpha_remainder;
      b_idx[j] += beta_pos * beta_remainder;
      if ((j + 1) % remainder == 0) {
        ++alpha_pos;
        ++beta_pos;
        if (alpha_pos == alpha_shape[i]) {
          alpha_pos = 0;
        }
        if (beta_pos == beta_shape[i]) {
          beta_pos = 0;
        }
      }
    }
  }
  int64_t seed = std::random_device()();
  std::mt19937 gen(seed);
  for (size_t i = 0; i < output_size; ++i) {
    std::gamma_distribution<float> gamma(alpha_data[a_idx[i]], beta_data[b_idx[i]]);
    output_data[i] = gamma(gen);
  }
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kGamma, GammaCpuKernel);
}  // namespace aicpu
