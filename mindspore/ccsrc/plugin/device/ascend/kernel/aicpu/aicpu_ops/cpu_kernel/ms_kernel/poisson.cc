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
#include "poisson.h"

#include <stdlib.h>
#include <memory.h>
#include <cfloat>
#include <random>
#include <securec.h>

#include "context/inc/cpu_kernel_utils.h"
#include "random/utils.h"
#include "inc/kernel_log.h"
#include "utils/kernel_util.h"

namespace aicpu {
namespace {
const uint32_t kPoissonShapeIndex = 0;
const uint32_t kPoissonMeanIndex = 1;
const uint32_t kPoissonSeedIndex = 2;
const uint32_t kPoissonSeed2Index = 3;
const uint32_t kPoissonOutputIndex = 4;
const char *kPoisson = "Poisson";

static uint64_t PoissonInferOutputShape(std::vector<uint64_t> *out_shape, std::vector<uint64_t> *shape,
                                        std::vector<uint64_t> *mean_shape, uint64_t *count) {
  uint64_t size = (*shape).size() > (*mean_shape).size() ? (*shape).size() : (*mean_shape).size();
  random::NormalizeShape(shape, size);
  random::NormalizeShape(mean_shape, size);
  for (uint64_t i = 0; i < size; ++i) {
    uint64_t shape_n = (*shape)[i] > (*mean_shape)[i] ? (*shape)[i] : (*mean_shape)[i];
    if (((*shape)[i] != 1 && (*shape)[i] != shape_n) || ((*mean_shape)[i] != 1 && (*mean_shape)[i] != shape_n)) {
      return KERNEL_STATUS_INNER_ERROR;
    }
    (*out_shape).push_back(shape_n);
    (*count) *= shape_n;
  }
  return KERNEL_STATUS_OK;
}
}  // namespace

uint32_t PoissonKernel::Compute(CpuKernelContext &ctx) {
  RETURN_IF_FAILURE(ParseKernelParam(ctx));
  int *tmp_out;
  if (out_count_ > 0) {
    tmp_out = static_cast<int *>(malloc(out_count_ * sizeof(int)));
    if (tmp_out == nullptr) {
      return KERNEL_STATUS_INNER_ERROR;
    }
  } else {
    return KERNEL_STATUS_INNER_ERROR;
  }

  uint64_t remainder = out_count_;
  uint64_t mean_remainder = mean_count_;
  std::vector<uint64_t> m_idx(out_count_);
  for (uint64_t i = 0; i < out_shape.size(); ++i) {
    uint64_t mean_pos = 0;
    remainder = remainder / out_shape[i];
    mean_remainder = mean_remainder / mean_shape[i];
    for (uint64_t j = 0; j < out_count_; ++j) {
      m_idx[j] += mean_pos * mean_remainder;
      if ((j + 1) % remainder == 0) {
        ++mean_pos;
        if (mean_pos == mean_shape[i]) {
          mean_pos = 0;
        }
      }
    }
  }

  uint64_t RNG_seed = random::GetSeed(static_cast<uint64_t>(*seed_), static_cast<uint64_t>(*seed2_));
  std::mt19937 gen(RNG_seed);
  for (uint64_t i = 0; i < out_count_; ++i) {
    std::poisson_distribution<> poisson(mean_[m_idx[i]]);
    tmp_out[i] = poisson(gen);
  }

  int ret = memcpy_s(ctx.Output(0)->GetData(), out_count_ * sizeof(int), tmp_out, out_count_ * sizeof(int));
  free(tmp_out);
  tmp_out = nullptr;
  if (ret < 0) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

uint32_t PoissonKernel::ParseKernelParam(CpuKernelContext &ctx) {
  auto seed_ptr = ctx.Input(kPoissonSeedIndex);
  auto seed2_ptr = ctx.Input(kPoissonSeed2Index);
  CUST_KERNEL_CHECK_FALSE(ctx, (seed_ptr->GetDataType() == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                          "'seed' only support int64_t input, but got [%s].",
                          DTypeStr(seed_ptr->GetDataType()).c_str());
  CUST_KERNEL_CHECK_FALSE(ctx, (seed2_ptr->GetDataType() == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                          "'seed2' only support int64_t input, but got [%s].",
                          DTypeStr(seed2_ptr->GetDataType()).c_str());
  seed_ = reinterpret_cast<int64_t *>(seed_ptr->GetData());
  seed2_ = reinterpret_cast<int64_t *>(seed2_ptr->GetData());

  // get inputs
  // shape of one batch
  auto input_shape_ptr = ctx.Input(kPoissonShapeIndex);
  const auto &shape_dt = input_shape_ptr->GetDataType();
  auto input_shape_size = input_shape_ptr->NumElements();
  if (shape_dt == DT_INT32) {
    auto input0 = reinterpret_cast<int32_t *>(input_shape_ptr->GetData());
    for (int64_t index = 0; index < input_shape_size; index++) {
      shape.push_back(input0[index]);
    }
  } else if (shape_dt == DT_INT64) {
    auto input0 = reinterpret_cast<int64_t *>(input_shape_ptr->GetData());
    for (int64_t index = 0; index < input_shape_size; index++) {
      shape.push_back(input0[index]);
    }
  } else {
    CUST_KERNEL_LOG_ERROR(ctx, "Input 'shape' only support int32_t and int64_t, but got [%s].",
                          DTypeStr(shape_dt).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // mean
  auto mean_tensor = ctx.Input(kPoissonMeanIndex);
  CUST_KERNEL_CHECK_FALSE(ctx, (mean_tensor->GetDataType() == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                          "Input 'mean' only support float, but got [%s].",
                          DTypeStr(mean_tensor->GetDataType()).c_str());
  mean_ = reinterpret_cast<float *>(mean_tensor->GetData());
  auto mean_tshape = mean_tensor->GetTensorShape()->GetDimSizes();
  mean_count_ = mean_tensor->NumElements();
  std::transform(mean_tshape.begin(), mean_tshape.end(), std::back_inserter(mean_shape),
                 [](int64_t x) { return static_cast<uint64_t>(x); });
  RETURN_IF_FAILURE(PoissonInferOutputShape(&out_shape, &shape, &mean_shape, &out_count_));
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kPoisson, PoissonKernel);
}  // namespace aicpu