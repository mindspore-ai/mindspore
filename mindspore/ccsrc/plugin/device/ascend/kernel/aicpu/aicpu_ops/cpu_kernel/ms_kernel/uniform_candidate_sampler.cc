/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "uniform_candidate_sampler.h"
#include <algorithm>
#include <securec.h>
#include "utils/range_sampler.h"
#include "random/utils.h"

namespace aicpu {
namespace {
const char *kUniformCandidateSampler = "UniformCandidateSampler";
const char *kLogUniformCandidateSampler = "CustLogUniformCandidateSampler";
const size_t kIndex2 = 2;
}  // namespace

uint32_t CandidateSamplerKernel::ParseKernelParam(CpuKernelContext &ctx) {
  auto num_true_attr = ctx.GetAttr("num_true");
  CUST_KERNEL_CHECK_NULLPTR(ctx, num_true_attr, KERNEL_STATUS_PARAM_INVALID, "Failed to get attr 'num_true'.");
  num_true_ = num_true_attr->GetInt();

  auto num_sampled_attr = ctx.GetAttr("num_sampled");
  CUST_KERNEL_CHECK_NULLPTR(ctx, num_sampled_attr, KERNEL_STATUS_PARAM_INVALID, "Failed to get attr 'num_sampled'.");
  num_sampled_ = num_sampled_attr->GetInt();

  auto unique_attr = ctx.GetAttr("unique");
  CUST_KERNEL_CHECK_NULLPTR(ctx, unique_attr, KERNEL_STATUS_PARAM_INVALID, "Failed to get attr 'unique'.");
  unique_ = unique_attr->GetBool();

  auto range_max_attr = ctx.GetAttr("range_max");
  CUST_KERNEL_CHECK_NULLPTR(ctx, range_max_attr, KERNEL_STATUS_PARAM_INVALID, "Failed to get attr 'range_max'.");
  range_max_ = range_max_attr->GetInt();

  auto seed_attr = ctx.GetAttr("seed");
  CUST_KERNEL_CHECK_NULLPTR(ctx, seed_attr, KERNEL_STATUS_PARAM_INVALID, "Failed to get attr 'seed'.");
  seed_ = seed_attr->GetInt();

  // input0: true_classes
  auto x_tensor = ctx.Input(0);
  x_dtype_ = x_tensor->GetDataType();
  x_shape_ = x_tensor->GetTensorShape()->GetDimSizes();
  if (x_shape_.size() != kIndex2) {
    CUST_AICPU_LOGE(ctx, "true_classes must be a matrix");
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (x_shape_[1] != num_true_) {
    CUST_AICPU_LOGE(ctx,
                    "true_classes must have "
                    "num_true columns, expected: ",
                    x_shape_[1], " was: ", num_true_);
    return KERNEL_STATUS_INNER_ERROR;
  }

  batch_size_ = x_shape_.front();
  if (x_dtype_ != DT_INT64 && x_dtype_ != DT_INT32) {
    CUST_AICPU_LOGE(ctx, "invalid type of x_dtype_: %d", x_dtype_);
    return KERNEL_STATUS_INNER_ERROR;
  }

  // output_2: sampled_candidates
  auto true_expected_count_tensor = ctx.Output(1);
  true_expected_count_dtype_ = true_expected_count_tensor->GetDataType();
  if (true_expected_count_dtype_ != DT_FLOAT) {
    CUST_AICPU_LOGE(ctx, "invalid type of true_expected_count_dtype_: %d", true_expected_count_dtype_);
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CandidateSamplerKernel::DoCompute(CpuKernelContext &ctx) {
  const int64_t batch_size = x_shape_[0];
  // input
  T *true_classes = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  std::vector<T> true_candidate_raw(true_classes, true_classes + batch_size * num_true_);
  std::vector<int64_t> true_candidate(true_candidate_raw.size());
  std::transform(true_candidate_raw.begin(), true_candidate_raw.end(), true_candidate.begin(),
                 [&](T x) { return static_cast<int64_t>(x); });
  std::vector<int64_t> sampled_candidate(num_sampled_);
  std::vector<T> sampled_candidate_raw(num_sampled_);
  std::vector<float> true_expected_count(batch_size * num_true_);
  std::vector<float> sampled_expected_count(num_sampled_);

  // get random generator seed
  uint64_t rng_seed = std::random_device()();
  rng_.seed(rng_seed);

  auto op_type = ctx.GetOpType();
  if (op_type == kUniformCandidateSampler) {
    set_sampler(new UniformSampler(range_max_));
  } else if (op_type == kLogUniformCandidateSampler) {
    set_sampler(new LogUniformSampler(range_max_));
  } else {
    CUST_KERNEL_LOG_ERROR(
      ctx, "CandidateSampler kernel only support op_type 'UniformSampler' and 'LogUniformSampler', but got %s.",
      op_type.c_str());
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (unique_ && num_sampled_ > sampler_->range()) {
    CUST_AICPU_LOGE(ctx, "For AICPU ops ", kernel_name_, ", the sampler's range is too small.");
    return KERNEL_STATUS_INNER_ERROR;
  }

  sampler_->SampleBatchGetExpectedCount(ctx, unique_, rng_seed, &sampled_candidate, &sampled_expected_count,
                                        true_candidate, &true_expected_count);

  std::transform(sampled_candidate.begin(), sampled_candidate.end(), sampled_candidate_raw.begin(),
                 [&](int64_t x) { return static_cast<T>(x); });
  int true_count_size = batch_size * num_true_ * sizeof(float);
  int ret1 =
    memcpy_s(reinterpret_cast<void *>(ctx.Output(0)->GetData()), num_sampled_ * sizeof(T),
             reinterpret_cast<void *>(&sampled_candidate_raw.front()), sampled_candidate_raw.size() * sizeof(T));
  int ret2 = memcpy_s(reinterpret_cast<void *>(ctx.Output(1)->GetData()), true_count_size,
                      reinterpret_cast<void *>(&true_expected_count.front()), true_count_size);
  int ret3 =
    memcpy_s(reinterpret_cast<void *>(ctx.Output(kIndex2)->GetData()), num_sampled_ * sizeof(float),
             reinterpret_cast<void *>(&sampled_expected_count.front()), sampled_expected_count.size() * sizeof(float));
  if (ret1 != EOK || ret2 != EOK || ret3 != EOK) {
    CUST_KERNEL_LOG_ERROR(ctx, "For 'CandidateSampler', memcpy_s failed.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

uint32_t CandidateSamplerKernel::Compute(CpuKernelContext &ctx) {
  RETURN_IF_FAILURE(ParseKernelParam(ctx));
  switch (x_dtype_) {
    case DT_INT32: {
      DoCompute<int>(ctx);
      break;
    }
    case DT_INT64: {
      DoCompute<int64_t>(ctx);
      break;
    }
    default: {
      CUST_AICPU_LOGE(ctx, "CandidateSampler op doesn't support input tensor types.");
      return KERNEL_STATUS_INNER_ERROR;
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kUniformCandidateSampler, CandidateSamplerKernel);
REGISTER_MS_CPU_KERNEL(kLogUniformCandidateSampler, CandidateSamplerKernel);
}  // namespace aicpu