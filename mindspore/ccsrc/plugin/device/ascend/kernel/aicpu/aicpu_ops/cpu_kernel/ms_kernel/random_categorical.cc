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
#include "ms_kernel/random_categorical.h"
#include <random>
#include <complex>
#include <algorithm>
#include "context/inc/cpu_kernel_utils.h"
#include "random/utils.h"

namespace {
const char *kRandomCategorical = "RandomCategorical";
constexpr size_t kRandomCategoricalInputNum = 3;
constexpr size_t kRandomCategoricalOutputNum = 1;
constexpr size_t kNumSamplesIdx = 1;

#define RANDOM_CATEGORICAL_COMPUTE_CASE(INPUT_DTYPE, INPUT_TYPE, OUTPUT_DTYPE, OUTPUT_TYPE) \
  case (INPUT_DTYPE + (OUTPUT_DTYPE << sizeof(DataType))): {                                \
    RandomCategoricalCompute<INPUT_TYPE, OUTPUT_TYPE>(ctx);                                 \
    break;                                                                                  \
  }
}  // namespace

namespace aicpu {
template <typename T, typename S>
uint32_t RandomCategoricalCpuKernel::RandomCategoricalCompute(CpuKernelContext &ctx) {
  auto logits_data = ctx.Input(0)->GetData();
  Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> input_map(static_cast<T *>(logits_data), batch_size_,
                                                                   num_classes_);
  const auto &logits = Eigen::Tensor<T, 2, Eigen::RowMajor>(input_map);
  int num_samples;
  auto num_samples_data = ctx.Input(kNumSamplesIdx)->GetData();
  if (num_sample_type_ == DT_INT64) {
    num_samples = static_cast<int>(*reinterpret_cast<int64_t *>(num_samples_data));
  } else {
    num_samples = static_cast<int>(*reinterpret_cast<int32_t *>(num_samples_data));
  }

  auto output_data = ctx.Output(0)->GetData();
  Eigen::TensorMap<Eigen::Tensor<S, 2, Eigen::RowMajor>> output(reinterpret_cast<S *>(output_data), batch_size_,
                                                                num_samples);

  int64_t limit_row = batch_size_;
  int64_t start_row = 0;
  Eigen::Tensor<T, 1, Eigen::RowMajor> cdf(num_classes_);
  for (int64_t b = start_row; b < limit_row; ++b) {
    const auto *logits_row = &logits(b, 0);

    T max = std::numeric_limits<T>::lowest();
    for (int64_t j = 0; j < num_classes_; ++j) {
      if (Eigen::numext::isfinite(logits_row[j])) {
        max = std::max(max, logits_row[j]);
      }
    }
    const T max_logit = static_cast<T>(max);
    cdf = (logits.chip(b, 0) - max_logit).exp();

    T running_total = static_cast<T>(0);
    for (int64_t j = 0; j < num_classes_; ++j) {
      if (Eigen::numext::isfinite(logits_row[j])) {
        running_total += cdf(j);
      }
      cdf(j) = running_total;
    }

    const T *cdf_begin = cdf.data();
    const T *cdf_end = cdf.data() + num_classes_;

    std::default_random_engine rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<> dist(0, 1);

    for (int64_t j = 0; j < num_samples; ++j) {
      const double to_find = dist(rng) * running_total;
      auto found_iter = std::upper_bound(cdf_begin, cdf_end, to_find);
      output(b, j) = static_cast<S>(std::distance(cdf_begin, found_iter));
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t RandomCategoricalCpuKernel::Compute(CpuKernelContext &ctx) {
  NormalCheck(ctx, kRandomCategoricalInputNum, kRandomCategoricalOutputNum);
  auto output = ctx.Output(0);
  output_type_ = output->GetDataType();

  auto logits = ctx.Input(0);
  auto logits_shape_ = logits->GetTensorShape()->GetDimSizes();
  CUST_KERNEL_CHECK_FALSE(ctx, logits_shape_.size() == 2, KERNEL_STATUS_INNER_ERROR,
                          "Logits must be 2D tensor, but got dim[%u].", logits_shape_.size());
  batch_size_ = logits_shape_[0];
  num_classes_ = logits_shape_[1];
  logits_type_ = logits->GetDataType();
  num_sample_type_ = ctx.Input(1)->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, num_sample_type_ == DT_INT32 || num_sample_type_ == DT_INT64, KERNEL_STATUS_INNER_ERROR,
                          "num_samples only supports int32_t and int64_t.");
  auto logits_dtype = logits->GetDataType();

  switch (logits_dtype + (output_type_ << sizeof(DataType))) {
    RANDOM_CATEGORICAL_COMPUTE_CASE(DT_FLOAT16, Eigen::half, DT_INT16, int16_t)
    RANDOM_CATEGORICAL_COMPUTE_CASE(DT_FLOAT, float, DT_INT16, int16_t)
    RANDOM_CATEGORICAL_COMPUTE_CASE(DT_DOUBLE, double, DT_INT16, int16_t)
    RANDOM_CATEGORICAL_COMPUTE_CASE(DT_FLOAT16, Eigen::half, DT_INT32, int32_t)
    RANDOM_CATEGORICAL_COMPUTE_CASE(DT_FLOAT, float, DT_INT32, int32_t)
    RANDOM_CATEGORICAL_COMPUTE_CASE(DT_DOUBLE, double, DT_INT32, int32_t)
    RANDOM_CATEGORICAL_COMPUTE_CASE(DT_FLOAT16, Eigen::half, DT_INT64, int64_t)
    RANDOM_CATEGORICAL_COMPUTE_CASE(DT_FLOAT, float, DT_INT64, int64_t)
    RANDOM_CATEGORICAL_COMPUTE_CASE(DT_DOUBLE, double, DT_INT64, int64_t)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "RandomCategorical kernel data type input: [%s], output: [%s] not support.",
                            DTypeStr(logits_dtype).c_str(), DTypeStr(output_type_).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kRandomCategorical, RandomCategoricalCpuKernel);
}  // namespace aicpu
