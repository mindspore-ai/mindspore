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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/random_categorical_kernels.h"

#include <memory.h>
#include <time.h>
#include <cfloat>
#include <ctime>
#include <random>
#include <limits>
#include <algorithm>

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "aicpu_sharder/aicpu_sharder.h"
#include "common/kernel_errcode.h"
#include "common/kernel_log.h"
#include "common/random_utils.h"

namespace aicpu {
namespace {
const uint32_t kCountsIndex = 3;
const uint32_t kStatesIndex = 4;
}  // namespace
uint32_t RandomCategoricalKernel::DoCompute() {
  switch (input_type_) {
    case aicpuops::DataType::MS_FLOAT32: {
      float input_type = 0;
      DoComputeWithOutputType(input_type);
      break;
    }
    case aicpuops::DataType::MS_FLOAT16: {
      // Eigen::half input_type = Eigen::half(0);
      float input_type = 0;
      DoComputeWithOutputType(input_type);
      break;
    }
    case aicpuops::DataType::MS_FLOAT64: {
      double input_type = 0;
      DoComputeWithOutputType(input_type);
      break;
    }
    default: {
      AICPU_LOGE("RandomCategorical op don't support input tensor types");
      return kAicpuKernelStateInvalid;
    }
  }
  return kAicpuKernelStateSucess;
}

template <typename T>
uint32_t RandomCategoricalKernel::DoComputeWithOutputType(T input_type) {
  switch (output_type_) {
    case aicpuops::DataType::MS_INT16: {
      int16_t output_type = 0;
      RandomCategoricalKernel::DoComputeForEachType(input_type, output_type);
      break;
    }
    case aicpuops::DataType::MS_INT32: {
      int output_type = 0;
      RandomCategoricalKernel::DoComputeForEachType(input_type, output_type);
      break;
    }
    case aicpuops::DataType::MS_INT64: {
      int64_t output_type = 0;
      RandomCategoricalKernel::DoComputeForEachType(input_type, output_type);
      break;
    }
    default: {
      AICPU_LOGE("RandomCategorical op don't support output tensor types");
      return kAicpuKernelStateInvalid;
    }
  }
  return kAicpuKernelStateSucess;
}

template <typename T, typename S>
uint32_t RandomCategoricalKernel::DoComputeForEachType(T input_type, S output_type) {
  clock_t start, end;
  start = clock();

  // start categorical
  // aux params
  int64_t batch_size = batch_size_;
  int num_classes = num_classes_;
  // input
  Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> input_map(reinterpret_cast<T *>(io_addrs_[0]), batch_size,
                                                                   num_classes);
  const auto &logits = Eigen::Tensor<T, 2, Eigen::RowMajor>(input_map);
  int num_samples = *reinterpret_cast<int *>(io_addrs_[1]);
  int seed = *reinterpret_cast<int *>(io_addrs_[2]);

  // output
  Eigen::TensorMap<Eigen::Tensor<S, 2, Eigen::RowMajor>> output(reinterpret_cast<S *>(io_addrs_[5]), batch_size,
                                                                num_samples);

  // get random generator seed
  uint32_t kernel_ret = 0;
  uint64_t rng_seed = random::GetKernelBaseRandomStates(
    io_addrs_, kCountsIndex, kStatesIndex, static_cast<uint64_t>(seed), 0, "RandomCategorical", &kernel_ret);
  if (kernel_ret != kAicpuKernelStateSucess) {
    return kAicpuKernelStateFailed;
  }
  // compute
  int64_t limit_row = batch_size;
  int64_t start_row = 0;
  Eigen::Tensor<T, 1, Eigen::RowMajor> cdf(num_classes);
  for (int64_t b = start_row; b < limit_row; ++b) {
    const auto *logits_row = &logits(b, 0);

    T max = std::numeric_limits<T>::lowest();
    for (int64_t j = 0; j < num_classes; ++j) {
      if (Eigen::numext::isfinite(logits_row[j])) {
        max = std::max(max, logits_row[j]);
      }
    }
    const T max_logit = static_cast<T>(max);
    cdf = (logits.chip(b, 0) - max_logit).exp();

    T running_total = 0;
    for (int64_t j = 0; j < num_classes; ++j) {
      if (Eigen::numext::isfinite(logits_row[j])) {
        running_total += cdf(j);
      }
      cdf(j) = running_total;
    }

    const T *cdf_begin = cdf.data();
    const T *cdf_end = cdf.data() + num_classes;

    rng_.seed(rng_seed);
    std::uniform_real_distribution<> dist(0, 1);

    for (int64_t j = 0; j < num_samples; ++j) {
      const double to_find = dist(rng_) * running_total;
      auto found_iter = std::upper_bound(cdf_begin, cdf_end, to_find);
      output(b, j) = static_cast<S>(std::distance(cdf_begin, found_iter));
    }
  }
  // end categorical

  end = clock();
  AICPU_LOGI("=====compute use time %fms=====", (float)(end - start) * 1000 / CLOCKS_PER_SEC);

  return kAicpuKernelStateSucess;
}

uint32_t RandomCategoricalKernel::ParseKernelParam() {
  clock_t start, end;
  start = clock();

  // start random categorical
  aicpuops::Tensor output_tensor = node_def_.outputs(0);
  output_type_ = static_cast<::aicpuops::DataType>(output_tensor.tensor_type());

  aicpuops::Tensor input_tensor = node_def_.inputs(0);
  aicpuops::TensorShape input_shape_ = input_tensor.tensor_shape();
  batch_size_ = input_shape_.dim(0).size();
  num_classes_ = input_shape_.dim(1).size();
  input_type_ = static_cast<::aicpuops::DataType>(input_tensor.tensor_type());
  // end random categorical

  end = clock();
  AICPU_LOGI("=====parse use time %fms=====", (float)(end - start) * 1000 / CLOCKS_PER_SEC);

  return kAicpuKernelStateSucess;
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t RandomCategorical(void *param) {
  aicpu::RandomCategoricalKernel randomCategoricalKernel;
  return randomCategoricalKernel.Compute(param);
}
}
