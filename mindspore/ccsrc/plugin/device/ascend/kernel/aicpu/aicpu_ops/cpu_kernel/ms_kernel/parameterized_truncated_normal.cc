/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "parameterized_truncated_normal.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

using namespace std;

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 5;
const char *kParameterizedTruncatedNormal = "ParameterizedTruncatedNormal";
using RNG_Engine = std::mt19937;
static constexpr int kMaxIterations = 1000;

#define BATCH_SIZE_CASE(DTYPE, TYPE, CTX)                   \
  case (DTYPE): {                                           \
    batch_size = int64_t(GetBatchSizeCheckDims<TYPE>(CTX)); \
    break;                                                  \
  }

// override functions for half
bool isinf(Eigen::half &data) { return Eigen::half_impl::isinf(data); }
void swap(Eigen::half &data1, Eigen::half &data2) {
  Eigen::half tmp = data1;
  data1 = data2;
  data2 = tmp;
}

Eigen::half exp(Eigen::half &data) { return Eigen::half_impl::exp(data); }
Eigen::half log(Eigen::half &data) { return Eigen::half_impl::log(data); }
}  // namespace

namespace aicpu {
template <typename T>
T GetBatchSizeCheckDims(CpuKernelContext &ctx) {
  auto output_shape = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  for (int i = 1; i < ctx.Input(0)->NumElements(); i++) {
    KERNEL_CHECK_FALSE((output_shape[i] >= 0), KERNEL_STATUS_PARAM_INVALID, "The output dimension must be >= 0.")
  }
  return output_shape[0];
}

template <typename T>
void Generate(int64_t size, T mean, T stddev, T minval, T maxval, T **output_ptr, RNG_Engine &rng) {
  auto output = *output_ptr;
  std::normal_distribution<double> normal_dist(0, 1);
  std::uniform_real_distribution<double> unifrom_dist(0, 1);
  // Vectorized intermediate calculations for uniform rejection sampling.
  const T stddev_inside_bound = T(1.3);

  /**
   * If possible, make one-sided bound be the lower bound, or make both
   * bounds positive. Otherwise, the bounds are on either side of the
   * mean.
   */
  if ((isinf(minval) && minval < T(0)) || maxval < mean) {
    // Reverse all calculations. norm_min and norm_max will be flipped.
    swap(minval, maxval);
    stddev = -stddev;
  }

  auto tmp_num = (stddev == static_cast<T>(0)) ? static_cast<T>(1) : stddev;
  // Calculate normalized samples, then convert them.
  const T norm_min = (minval - mean) / tmp_num;
  const T norm_max = (maxval - mean) / tmp_num;
  int sample_num = 0;

  // Determine the method to use.
  const T sqrt_factor = sqrt((norm_min * norm_min) + T(4));
  const T cutoff = T(2) * exp(T(0.5) + (norm_min * (norm_min - sqrt_factor)) / T(4)) / (norm_min + sqrt_factor);
  const T diff = norm_max - norm_min;

  if (((norm_min < -stddev_inside_bound) && (norm_max >= T(0.))) ||
      ((norm_max > stddev_inside_bound) && (norm_min <= T(0.)))) {
    /**
     * If the bounds are a least 3 standard deviations from the mean
     * on at least one side then we rejection sample by sampling
     * from the normal distribution and rejecting samples outside
     * the bounds.
     * Under this condition the acceptance rate per iteration should
     * always be ~ 50%. This sampler is more efficient (and more
     * numerically stable when one or both bounds is far from the mean).
     */
    while (sample_num < size) {
      for (int iter = 0; iter <= kMaxIterations;) {
        T normal_sample = T(normal_dist(rng));

        if ((normal_sample >= norm_min) && (normal_sample <= norm_max)) {
          *output = normal_sample * stddev + mean;
          if (stddev <= static_cast<T>(0)) {
            *output = static_cast<T>(INFINITY);
          } else {
            output = output + 1;
          }
          sample_num++;
          break;
        } else {
          iter++;
          if (iter > kMaxIterations) {
            /**
             * This should never occur because this sampler should
             * (by the selection criteria above) be used if at least 3
             * standard deviations of one side of the distribution
             * is within the limits (so acceptance probability per
             * iterations >~ 1/2 per iteration).
             */
            KERNEL_LOG_ERROR(
              "TruncatedNormal randn rejection sampler "
              "exceeded maximum iterations");
            *output_ptr = output;
            return;
          }
        }
      }
    }
  } else if (diff < cutoff) {
    // Sample from a uniform distribution on [norm_min, norm_max].
    const T plus_Factor = (norm_min < T(0)) ? T(0) : norm_min * norm_min;

    while (sample_num < size) {
      for (int iter = 0; iter <= kMaxIterations;) {
        T uniform_sample = T(unifrom_dist(rng));

        T z = uniform_sample * diff + norm_min;
        T g = (plus_Factor - z * z) / T(2.0);

        bool accept = T(unifrom_dist(rng)) <= exp(g);

        if (accept || iter + 1 >= kMaxIterations) {
          if (!accept) {
            KERNEL_LOG_ERROR(
              "TruncatedNormal uniform rejection sampler "
              "exceeded max iterations. Sample may contain outliers.");
            *output_ptr = output;
            return;
          }

          *output = z * stddev + mean;
          if (stddev <= static_cast<T>(0)) {
            *output = static_cast<T>(INFINITY);
          } else {
            output = output + 1;
          }
          sample_num++;
          break;

        } else {
          iter++;
        }
      }
    }
  } else {
    /**
     * Sample from an exponential distribution with alpha maximizing
     * acceptance probability, offset by norm_min from the origin.
     * Accept only if less than norm_max.
     */
    const T alpha = (norm_min + sqrt((norm_min * norm_min) + T(4))) / T(2);
    while (sample_num < size) {
      for (int iter = 0; iter <= kMaxIterations;) {
        T uniform_sample = T(unifrom_dist(rng));
        T z = -log(uniform_sample) / alpha + norm_min;
        const T x = norm_min < alpha ? alpha - z : norm_min - alpha;
        const T g = exp(-x * x / T(2.0));

        const T u = T(unifrom_dist(rng));

        bool accept = (u <= g && z < norm_max);
        if (accept || iter + 1 >= kMaxIterations) {
          if (!accept) {
            KERNEL_LOG_ERROR(
              "TruncatedNormal exponential distribution "
              "rejection sampler exceeds max iterations. "
              "Sample may contain outliers.");
            *output_ptr = output;
            return;
          }
          *output = z * stddev + mean;
          output = output + 1;
          sample_num++;
          break;
        } else {
          iter++;
        }
      }
    }
  }

  *output_ptr = output;
  return;
}

template <typename T_shape, typename T_val>
uint32_t BatchGenerate(CpuKernelContext &ctx) {
  Tensor *input_0 = ctx.Input(0);
  auto output_shape = reinterpret_cast<T_shape *>(input_0->GetData());
  // check shape
  auto batch_size = output_shape[0];
  int sample_size = 1;
  for (int i = 1; i < ctx.Input(0)->NumElements(); i++) {
    sample_size *= output_shape[i];
  }

  Tensor *input_3 = ctx.Input(3);
  Tensor *input_4 = ctx.Input(4);
  Tensor *input_1 = ctx.Input(1);
  Tensor *input_2 = ctx.Input(2);
  Tensor *output = ctx.Output(0);

  auto output_data = reinterpret_cast<T_val *>(output->GetData());
  auto means = reinterpret_cast<T_val *>(input_1->GetData());
  auto stdevs = reinterpret_cast<T_val *>(input_2->GetData());
  auto minvals = reinterpret_cast<T_val *>(input_3->GetData());
  auto maxvals = reinterpret_cast<T_val *>(input_4->GetData());

  // setup seed
  int64_t final_seed = 0;
  auto attr_seed = ctx.GetAttr("seed");
  if (attr_seed != nullptr) {
    final_seed = attr_seed->GetInt();
  }
  if (final_seed == 0) {
    auto attr_seed2 = ctx.GetAttr("seed2");
    if (attr_seed2 != nullptr) {
      final_seed = attr_seed2->GetInt();
    }
  }

  // setup random engine
  std::random_device r;
  RNG_Engine rng;
  final_seed = final_seed ? final_seed : r();
  rng.seed(final_seed);

  vector<T_val *> params = {means, stdevs, minvals, maxvals};

  vector<int> params_idx;
  if (input_1->NumElements() > 1) {
    params_idx.push_back(0);
  }
  if (input_2->NumElements() > 1) {
    params_idx.push_back(1);
  }
  if (input_3->NumElements() > 1) {
    params_idx.push_back(2);
  }
  if (input_4->NumElements() > 1) {
    params_idx.push_back(3);
  }

  for (int batch = 0; batch < batch_size; batch++) {
    auto maxval = *params[3];
    auto minval = *params[2];
    auto stdevs_val = *params[1];
    KERNEL_CHECK_FALSE(stdevs_val >= static_cast<T_val>(0), KERNEL_STATUS_PARAM_INVALID,
                       "'stdevs' must be greater than 0.")
    KERNEL_CHECK_FALSE((maxval > minval), KERNEL_STATUS_PARAM_INVALID,
                       "Max value must be greater than min value in each batch")
    Generate<T_val>(int64_t(sample_size), *params[0], *params[1], minval, maxval, &output_data, rng);
    for (auto i : params_idx) {
      params[i] = params[i] + 1;
    }
  }

  return KERNEL_STATUS_OK;
}

uint32_t ParameterizedTruncatedNormalCpuKernel::ParameterizedTruncatedNormalCheck(CpuKernelContext &ctx) {
  DataType val_datatype = ctx.Input(1)->GetDataType();
  DataType shape_datatype = ctx.Input(0)->GetDataType();

  for (uint32_t i = 0; i < kInputNum; i++) {
    Tensor *input = ctx.Input(i);

    // check input datatype
    DataType input_datatype = input->GetDataType();
    switch (i) {
      case 0:
        KERNEL_CHECK_FALSE((input_datatype == DT_INT32 || input_datatype == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                           "Input[0] data type must DT_INT32 or DT_INT64,"
                           "but got data type[%s].",
                           DTypeStr(input_datatype).c_str());
        break;
      case 1:
        KERNEL_CHECK_FALSE((input_datatype == DT_FLOAT16 || input_datatype == DT_FLOAT || input_datatype == DT_DOUBLE),
                           KERNEL_STATUS_PARAM_INVALID,
                           "Input[1] data type must DT_FLOAT16 or DT_FLOAT or DT_DOUBLE,"
                           "but got data type[%s].",
                           DTypeStr(input_datatype).c_str());
        break;
      default:
        KERNEL_CHECK_FALSE((input_datatype == val_datatype), KERNEL_STATUS_PARAM_INVALID,
                           "The data type of input[%u] [%s] need be same with input[1] [%s].", i,
                           DTypeStr(input_datatype).c_str(), DTypeStr(val_datatype).c_str())
    }

    // check input dimension
    auto input_dims = input->GetTensorShape()->GetDims();

    int64_t batch_size = 0;
    switch (shape_datatype) {
      BATCH_SIZE_CASE(DT_INT32, int32_t, ctx)
      BATCH_SIZE_CASE(DT_INT64, int64_t, ctx)
      default:
        KERNEL_LOG_ERROR("input0 data type [%u] not support.", shape_datatype);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_CHECK_FALSE((batch_size >= 0), KERNEL_STATUS_PARAM_INVALID, "The batch size must be >= 0.")

    switch (i) {
      case 0:
        KERNEL_CHECK_FALSE((input_dims == 1), KERNEL_STATUS_PARAM_INVALID,
                           "Input[0] should be rank 1, but got rank [%d].", input_dims);
        break;

      default:
        KERNEL_CHECK_FALSE((input_dims <= 1), KERNEL_STATUS_PARAM_INVALID,
                           "Input[%u] should be at most rank 1, but got rank [%d].", i, input_dims);
        if (input_dims == 1) {
          auto num_of_elems = input->NumElements();

          KERNEL_CHECK_FALSE((num_of_elems == 1 || num_of_elems == batch_size), KERNEL_STATUS_PARAM_INVALID,
                             "Input[%u] length should be 1 or equal to the "
                             "batch size, got %d.",
                             i, num_of_elems);
        }
    }
  }
  return KERNEL_STATUS_OK;
}

void ParameterizedTruncatedNormalCpuKernel::SetMap() {
  calls_[DT_INT32][DT_FLOAT16] = BatchGenerate<int32_t, Eigen::half>;
  calls_[DT_INT32][DT_FLOAT] = BatchGenerate<int32_t, float>;
  calls_[DT_INT32][DT_DOUBLE] = BatchGenerate<int32_t, double>;
  calls_[DT_INT64][DT_FLOAT16] = BatchGenerate<int64_t, Eigen::half>;
  calls_[DT_INT64][DT_FLOAT] = BatchGenerate<int64_t, float>;
  calls_[DT_INT64][DT_DOUBLE] = BatchGenerate<int64_t, double>;
}

uint32_t ParameterizedTruncatedNormalCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "ParameterizedTruncatedNormal check input and output number failed.");

  KERNEL_HANDLE_ERROR(ParameterizedTruncatedNormalCheck(ctx), "ParameterizedTruncatedNormal check params failed.");

  DataType val_datatype = ctx.Input(1)->GetDataType();
  DataType shape_datatype = ctx.Input(0)->GetDataType();

  SetMap();
  auto ret = calls_[shape_datatype][val_datatype](ctx);
  calls_.clear();

  return ret;
}

REGISTER_CPU_KERNEL(kParameterizedTruncatedNormal, ParameterizedTruncatedNormalCpuKernel);
}  // namespace aicpu
