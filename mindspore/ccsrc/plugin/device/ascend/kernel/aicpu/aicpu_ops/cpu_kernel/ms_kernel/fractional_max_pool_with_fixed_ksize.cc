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

#include "fractional_max_pool_with_fixed_ksize.h"

#include <cmath>
#include <limits>
#include <vector>
#include "Eigen/Dense"

#include "cpu_kernel_utils.h"
#include "kernel_log.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 2;
const char *kFractionalMaxPoolWithFixedKsize = "FractionalMaxPoolWithFixedKsize";
constexpr int64_t kParallelDataNums = 128 * 1024;

#define FRACTIONALMAXPOOLWITHFIXEDKSIZE_COMPUTE_CASE(DTYPE, TYPE, RANDOM_SAMPLES_TYPE, X, RANDOM_SAMPLES, INPUT_N,     \
                                                     INPUT_C, INPUT_H, INPUT_W, OUTPUT_H, OUTPUT_W, POOL_H, POOL_W,    \
                                                     CTX)                                                              \
  case (DTYPE): {                                                                                                      \
    uint32_t result = KERNEL_STATUS_PARAM_INVALID;                                                                     \
    if (RANDOM_SAMPLES_TYPE == DT_FLOAT16) {                                                                           \
      result = FractionalMaxPoolWithFixedKsizeCompute<TYPE, Eigen::half>(                                              \
        X, RANDOM_SAMPLES, INPUT_N, INPUT_C, INPUT_H, INPUT_W, OUTPUT_H, OUTPUT_W, POOL_H, POOL_W, CTX);               \
    } else if (RANDOM_SAMPLES_TYPE == DT_FLOAT) {                                                                      \
      result = FractionalMaxPoolWithFixedKsizeCompute<TYPE, float>(X, RANDOM_SAMPLES, INPUT_N, INPUT_C, INPUT_H,       \
                                                                   INPUT_W, OUTPUT_H, OUTPUT_W, POOL_H, POOL_W, CTX);  \
    } else if (RANDOM_SAMPLES_TYPE == DT_DOUBLE) {                                                                     \
      result = FractionalMaxPoolWithFixedKsizeCompute<TYPE, double>(X, RANDOM_SAMPLES, INPUT_N, INPUT_C, INPUT_H,      \
                                                                    INPUT_W, OUTPUT_H, OUTPUT_W, POOL_H, POOL_W, CTX); \
    } else {                                                                                                           \
      KERNEL_LOG_ERROR(                                                                                                \
        "FractionalMaxPoolWithFixedKsize kernel random_samples type [%s] "                                             \
        "not support.",                                                                                                \
        DTypeStr(RANDOM_SAMPLES_TYPE).c_str());                                                                        \
      return KERNEL_STATUS_PARAM_INVALID;                                                                              \
    }                                                                                                                  \
    if (result != KERNEL_STATUS_OK) {                                                                                  \
      KERNEL_LOG_ERROR("FractionalMaxPoolWithFixedKsize kernel compute failed.");                                      \
      return result;                                                                                                   \
    }                                                                                                                  \
    break;                                                                                                             \
  }
}  // namespace

namespace aicpu {
uint32_t FractionalMaxPoolWithFixedKsize::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "FractionalMaxPoolWithFixedKsize check input and output number failed.");

  AttrValue *ksize = ctx.GetAttr("ksize");
  KERNEL_CHECK_NULLPTR(ksize, KERNEL_STATUS_PARAM_INVALID, "FractionalMaxPoolWithFixedKsize get attr[ksize] failed.");
  std::vector<int64_t> ksize_values = ksize->GetListInt();
  KERNEL_CHECK_FALSE(ksize_values.size() == 1 || ksize_values.size() == 2, KERNEL_STATUS_PARAM_INVALID,
                     "The size of attr[ksize] must be 1 or 2.");
  int64_t pool_h = ksize_values[0];
  int64_t pool_w = ksize_values.size() == 1 ? ksize_values[0] : ksize_values[1];

  AttrValue *output_shape = ctx.GetAttr("output_shape");
  std::vector<int64_t> output_shape_values = output_shape->GetListInt();
  int64_t output_h = output_shape_values[0];
  int64_t output_w = output_shape_values.size() == 1 ? output_shape_values[0] : output_shape_values[1];

  std::string x_format = "NCHW";
  AttrValue *data_format = ctx.GetAttr("data_format");
  if (data_format != nullptr) {
    x_format = data_format->GetString();
    KERNEL_CHECK_FALSE(x_format.compare("NCHW") == 0, KERNEL_STATUS_PARAM_INVALID,
                       "data_format of input[x] must be NCHW.");
  }

  Tensor *x = ctx.Input(0);
  auto x_shape = x->GetTensorShape();
  int64_t input_n = x_shape->GetDimSize(0);
  int64_t input_c = x_shape->GetDimSize(1);
  int64_t input_h = x_shape->GetDimSize(2);
  int64_t input_w = x_shape->GetDimSize(3);

  KERNEL_CHECK_FALSE(output_h + pool_h - 1 <= input_h, KERNEL_STATUS_PARAM_INVALID,
                     "FractionalMaxPoolWithFixedKsize pool height[%d] too "
                     "large relative to input height[%d].",
                     pool_h, input_h);
  KERNEL_CHECK_FALSE(output_w + pool_w - 1 <= input_w, KERNEL_STATUS_PARAM_INVALID,
                     "FractionalMaxPoolWithFixedKsize pool width[%d] too large "
                     "relative to input width[%d].",
                     pool_w, input_w);

  Tensor *random_samples = ctx.Input(1);
  auto random_samples_shape = random_samples->GetTensorShape();
  int32_t random_samples_dims = random_samples_shape->GetDims();
  KERNEL_CHECK_FALSE(random_samples_dims == 3, KERNEL_STATUS_PARAM_INVALID,
                     "The dim of input[random_samples] must be 3.");
  KERNEL_CHECK_FALSE(x_shape->GetDimSize(0) == random_samples_shape->GetDimSize(0), KERNEL_STATUS_PARAM_INVALID,
                     "The first dim of input[x] and input[random_samples] must be equal, but "
                     "got x=[%d] and random_samples=[%d].",
                     x_shape->GetDimSize(0), random_samples_shape->GetDimSize(0));
  KERNEL_CHECK_FALSE(x_shape->GetDimSize(1) == random_samples_shape->GetDimSize(1), KERNEL_STATUS_PARAM_INVALID,
                     "The second dim of input[x] and input[random_samples] must be equal, but "
                     "got x=[%d] and random_samples=[%d].",
                     x_shape->GetDimSize(1), random_samples_shape->GetDimSize(1));
  KERNEL_CHECK_FALSE(random_samples_shape->GetDimSize(2) == 2, KERNEL_STATUS_PARAM_INVALID,
                     "The third dim of input[random_samples] must be 2, but got [%d].",
                     random_samples_shape->GetDimSize(2));

  auto random_samples_type = random_samples->GetDataType();
  auto data_type = x->GetDataType();
  switch (data_type) {
    FRACTIONALMAXPOOLWITHFIXEDKSIZE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, random_samples_type, x, random_samples,
                                                 input_n, input_c, input_h, input_w, output_h, output_w, pool_h, pool_w,
                                                 ctx)
    FRACTIONALMAXPOOLWITHFIXEDKSIZE_COMPUTE_CASE(DT_FLOAT, float, random_samples_type, x, random_samples, input_n,
                                                 input_c, input_h, input_w, output_h, output_w, pool_h, pool_w, ctx)
    FRACTIONALMAXPOOLWITHFIXEDKSIZE_COMPUTE_CASE(DT_DOUBLE, double, random_samples_type, x, random_samples, input_n,
                                                 input_c, input_h, input_w, output_h, output_w, pool_h, pool_w, ctx)
    FRACTIONALMAXPOOLWITHFIXEDKSIZE_COMPUTE_CASE(DT_INT32, int32_t, random_samples_type, x, random_samples, input_n,
                                                 input_c, input_h, input_w, output_h, output_w, pool_h, pool_w, ctx)
    FRACTIONALMAXPOOLWITHFIXEDKSIZE_COMPUTE_CASE(DT_INT64, int64_t, random_samples_type, x, random_samples, input_n,
                                                 input_c, input_h, input_w, output_h, output_w, pool_h, pool_w, ctx)
    default:
      KERNEL_LOG_ERROR(
        "FractionalMaxPoolWithFixedKsize kernel input[x] type "
        "[%s] not support.",
        DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T, typename SType>
uint32_t FractionalMaxPoolWithFixedKsize::FractionalMaxPoolWithFixedKsizeCompute(
  Tensor *x, Tensor *random_samples, const int input_n, const int input_c, const int input_h, const int input_w,
  const int output_h, const int output_w, const int pool_h, const int pool_w, CpuKernelContext &ctx) {
  T *x_addr = reinterpret_cast<T *>(x->GetData());
  SType *random_samples_addr = reinterpret_cast<SType *>(random_samples->GetData());

  Tensor *y = ctx.Output(0);
  T *y_addr = reinterpret_cast<T *>(y->GetData());
  Tensor *argmax = ctx.Output(1);
  int64_t *argmax_addr = reinterpret_cast<int64_t *>(argmax->GetData());

  int64_t data_nums = x->NumElements();
  if (data_nums < kParallelDataNums || input_n == 1) {
    for (int n = 0; n < input_n; n++) {
      T *x_single_batch_addr = x_addr + n * input_c * input_h * input_w;
      SType *random_samples_single_batch_addr = random_samples_addr + n * input_c * 2;
      T *y_single_batch_addr = y_addr + n * input_c * output_h * output_w;
      int64_t *argmax_single_batch_addr = argmax_addr + n * input_c * output_h * output_w;
      ComputeSingleBatch<T, SType>(x_single_batch_addr, random_samples_single_batch_addr, y_single_batch_addr,
                                   argmax_single_batch_addr, input_c, input_h, input_w, output_h, output_w, pool_h,
                                   pool_w);
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > input_n) {
      max_core_num = input_n;
    }
    auto shared_computeN = [&](size_t start, size_t end) {
      for (size_t n = start; n < end; n++) {
        T *x_single_batch_addr = x_addr + n * input_c * input_h * input_w;
        SType *random_samples_single_batch_addr = random_samples_addr + n * input_c * 2;
        T *y_single_batch_addr = y_addr + n * input_c * output_h * output_w;
        int64_t *argmax_single_batch_addr = argmax_addr + n * input_c * output_h * output_w;

        ComputeSingleBatch<T, SType>(x_single_batch_addr, random_samples_single_batch_addr, y_single_batch_addr,
                                     argmax_single_batch_addr, input_c, input_h, input_w, output_h, output_w, pool_h,
                                     pool_w);
      }
    };
    uint32_t ret = CpuKernelUtils::ParallelFor(ctx, input_n, input_n / max_core_num, shared_computeN);
    if (ret != KERNEL_STATUS_OK) {
      KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor shared_computeN failed.");
      return KERNEL_STATUS_INNER_ERROR;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename SType>
uint32_t FractionalMaxPoolWithFixedKsize::ComputeSingleBatch(T *x_addr, SType *random_samples_addr, T *y_addr,
                                                             int64_t *argmax_addr, const int input_c, const int input_h,
                                                             const int input_w, const int output_h, const int output_w,
                                                             const int pool_h, const int pool_w) {
  for (auto plane = 0; plane < input_c; plane++) {
    SType *random_samples_plane = random_samples_addr + plane * 2;
    auto sequence_w =
      FractionalMaxPoolWithFixedKsizeGenerateIntervals<SType>(random_samples_plane[0], input_w, output_w, pool_w);
    auto sequence_h =
      FractionalMaxPoolWithFixedKsizeGenerateIntervals<SType>(random_samples_plane[1], input_h, output_h, pool_h);

    T *x_plane_addr = x_addr + plane * input_h * input_w;
    T *y_plane_addr = y_addr + plane * output_h * output_w;
    int64_t *argmax_plane_addr = argmax_addr + plane * output_h * output_w;
    int h, w;
    for (h = 0; h < output_h; h++) {
      int pool_h_start = sequence_h[h];
      for (w = 0; w < output_w; w++) {
        int pool_w_start = sequence_w[w];
        int h2 = pool_h_start;
        int w2 = pool_w_start;
        T max_value = -std::numeric_limits<T>::infinity();
        int64_t max_index = h2 * input_w + w2;

        for (h2 = pool_h_start; h2 < pool_h_start + pool_h; h2++) {
          for (w2 = pool_w_start; w2 < pool_w_start + pool_w; w2++) {
            KERNEL_CHECK_FALSE(h2 >= 0 && h2 < input_h, KERNEL_STATUS_PARAM_INVALID,
                               "FractionalMaxPoolWithFixedKsizeCompute index H out of bound.");
            KERNEL_CHECK_FALSE(w2 >= 0 && w2 < input_w, KERNEL_STATUS_PARAM_INVALID,
                               "FractionalMaxPoolWithFixedKsizeCompute index W out of bound.");
            int index = h2 * input_w + w2;
            T value = x_plane_addr[index];
            if (value > max_value) {
              max_value = value;
              max_index = index;
            }
          }
        }
        y_plane_addr[h * output_w + w] = max_value;
        argmax_plane_addr[h * output_w + w] = max_index;
      }
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename SType>
std::vector<int> FractionalMaxPoolWithFixedKsize::FractionalMaxPoolWithFixedKsizeGenerateIntervals(
  SType sample, const int input_size, const int output_size, const int pool_size) {
  std::vector<int> sequence(output_size);
  if (output_size > 1) {
    SType alpha = static_cast<SType>(input_size - pool_size) / static_cast<SType>(output_size - 1);

    for (int i = 0; i < output_size - 1; i++) {
      sequence[i] = static_cast<int>((static_cast<SType>(i) + sample) * alpha) - static_cast<int>(sample * alpha);
    }
  }
  sequence[output_size - 1] = input_size - pool_size;

  return sequence;
}

REGISTER_CPU_KERNEL(kFractionalMaxPoolWithFixedKsize, FractionalMaxPoolWithFixedKsize);
}  // namespace aicpu