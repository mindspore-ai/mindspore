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

#include "fractional_max_pool_grad_with_fixed_ksize.h"

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
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
const char *kFractionalMaxPoolGradWithFixedKsize = "FractionalMaxPoolGradWithFixedKsize";
constexpr int64_t kParallelDataNums = 128 * 1024;

#define FRACTIONALMAXPOOLGRADWITHFIXEDKSIZE_COMPUTE_CASE(DTYPE, TYPE, OUT_BACKPROP, ARGMAX, DATA_NUMS, N_SIZE, C_SIZE, \
                                                         INPUT_H, INPUT_W, OUTPUT_H, OUTPUT_W, CTX)                    \
  case (DTYPE): {                                                                                                      \
    uint32_t result = FractionalMaxPoolGradWithFixedKsizeCompute<TYPE>(                                                \
      OUT_BACKPROP, ARGMAX, DATA_NUMS, N_SIZE, C_SIZE, INPUT_H, INPUT_W, OUTPUT_H, OUTPUT_W, CTX);                     \
    if (result != KERNEL_STATUS_OK) {                                                                                  \
      KERNEL_LOG_ERROR("FractionalMaxPoolGradWithFixedKsize kernel compute failed.");                                  \
      return result;                                                                                                   \
    }                                                                                                                  \
    break;                                                                                                             \
  }

}  // namespace

namespace aicpu {
uint32_t FractionalMaxPoolGradWithFixedKsize::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "FractionalMaxPoolGradWithFixedKsize check input and "
                      "output number failed.");

  Tensor *origin_input = ctx.Input(0);
  int64_t data_nums = origin_input->NumElements();
  auto origin_input_shape = origin_input->GetTensorShape();
  int32_t origin_input_dim = origin_input_shape->GetDims();
  KERNEL_CHECK_FALSE(origin_input_dim == 4, KERNEL_STATUS_PARAM_INVALID,
                     "The dim of input[origin_input] must be 4, but got [%d].", origin_input_dim);

  Tensor *out_backprop = ctx.Input(1);
  auto out_backprop_shape = out_backprop->GetTensorShape();
  int32_t out_backprop_dim = out_backprop_shape->GetDims();
  KERNEL_CHECK_FALSE(out_backprop_dim == 4, KERNEL_STATUS_PARAM_INVALID,
                     "The dim of input[out_backprop] must be 4, but got [%d].", out_backprop_dim);
  Tensor *argmax = ctx.Input(2);
  auto argmax_shape = argmax->GetTensorShape();
  int32_t argmax_dim = argmax_shape->GetDims();
  KERNEL_CHECK_FALSE(argmax_dim == 4, KERNEL_STATUS_PARAM_INVALID, "The dim of input[argmax] must be 4, but got [%d].",
                     argmax_dim);
  std::vector<int64_t> out_backprop_dim_sizes = out_backprop_shape->GetDimSizes();
  std::vector<int64_t> argmax_dim_sizes = argmax_shape->GetDimSizes();
  KERNEL_CHECK_FALSE(out_backprop_dim_sizes == argmax_dim_sizes, KERNEL_STATUS_PARAM_INVALID,
                     "The shape of input[out_backprop] and input[argmax] must be equal.");
  int64_t n_size = out_backprop_dim_sizes[0];
  int64_t c_size = out_backprop_dim_sizes[1];
  int64_t input_h = out_backprop_dim_sizes[2];
  int64_t input_w = out_backprop_dim_sizes[3];

  std::vector<int64_t> origin_input_dim_sizes = origin_input_shape->GetDimSizes();
  KERNEL_CHECK_FALSE(origin_input_dim_sizes[0] == n_size, KERNEL_STATUS_PARAM_INVALID,
                     "The first dim of input[origin_input] and "
                     "input[out_backprop] must be equal,"
                     "but got origin_input=[%d] and out_backprop=[%d].",
                     origin_input_dim_sizes[0], n_size);
  KERNEL_CHECK_FALSE(origin_input_dim_sizes[1] == c_size, KERNEL_STATUS_PARAM_INVALID,
                     "The second dim of input[origin_input] and "
                     "input[out_backprop] must be equal,"
                     "but got origin_input=[%d] and out_backprop=[%d].",
                     origin_input_dim_sizes[1], c_size);
  int64_t output_h = origin_input_dim_sizes[2];
  int64_t output_w = origin_input_dim_sizes[3];

  auto data_type = out_backprop->GetDataType();
  switch (data_type) {
    FRACTIONALMAXPOOLGRADWITHFIXEDKSIZE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, out_backprop, argmax, data_nums, n_size,
                                                     c_size, input_h, input_w, output_h, output_w, ctx)
    FRACTIONALMAXPOOLGRADWITHFIXEDKSIZE_COMPUTE_CASE(DT_FLOAT, float, out_backprop, argmax, data_nums, n_size, c_size,
                                                     input_h, input_w, output_h, output_w, ctx)
    FRACTIONALMAXPOOLGRADWITHFIXEDKSIZE_COMPUTE_CASE(DT_DOUBLE, double, out_backprop, argmax, data_nums, n_size, c_size,
                                                     input_h, input_w, output_h, output_w, ctx)
    FRACTIONALMAXPOOLGRADWITHFIXEDKSIZE_COMPUTE_CASE(DT_INT32, int32_t, out_backprop, argmax, data_nums, n_size, c_size,
                                                     input_h, input_w, output_h, output_w, ctx)
    FRACTIONALMAXPOOLGRADWITHFIXEDKSIZE_COMPUTE_CASE(DT_INT64, int64_t, out_backprop, argmax, data_nums, n_size, c_size,
                                                     input_h, input_w, output_h, output_w, ctx)
    default:
      KERNEL_LOG_ERROR(
        "FractionalMaxPoolGradWithFixedKsize kernel input[out_backprop] type "
        "[%s] not support.",
        DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t FractionalMaxPoolGradWithFixedKsize::FractionalMaxPoolGradWithFixedKsizeCompute(
  Tensor *out_backprop, Tensor *argmax, const int64_t data_nums, const int n_size, const int c_size, const int input_h,
  const int input_w, const int output_h, const int output_w, CpuKernelContext &ctx) {
  T *out_backprop_addr = reinterpret_cast<T *>(out_backprop->GetData());
  int64_t *argmax_addr = reinterpret_cast<int64_t *>(argmax->GetData());

  Tensor *y = ctx.Output(0);
  T *y_addr = reinterpret_cast<T *>(y->GetData());

  if (data_nums < kParallelDataNums || n_size == 1) {
    for (int n = 0; n < n_size; n++) {
      T *out_backprop_single_batch_addr = out_backprop_addr + n * c_size * input_h * input_w;
      int64_t *argmax_single_batch_addr = argmax_addr + n * c_size * input_h * input_w;
      T *y_single_batch_addr = y_addr + n * c_size * output_h * output_w;

      ComputeSingleBatch<T>(out_backprop_single_batch_addr, argmax_single_batch_addr, y_single_batch_addr, c_size,
                            input_h, input_w, output_h, output_w);
    }
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > (uint32_t)n_size) {
      max_core_num = n_size;
    }
    auto shared_computeN = [&](size_t start, size_t end) {
      for (size_t n = start; n < end; n++) {
        T *out_backprop_single_batch_addr = out_backprop_addr + n * c_size * input_h * input_w;
        int64_t *argmax_single_batch_addr = argmax_addr + n * c_size * input_h * input_w;
        T *y_single_batch_addr = y_addr + n * c_size * output_h * output_w;

        ComputeSingleBatch<T>(out_backprop_single_batch_addr, argmax_single_batch_addr, y_single_batch_addr, c_size,
                              input_h, input_w, output_h, output_w);
      }
    };
    uint32_t ret = CpuKernelUtils::ParallelFor(ctx, n_size, n_size / max_core_num, shared_computeN);
    if (ret != KERNEL_STATUS_OK) {
      KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor shared_computeN failed.");
      return KERNEL_STATUS_INNER_ERROR;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t FractionalMaxPoolGradWithFixedKsize::ComputeSingleBatch(T *out_backprop_single_batch_addr,
                                                                 int64_t *argmax_single_batch_addr,
                                                                 T *y_single_batch_addr, const int c_size,
                                                                 const int input_h, const int input_w,
                                                                 const int output_h, const int output_w) {
  for (int plane = 0; plane < c_size; plane++) {
    T *out_backprop_plane_addr = out_backprop_single_batch_addr + plane * input_h * input_w;
    int64_t *argmax_plane_addr = argmax_single_batch_addr + plane * input_h * input_w;
    T *y_plane_addr = y_single_batch_addr + plane * output_h * output_w;

    for (int i = 0; i < output_h; i++) {
      for (int j = 0; j < output_w; j++) {
        y_plane_addr[i * output_w + j] = static_cast<T>(0);
      }
    }

    for (int h = 0; h < input_h; h++) {
      for (int w = 0; w < input_w; w++) {
        int input_index = h * input_w + w;
        KERNEL_CHECK_FALSE((input_index >= 0 && input_index < input_h * input_w), KERNEL_STATUS_PARAM_INVALID,
                           "The input_index[%d] out of the length of argmax.", input_index);
        int output_index = argmax_plane_addr[input_index];
        KERNEL_CHECK_FALSE((output_index >= 0 && output_index < output_h * output_w), KERNEL_STATUS_PARAM_INVALID,
                           "The output_index[%d] out of the length of y.", output_index);

        y_plane_addr[output_index] += out_backprop_plane_addr[input_index];
      }
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kFractionalMaxPoolGradWithFixedKsize, FractionalMaxPoolGradWithFixedKsize);
}  // namespace aicpu