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

#include "cpu_kernel/ms_kernel/layer_norm_grad_grad.h"

#include <cmath>
#include <numeric>
#include <vector>

#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 3;
const uint32_t kInputNum = 10;
const char *kLayerNormGradGrad = "LayerNormGradGrad";

#define LAYERNORMGRADGRAD_COMPUTE_CASE(DTYPE, TYPE, CTX, NUM)                 \
  case (DTYPE): {                                                             \
    uint32_t result = LayerNormGradGradCompute<TYPE>(CTX, NUM);               \
    if (result != KERNEL_STATUS_OK) {                                         \
      CUST_KERNEL_LOG_ERROR(ctx, "LayerNormGradGrad kernel compute failed."); \
      return result;                                                          \
    }                                                                         \
    break;                                                                    \
  }

#define SWITCH_PARALLEL(ctx, SHARD, data_num, thread_num)                                 \
  if (data_num <= ParallelDataNums) {                                                     \
    for (size_t i = 0; i < thread_num; i++) {                                             \
      SHARD(i, i + 1);                                                                    \
    }                                                                                     \
  } else {                                                                                \
    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, thread_num, 1, SHARD), \
                             "LayerNormGradGrad ParallelFor Compute failed.");            \
  }

Eigen::half sqrt(Eigen::half &data) { return Eigen::half_impl::sqrt(data); }
}  // namespace

namespace aicpu {
uint32_t LayerNormGradGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "LayerNormGradGrad check input and output number failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    LAYERNORMGRADGRAD_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx, 512)
    LAYERNORMGRADGRAD_COMPUTE_CASE(DT_FLOAT, float, ctx, 4 * 1024)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "LayerNormGradGrad kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SwitchParallelCompute0(CpuKernelContext &ctx, size_t ParallelDataNums, const size_t &num, const size_t &g_num,
                                const size_t &mean_num, T *input_x, T *input_dy, T *input_mean, T *input_gamma,
                                T *input_d_dx, T *inv_std, T *x_hat, T *dy_gamma, T *sum1, T *sum2, T *sum3, T *sum4) {
  auto shard_inner_mean = [&](size_t start, size_t end) {
    for (size_t sum_idx = start; sum_idx < end; sum_idx++) {
      for (size_t g_idx = 0; g_idx < g_num; g_idx++) {
        size_t i = g_idx + sum_idx * g_num;  // value of sum_idx = i / g_num;
        sum1[sum_idx] -= inv_std[sum_idx] * input_d_dx[i] / static_cast<T>(g_num);

        T cur_x_hat = (input_x[i] - input_mean[sum_idx]) * inv_std[sum_idx];
        x_hat[i] = cur_x_hat;
        sum2[sum_idx] -= cur_x_hat * inv_std[sum_idx] * input_d_dx[i] / static_cast<T>(g_num);

        T cur_dy_gamma = input_dy[i] * input_gamma[g_idx];
        dy_gamma[i] = cur_dy_gamma;
        sum3[sum_idx] += cur_dy_gamma / static_cast<T>(g_num);

        sum4[sum_idx] += cur_dy_gamma * cur_x_hat / static_cast<T>(g_num);
      }
    }
  };
  SWITCH_PARALLEL(ctx, shard_inner_mean, num, mean_num);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SwitchParallelCompute1(CpuKernelContext &ctx, size_t ParallelDataNums, const size_t &num, const size_t &g_num,
                                const size_t &mean_num, T *input_x, T *input_dy, T *input_mean, T *input_gamma,
                                T *input_d_dx, T *input_d_dg, T *inv_std, T *x_hat, T *dy_gamma, T *sum2, T *sum3,
                                T *sum4, T *sum5, T *sum6, T *sum7, T *part3) {
  auto shard_outer_mean = [&](size_t start, size_t end) {
    for (size_t sum_idx = start; sum_idx < end; sum_idx++) {
      for (size_t g_idx = 0; g_idx < g_num; g_idx++) {
        size_t i = g_idx + sum_idx * g_num;  // value of sum_idx is i / g_num;
        T part_sum1 = dy_gamma[i] - sum3[sum_idx] - x_hat[i] * sum4[sum_idx];
        T part_sum2 = dy_gamma[i] * sum2[sum_idx] - sum4[sum_idx] * input_d_dx[i] * inv_std[sum_idx] +
                      input_dy[i] * input_d_dg[g_idx];
        sum5[sum_idx] += input_d_dx[i] * part_sum1 / static_cast<T>(g_num);

        sum6[sum_idx] += (input_x[i] - input_mean[sum_idx]) * part_sum2 / static_cast<T>(g_num);

        T cur_part3 = inv_std[sum_idx] * part_sum2;
        part3[i] = cur_part3;
        sum7[sum_idx] -= cur_part3 / static_cast<T>(g_num);
      }
    }
  };
  SWITCH_PARALLEL(ctx, shard_outer_mean, num, mean_num);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SwitchParallelCompute2(CpuKernelContext &ctx, size_t ParallelDataNums, const size_t &num, const size_t &g_num,
                                const size_t &mean_num, T *input_gamma, T *input_d_dx, T *input_d_dg, T *input_d_db,
                                T *inv_std, T *output_sopd_x, T *output_sopd_dy, T *x_hat, T *sum1, T *sum2, T *sum5,
                                T *sum6, T *sum7, T *part3) {
  auto shard_input_prop = [&](size_t start, size_t end) {
    for (size_t sum_idx = start; sum_idx < end; sum_idx++) {
      for (size_t g_idx = 0; g_idx < g_num; g_idx++) {
        size_t i = g_idx + sum_idx * g_num;  // value of sum_idx is i / g_num;
        T cur_part4 = -x_hat[i] * inv_std[sum_idx] * inv_std[sum_idx] * (sum5[sum_idx] + sum6[sum_idx]);
        output_sopd_x[i] = part3[i] + cur_part4 + sum7[sum_idx];
        T cur_part5 = input_gamma[g_idx] * input_d_dx[i] * inv_std[sum_idx];
        T cur_part6 = input_gamma[g_idx] * sum1[sum_idx];
        T cur_part7 = input_gamma[g_idx] * x_hat[i] * sum2[sum_idx];
        T cur_part8 = x_hat[i] * input_d_dg[g_idx];
        output_sopd_dy[i] = cur_part5 + cur_part6 + cur_part7 + cur_part8 + input_d_db[g_idx];
      }
    }
  };
  SWITCH_PARALLEL(ctx, shard_input_prop, num, mean_num);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SwitchParallelCompute3(CpuKernelContext &ctx, size_t ParallelDataNums, const size_t &num, const size_t &g_num,
                                const size_t &mean_num, T *input_dy, T *input_d_dx, T *inv_std, T *output_sopd_g,
                                T *x_hat, T *sum1, T *sum2) {
  auto shard_param_prop = [&](size_t start, size_t end) {
    for (size_t g_idx = start; g_idx < end; g_idx++) {
      for (size_t sum_idx = 0; sum_idx < mean_num; sum_idx++) {
        size_t i = g_idx + sum_idx * g_num;  // value of sum_idx is i / g_num;
        T cur_part9 = input_dy[i] * x_hat[i] * sum2[sum_idx];
        T cur_part10 = input_dy[i] * sum1[sum_idx];
        T cur_part11 = input_dy[i] * input_d_dx[i] * inv_std[sum_idx];
        output_sopd_g[g_idx] += cur_part9 + cur_part10 + cur_part11;
      }
    }
  };
  SWITCH_PARALLEL(ctx, shard_param_prop, num, g_num);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t LayerNormGradGradCpuKernel::LayerNormGradGradCompute(CpuKernelContext &ctx, size_t ParallelDataNums) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input_dy = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto input_var = reinterpret_cast<T *>(ctx.Input(2)->GetData());
  auto input_mean = reinterpret_cast<T *>(ctx.Input(3)->GetData());
  auto input_gamma = reinterpret_cast<T *>(ctx.Input(4)->GetData());
  auto input_d_dx = reinterpret_cast<T *>(ctx.Input(5)->GetData());
  auto input_d_dg = reinterpret_cast<T *>(ctx.Input(6)->GetData());
  auto input_d_db = reinterpret_cast<T *>(ctx.Input(7)->GetData());
  auto output_sopd_x = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto output_sopd_dy = reinterpret_cast<T *>(ctx.Output(1)->GetData());
  auto output_sopd_g = reinterpret_cast<T *>(ctx.Output(2)->GetData());
  size_t num = static_cast<size_t>(ctx.Input(0)->NumElements());
  size_t g_num = static_cast<size_t>(ctx.Input(4)->NumElements());
  size_t mean_num = static_cast<size_t>(ctx.Input(3)->NumElements());

  uint32_t res = static_cast<uint32_t>(KERNEL_STATUS_OK);
  CUST_KERNEL_CHECK_FALSE(ctx, (g_num > 0), KERNEL_STATUS_PARAM_INVALID, "gamma should not be empty");

  T *inv_std = new T[mean_num];
  for (size_t i = 0; i < mean_num; i++) {
    if (input_var[i] <= T(0)) {
      delete[] inv_std;
      CUST_KERNEL_LOG_ERROR(ctx, "variance must be greater than zero");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    inv_std[i] = T(1) / sqrt(input_var[i]);
  }

  T *x_hat = new T[num];
  T *dy_gamma = new T[num];
  T *sum1 = new T[mean_num];
  std::fill_n(sum1, mean_num, T(0));
  T *sum2 = new T[mean_num];
  std::fill_n(sum2, mean_num, T(0));
  T *sum3 = new T[mean_num];
  std::fill_n(sum3, mean_num, T(0));
  T *sum4 = new T[mean_num];
  std::fill_n(sum4, mean_num, T(0));
  res = SwitchParallelCompute0(ctx, ParallelDataNums, num, g_num, mean_num, input_x, input_dy, input_mean, input_gamma,
                               input_d_dx, inv_std, x_hat, dy_gamma, sum1, sum2, sum3, sum4);
  if (res != static_cast<uint32_t>(KERNEL_STATUS_OK)) {
    delete[] inv_std;
    delete[] x_hat;
    delete[] dy_gamma;
    delete[] sum1;
    delete[] sum2;
    delete[] sum3;
    delete[] sum4;
    CUST_KERNEL_LOG_ERROR(ctx, "LayerNormGradGrad kernel SwitchParallelCompute0 failed.");
    return res;
  }

  T *sum5 = new T[mean_num];
  std::fill_n(sum5, mean_num, T(0));
  T *sum6 = new T[mean_num];
  std::fill_n(sum6, mean_num, T(0));
  T *sum7 = new T[mean_num];
  std::fill_n(sum7, mean_num, T(0));
  T *part3 = new T[num];
  res =
    SwitchParallelCompute1(ctx, ParallelDataNums, num, g_num, mean_num, input_x, input_dy, input_mean, input_gamma,
                           input_d_dx, input_d_dg, inv_std, x_hat, dy_gamma, sum2, sum3, sum4, sum5, sum6, sum7, part3);
  delete[] sum3;
  delete[] sum4;
  delete[] dy_gamma;
  if (res != KERNEL_STATUS_OK) {
    delete[] inv_std;
    delete[] x_hat;
    delete[] sum1;
    delete[] sum2;
    delete[] sum5;
    delete[] sum6;
    delete[] sum7;
    delete[] part3;
    CUST_KERNEL_LOG_ERROR(ctx, "LayerNormGradGrad kernel SwitchParallelCompute1 failed.");
    return res;
  }

  res =
    SwitchParallelCompute2(ctx, ParallelDataNums, num, g_num, mean_num, input_gamma, input_d_dx, input_d_dg, input_d_db,
                           inv_std, output_sopd_x, output_sopd_dy, x_hat, sum1, sum2, sum5, sum6, sum7, part3);
  delete[] sum5;
  delete[] sum6;
  delete[] sum7;
  if (res != static_cast<uint32_t>(KERNEL_STATUS_OK)) {
    delete[] inv_std;
    delete[] x_hat;
    delete[] sum1;
    delete[] sum2;
    delete[] part3;
    CUST_KERNEL_LOG_ERROR(ctx, "LayerNormGradGrad kernel SwitchParallelCompute2 failed.");
    return res;
  }

  std::fill_n(output_sopd_g, g_num, T(0));
  res = SwitchParallelCompute3(ctx, ParallelDataNums, num, g_num, mean_num, input_dy, input_d_dx, inv_std,
                               output_sopd_g, x_hat, sum1, sum2);
  delete[] inv_std;
  delete[] x_hat;
  delete[] sum1;
  delete[] sum2;
  delete[] part3;
  if (res != static_cast<uint32_t>(KERNEL_STATUS_OK)) {
    CUST_KERNEL_LOG_ERROR(ctx, "LayerNormGradGrad kernel SwitchParallelCompute3 failed.");
    return res;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kLayerNormGradGrad, LayerNormGradGradCpuKernel);
}  // namespace aicpu
