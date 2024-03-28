/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "cpu_kernel/ms_kernel/batch_norm_grad_grad.h"
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 8;
const uint32_t kOutputNum = 3;

// when size of 1D tensor is more than kParallelDataNum, use Parallel func
const int kParallelDataNum = 4;
const int kParallelDataNumMid = 16;

const char *const kBatchNormGradGrad = "BatchNormGradGrad";

#define BATCHNORMGRADGRAD_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                             \
    uint32_t result = ParallelCompute<TYPE>(CTX);                             \
    if (result != KERNEL_STATUS_OK) {                                         \
      CUST_KERNEL_LOG_ERROR(ctx, "BatchNormGradGrad kernel compute failed."); \
      return result;                                                          \
    }                                                                         \
    break;                                                                    \
  }
}  // namespace

namespace aicpu {
uint32_t BatchNormGradGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "BatchNormGradGrad check input and output number failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    BATCHNORMGRADGRAD_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    BATCHNORMGRADGRAD_COMPUTE_CASE(DT_FLOAT, float, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "BatchNormGradGrad kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t BatchNormGradGradCpuKernel::ParallelCompute(CpuKernelContext &ctx) {
  // handle attr
  AttrValue *Attr_data_format = ctx.GetAttr("data_format");
  std::string data_format = (Attr_data_format == nullptr) ? "NHWC" : Attr_data_format->GetString();
  AttrValue *Attr_is_training = ctx.GetAttr("is_training");
  bool is_training = (Attr_is_training == nullptr) ? true : Attr_is_training->GetBool();

  // check param
  int C_num = static_cast<int>(ctx.Input(2)->NumElements());
  auto reserve_space_2 = reinterpret_cast<float *>(ctx.Input(4)->GetData());  // fp32
  for (int j = 0; j < C_num; j++) {
    if (*(reserve_space_2 + j) < 0) {
      CUST_KERNEL_LOG_ERROR(ctx, "'reserve_space_2' must be no less than zero");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  // cannot use parallelfunc in NCHW cases
  if (data_format == "NCHW") {
    if (is_training) {
      TrainingComputeNCHW<T>(ctx);
    } else {
      InferenceComputeNCHW<T>(ctx);
    }
  } else if (data_format == "NHWC") {
    // parallel check
    if (C_num >= kParallelDataNum) {
      uint32_t min_core_num = 1;
      int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

      if (C_num <= kParallelDataNumMid) {
        max_core_num = std::min(max_core_num, static_cast<int64_t>(4));  // up to 4 cpu cores
      }

      if (max_core_num > C_num) {
        max_core_num = C_num;
      }

      auto sharder_BNGG = [&](int start, int end) {
        if (is_training) {
          TrainingComputeNHWC<T>(ctx, start, end);
        } else {
          InferenceComputeNHWC<T>(ctx, start, end);
        }
      };

      CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, C_num, C_num / max_core_num, sharder_BNGG),
                               "BatchNormGradGrad Compute failed.");
    } else {
      if (is_training) {
        TrainingComputeNHWC<T>(ctx, 0, C_num);
      } else {
        InferenceComputeNHWC<T>(ctx, 0, C_num);
      }
    }
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
void BatchNormGradGradCpuKernel::TrainingComputeNHWC(CpuKernelContext &ctx, int start, int end) {
  auto x_ori = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto dy_ori = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto scale = reinterpret_cast<float *>(ctx.Input(2)->GetData());            // fp32
  auto reserve_space_1 = reinterpret_cast<float *>(ctx.Input(3)->GetData());  // batch_mean  fp32
  auto reserve_space_2 = reinterpret_cast<float *>(ctx.Input(4)->GetData());  // batch_var  fp32
  auto ddx_ori = reinterpret_cast<T *>(ctx.Input(5)->GetData());
  auto ddscale = reinterpret_cast<float *>(ctx.Input(6)->GetData());   // fp32
  auto ddoffset = reinterpret_cast<float *>(ctx.Input(7)->GetData());  // fp32
  auto dx = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto ddy = reinterpret_cast<T *>(ctx.Output(1)->GetData());
  auto dscale = reinterpret_cast<float *>(ctx.Output(2)->GetData());  // fp32

  int x_num = static_cast<int>(ctx.Input(0)->NumElements());
  int C_num = static_cast<int>(ctx.Input(2)->NumElements());
  int N_num = static_cast<int>(ctx.Input(0)->GetTensorShape()->GetDimSize(0));
  int NHW_num = static_cast<int>(x_num / C_num);
  int HW_num = static_cast<int>(NHW_num / N_num);

  // change dtype from 'T' to 'fp32'
  float *x = new float[x_num];
  float *dy = new float[x_num];
  float *ddx = new float[x_num];

  for (int i = 0; i < N_num; i++) {
    for (int k = 0; k < HW_num; k++) {
      for (int j = start; j < end; j++) {
        int index = i * HW_num * C_num + k * C_num + j;
        *(x + index) = static_cast<float>(*(x_ori + index));
        *(dy + index) = static_cast<float>(*(dy_ori + index));
        *(ddx + index) = static_cast<float>(*(ddx_ori + index));
      }
    }
  }

  AttrValue *Attr_epsilon = ctx.GetAttr("epsilon");
  float epsilon = (Attr_epsilon == nullptr) ? 0.0001 : Attr_epsilon->GetFloat();

  float num_1 = 1.0;
  float num_3 = 3.0;
  float M = static_cast<float>((NHW_num));

  // create intermediate variables
  float *inv_std = new float[C_num];
  float *x_hat = new float[x_num];

  for (int j = start; j < end; j++) {
    *(inv_std + j) = num_1 / sqrt(*(reserve_space_2 + j) + epsilon);
  }

  for (int i = 0; i < N_num; i++) {
    for (int k = 0; k < HW_num; k++) {
      for (int j = start; j < end; j++) {
        int index = i * HW_num * C_num + k * C_num + j;
        *(x_hat + index) = (*(inv_std + j)) * (*(x + index) - *(reserve_space_1 + j));
      }
    }
  }

  // create intermediate variables
  float *sum_dy = new float[C_num];
  float *sum_dy_x_hat = new float[C_num];
  float *sum_ddx = new float[C_num];
  float *sum_ddx_x_hat = new float[C_num];
  float *sum_dy_ddx = new float[C_num];

  // initialize
  for (int j = start; j < end; j++) {
    *(sum_dy + j) = 0;
    *(sum_dy_x_hat + j) = 0;
    *(sum_ddx + j) = 0;
    *(sum_ddx_x_hat + j) = 0;
    *(sum_dy_ddx + j) = 0;
  }

  // compute np.sum(to C dim)
  for (int i = 0; i < N_num; i++) {
    for (int k = 0; k < HW_num; k++) {
      for (int j = start; j < end; j++) {
        int index = i * HW_num * C_num + k * C_num + j;
        *(sum_dy + j) += *(dy + index);
        *(sum_dy_x_hat + j) += (*(x_hat + index)) * (*(dy + index));
        *(sum_ddx + j) += *(ddx + index);
        *(sum_ddx_x_hat + j) += (*(x_hat + index)) * (*(ddx + index));
        *(sum_dy_ddx + j) += (*(dy + index)) * (*(ddx + index));
      }
    }
  }

  // create intermediate variables
  float *dx_term_0 = new float[x_num];
  float *dx_term_1 = new float[x_num];
  float *dx_term_2 = new float[x_num];
  float *dx_term = new float[x_num];
  float *scale_term = new float[x_num];

  // calculate dx
  for (int i = 0; i < N_num; i++) {
    for (int k = 0; k < HW_num; k++) {
      for (int j = start; j < end; j++) {
        int index = i * HW_num * C_num + k * C_num + j;
        *(dx_term_0 + index) = (*(x_hat + index)) *
                               ((*(sum_ddx + j)) * (*(sum_dy + j)) / M - (*(sum_dy_ddx + j)) +
                                num_3 * (*(sum_dy_x_hat + j)) * (*(sum_ddx_x_hat + j)) / M) /
                               M;
        *(dx_term_1 + index) = (*(sum_ddx_x_hat + j)) * ((*(sum_dy + j)) / M - (*(dy + index))) / M;
        *(dx_term_2 + index) = (*(sum_dy_x_hat + j)) * ((*(sum_ddx + j)) / M - (*(ddx + index))) / M;
      }
    }
  }

  for (int i = 0; i < N_num; i++) {
    for (int k = 0; k < HW_num; k++) {
      for (int j = start; j < end; j++) {
        int index = i * HW_num * C_num + k * C_num + j;
        *(dx_term + index) = (*(scale + j)) / ((*(reserve_space_2 + j)) + epsilon) *
                             ((*(dx_term_0 + index)) + (*(dx_term_1 + index)) + (*(dx_term_2 + index)));
        *(scale_term + index) =
          (*(ddscale + j)) * (*(inv_std + j)) *
          ((*(dy + index)) - (*(sum_dy + j)) / M - (*(sum_dy_x_hat + j)) / M * (*(x_hat + index)));
      }
    }
  }

  for (int i = 0; i < N_num; i++) {
    for (int k = 0; k < HW_num; k++) {
      for (int j = start; j < end; j++) {
        int index = i * HW_num * C_num + k * C_num + j;
        *(dx + index) = static_cast<T>((*(dx_term + index)) + (*(scale_term + index)));
      }
    }
  }

  // calculate ddy
  for (int i = 0; i < N_num; i++) {
    for (int k = 0; k < HW_num; k++) {
      for (int j = start; j < end; j++) {
        int index = i * HW_num * C_num + k * C_num + j;
        *(ddy + index) =
          static_cast<T>((*(scale + j)) * (*(inv_std + j)) / M *
                           (M * (*(ddx + index)) - (*(sum_ddx + j)) - (*(x_hat + index)) * (*(sum_ddx_x_hat + j))) +
                         (*(ddscale + j)) * (*(x_hat + index)) + (*(ddoffset + j)));
      }
    }
  }

  // calculate dscale
  for (int j = start; j < end; j++) {
    *(dscale + j) = 0;
  }

  for (int i = 0; i < N_num; i++) {
    for (int k = 0; k < HW_num; k++) {
      for (int j = start; j < end; j++) {
        int index = i * HW_num * C_num + k * C_num + j;
        *(dscale + j) += (*(ddx + index)) * (*(inv_std + j)) *
                         ((*(dy + index)) - (*(sum_dy + j)) / M - (*(sum_dy_x_hat + j)) / M * (*(x_hat + index)));
      }
    }
  }

  // free memory
  delete[] x;
  delete[] dy;
  delete[] ddx;
  delete[] inv_std;
  delete[] x_hat;
  delete[] sum_dy;
  delete[] sum_dy_x_hat;
  delete[] sum_ddx;
  delete[] sum_ddx_x_hat;
  delete[] sum_dy_ddx;
  delete[] dx_term_0;
  delete[] dx_term_1;
  delete[] dx_term_2;
  delete[] dx_term;
  delete[] scale_term;

  return;
}

template <typename T>
void BatchNormGradGradCpuKernel::InferenceComputeNHWC(CpuKernelContext &ctx, int start, int end) {
  auto x_ori = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto dy_ori = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto scale = reinterpret_cast<float *>(ctx.Input(2)->GetData());            // fp32
  auto reserve_space_1 = reinterpret_cast<float *>(ctx.Input(3)->GetData());  // running_mean  fp32
  auto reserve_space_2 = reinterpret_cast<float *>(ctx.Input(4)->GetData());  // running_var  fp32
  auto ddx_ori = reinterpret_cast<T *>(ctx.Input(5)->GetData());
  auto ddscale = reinterpret_cast<float *>(ctx.Input(6)->GetData());   // fp32
  auto ddoffset = reinterpret_cast<float *>(ctx.Input(7)->GetData());  // fp32
  auto dx = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto ddy = reinterpret_cast<T *>(ctx.Output(1)->GetData());
  auto dscale = reinterpret_cast<float *>(ctx.Output(2)->GetData());  // fp32

  int x_num = static_cast<int>(ctx.Input(0)->NumElements());
  int C_num = static_cast<int>(ctx.Input(2)->NumElements());
  int N_num = static_cast<int>(ctx.Input(0)->GetTensorShape()->GetDimSize(0));
  int HW_num = static_cast<int>(x_num / (N_num * C_num));

  // change dtype from 'T' to 'fp32'
  float *x = new float[x_num];
  float *dy = new float[x_num];
  float *ddx = new float[x_num];

  for (int i = 0; i < N_num; i++) {
    for (int k = 0; k < HW_num; k++) {
      for (int j = start; j < end; j++) {
        int index = i * HW_num * C_num + k * C_num + j;
        *(x + index) = static_cast<float>(*(x_ori + index));
        *(dy + index) = static_cast<float>(*(dy_ori + index));
        *(ddx + index) = static_cast<float>(*(ddx_ori + index));
      }
    }
  }

  AttrValue *Attr_epsilon = ctx.GetAttr("epsilon");
  float epsilon = (Attr_epsilon == nullptr) ? 0.0001 : Attr_epsilon->GetFloat();

  float num_1 = 1.0;

  // create intermediate variables
  float *inv_std = new float[C_num];
  float *x_hat = new float[x_num];

  for (int j = start; j < end; j++) {
    *(inv_std + j) = num_1 / sqrt(*(reserve_space_2 + j) + epsilon);
  }

  for (int i = 0; i < N_num; i++) {
    for (int k = 0; k < HW_num; k++) {
      for (int j = start; j < end; j++) {
        int index = i * HW_num * C_num + k * C_num + j;
        *(x_hat + index) = (*(inv_std + j)) * (*(x + index) - *(reserve_space_1 + j));
      }
    }
  }

  // initialize dscale
  for (int j = start; j < end; j++) {
    *(dscale + j) = 0;
  }

  // calculate dx, ddy, dscale
  for (int i = 0; i < N_num; i++) {
    for (int k = 0; k < HW_num; k++) {
      for (int j = start; j < end; j++) {
        int index = i * HW_num * C_num + k * C_num + j;
        *(dx + index) = static_cast<T>((*(ddscale + j)) * (*(inv_std + j)) * (*(dy + index)));
        *(ddy + index) = static_cast<T>((*(ddx + index)) * (*(inv_std + j)) * (*(scale + j)) +
                                        (*(ddscale + j)) * (*(x_hat + index)) + *(ddoffset + j));
        *(dscale + j) += (*(ddx + index)) * (*(dy + index)) * (*(inv_std + j));
      }
    }
  }

  // free memory
  delete[] x;
  delete[] dy;
  delete[] ddx;
  delete[] inv_std;
  delete[] x_hat;

  return;
}

template <typename T>
void BatchNormGradGradCpuKernel::TrainingComputeNCHW(CpuKernelContext &ctx) {
  auto x_ori = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto dy_ori = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto scale = reinterpret_cast<float *>(ctx.Input(2)->GetData());            // fp32
  auto reserve_space_1 = reinterpret_cast<float *>(ctx.Input(3)->GetData());  // batch_mean  fp32
  auto reserve_space_2 = reinterpret_cast<float *>(ctx.Input(4)->GetData());  // batch_var  fp32
  auto ddx_ori = reinterpret_cast<T *>(ctx.Input(5)->GetData());
  auto ddscale = reinterpret_cast<float *>(ctx.Input(6)->GetData());   // fp32
  auto ddoffset = reinterpret_cast<float *>(ctx.Input(7)->GetData());  // fp32
  auto dx = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto ddy = reinterpret_cast<T *>(ctx.Output(1)->GetData());
  auto dscale = reinterpret_cast<float *>(ctx.Output(2)->GetData());  // fp32

  int x_num = static_cast<int>(ctx.Input(0)->NumElements());
  int N_num = static_cast<int>(ctx.Input(0)->GetTensorShape()->GetDimSize(0));
  int C_num = static_cast<int>(ctx.Input(2)->NumElements());
  int CHW_num = static_cast<int>(x_num / N_num);
  int NHW_num = static_cast<int>(x_num / C_num);
  int HW_num = static_cast<int>(NHW_num / N_num);

  // change dtype from 'T' to 'fp32'
  float *x = new float[x_num];
  float *dy = new float[x_num];
  float *ddx = new float[x_num];

  for (int i = 0; i < N_num; i++) {
    for (int k = 0; k < HW_num; k++) {
      for (int j = 0; j < C_num; j++) {
        int index = i * HW_num * C_num + k * C_num + j;
        *(x + index) = static_cast<float>(*(x_ori + index));
        *(dy + index) = static_cast<float>(*(dy_ori + index));
        *(ddx + index) = static_cast<float>(*(ddx_ori + index));
      }
    }
  }

  AttrValue *Attr_epsilon = ctx.GetAttr("epsilon");
  float epsilon = (Attr_epsilon == nullptr) ? 0.0001 : Attr_epsilon->GetFloat();

  float num_1 = 1.0;
  float num_3 = 3.0;
  float M = static_cast<float>((NHW_num));

  // create intermediate variables
  float *inv_std = new float[C_num];
  float *x_hat = new float[x_num];

  for (int j = 0; j < C_num; j++) {
    *(inv_std + j) = num_1 / sqrt(*(reserve_space_2 + j) + epsilon);
  }

  for (int i = 0; i < N_num; i++) {
    for (int j = 0; j < C_num; j++) {
      for (int k = 0; k < HW_num; k++) {
        int index = i * CHW_num + j * HW_num + k;
        *(x_hat + index) = (*(inv_std + j)) * (*(x + index) - *(reserve_space_1 + j));
      }
    }
  }

  // create intermediate variables
  float *sum_dy = new float[C_num];
  float *sum_dy_x_hat = new float[C_num];
  float *sum_ddx = new float[C_num];
  float *sum_ddx_x_hat = new float[C_num];
  float *sum_dy_ddx = new float[C_num];

  // initialize
  for (int j = 0; j < C_num; j++) {
    *(sum_dy + j) = 0;
    *(sum_dy_x_hat + j) = 0;
    *(sum_ddx + j) = 0;
    *(sum_ddx_x_hat + j) = 0;
    *(sum_dy_ddx + j) = 0;
  }

  // compute np.sum(to C dim)
  for (int i = 0; i < N_num; i++) {
    for (int j = 0; j < C_num; j++) {
      for (int k = 0; k < HW_num; k++) {
        int index = i * CHW_num + j * HW_num + k;
        *(sum_dy + j) += *(dy + index);
        *(sum_dy_x_hat + j) += (*(x_hat + index)) * (*(dy + index));
        *(sum_ddx + j) += *(ddx + index);
        *(sum_ddx_x_hat + j) += (*(x_hat + index)) * (*(ddx + index));
        *(sum_dy_ddx + j) += (*(dy + index)) * (*(ddx + index));
      }
    }
  }

  // create intermediate variables
  float *dx_term_0 = new float[x_num];
  float *dx_term_1 = new float[x_num];
  float *dx_term_2 = new float[x_num];
  float *dx_term = new float[x_num];
  float *scale_term = new float[x_num];

  // calculate dx
  for (int i = 0; i < N_num; i++) {
    for (int j = 0; j < C_num; j++) {
      for (int k = 0; k < HW_num; k++) {
        int index = i * CHW_num + j * HW_num + k;
        *(dx_term_0 + index) = (*(x_hat + index)) *
                               ((*(sum_ddx + j)) * (*(sum_dy + j)) / M - (*(sum_dy_ddx + j)) +
                                num_3 * (*(sum_dy_x_hat + j)) * (*(sum_ddx_x_hat + j)) / M) /
                               M;
        *(dx_term_1 + index) = (*(sum_ddx_x_hat + j)) * ((*(sum_dy + j)) / M - (*(dy + index))) / M;
        *(dx_term_2 + index) = (*(sum_dy_x_hat + j)) * ((*(sum_ddx + j)) / M - (*(ddx + index))) / M;
      }
    }
  }

  for (int i = 0; i < N_num; i++) {
    for (int j = 0; j < C_num; j++) {
      for (int k = 0; k < HW_num; k++) {
        int index = i * CHW_num + j * HW_num + k;
        *(dx_term + index) = (*(scale + j)) / ((*(reserve_space_2 + j)) + epsilon) *
                             ((*(dx_term_0 + index)) + (*(dx_term_1 + index)) + (*(dx_term_2 + index)));
        *(scale_term + index) =
          (*(ddscale + j)) * (*(inv_std + j)) *
          ((*(dy + index)) - (*(sum_dy + j)) / M - (*(sum_dy_x_hat + j)) / M * (*(x_hat + index)));
      }
    }
  }

  for (int i = 0; i < N_num; i++) {
    for (int j = 0; j < C_num; j++) {
      for (int k = 0; k < HW_num; k++) {
        int index = i * CHW_num + j * HW_num + k;
        *(dx + index) = static_cast<T>((*(dx_term + index)) + (*(scale_term + index)));
      }
    }
  }

  // calculate ddy
  for (int i = 0; i < N_num; i++) {
    for (int j = 0; j < C_num; j++) {
      for (int k = 0; k < HW_num; k++) {
        int index = i * CHW_num + j * HW_num + k;
        *(ddy + index) =
          static_cast<T>((*(scale + j)) * (*(inv_std + j)) / M *
                           (M * (*(ddx + index)) - (*(sum_ddx + j)) - (*(x_hat + index)) * (*(sum_ddx_x_hat + j))) +
                         (*(ddscale + j)) * (*(x_hat + index)) + (*(ddoffset + j)));
      }
    }
  }

  // calculate dscale
  for (int j = 0; j < C_num; j++) {
    *(dscale + j) = 0;
  }

  for (int i = 0; i < N_num; i++) {
    for (int j = 0; j < C_num; j++) {
      for (int k = 0; k < HW_num; k++) {
        int index = i * CHW_num + j * HW_num + k;
        *(dscale + j) += (*(ddx + index)) * (*(inv_std + j)) *
                         ((*(dy + index)) - (*(sum_dy + j)) / M - (*(sum_dy_x_hat + j)) / M * (*(x_hat + index)));
      }
    }
  }

  // free memory
  delete[] x;
  delete[] dy;
  delete[] ddx;
  delete[] inv_std;
  delete[] x_hat;
  delete[] sum_dy;
  delete[] sum_dy_x_hat;
  delete[] sum_ddx;
  delete[] sum_ddx_x_hat;
  delete[] sum_dy_ddx;
  delete[] dx_term_0;
  delete[] dx_term_1;
  delete[] dx_term_2;
  delete[] dx_term;
  delete[] scale_term;

  return;
}

template <typename T>
void BatchNormGradGradCpuKernel::InferenceComputeNCHW(CpuKernelContext &ctx) {
  auto x_ori = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto dy_ori = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto scale = reinterpret_cast<float *>(ctx.Input(2)->GetData());            // fp32
  auto reserve_space_1 = reinterpret_cast<float *>(ctx.Input(3)->GetData());  // running_mean  fp32
  auto reserve_space_2 = reinterpret_cast<float *>(ctx.Input(4)->GetData());  // running_var  fp32
  auto ddx_ori = reinterpret_cast<T *>(ctx.Input(5)->GetData());
  auto ddscale = reinterpret_cast<float *>(ctx.Input(6)->GetData());   // fp32
  auto ddoffset = reinterpret_cast<float *>(ctx.Input(7)->GetData());  // fp32
  auto dx = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto ddy = reinterpret_cast<T *>(ctx.Output(1)->GetData());
  auto dscale = reinterpret_cast<float *>(ctx.Output(2)->GetData());  // fp32

  int x_num = static_cast<int>(ctx.Input(0)->NumElements());
  int N_num = static_cast<int>(ctx.Input(0)->GetTensorShape()->GetDimSize(0));
  int C_num = static_cast<int>(ctx.Input(2)->NumElements());
  int CHW_num = static_cast<int>(x_num / N_num);
  int HW_num = static_cast<int>(CHW_num / C_num);

  // change dtype from 'T' to 'fp32'
  float *x = new float[x_num];
  float *dy = new float[x_num];
  float *ddx = new float[x_num];

  for (int i = 0; i < N_num; i++) {
    for (int k = 0; k < HW_num; k++) {
      for (int j = 0; j < C_num; j++) {
        int index = i * HW_num * C_num + k * C_num + j;
        *(x + index) = static_cast<float>(*(x_ori + index));
        *(dy + index) = static_cast<float>(*(dy_ori + index));
        *(ddx + index) = static_cast<float>(*(ddx_ori + index));
      }
    }
  }

  AttrValue *Attr_epsilon = ctx.GetAttr("epsilon");
  float epsilon = (Attr_epsilon == nullptr) ? 0.0001 : Attr_epsilon->GetFloat();

  float num_1 = 1.0;

  // create intermediate variables
  float *inv_std = new float[C_num];
  float *x_hat = new float[x_num];

  for (int j = 0; j < C_num; j++) {
    *(inv_std + j) = num_1 / sqrt(*(reserve_space_2 + j) + epsilon);
  }

  for (int i = 0; i < N_num; i++) {
    for (int j = 0; j < C_num; j++) {
      for (int k = 0; k < HW_num; k++) {
        int index = i * CHW_num + j * HW_num + k;
        *(x_hat + index) = (*(inv_std + j)) * (*(x + index) - *(reserve_space_1 + j));
      }
    }
  }

  // initialize dscale
  for (int j = 0; j < C_num; j++) {
    *(dscale + j) = 0;
  }

  // calculate dx, ddy, dscale
  for (int i = 0; i < N_num; i++) {
    for (int j = 0; j < C_num; j++) {
      for (int k = 0; k < HW_num; k++) {
        int index = i * CHW_num + j * HW_num + k;
        *(dx + index) = static_cast<T>((*(ddscale + j)) * (*(inv_std + j)) * (*(dy + index)));
        *(ddy + index) = static_cast<T>((*(ddx + index)) * (*(inv_std + j)) * (*(scale + j)) +
                                        (*(ddscale + j)) * (*(x_hat + index)) + *(ddoffset + j));
        *(dscale + j) += (*(ddx + index)) * (*(dy + index)) * (*(inv_std + j));
      }
    }
  }

  // free memory
  delete[] x;
  delete[] dy;
  delete[] ddx;
  delete[] inv_std;
  delete[] x_hat;

  return;
}

REGISTER_MS_CPU_KERNEL(kBatchNormGradGrad, BatchNormGradGradCpuKernel);
}  // namespace aicpu
