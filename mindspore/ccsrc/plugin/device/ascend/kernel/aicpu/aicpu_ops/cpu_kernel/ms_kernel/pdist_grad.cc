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

#include "pdist_grad.h"

#include <algorithm>
#include <math.h>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "kernel_log.h"
#include "status.h"

namespace {
const char *kPdistGrad = "PdistGrad";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;
constexpr int64_t kParallelDataNums = 16 * 1024;
constexpr int64_t kParallelDataNumsMid = 7 * 1024;

#define SWITCH_PARALLEL(SHARD, end_num, divisor)                                                  \
  if (end_num >= (kParallelDataNumsMid / divisor)) {                                              \
    uint32_t min_core_num = 1;                                                                    \
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);     \
    if (end_num < (kParallelDataNums / divisor)) {                                                \
      max_core_num = std::min(max_core_num, 4L);                                                  \
    }                                                                                             \
    if (max_core_num > end_num) {                                                                 \
      max_core_num = end_num;                                                                     \
    }                                                                                             \
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, end_num, end_num / max_core_num, SHARD), \
                        "PdistGrad #SHARD Compute failed.");                                      \
  } else {                                                                                        \
    SHARD(0, end_num);                                                                            \
  }
}  // namespace

namespace aicpu {
template <typename T>
struct Grad {
  static inline T abs(T x) { return static_cast<T>(std::abs(*((float *)&x))); }

  static inline T pow(T x, float p) { return static_cast<T>(std::pow(*((float *)&x), p)); }

  static inline T sign(T x) { return x > T{0.0f} ? T{1.0f} : T{-1.0f}; }

  struct o_grad {
    static inline T backward(T diff, T grad, T dist, float p) { return diff > T{0.0f} ? grad : -grad; }
  };

  struct t_grad {
    static inline float backward(float diff, float grad, float dist, float p) {
      return dist == 0.0f ? 0.0f : grad * diff / dist;
    }

    static inline Eigen::half backward(Eigen::half diff, Eigen::half grad, Eigen::half dist, float p) {
      return dist == Eigen::half{0.0f} ? Eigen::half{0.0f}
                                       : sign(diff) * pow(abs(diff), p - 1) * grad / pow(dist, p - 1);
    }
  };

  struct p_grad {
    static inline T backward(T diff, T grad, T dist, float p) {
      return dist == T{0.0f} ? T{0.0f} : sign(diff) * pow(abs(diff), p - 1) * grad / pow(dist, p - 1);
    }
  };

  struct i_grad {
    static inline T backward(T diff, T grad, T dist, float p) {
      return (diff == dist || -diff == dist) ? sign(diff) * grad : T{0.0f};
    }
  };

  template <typename S>
  static uint32_t ParallelForPdistGrad(T *grad, T *x, T *dist, T *y, float p, CpuKernelContext &ctx) {
    int64_t data_num = ctx.Input(1)->NumElements();
    int64_t n = ctx.Input(1)->GetTensorShape()->GetDimSize(0);
    int64_t m = ctx.Input(1)->GetTensorShape()->GetDimSize(1);
    auto shard_pdistgrad = [&](int64_t start, int64_t end) {
      int64_t index;
      for (int64_t col = start; col < end; col++) {
        index = 0;
        for (int64_t i = col; i < data_num; i += m) {
          for (int64_t j = i + m; j < data_num; j += m) {
            T diff = x[i] - x[j];
            if (diff == T{0.0f}) {
              index++;
              continue;
            }
            T result = S::backward(diff, grad[index], dist[index], p);
            *(y + i) += result;
            *(y + j) -= result;
            index++;
          }
        }
      }
    };
    SWITCH_PARALLEL(shard_pdistgrad, m, n);
    return KERNEL_STATUS_OK;
  }

  static inline uint32_t PdistGradComputeKernel(T *grad, T *x, T *dist, T *y, float p, CpuKernelContext &ctx) {
    int64_t data_num = ctx.Input(1)->NumElements();
    T zero = T{0};
    auto shard_fill = [&](int64_t start, int64_t end) { std::fill(y + start, y + end, zero); };
    SWITCH_PARALLEL(shard_fill, data_num, 1);
    if (p == 0.0) {
      return KERNEL_STATUS_OK;
    } else if (p == 1.0) {
      return ParallelForPdistGrad<o_grad>(grad, x, dist, y, p, ctx);
    } else if (p == 2.0) {
      return ParallelForPdistGrad<t_grad>(grad, x, dist, y, p, ctx);
    } else if (std::isinf(p)) {
      return ParallelForPdistGrad<i_grad>(grad, x, dist, y, p, ctx);
    } else {
      return ParallelForPdistGrad<p_grad>(grad, x, dist, y, p, ctx);
    }
  }
};  // Grad

uint32_t PdistGradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "PdistGrad check input and output number failed.");
  DataType input_type = ctx.Input(1)->GetDataType();
  DataType output_type = ctx.Output(0)->GetDataType();
  KERNEL_CHECK_FALSE((input_type == output_type), KERNEL_STATUS_PARAM_INVALID,
                     "Input data type[%s] is not equal to output data type[%s].", DTypeStr(input_type).c_str(),
                     DTypeStr(output_type).c_str());
  uint64_t input_size = ctx.Input(1)->GetDataSize();
  uint64_t output_size = ctx.Output(0)->GetDataSize();
  KERNEL_CHECK_FALSE((input_size == output_size), KERNEL_STATUS_PARAM_INVALID,
                     "Input data size[%llu] is not equal to output data size[%llu].", input_size, output_size);
  switch (input_type) {
    case DT_FLOAT16:
      return PdistGradCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return PdistGradCompute<float>(ctx);
    default:
      KERNEL_LOG_ERROR("PdistGrad kernel data type [%s] not support.", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t PdistGradCpuKernel::PdistGradCompute(CpuKernelContext &ctx) {
  Tensor *grad_tensor = ctx.Input(0);
  Tensor *x_tensor = ctx.Input(1);
  Tensor *pdist_tensor = ctx.Input(2);
  Tensor *y_tensor = ctx.Output(0);

  T *grad = reinterpret_cast<T *>(grad_tensor->GetData());
  T *x = reinterpret_cast<T *>(x_tensor->GetData());
  T *pdist = reinterpret_cast<T *>(pdist_tensor->GetData());
  T *y = reinterpret_cast<T *>(y_tensor->GetData());

  float p = 2.0;
  AttrValue *p_attr = ctx.GetAttr("p");
  if (p_attr != nullptr) {
    p = p_attr->GetFloat();
  }
  KERNEL_CHECK_FALSE((p >= 0), KERNEL_STATUS_PARAM_INVALID, "Attr[p] data cannot be less than 0.");

  uint32_t ret = Grad<T>::PdistGradComputeKernel(grad, x, pdist, y, p, ctx);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kPdistGrad, PdistGradCpuKernel);
}  // namespace aicpu
