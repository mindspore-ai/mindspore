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
#include "exp.h"

#include "cmath"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kExp = "Exp";
constexpr size_t kDataNum = 1024;

#define EXP_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                     \
    uint32_t result = ExpCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                 \
      KERNEL_LOG_ERROR("Exp kernel compute failed."); \
      return result;                                  \
    }                                                 \
    break;                                            \
  }

#define EXP_COMPUTE_CASE2(DTYPE, TYPE, CTX)           \
  case (DTYPE): {                                     \
    uint32_t result = ExpCompute2(CTX);               \
    if (result != KERNEL_STATUS_OK) {                 \
      KERNEL_LOG_ERROR("Exp kernel compute failed."); \
      return result;                                  \
    }                                                 \
    break;                                            \
  }
}  // namespace

namespace aicpu {
uint32_t ExpCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kExp);
  KERNEL_HANDLE_ERROR(ExpCheck(ctx), "[%s] check params failed.", kExp);
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    EXP_COMPUTE_CASE2(DT_FLOAT16, Eigen::half, ctx)
    EXP_COMPUTE_CASE(DT_FLOAT, float, ctx)
    EXP_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    EXP_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    EXP_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("Exp kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t ExpCpuKernel::ExpCheck(CpuKernelContext &ctx) {
  auto input_0 = ctx.Input(0);
  auto output_0 = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(input_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
  KERNEL_CHECK_NULLPTR(output_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
  KERNEL_CHECK_NULLPTR(input_0->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get input tensor shape failed.")
  std::vector<int64_t> shape_x = input_0->GetTensorShape()->GetDimSizes();
  size_t shape_size = shape_x.size();
  KERNEL_CHECK_FALSE((shape_size >= 0), KERNEL_STATUS_PARAM_INVALID, "Input must be at least rank 0, got [%zu].",
                     shape_x.size())
  if (shape_size > 0) {
    KERNEL_CHECK_FALSE((shape_x[shape_size - 1] > 0), KERNEL_STATUS_PARAM_INVALID,
                       "Input last dimension must be at least 1.")
  }
  AttrValue *base_ptr = ctx.GetAttr("base");
  KERNEL_CHECK_NULLPTR(base_ptr, KERNEL_STATUS_PARAM_INVALID, "Get attr base failed.");
  float base_ = base_ptr->GetFloat();
  KERNEL_CHECK_FALSE(((base_ > 0) || base_ == -1.0), KERNEL_STATUS_PARAM_INVALID,
                     "Attr base must be -1.0  or base > 0, but got attr base[%lld]", base_);
  AttrValue *scale_ptr = ctx.GetAttr("scale");
  KERNEL_CHECK_NULLPTR(scale_ptr, KERNEL_STATUS_PARAM_INVALID, "Get attr scale failed.");
  AttrValue *shift_ptr = ctx.GetAttr("shift");
  KERNEL_CHECK_NULLPTR(shift_ptr, KERNEL_STATUS_PARAM_INVALID, "Get attr shift failed.");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ExpCpuKernel::ExpCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  AttrValue *base_ptr = ctx.GetAttr("base");
  T base_;
  base_ = static_cast<T>(base_ptr->GetFloat());
  if (base_ == static_cast<T>(-1.0)) {
    base_ = static_cast<T>(exp(1.0));
  }
  AttrValue *scale_ptr = ctx.GetAttr("scale");
  T scale_;
  scale_ = static_cast<T>(scale_ptr->GetFloat());
  AttrValue *shift_ptr = ctx.GetAttr("shift");
  T shift_;
  shift_ = static_cast<T>(shift_ptr->GetFloat());

  size_t data_num = ctx.Input(0)->NumElements();
  if (data_num <= kDataNum) {
    for (size_t i = 0; i < data_num; i++) {
      *(output_y + i) = pow(base_, shift_ + scale_ * (*(input_x + i)));
    }
  } else {
    uint32_t min_core_num = 1;
    size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_exp = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        *(output_y + i) = pow(base_, shift_ + scale_ * (*(input_x + i)));
      }
    };
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_exp),
                        "Exp Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

uint32_t ExpCpuKernel::ExpCompute2(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<Eigen::half *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<Eigen::half *>(ctx.Output(0)->GetData());
  size_t data_num = ctx.Input(0)->NumElements();
  using ArrayxXd = Eigen::Array<Eigen::half, Eigen::Dynamic, Eigen::Dynamic>;
  ArrayxXd array_x(1, data_num);
  ArrayxXd array_y(1, data_num);
  ArrayxXd array_base_(1, data_num);
  AttrValue *base_ptr = ctx.GetAttr("base");
  Eigen::half base_;
  base_ = static_cast<Eigen::half>(base_ptr->GetFloat());
  AttrValue *scale_ptr = ctx.GetAttr("scale");
  Eigen::half scale_;
  scale_ = static_cast<Eigen::half>(scale_ptr->GetFloat());
  AttrValue *shift_ptr = ctx.GetAttr("shift");
  Eigen::half shift_;
  shift_ = static_cast<Eigen::half>(shift_ptr->GetFloat());
  if (base_ == static_cast<Eigen::half>(-1.0)) {
    base_ = static_cast<Eigen::half>(exp(1.0));
    for (size_t i = 0; i < data_num; i++) {
      array_x(0, i) = *(input_x + i);
    }
    array_x = array_x * scale_;
    array_x = array_x + shift_;
    if (data_num <= kDataNum) {
      array_y = array_x.exp();
      for (size_t i = 0; i < data_num; i++) {
        *(output_y + i) = array_y(0, i);
      }
    } else {
      uint32_t min_core_num = 1;
      size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
      if (max_core_num > data_num) {
        max_core_num = data_num;
      }
      auto shard_exp = [&](size_t start, size_t end) {
        array_y = array_x.exp();
        for (size_t i = start; i < end; i++) {
          *(output_y + i) = array_y(0, i);
        }
      };
      if (max_core_num == 0) {
        KERNEL_LOG_ERROR("max_core_num could not be 0.");
      }
      KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_exp),
                          "Exp Compute failed.");
    }
  } else {
    for (size_t i = 0; i < data_num; i++) {
      array_x(0, i) = *(input_x + i);
      array_base_(0, i) = base_;
    }
    array_x = array_x * scale_;
    array_x = array_x + shift_;
    if (data_num <= kDataNum) {
      array_y = array_base_.pow(array_x);
      for (size_t i = 0; i < data_num; i++) {
        *(output_y + i) = array_y(0, i);
      }
    } else {
      uint32_t min_core_num = 1;
      size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
      if (max_core_num > data_num) {
        max_core_num = data_num;
      }
      auto shard_exp = [&](size_t start, size_t end) {
        array_y = array_base_.pow(array_x);
        for (size_t i = start; i < end; i++) {
          *(output_y + i) = array_y(0, i);
        }
      };
      if (max_core_num == 0) {
        KERNEL_LOG_ERROR("max_core_num could not be 0.");
      }
      KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_exp),
                          "Exp Compute failed.");
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kExp, ExpCpuKernel);
}  // namespace aicpu
