/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

#include "glu.h"

#include <cmath>
#include <iostream>

#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *GLU = "Glu";
const int64_t kParallelDataNum = 16 * 1024;
}  // namespace

namespace aicpu {
uint32_t GluCpuKernel::CheckAndInitParams(CpuKernelContext &ctx) {
  // get input value
  Tensor *value_ptr = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(value_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input value failed.");
  value_data_ptr_ = value_ptr->GetData();
  KERNEL_CHECK_NULLPTR(value_data_ptr_, KERNEL_STATUS_PARAM_INVALID, "Get input value data failed.");
  auto value_shape_ptr = value_ptr->GetTensorShape();
  KERNEL_CHECK_NULLPTR(value_shape_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input value shape failed.");
  value_dim_ = value_shape_ptr->GetDims();
  // get Attr axis
  AttrValue *split_dim_ptr = ctx.GetAttr("axis");
  if (split_dim_ptr != nullptr) {
    split_dim_ = split_dim_ptr->GetInt();
  } else {
    split_dim_ = -1;
  }
  KERNEL_CHECK_FALSE((value_dim_ > split_dim_) && (-value_dim_ <= split_dim_), KERNEL_STATUS_PARAM_INVALID,
                     "Dim of Input value must lesser than value_dim_ and greater than or "
                     "equal to minus value_dim_, split_dim_ is [%d], value_dim_ is [%d].",
                     split_dim_, value_dim_);
  if (split_dim_ < 0) {
    split_dim_ += value_dim_;
  }
  data_type_ = value_ptr->GetDataType();
  value_num_ = value_ptr->NumElements();
  value_shape_vec_ = value_shape_ptr->GetDimSizes();
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Glu check input and output number failed.");

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GluCpuKernel::SplitWithDimZero(const CpuKernelContext &ctx, T *input_data_ptr, T *output_data_ptr) {
  int64_t copy_num = value_num_ / value_shape_vec_[0];
  T *input_copy_ptr = input_data_ptr;
  KERNEL_CHECK_FALSE((value_shape_vec_[0] % 2 == 0), KERNEL_STATUS_PARAM_INVALID,
                     "The length of the split dimension must be even.", value_shape_vec_[0]);
  int64_t size_split = value_shape_vec_[0] / 2;
  // set output[0]
  int64_t copy_size_per = size_split * copy_num;
  input_copy_ptr += copy_size_per;
  // set output[1]
  uint32_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
  auto sharder_glu = [&](int64_t start, int64_t end) {
    for (int64_t k = start; k < end; k++) {
      T val = *(input_copy_ptr + k);
      *(output_data_ptr + k) = (T(1) / (T(1) + exp(-val))) * (*(input_data_ptr + k));
    }
  };
  if (copy_size_per < kParallelDataNum) {
    sharder_glu(0, copy_size_per);
  } else {
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, copy_size_per, copy_size_per / max_core_num, sharder_glu),
                        "Glu Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GluCpuKernel::SplitCompute(const CpuKernelContext &ctx, T *input_data_ptr, T *output_data_ptr) {
  uint32_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
  int64_t prefix = 1;
  for (int32_t i = 0; i < split_dim_; i++) {
    prefix *= value_shape_vec_[i];
  }
  int64_t midfix = value_shape_vec_[split_dim_];
  KERNEL_CHECK_FALSE((midfix % 2 == 0), KERNEL_STATUS_PARAM_INVALID, "The length of the split dimension must be even.",
                     midfix);
  int64_t size_split = midfix / 2;
  int64_t subfix = 1;
  for (size_t i = split_dim_ + 1; i < value_shape_vec_.size(); i++) {
    subfix *= value_shape_vec_[i];
  }
  int64_t offset = 0;
  T *input_copy_ptr = input_data_ptr;
  // set output[0]
  int64_t copy_num = subfix * size_split;
  offset += copy_num;
  // output[1]
  input_data_ptr = input_data_ptr + offset;
  auto sharder_glu = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      auto sharder_input_data = input_data_ptr + (i) * (subfix * midfix);
      auto sharder_input_copy = input_copy_ptr + (i) * (subfix * midfix);
      auto sharder_output = output_data_ptr + (i)*copy_num;
      for (int64_t k = 0; k < copy_num; k++) {
        T val = *(sharder_input_data + k);
        *(sharder_output + k) = (T(1) / (T(1) + exp(-val))) * (*(sharder_input_copy + k));
      }
    }
  };
  if (prefix < kParallelDataNum) {
    sharder_glu(0, prefix);
  } else {
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, prefix, prefix / max_core_num, sharder_glu),
                        "Glu Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GluCpuKernel::DoCompute(CpuKernelContext &ctx) {
  Tensor *output_ptr = ctx.Output(0);
  auto dest_output_data_ptr = output_ptr->GetData();
  T *dest_output_backup_ptr = reinterpret_cast<T *>(dest_output_data_ptr);
  T *input_data_ptr = reinterpret_cast<T *>(value_data_ptr_);
  if (split_dim_ == 0) {
    KERNEL_CHECK_FALSE((SplitWithDimZero<T>(ctx, input_data_ptr, dest_output_backup_ptr) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "SplitWithDimZero failed.");
    return KERNEL_STATUS_OK;
  } else {
    KERNEL_CHECK_FALSE((SplitCompute<T>(ctx, input_data_ptr, dest_output_backup_ptr) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "Split Compute failed.");
    return KERNEL_STATUS_OK;
  }
}

uint32_t GluCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_CHECK_FALSE((CheckAndInitParams(ctx) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "CheckAndInitParams failed.");
  switch (data_type_) {
    case DT_FLOAT16:
      return DoCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return DoCompute<float>(ctx);
    case DT_DOUBLE:
      return DoCompute<double>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported datatype[%s]", DTypeStr(data_type_).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_CPU_KERNEL(GLU, GluCpuKernel);
}  // namespace aicpu
