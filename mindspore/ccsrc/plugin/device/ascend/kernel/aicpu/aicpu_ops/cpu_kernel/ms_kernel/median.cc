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

#include "median.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

#include <algorithm>

namespace {
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 2;
const char *kMedian = "Median";

#define MEDIAN_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                        \
    uint32_t result = MedianCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                    \
      KERNEL_LOG_ERROR("Median kernel compute failed."); \
      return result;                                     \
    }                                                    \
    break;                                               \
  }

#define GLOBAL_MEDIAN_COMPUTE_CASE(DTYPE, TYPE, CTX)     \
  case (DTYPE): {                                        \
    uint32_t result = GlobalMedianCompute<TYPE>(CTX);    \
    if (result != KERNEL_STATUS_OK) {                    \
      KERNEL_LOG_ERROR("Median kernel compute failed."); \
      return result;                                     \
    }                                                    \
    break;                                               \
  }
}  // namespace

namespace aicpu {
uint32_t MedianCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(MedianCheck(ctx), "Median check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  AttrValue *global_ptr = ctx.GetAttr("global_median");
  bool global_median_bool = global_ptr->GetBool();
  if (global_median_bool == false) {
    switch (data_type) {
      MEDIAN_COMPUTE_CASE(DT_INT16, int16_t, ctx)
      MEDIAN_COMPUTE_CASE(DT_INT32, int32_t, ctx)
      MEDIAN_COMPUTE_CASE(DT_INT64, int64_t, ctx)
      MEDIAN_COMPUTE_CASE(DT_FLOAT, float, ctx)
      MEDIAN_COMPUTE_CASE(DT_DOUBLE, double, ctx)
      default:
        KERNEL_LOG_ERROR("Median kernel data type [%s] not support.", DTypeStr(data_type).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
  } else {
    switch (data_type) {
      GLOBAL_MEDIAN_COMPUTE_CASE(DT_INT16, int16_t, ctx)
      GLOBAL_MEDIAN_COMPUTE_CASE(DT_INT32, int32_t, ctx)
      GLOBAL_MEDIAN_COMPUTE_CASE(DT_INT64, int64_t, ctx)
      GLOBAL_MEDIAN_COMPUTE_CASE(DT_FLOAT, float, ctx)
      GLOBAL_MEDIAN_COMPUTE_CASE(DT_DOUBLE, double, ctx)
      default:
        KERNEL_LOG_ERROR("Median kernel data type [%s] not support.", DTypeStr(data_type).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t MedianCpuKernel::MedianCheck(CpuKernelContext &ctx) {
  auto global_median = ctx.GetAttr("global_median");
  KERNEL_CHECK_NULLPTR(global_median, KERNEL_STATUS_PARAM_INVALID, "Get attr global_median failed.");
  bool global_median_value = global_median->GetBool();
  if (global_median_value == false) {
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Median check input and output number failed.");
    auto input_shape_ptr = ctx.Input(0)->GetTensorShape();
    int64_t input_shape_dims = input_shape_ptr->GetDims();
    int64_t dim_num = 0;
    AttrValue *dim_ptr = ctx.GetAttr("axis");
    if (dim_ptr != nullptr) dim_num = dim_ptr->GetInt();
    if (input_shape_dims != 0) {
      KERNEL_CHECK_FALSE((dim_num >= (0 - input_shape_dims) && dim_num <= (input_shape_dims - 1)),
                         KERNEL_STATUS_PARAM_INVALID,
                         "IndexError: Dimension out of range "
                         "(expected to be in range of [[%lld], [%lld]], but got [%lld])",
                         (0 - input_shape_dims), (input_shape_dims - 1), dim_num);
    } else {
      KERNEL_CHECK_FALSE((dim_num >= -1 && dim_num <= 0), KERNEL_STATUS_PARAM_INVALID,
                         "IndexError: Dimension out of range "
                         "(expected to be in range of [[%lld], [%lld]], but got [%lld])",
                         -1, 0, dim_num);
    }
  } else {
    Tensor *input_0 = ctx.Input(0);
    KERNEL_CHECK_NULLPTR(input_0, KERNEL_STATUS_PARAM_INVALID, "Get input failed.");
    KERNEL_CHECK_NULLPTR(input_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.");
    Tensor *output_0 = ctx.Output(0);
    KERNEL_CHECK_NULLPTR(output_0, KERNEL_STATUS_PARAM_INVALID, "Get output_0 failed.");
    KERNEL_CHECK_NULLPTR(output_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data 0 failed.");
  }
  if (global_median_value == false) {
    KERNEL_LOG_DEBUG(
      "MedianCpuKernel[%s], input0: size[%llu];"
      "output0: size[%llu], output1: size[%llu].",
      ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize(), ctx.Output(1)->GetDataSize());
  } else {
    KERNEL_LOG_DEBUG(
      "MedianCpuKernel[%s], input0: size[%llu];"
      "output0: size[%llu].",
      ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MedianCpuKernel::GlobalMedianCompute(CpuKernelContext &ctx) {
  auto input_x0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y0 = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  size_t data_num = ctx.Input(0)->GetTensorShape()->NumElements();
  const int64_t half = 2;
  std::nth_element(input_x0, input_x0 + static_cast<int64_t>((data_num - 1) / half), input_x0 + data_num);
  *output_y0 = *(input_x0 + static_cast<int64_t>((data_num - 1) / half));
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MedianCpuKernel::MedianCompute(CpuKernelContext &ctx) {
  auto input_x0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y0 = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto output_y1 = reinterpret_cast<int64_t *>(ctx.Output(1)->GetData());

  auto input_shape_ptr = ctx.Input(0)->GetTensorShape();
  int64_t input_shape_dims = input_shape_ptr->GetDims();
  if (input_shape_dims == 0) {
    *output_y0 = *input_x0;
    *output_y1 = 0;
    return KERNEL_STATUS_OK;
  }

  int64_t dim_num = 0;
  AttrValue *dim_ptr = ctx.GetAttr("axis");
  if (dim_ptr != nullptr) {
    dim_num = dim_ptr->GetInt();
  }
  if (dim_num < 0) {
    dim_num += input_shape_dims;
  }
  auto input_shape_0 = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  int64_t dim_data_num = input_shape_0[dim_num];
  T *temp_median_vec = new T[dim_data_num];
  int64_t *temp_median_index_vec = new int64_t[dim_data_num];
  int64_t group = 1;
  int64_t jump = 1;

  int64_t median_pos = static_cast<int64_t>((dim_data_num - 1) / 2);

  if (dim_num != 0) {
    for (int64_t i = 0; i < dim_num; i++) {
      group *= input_shape_0[i];
    }
  }
  if (dim_num != input_shape_dims - 1) {
    for (int64_t i = dim_num + 1; i < input_shape_dims; i++) {
      jump *= input_shape_0[i];
    }
  }

  T *start = input_x0;
  for (int64_t i = 0; i < group; i++) {
    for (int64_t j = 0; j < jump; j++) {
      for (int64_t k = 0; k < dim_data_num; k++) {
        auto num_index = start + k * jump + j;
        temp_median_index_vec[k] = k;
        temp_median_vec[k] = *num_index;
      }
      std::nth_element(temp_median_index_vec, temp_median_index_vec + median_pos, temp_median_index_vec + dim_data_num,
                       [&temp_median_vec, dim_data_num](int64_t pos1, int64_t pos2) {
                         return (pos1 >= 0 && pos1 < dim_data_num && pos1 < pos2) &&
                                (*(temp_median_vec + pos1) < *(temp_median_vec + pos2) ||
                                 *(temp_median_vec + pos1) == *(temp_median_vec + pos2));
                       });
      std::nth_element(temp_median_vec, temp_median_vec + median_pos, temp_median_vec + dim_data_num);
      *(output_y0 + i * jump + j) = *(temp_median_vec + median_pos);
      *(output_y1 + i * jump + j) = *(temp_median_index_vec + median_pos);
    }
    if (i != group - 1) {
      start += jump * dim_data_num;
    }
  };

  delete[] temp_median_vec;
  delete[] temp_median_index_vec;

  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kMedian, MedianCpuKernel);
}  // namespace aicpu
