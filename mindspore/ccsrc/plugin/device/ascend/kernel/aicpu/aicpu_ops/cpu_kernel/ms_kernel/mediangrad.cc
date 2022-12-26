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

#include "mediangrad.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kMedianGrad = "MedianGrad";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 4;
const uint32_t kGlobalOutputNum = 1;
const uint32_t kGlobalInputNum = 3;
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;

#define MEDIANGRAD_COMPUTE_CASE(DTYPE, TYPE, TYPE2, CTX)     \
  case (DTYPE): {                                            \
    uint32_t result = MedianGradCompute<TYPE, TYPE2>(CTX);   \
    if (result != KERNEL_STATUS_OK) {                        \
      KERNEL_LOG_ERROR("MedianGrad kernel compute failed."); \
      return result;                                         \
    }                                                        \
    break;                                                   \
  }

#define GLOBALMEDIANGRAD_COMPUTE_CASE(DTYPE, TYPE, TYPE2, CTX)     \
  case (DTYPE): {                                                  \
    uint32_t result = GlobalMedianGradCompute<TYPE, TYPE2>(CTX);   \
    if (result != KERNEL_STATUS_OK) {                              \
      KERNEL_LOG_ERROR("GlobalMedianGrad kernel compute failed."); \
      return result;                                               \
    }                                                              \
    break;                                                         \
  }
}  // namespace

namespace aicpu {
uint32_t MedianGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(MedianGradParamCheck(ctx), "MedianGrad check params failed.");
  auto data_type_x = ctx.Input(1)->GetDataType();
  AttrValue *global_median_ptr = ctx.GetAttr("global_median");
  bool global_median = global_median_ptr->GetBool();
  if (global_median == false) {
    switch (data_type_x) {
      MEDIANGRAD_COMPUTE_CASE(DT_INT16, int16_t, float, ctx)
      MEDIANGRAD_COMPUTE_CASE(DT_INT32, int32_t, float, ctx)
      MEDIANGRAD_COMPUTE_CASE(DT_INT64, int64_t, float, ctx)
      MEDIANGRAD_COMPUTE_CASE(DT_FLOAT, float, float, ctx)
      MEDIANGRAD_COMPUTE_CASE(DT_DOUBLE, double, double, ctx)
      default:
        KERNEL_LOG_ERROR("MedianGrad kernel data type [%s] of input x not support.", DTypeStr(data_type_x).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
  } else {
    switch (data_type_x) {
      GLOBALMEDIANGRAD_COMPUTE_CASE(DT_INT16, int16_t, float, ctx)
      GLOBALMEDIANGRAD_COMPUTE_CASE(DT_INT32, int32_t, float, ctx)
      GLOBALMEDIANGRAD_COMPUTE_CASE(DT_INT64, int64_t, float, ctx)
      GLOBALMEDIANGRAD_COMPUTE_CASE(DT_FLOAT, float, float, ctx)
      GLOBALMEDIANGRAD_COMPUTE_CASE(DT_DOUBLE, double, double, ctx)
      default:
        KERNEL_LOG_ERROR("GlobalMedianGrad kernel data type [%s] of input x not support.",
                         DTypeStr(data_type_x).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t MedianGradCpuKernel::MedianGradParamCheck(CpuKernelContext &ctx) {
  auto global_median_ptr = ctx.GetAttr("global_median");
  KERNEL_CHECK_NULLPTR(global_median_ptr, KERNEL_STATUS_PARAM_INVALID, "Get attr global_median failed.");
  bool global_median = global_median_ptr->GetBool();

  if (global_median == false) {
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "MedianGrad check input and output number failed.");
  } else {
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kGlobalInputNum, kGlobalOutputNum),
                        "GlobalMedianGrad check input and output number failed.");
  }

  Tensor *input_y_grad = ctx.Input(0);
  Tensor *input_x = ctx.Input(1);
  Tensor *input_y = ctx.Input(2);
  Tensor *output_x_grad = ctx.Output(0);

  int64_t y_grad_num = ctx.Input(0)->GetTensorShape()->NumElements();
  int64_t y_num = ctx.Input(2)->GetTensorShape()->NumElements();
  KERNEL_CHECK_FALSE((y_num == y_grad_num), KERNEL_STATUS_PARAM_INVALID,
                     "The data num of input y_grad [%llu] is different from y [%llu].", y_grad_num, y_num)
  auto data_type_x = ctx.Input(1)->GetDataType();
  auto data_type_y_grad = ctx.Input(0)->GetDataType();
  KERNEL_CHECK_FALSE((data_type_y_grad == data_type_x), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input y_grad [%s] is different from x [%s].", DTypeStr(data_type_y_grad).c_str(),
                     DTypeStr(data_type_x).c_str())

  if (global_median == false) {
    Tensor *input_indices = ctx.Input(3);
    KERNEL_LOG_DEBUG(
      "MedianGradCpuKernel[%s], input_y_grad: size[%llu],"
      "input_x: size[%llu], input_y: size[%llu],"
      "input_indices: size[%llu], output_x_grad: size[%llu].",
      ctx.GetOpType().c_str(), input_y_grad->GetDataSize(), input_x->GetDataSize(), input_y->GetDataSize(),
      input_indices->GetDataSize(), output_x_grad->GetDataSize());
  } else {
    KERNEL_LOG_DEBUG(
      "MedianGradCpuKernel[%s], input_y_grad: size[%llu],"
      "input_x: size[%llu], input_y: size[%llu],"
      "output_x_grad: size[%llu].",
      ctx.GetOpType().c_str(), input_y_grad->GetDataSize(), input_x->GetDataSize(), input_y->GetDataSize(),
      output_x_grad->GetDataSize());
  }

  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
uint32_t MedianGradCpuKernel::GlobalMedianGradCompute(CpuKernelContext &ctx) {
  auto y_grad = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto x = reinterpret_cast<T1 *>(ctx.Input(1)->GetData());
  auto y = reinterpret_cast<T1 *>(ctx.Input(2)->GetData());
  auto x_grad = reinterpret_cast<T2 *>(ctx.Output(0)->GetData());
  int64_t output_data_num = ctx.Output(0)->NumElements();
  int64_t input_data_num = ctx.Input(1)->NumElements();

  T2 count_repeat = 0;
  for (int64_t i = 0; i < input_data_num; i++) {
    count_repeat += (*(x + i) == *y) ? 1 : 0;
  }

  if (output_data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

    if (output_data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > output_data_num) {
      max_core_num = output_data_num;
    }

    auto sharder_mediangrad = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        *(x_grad + i) = (*(x + i) == *y) ? (*y_grad / count_repeat) : 0;
      }
    };
    KERNEL_HANDLE_ERROR(
      CpuKernelUtils::ParallelFor(ctx, output_data_num, output_data_num / max_core_num, sharder_mediangrad),
      "MedianGrad Compute failed.");
  } else {
    for (int64_t i = 0; i < output_data_num; i++) {
      *(x_grad + i) = (*(x + i) == *y) ? (*y_grad / count_repeat) : 0;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
uint32_t MedianGradCpuKernel::MedianGradCompute(CpuKernelContext &ctx) {
  auto y_grad = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto indices = reinterpret_cast<int64_t *>(ctx.Input(3)->GetData());
  auto x_grad = reinterpret_cast<T2 *>(ctx.Output(0)->GetData());
  int64_t output_data_num = ctx.Output(0)->NumElements();
  int64_t need_calculate_num = ctx.Input(0)->NumElements();

  for (int64_t i = 0; i < output_data_num; i++) {
    *(x_grad + i) = 0;
  }

  AttrValue *axis_ptr = ctx.GetAttr("axis");
  int64_t axis = axis_ptr == nullptr ? 0 : axis_ptr->GetInt();

  std::vector<int64_t> shape_x = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_y = ctx.Input(2)->GetTensorShape()->GetDimSizes();

  std::vector<int64_t> shape_keepdim;
  int64_t dim_num_x = ctx.Input(1)->GetTensorShape()->GetDims();
  axis = axis >= 0 ? axis : axis + dim_num_x;
  for (int64_t i = 0; i < dim_num_x; i++) {
    if (i == axis) {
      shape_keepdim.push_back(1);
    } else {
      shape_keepdim.push_back(shape_x[i]);
    }
  }

  std::vector<int64_t> element_num_each_dim_x;
  std::vector<int64_t> element_num_each_dim_y;
  int64_t element_num_y = 1;
  int64_t element_num_x = 1;
  for (int64_t i = shape_keepdim.size() - 1; i >= 0; i--) {
    element_num_each_dim_x.insert(element_num_each_dim_x.begin(), element_num_x);
    element_num_x *= shape_x[i];
    element_num_each_dim_y.insert(element_num_each_dim_y.begin(), element_num_y);
    element_num_y *= shape_keepdim[i];
  }

  if (need_calculate_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

    if (need_calculate_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > need_calculate_num) {
      max_core_num = need_calculate_num;
    }

    auto sharder_mediangrad = [&](int64_t start, int64_t end) {
      std::vector<int64_t> dim_vec;
      for (int64_t i = 0; i < dim_num_x; i++) {
        dim_vec.push_back(0);
      }
      for (int64_t nth_element = start; nth_element < end; nth_element++) {
        int64_t elements_remain = nth_element;
        for (int64_t i = 0; i < dim_num_x; i++) {
          dim_vec[i] = elements_remain / element_num_each_dim_y[i];
          elements_remain %= element_num_each_dim_y[i];
        }
        int64_t update_element_pos = 0;
        for (int64_t i = 0; i < dim_num_x; i++) {
          if (i == axis) {
            update_element_pos += *(indices + nth_element) * element_num_each_dim_x[i];
          } else {
            update_element_pos += dim_vec[i] * element_num_each_dim_x[i];
          }
        }
        *(x_grad + update_element_pos) = *(y_grad + nth_element);
      }
    };
    KERNEL_HANDLE_ERROR(
      CpuKernelUtils::ParallelFor(ctx, need_calculate_num, need_calculate_num / max_core_num, sharder_mediangrad),
      "MedianGrad Compute failed.");
  } else {
    std::vector<int64_t> dim_vec;
    for (int64_t i = 0; i < dim_num_x; i++) {
      dim_vec.push_back(0);
    }
    for (int64_t nth_element = 0; nth_element < need_calculate_num; nth_element++) {
      int64_t elements_remain = nth_element;
      for (int64_t i = 0; i < dim_num_x; i++) {
        dim_vec[i] = elements_remain / element_num_each_dim_y[i];
        elements_remain %= element_num_each_dim_y[i];
      }
      int64_t update_element_pos = 0;
      for (int64_t i = 0; i < dim_num_x; i++) {
        if (i == axis) {
          update_element_pos += *(indices + nth_element) * element_num_each_dim_x[i];
        } else {
          update_element_pos += dim_vec[i] * element_num_each_dim_x[i];
        }
      }
      *(x_grad + update_element_pos) = *(y_grad + nth_element);
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMedianGrad, MedianGradCpuKernel);
}  // namespace aicpu
