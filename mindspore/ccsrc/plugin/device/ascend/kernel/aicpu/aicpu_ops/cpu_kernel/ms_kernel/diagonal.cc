/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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

#include "diagonal.h"

#include "context/inc/cpu_kernel_utils.h"
#include "inc/kernel_log.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#define N2 2
#define N3 3
#define N4 4

using namespace std;

namespace {
const size_t kOutputNum = 1;
const size_t kInputNum = 1;
const char *kDiagonal = "Diagonal";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 400;
const int64_t kParallelDataNumMid = 2 * 1024;
const uint32_t min_core_num = 1;

template <typename T>
T mul_sum(std::vector<T> v1, std::vector<T> v2) {
  T output = 0;
  if (v1.size() != v2.size()) {
    return false;
  } else {
    for (unsigned int i = 0; i < v1.size(); i++) {
      output += v1[i] * v2[i];
    }
    return output;
  }
}

template <typename T>
std::vector<T> construct_stride(std::vector<T> t_shape) {
  std::vector<T> t_stride(t_shape.size(), 1);
  int initial = 1;
  for (unsigned int i = t_shape.size(); i > 0; i--) {
    t_stride[i - 1] = initial;
    initial = initial * t_shape[i - 1];
  }
  return t_stride;
}

int64_t diag_size(const int64_t &offset, const int64_t &dim1, const int64_t &dim2, std::vector<int64_t> x_shape) {
  int64_t dsize = 0;
  if (offset >= 0) {
    dsize = std::max<int64_t>(std::min(x_shape.at(dim1), x_shape.at(dim2) - offset), 0);
  } else {
    dsize = std::max<int64_t>(std::min(x_shape.at(dim1) + offset, x_shape.at(dim2)), 0);
  }
  return dsize;
}

int64_t maybe_wrap_dim(int64_t dim, int64_t dim_post_expr) {
  if (dim < 0) {
    dim += dim_post_expr;
  }
  return dim;
}

template <typename T>
T get_data(int64_t basepos, int64_t offset, int64_t *ar, T *dptr) {
  if (offset >= 0) {
    return dptr[basepos + offset * ar[1]];
  } else {
    return dptr[basepos - offset * ar[0]];
  }
}

template <typename T>
std::vector<T> construct_index(int num, std::vector<T> &stride) {
  std::vector<T> idx;
  int tmp_num = num;
  for (uint32_t i = 0; i < stride.size(); i++) {
    idx.push_back(tmp_num / stride[i]);
    tmp_num = tmp_num % stride[i];
  }
  return idx;
}
}  // namespace

namespace aicpu {
template <typename T>
void DiagonalCpuKernel::set_output(int64_t *ar, T *dptr, T *y_dptr) {
  for (int i = 0; i < dsize; i++) {
    y_dptr[ar[N3] + i] = get_data(ar[N2] + i * (ar[0] + ar[1]), offset_, ar, dptr);
  }
}

template <typename T>
uint32_t DiagonalCpuKernel::DoComputeType(CpuKernelContext &ctx) {
  // Get the inuput and output
  Tensor *input_x = ctx.Input(0);
  // Get some information of input
  int32_t x_NumE = input_x->NumElements();
  auto x_shape = input_x->GetTensorShape();
  std::vector<int64_t> x_shape_ = x_shape->GetDimSizes();
  const int64_t x_dim = x_shape->GetDims();
  auto dataptr = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto y_dataptr = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  // Compute
  dsize = diag_size(offset_, dim1_, dim2_, x_shape_);
  std::vector<int64_t> x_stride = construct_stride<int64_t>(x_shape_);
  if (x_dim != N2 && x_NumE > 0) {
    // set the vx_shape and vx_stride, which is x_shape_ and x_stride of
    // position dim1_ and dim2_ removed.
    std::vector<int64_t> vx_shape;
    std::vector<int64_t> vx_stride;
    for (unsigned int tmp_dim = 0; tmp_dim < x_shape_.size(); tmp_dim++) {
      if (tmp_dim != dim1_ && tmp_dim != dim2_) {
        vx_shape.push_back(x_shape_[tmp_dim]);
        vx_stride.push_back(x_stride[tmp_dim]);
      }
    }
    // set the y_shape (the output shape), y_stride(the output stride),
    // vy_stride(the y_stride without the last dim)
    std::vector<int64_t> y_shape = vx_shape;
    y_shape.push_back(dsize);
    std::vector<int64_t> y_stride = construct_stride<int64_t>(y_shape);
    std::vector<int64_t> vy_stride = y_stride;
    vy_stride.pop_back();
    // diagonal
    int32_t task_num = x_NumE / x_shape_[dim1_] / x_shape_[dim2_];
    if (task_num >= kParallelDataNum) {
      int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
      if (task_num <= kParallelDataNumMid) {
        max_core_num = std::min(max_core_num, static_cast<int64_t>(N4));
      }
      max_core_num = max_core_num > task_num ? task_num : max_core_num;
      auto sharder_diagonal = [&](int64_t start, int64_t end) {
        for (int64_t j = start; j < end; j++) {
          std::vector<int64_t> v_s_stride = construct_stride<int64_t>(vx_shape);
          auto p = construct_index<int64_t>(j, v_s_stride);
          int64_t arr[N4] = {x_stride[dim1_], x_stride[dim2_], mul_sum<int64_t>(p, vx_stride),
                             mul_sum<int64_t>(p, vy_stride)};
          set_output(arr, dataptr, y_dataptr);
        }
      };
      if (max_core_num != 0) {
        int64_t per_unit = task_num / max_core_num;
        CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, task_num, per_unit, sharder_diagonal),
                                 "Diagonal failed.");
      }
    } else {
      for (int64_t j = 0; j < task_num; j++) {
        std::vector<int64_t> v_s_stride = construct_stride<int64_t>(vx_shape);
        auto p = construct_index<int64_t>(j, v_s_stride);
        int64_t arr[N4] = {x_stride[dim1_], x_stride[dim2_], mul_sum<int64_t>(p, vx_stride),
                           mul_sum<int64_t>(p, vy_stride)};
        set_output(arr, dataptr, y_dataptr);
      }
    }
  } else if (x_dim == N2) {
    int64_t arr[N4] = {x_stride[dim1_], x_stride[dim2_], 0, 0};
    set_output(arr, dataptr, y_dataptr);
  } else {
    y_dataptr = dataptr;
  }
  return KERNEL_STATUS_OK;
}

uint32_t DiagonalCpuKernel::ComputeWithType(CpuKernelContext &ctx) {
  Tensor *input_x = ctx.Input(0);
  auto data_type = input_x->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      return DoComputeType<float>(ctx);
    case DT_DOUBLE:
      return DoComputeType<double>(ctx);
    case DT_BOOL:
      return DoComputeType<bool>(ctx);
    case DT_INT8:
      return DoComputeType<std::int8_t>(ctx);
    case DT_INT16:
      return DoComputeType<std::int16_t>(ctx);
    case DT_INT32:
      return DoComputeType<std::int32_t>(ctx);
    case DT_INT64:
      return DoComputeType<std::int64_t>(ctx);
    case DT_UINT8:
      return DoComputeType<std::uint8_t>(ctx);
    case DT_UINT16:
      return DoComputeType<std::uint16_t>(ctx);
    case DT_UINT32:
      return DoComputeType<std::uint32_t>(ctx);
    case DT_UINT64:
      return DoComputeType<std::uint64_t>(ctx);
    case DT_FLOAT16:
      return DoComputeType<Eigen::half>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "[Diagonal]: Diagonal kernel data type [%s] not support.",
                            DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t DiagonalCpuKernel::Compute(CpuKernelContext &ctx) {
  // Check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "Diagonal check input and output number failed.");
  // Get the inuput
  Tensor *input_x = ctx.Input(0);
  auto input_size = input_x->GetTensorShape()->GetDims();
  // Check the input dims
  if (input_size < N2) {
    CUST_KERNEL_LOG_ERROR(ctx, "[Diagonal]: the input tensor must is at least 2-dimensional.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // Get the attr
  AttrValue *offset = ctx.GetAttr("offset");
  offset_ = (offset == nullptr) ? 0 : (offset->GetInt());
  AttrValue *dim1 = ctx.GetAttr("dim1");
  dim1_ = (dim1 == nullptr) ? 0 : (dim1->GetInt());
  AttrValue *dim2 = ctx.GetAttr("dim2");
  dim2_ = (dim2 == nullptr) ? 1 : (dim2->GetInt());
  int64_t min_d = -input_size;
  int64_t max_d = input_size - 1;
  // Check the attr
  if (dim1_ < min_d || dim1_ > max_d || dim2_ < min_d || dim2_ > max_d) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "[Diagonal]: Dimension out of range (expected to be in range of [%d, "
                          "%d]).",
                          min_d, max_d);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // Represent the dim in uniform standard form and Check the dim
  dim1_ = maybe_wrap_dim(dim1_, input_size);
  dim2_ = maybe_wrap_dim(dim2_, input_size);
  if (dim1_ == dim2_) {
    CUST_KERNEL_LOG_ERROR(ctx, "[Diagonal]:Diagonal dimensions cannot be identical.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return ComputeWithType(ctx);
}

REGISTER_MS_CPU_KERNEL(kDiagonal, DiagonalCpuKernel);
}  // namespace aicpu
