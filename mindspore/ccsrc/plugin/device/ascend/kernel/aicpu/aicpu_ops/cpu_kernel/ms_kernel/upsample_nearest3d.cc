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

#include "upsample_nearest3d.h"
#include <stdint.h>
#include <cstring>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_log.h"
#include "securec.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr float kValueZero = 0.;
constexpr int32_t kValue3 = 3;
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
constexpr uint32_t kIndex0 = 0;
constexpr uint32_t kIndex1 = 1;
constexpr uint32_t kIndex2 = 2;
constexpr uint32_t kIndex3 = 3;
constexpr uint32_t kIndex4 = 4;
constexpr int32_t kDims5 = 5;
const char *const kUpsampleNearest3d = "UpsampleNearest3d";
}  // namespace

namespace aicpu {
template <typename T>
static inline T DataIndexInit(T offset) {
  return offset;
}

template <typename T, typename... Args>
static inline T DataIndexInit(T offset, T *x, const T &X, Args &&... args) {
  offset = DataIndexInit(offset, std::forward<Args>(args)...);
  if (X > 0) {
    *x = offset % X;
    return offset / X;
  }
  return 0;
}

static inline bool DataIndexStep() { return true; }

template <typename T, typename... Args>
static inline bool DataIndexStep(T *x, const T &X, Args &&... args) {
  if (DataIndexStep(std::forward<Args>(args)...)) {
    *x = ((*x + 1) == X) ? 0 : (*x + 1);
    return *x == 0;
  }
  return false;
}
template <typename T>
static inline T ComputeScales(const double &scale, const size_t &input_size, const size_t &output_size) {
  if (scale > 0) {
    return static_cast<T>(1.0 / scale);
  } else if (output_size > 0) {
    return (static_cast<T>(input_size) / output_size);
  }
  return 0;
}

static inline size_t NearestNeighborSourceIndex(const float &scale, const size_t &dst_index, const size_t &input_size) {
  size_t src_index = std::min(static_cast<size_t>(floorf(dst_index * scale)), input_size - 1);
  return src_index;
}

static inline size_t NearestIndex(const size_t &output_index, const size_t &input_size, const size_t &output_size,
                                  const double &scales) {
  constexpr size_t kNumberTwo = 2;
  if (output_size == input_size) {
    return output_index;
  } else if (output_size == kNumberTwo * input_size) {
    return output_index >> 1;
  } else {
    float scale = ComputeScales<float>(scales, input_size, output_size);
    return NearestNeighborSourceIndex(scale, output_index, input_size);
  }
}

uint32_t UpsampleNearest3dCpuKernel::UpsampleNearest3dParamCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check params failed.", kUpsampleNearest3d);
  Tensor *x_ptr = ctx.Input(0);
  auto x_dims = x_ptr->GetTensorShape()->GetDims();
  KERNEL_CHECK_FALSE(x_dims == kDims5, KERNEL_STATUS_PARAM_INVALID,
                     "Upsample with NHWC format supports tensors with 5 "
                     "dims. but got %d dim(s).",
                     x_dims);

  auto none_list_ptr = ctx.GetAttr("none_list");
  KERNEL_CHECK_NULLPTR(none_list_ptr, KERNEL_STATUS_PARAM_INVALID, "[%s] get attr 'none_list' failed.",
                       ctx.GetOpType().c_str());
  none_list = none_list_ptr->GetListInt();
  KERNEL_CHECK_FALSE(none_list.size() == 1, KERNEL_STATUS_PARAM_INVALID,
                     "For 'UpsampleNearest3D', only one of output_size or scales should be specified.");

  return KERNEL_STATUS_OK;
}

uint32_t UpsampleNearest3dCpuKernel::Compute(CpuKernelContext &ctx) {
  if (UpsampleNearest3dParamCheck(ctx) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *x = ctx.Input(0);
  DataType data_type = x->GetDataType();
  uint32_t res = KERNEL_STATUS_OK;
  switch (data_type) {
    case DT_UINT8:
      res = UpsampleNearest3dCompute<uint8_t>(ctx);
      break;
    case DT_FLOAT16:
      res = UpsampleNearest3dCompute<Eigen::half>(ctx);
      break;
    case DT_FLOAT:
      res = UpsampleNearest3dCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      res = UpsampleNearest3dCompute<double>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("UpsampleNearest3d invalid input type [%s].", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Failed launching UpsampleNearest3d.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t UpsampleNearest3dCpuKernel::UpsampleNearest3dCompute(const CpuKernelContext &ctx) {
  Tensor *input_x = ctx.Input(0);
  Tensor *output_y = ctx.Output(0);
  std::vector<int64_t> x_shape_ = input_x->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> y_shape_ = output_y->GetTensorShape()->GetDimSizes();

  std::vector<float> scales(kIndex3, kValueZero);
  if (none_list[kIndex0] != static_cast<int64_t>(kIndex2)) {
    auto scales_ptr = reinterpret_cast<float *>(ctx.Input(kIndex1)->GetData());
    for (int i = 0; i < kValue3; ++i) {
      scales[i] = scales_ptr[i];
    }
  }

  size_t batch_size = static_cast<size_t>(x_shape_[kIndex0]);
  size_t channels = static_cast<size_t>(x_shape_[kIndex1]);
  size_t input_depth = static_cast<size_t>(x_shape_[kIndex2]);
  size_t input_height = static_cast<size_t>(x_shape_[kIndex3]);
  size_t input_width = static_cast<size_t>(x_shape_[kIndex4]);
  size_t output_depth = static_cast<size_t>(y_shape_[kIndex2]);
  size_t output_height = static_cast<size_t>(y_shape_[kIndex3]);
  size_t output_width = static_cast<size_t>(y_shape_[kIndex4]);

  channels = channels * batch_size;
  auto x_ptr = reinterpret_cast<T *>(input_x->GetData());
  auto y_ptr = reinterpret_cast<T *>(output_y->GetData());
  size_t y_size = output_y->GetDataSize();
  if (input_depth == output_depth && input_height == output_height && input_width == output_width) {
    auto ret = memcpy_s(y_ptr, y_size, x_ptr, channels * input_depth * input_height * input_width * sizeof(T));
    KERNEL_CHECK_FALSE((ret == EOK), KERNEL_STATUS_INNER_ERROR,
                       "UpsampleNearest3D memcpy failed, dst len is %zu, src size is %zu.", y_size,
                       channels * input_depth * input_height * input_width * sizeof(T));
    return KERNEL_STATUS_OK;
  }
  auto loop3d = [&](size_t begin, size_t end) {
    size_t n = 0;
    size_t od = 0;
    size_t oh = 0;
    size_t ow = 0;
    (void)DataIndexInit(begin, &n, channels, &od, output_depth, &oh, output_height, &ow, output_width);
    for (size_t idx = begin; idx < end; ++idx) {
      size_t id = NearestIndex(od, input_depth, output_depth, static_cast<double>(scales[kIndex0]));
      size_t ih = NearestIndex(oh, input_height, output_height, static_cast<double>(scales[kIndex1]));
      size_t iw = NearestIndex(ow, input_width, output_width, static_cast<double>(scales[kIndex2]));
      auto output_ptr = y_ptr + idx;
      auto input_ptr =
        x_ptr + n * input_depth * input_height * input_width + id * input_height * input_width + ih * input_width + iw;
      output_ptr[0] = input_ptr[0];
      (void)DataIndexStep(&n, channels, &od, output_depth, &oh, output_height, &ow, output_width);
    }
  };
  std::int64_t total{output_y->NumElements()};
  std::uint32_t cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, output_y->NumElements(), per_unit_size, loop3d),
                      "loop3d Compute failed.");
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kUpsampleNearest3d, UpsampleNearest3dCpuKernel);
}  // namespace aicpu
