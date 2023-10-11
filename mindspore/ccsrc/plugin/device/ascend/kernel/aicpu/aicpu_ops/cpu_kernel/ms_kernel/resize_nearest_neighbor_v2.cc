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

#include "ms_kernel/resize_nearest_neighbor_v2.h"

#include <securec.h>
#include <stdint.h>
#include <algorithm>
#include <vector>

#include "common/cpu_kernel_utils.h"
#include "common/kernel_log.h"
#include "common/status.h"
#include "inc/cpu_types.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
constexpr uint32_t kDim1 = 1;
constexpr uint32_t kDim4 = 4;
constexpr uint32_t kValue0 = 0;
constexpr uint32_t kIndex0 = 0;
constexpr uint32_t kIndex1 = 1;
constexpr uint32_t kIndex2 = 2;
constexpr uint32_t kIndex3 = 3;
constexpr uint32_t kNumElements2 = 2;
constexpr uint32_t kMaxValue = 24;
const char *kResizeNearestNeighborV2 = "ResizeNearestNeighborV2";
}  // namespace

namespace aicpu {
inline float Scaler(const int x, const float scale, bool half_pixel_centers) {
  if (half_pixel_centers) {
    return (static_cast<float>(x) + 0.5f) * scale;
  } else {
    return static_cast<float>(x) * scale;
  }
}

inline float CalculateResizeScale(int64_t in_size, int64_t out_size, bool align_corners) {
  return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                         : in_size / static_cast<float>(out_size);
}

uint32_t ResizeNearestNeighborV2CpuKernel::ResizeNearestNeighborV2ParamCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check params failed.", kResizeNearestNeighborV2);
  Tensor *x_ptr = ctx.Input(0);
  Tensor *size_ptr = ctx.Input(1);
  auto format = x_ptr->GetTensorShape()->GetFormat();
  is_nchw = (format == Format::FORMAT_NCHW);
  if (is_nchw) {
    c_idx = 1;
    h_idx = 2;
    w_idx = 3;
  } else {
    c_idx = 3;
    h_idx = 1;
    w_idx = 2;
  }

  AttrValue *align_corners_ptr = ctx.GetAttr("align_corners");
  AttrValue *half_pixel_centers_ptr = ctx.GetAttr("half_pixel_centers");

  auto align_corners = false;
  auto half_pixel_centers = false;
  if (align_corners_ptr != nullptr) {
    align_corners = align_corners_ptr->GetBool();
  }
  if (half_pixel_centers_ptr != nullptr) {
    half_pixel_centers = half_pixel_centers_ptr->GetBool();
  }
  auto x_shape = x_ptr->GetTensorShape()->GetDimSizes();
  auto x_dims = x_ptr->GetTensorShape()->GetDims();
  auto size_dims = size_ptr->GetTensorShape()->GetDims();
  auto size_data = static_cast<int32_t *>(size_ptr->GetData());

  KERNEL_CHECK_FALSE(!(half_pixel_centers && align_corners), KERNEL_STATUS_PARAM_INVALID,
                     "If half_pixel_centers is True, "
                     "align_corners must be False, but got half_pixel_centers %s, "
                     "align_corners %s.",
                     half_pixel_centers == true ? "True" : "False", align_corners == true ? "True" : "False");
  KERNEL_CHECK_FALSE(x_dims == kDim4, KERNEL_STATUS_PARAM_INVALID, "x must be 4-dimensional but got %d-dimensional.",
                     x_dims);
  auto channels = x_shape[c_idx];
  KERNEL_CHECK_FALSE(channels > kValue0, KERNEL_STATUS_PARAM_INVALID,
                     "image must have at least one channel but got %d channel.", channels);
  auto height = x_shape[h_idx];
  auto width = x_shape[w_idx];
  KERNEL_CHECK_FALSE(height > kValue0 && width > kValue0, KERNEL_STATUS_PARAM_INVALID,
                     "x image must be of non-zero size but got height %d, width %d.", height, width);
  KERNEL_CHECK_FALSE(height < INT32_MAX && width < INT32_MAX, KERNEL_STATUS_PARAM_INVALID,
                     "x sizes must be between 0 and max int32 but got but "
                     "got height %d, width %d.",
                     height, width);
  auto in_height = static_cast<int32_t>(height);
  auto in_width = static_cast<int32_t>(width);
  KERNEL_CHECK_FALSE(size_dims == kDim1, KERNEL_STATUS_PARAM_INVALID, "size_shape must be 1-dimensional but got %d.",
                     size_dims);
  KERNEL_CHECK_FALSE(size_ptr->NumElements() == kNumElements2, KERNEL_STATUS_PARAM_INVALID,
                     "shape_t must have two elements but got %d element(s).", size_ptr->NumElements());
  KERNEL_CHECK_FALSE(size_data[kIndex0] > 0 && size_data[kIndex1] > 0, KERNEL_STATUS_PARAM_INVALID,
                     "output dimensions must be positive but got height %d, width %d.", size_data[kIndex0],
                     size_data[kIndex1]);
  auto out_height = size_data[0];
  auto out_width = size_data[1];

  auto height_scale = CalculateResizeScale(in_height, out_height, align_corners);
  auto width_scale = CalculateResizeScale(in_width, out_width, align_corners);
  KERNEL_CHECK_FALSE(ceilf((out_height - 1) * height_scale) <= static_cast<float>(INT64_MAX),
                     KERNEL_STATUS_PARAM_INVALID, "input image height scale would cause an overflow.");
  KERNEL_CHECK_FALSE(ceilf((out_width - 1) * width_scale) <= static_cast<float>(INT64_MAX), KERNEL_STATUS_PARAM_INVALID,
                     "input image width scale would cause an overflow.");
  KERNEL_CHECK_FALSE(in_height < (1 << kMaxValue) && in_width < (1 << kMaxValue), KERNEL_STATUS_PARAM_INVALID,
                     "nearest neighbor requires max height "
                     "& width of 2^24.");

  // Set Output Shape
  std::vector<int64_t> y_shape(x_shape);
  y_shape[h_idx] = out_height;
  y_shape[w_idx] = out_width;
  ctx.Output(0)->GetTensorShape()->SetDimSizes(y_shape);

  return KERNEL_STATUS_OK;
}

uint32_t ResizeNearestNeighborV2CpuKernel::Compute(CpuKernelContext &ctx) {
  if (ResizeNearestNeighborV2ParamCheck(ctx) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *x = ctx.Input(0);
  DataType data_type = DataType(x->GetDataType());
  uint32_t res = KERNEL_STATUS_OK;
  switch (data_type) {
    case DT_UINT8:
      res = ResizeNearestNeighborV2Compute<uint8_t>(ctx);
      break;
    case DT_FLOAT16:
      res = ResizeNearestNeighborV2Compute<Eigen::half>(ctx);
      break;
    case DT_FLOAT:
      res = ResizeNearestNeighborV2Compute<float>(ctx);
      break;
    case DT_DOUBLE:
      res = ResizeNearestNeighborV2Compute<double>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("For ResizeNearestNeighborV2, invalid input type [%s].", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return res;
}
template <typename T>
void ResizeNearestNeighborV2CpuKernel::InnerCompute(
  Eigen::Index b, Eigen::Index y,
  Eigen::TensorMap<Eigen::Tensor<T, kValue4, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> x_4d,
  Eigen::TensorMap<Eigen::Tensor<T, kValue4, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> y_4d) {
  Eigen::Index in_y =
    std::min((align_corners) ? static_cast<Eigen::Index>(roundf(Scaler(y, height_scale, half_pixel_centers)))
                             : static_cast<Eigen::Index>(floorf(Scaler(y, height_scale, half_pixel_centers))),
             in_height - 1);
  if (half_pixel_centers) {
    in_y = std::max(static_cast<Eigen::Index>(0), in_y);
  }
  for (Eigen::Index x = 0; x < out_width; ++x) {
    Eigen::Index in_x =
      std::min((align_corners) ? static_cast<Eigen::Index>(roundf(Scaler(x, width_scale, half_pixel_centers)))
                               : static_cast<Eigen::Index>(floorf(Scaler(x, width_scale, half_pixel_centers))),
               in_width - 1);
    if (half_pixel_centers) {
      in_x = std::max(static_cast<Eigen::Index>(0), in_x);
    }
    if (is_nchw) {
      // data_format = NCHW
      for (Eigen::Index c = 0; c < channels; ++c) {
        y_4d(b, c, y, x) = x_4d(b, c, in_y, in_x);
      }
    } else {
      for (Eigen::Index c = 0; c < channels; ++c) {
        y_4d(b, y, x, c) = x_4d(b, in_y, in_x, c);
      }
    }
  }
}

template <typename T>
uint32_t ResizeNearestNeighborV2CpuKernel::ResizeNearestNeighborV2Compute(const CpuKernelContext &ctx) {
  Tensor *input_x = ctx.Input(0);
  Tensor *output_y = ctx.Output(0);
  std::vector<int64_t> x_shape = input_x->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> y_shape = output_y->GetTensorShape()->GetDimSizes();
  AttrValue *align_corners_ptr = ctx.GetAttr("align_corners");
  AttrValue *half_pixel_centers_ptr = ctx.GetAttr("half_pixel_centers");
  align_corners = false;
  half_pixel_centers = false;
  if (align_corners_ptr != nullptr) {
    align_corners = align_corners_ptr->GetBool();
  }
  if (half_pixel_centers_ptr != nullptr) {
    half_pixel_centers = half_pixel_centers_ptr->GetBool();
  }

  batch_size = x_shape[n_idx];
  channels = x_shape[c_idx];
  in_height = x_shape[h_idx];
  in_width = x_shape[w_idx];

  out_height = y_shape[h_idx];
  out_width = y_shape[w_idx];

  height_scale = CalculateResizeScale(in_height, out_height, align_corners);
  width_scale = CalculateResizeScale(in_width, out_width, align_corners);
  EigenTensor x_et(input_x, input_x->GetData());
  EigenTensor y_et(output_y, output_y->GetData());
  auto x_4d = x_et.tensor<T, 4>();
  auto y_4d = y_et.tensor<T, 4>();
  for (Eigen::Index b = 0; b < batch_size; ++b) {
    for (Eigen::Index y = 0; y < out_height; ++y) {
      InnerCompute<T>(b, y, x_4d, y_4d);
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kResizeNearestNeighborV2, ResizeNearestNeighborV2CpuKernel);
}  // namespace aicpu
