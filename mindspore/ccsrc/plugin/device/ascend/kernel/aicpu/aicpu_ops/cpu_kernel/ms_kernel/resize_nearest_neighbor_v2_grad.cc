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

#include "resize_nearest_neighbor_v2_grad.h"

#include <stdint.h>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"
#include "kernel_log.h"
#include "securec.h"
#include "status.h"
#include "utils/eigen_tensor.h"

namespace {
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
constexpr uint32_t kDim1 = 1;
constexpr uint32_t kDim4 = 4;
constexpr uint32_t kIndex0 = 0;
constexpr uint32_t kIndex1 = 1;
constexpr uint32_t kNumElements2 = 2;
const char *kResizeNearestNeighborV2Grad = "ResizeNearestNeighborV2Grad";
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

uint32_t ResizeNearestNeighborV2GradCpuKernel::ResizeNearestNeighborV2GradParamCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check params failed.",
                      kResizeNearestNeighborV2Grad);
  Tensor *grads_ptr = ctx.Input(0);
  Tensor *size_ptr = ctx.Input(1);

  auto grads_shape = grads_ptr->GetTensorShape()->GetDimSizes();
  auto grads_dims = grads_ptr->GetTensorShape()->GetDims();
  auto size_shape = size_ptr->GetTensorShape()->GetDimSizes();
  auto size_dims = size_ptr->GetTensorShape()->GetDims();
  auto size_data = static_cast<int32_t *>(size_ptr->GetData());

  KERNEL_CHECK_FALSE(grads_dims == kDim4, KERNEL_STATUS_PARAM_INVALID,
                     "grads must be 4-dimensional but got %d-dimensional.", grads_dims);

  KERNEL_CHECK_FALSE(size_dims == kDim1, KERNEL_STATUS_PARAM_INVALID, "size_shape must be 1-dimensional but got %d.",
                     size_dims);

  KERNEL_CHECK_FALSE(size_ptr->NumElements() == kNumElements2, KERNEL_STATUS_PARAM_INVALID,
                     "size must have two elements but got %d element(s).", size_ptr->NumElements());
  KERNEL_CHECK_FALSE(size_data[kIndex0] > 0 && size_data[kIndex1] > 0, KERNEL_STATUS_PARAM_INVALID,
                     "size elements must be positive but got height %d, width %d.", size_data[kIndex0],
                     size_data[kIndex1]);
  return KERNEL_STATUS_OK;
}

template <typename T>
void ResizeNearestNeighborV2GradCpuKernel::InnerCompute(
  Eigen::Index y, Eigen::Index out_y, Eigen::Index x,
  Eigen::TensorMap<Eigen::Tensor<T, kValue4, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> grads_4d,
  Eigen::TensorMap<Eigen::Tensor<T, kValue4, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> y_4d) {
  const Eigen::Index out_x =
    std::min((align_corners) ? static_cast<Eigen::Index>(roundf(Scaler(x, width_scale, half_pixel_centers)))
                             : static_cast<Eigen::Index>(floorf(Scaler(x, width_scale, half_pixel_centers))),
             out_width - 1);
  for (Eigen::Index b = 0; b < batch_size; ++b) {
    for (Eigen::Index c = 0; c < channels; ++c) {
      if (data_format == "NHWC") {
        y_4d(b, out_y, out_x, c) += grads_4d(b, y, x, c);
      } else {
        // data_format = NCHW
        y_4d(b, c, out_y, out_x) += grads_4d(b, c, y, x);
      }
    }
  }
}

uint32_t ResizeNearestNeighborV2GradCpuKernel::Compute(CpuKernelContext &ctx) {
  if (ResizeNearestNeighborV2GradParamCheck(ctx) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *grads = ctx.Input(0);
  DataType data_type = DataType(grads->GetDataType());
  uint32_t res = KERNEL_STATUS_OK;
  switch (data_type) {
    case DT_INT8:
      res = ResizeNearestNeighborV2GradCompute<int8_t>(ctx);
      break;
    case DT_UINT8:
      res = ResizeNearestNeighborV2GradCompute<uint8_t>(ctx);
      break;
    case DT_INT16:
      res = ResizeNearestNeighborV2GradCompute<int16_t>(ctx);
      break;
    case DT_UINT16:
      res = ResizeNearestNeighborV2GradCompute<uint16_t>(ctx);
      break;
    case DT_INT32:
      res = ResizeNearestNeighborV2GradCompute<int32_t>(ctx);
      break;
    case DT_INT64:
      res = ResizeNearestNeighborV2GradCompute<int64_t>(ctx);
      break;
    case DT_FLOAT16:
      res = ResizeNearestNeighborV2GradCompute<Eigen::half>(ctx);
      break;
    case DT_FLOAT:
      res = ResizeNearestNeighborV2GradCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      res = ResizeNearestNeighborV2GradCompute<double>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("For ResizeNearestNeighborV2Grad, invalid input type [%s].", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return res;
}

template <typename T>
uint32_t ResizeNearestNeighborV2GradCpuKernel::ResizeNearestNeighborV2GradCompute(CpuKernelContext &ctx) {
  Tensor *input_grads = ctx.Input(0);
  Tensor *output_y = ctx.Output(0);
  std::vector<int64_t> grads_shape = input_grads->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> y_shape = output_y->GetTensorShape()->GetDimSizes();
  AttrValue *align_corners_ptr = ctx.GetAttr("align_corners");
  AttrValue *half_pixel_centers_ptr = ctx.GetAttr("half_pixel_centers");
  AttrValue *data_format_ptr = ctx.GetAttr("data_format");
  if (data_format_ptr != nullptr) {
    data_format = data_format_ptr->GetString();
  }
  if (data_format == "NHWC") {
    dim_idx_map_ = {
      {'N', kFormatNHWCIndexN}, {'H', kFormatNHWCIndexH}, {'W', kFormatNHWCIndexW}, {'C', kFormatNHWCIndexC}};
  } else if (data_format == "NCHW") {
    dim_idx_map_ = {
      {'N', kFormatNCHWIndexN}, {'C', kFormatNCHWIndexC}, {'H', kFormatNCHWIndexH}, {'W', kFormatNCHWIndexW}};
  } else {
    KERNEL_LOG_ERROR(
      "For ResizeNearestNeighborV2Grad, data_format only support [NCHW, "
      "NHWC], but get [%s].",
      data_format);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  align_corners = false;
  half_pixel_centers = false;
  if (align_corners_ptr != nullptr) {
    align_corners = align_corners_ptr->GetBool();
  }
  if (half_pixel_centers_ptr != nullptr) {
    half_pixel_centers = half_pixel_centers_ptr->GetBool();
  }
  batch_size = grads_shape[dim_idx_map_['N']];
  in_height = grads_shape[dim_idx_map_['H']];
  in_width = grads_shape[dim_idx_map_['W']];
  channels = grads_shape[dim_idx_map_['C']];

  out_height = y_shape[dim_idx_map_['H']];
  out_width = y_shape[dim_idx_map_['W']];

  height_scale = CalculateResizeScale(out_height, in_height, align_corners);
  width_scale = CalculateResizeScale(out_width, in_width, align_corners);

  EigenTensor grads_et(input_grads, input_grads->GetData());
  EigenTensor y_et(output_y, output_y->GetData());
  auto grads_4d = grads_et.tensor<T, 4>();
  auto y_4d = y_et.tensor<T, 4>();
  y_4d.setZero();
  for (Eigen::Index y = 0; y < in_height; ++y) {
    const Eigen::Index out_y =
      std::min((align_corners) ? static_cast<Eigen::Index>(roundf(Scaler(y, height_scale, half_pixel_centers)))
                               : static_cast<Eigen::Index>(floorf(Scaler(y, height_scale, half_pixel_centers))),
               out_height - 1);
    for (Eigen::Index x = 0; x < in_width; ++x) {
      InnerCompute(y, out_y, x, grads_4d, y_4d);
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kResizeNearestNeighborV2Grad, ResizeNearestNeighborV2GradCpuKernel);
}  // namespace aicpu
