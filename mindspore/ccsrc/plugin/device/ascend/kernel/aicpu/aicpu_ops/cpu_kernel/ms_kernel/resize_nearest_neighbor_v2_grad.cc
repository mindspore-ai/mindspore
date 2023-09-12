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

#include "ms_kernel/resize_nearest_neighbor_v2_grad.h"

#include <securec.h>
#include <stdint.h>
#include <vector>
#include <algorithm>

#include "common/cpu_kernel_utils.h"
#include "inc/cpu_types.h"
#include "common/kernel_log.h"
#include "common/status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
constexpr uint32_t kDim1 = 1;
constexpr uint32_t kDim4 = 4;
constexpr uint32_t kIndex0 = 0;
constexpr uint32_t kIndex1 = 1;
constexpr uint32_t kIndex2 = 2;
constexpr uint32_t kIndex3 = 3;
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

  grads_shape = grads_ptr->GetTensorShape()->GetDimSizes();
  auto grads_dims = grads_ptr->GetTensorShape()->GetDims();
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
  return KERNEL_STATUS_OK;
}

uint32_t ResizeNearestNeighborV2GradCpuKernel::Compute(CpuKernelContext &ctx) {
  if (ResizeNearestNeighborV2GradParamCheck(ctx) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *grads = ctx.Input(0);
  DataType data_type = DataType(grads->GetDataType());
  uint32_t res = KERNEL_STATUS_OK;
  switch (data_type) {
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
uint32_t ResizeNearestNeighborV2GradCpuKernel::ResizeNearestNeighborV2GradCompute(const CpuKernelContext &ctx) {
  Tensor *input_grads = ctx.Input(0);
  Tensor *output_y = ctx.Output(0);
  grads_shape = input_grads->GetTensorShape()->GetDimSizes();
  y_shape = output_y->GetTensorShape()->GetDimSizes();
  y_size = output_y->GetTensorShape()->NumElements();
  bool is_fp16 = std::is_same<T, Eigen::half>::value;
  auto grads_4d = reinterpret_cast<T *>(input_grads->GetData());
  auto y_4d = reinterpret_cast<T *>(output_y->GetData());
  if (is_fp16) {
    // define for fp16
    std::vector<float> y_work_copy(1);
    y_work_copy.resize(y_size);
    int ret = memset_s(y_work_copy.data(), y_size * sizeof(float), 0, y_size * sizeof(float));
    KERNEL_CHECK_FALSE(ret == EOK, KERNEL_STATUS_INNER_ERROR,
                       "For 'ResizeNearestNeighborV2Grad', memset_s error. Error no: %d.", ret);
    RealCompute<T, float>(grads_4d, y_work_copy.data());
    for (size_t idx = 0; idx < y_size; ++idx) {
      y_4d[idx] = static_cast<T>(y_work_copy[idx]);
    }
  } else {
    int ret = memset_s(y_4d, y_size * sizeof(T), 0, y_size * sizeof(T));
    KERNEL_CHECK_FALSE(ret == EOK, KERNEL_STATUS_INNER_ERROR,
                       "For 'ResizeNearestNeighborV2Grad', memset_s error. Error no: %d.", ret);
    RealCompute<T, T>(grads_4d, y_4d);
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename S>
void ResizeNearestNeighborV2GradCpuKernel::RealCompute(T *const grads_4d, S *const y_4d) {
  const int64_t batch_size = grads_shape[kIndex0];
  const int64_t channels = grads_shape[kIndex1];
  const int64_t in_height = grads_shape[kIndex2];
  const int64_t in_width = grads_shape[kIndex3];
  const int64_t in_hw = in_height * in_width;

  const int64_t out_height = y_shape[kIndex2];
  const int64_t out_width = y_shape[kIndex3];
  const int64_t out_hw = out_height * out_width;

  const float height_scale = CalculateResizeScale(out_height, in_height, align_corners);
  const float width_scale = CalculateResizeScale(out_width, in_width, align_corners);

  for (int64_t y = 0; y < in_height; ++y) {
    int64_t out_y =
      std::min((align_corners) ? static_cast<int64_t>(roundf(Scaler(y, height_scale, half_pixel_centers)))
                               : static_cast<int64_t>(floorf(Scaler(y, height_scale, half_pixel_centers))),
               out_height - 1);
    for (int64_t x = 0; x < in_width; ++x) {
      int64_t out_x =
        std::min((align_corners) ? static_cast<int64_t>(roundf(Scaler(x, width_scale, half_pixel_centers)))
                                 : static_cast<int64_t>(floorf(Scaler(x, width_scale, half_pixel_centers))),
                 out_width - 1);
      for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
          y_4d[(b * channels + c) * out_hw + out_y * out_width + out_x] +=
            static_cast<S>(grads_4d[(b * channels + c) * in_hw + y * in_width + x]);
        }
      }
    }
  }
}

REGISTER_CPU_KERNEL(kResizeNearestNeighborV2Grad, ResizeNearestNeighborV2GradCpuKernel);
}  // namespace aicpu
