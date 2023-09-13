/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
#include "cpu_kernel/ms_kernel/upsample_trilinear3d.h"

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "cpu_kernel/inc/cpu_types.h"
#include "utils/kernel_util.h"
#include "common/kernel_log.h"
#include "securec/include/securec.h"
#include "cpu_kernel/common/status.h"
#include "utils/eigen_tensor.h"

namespace {
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
constexpr int32_t kValue3 = 3;
constexpr int32_t kDims5 = 5;
constexpr uint32_t kIndex0 = 0;
constexpr uint32_t kIndex1 = 1;
constexpr uint32_t kIndex2 = 2;
constexpr uint32_t kIndex3 = 3;
constexpr uint32_t kIndex4 = 4;
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 7 * 1024;
const int64_t kParallelDataNumMid = 35 * 1024;
const int64_t kReserveCpu = 4;
const char *const kUpsampleTrilinear3d = "UpsampleTrilinear3d";
}  // namespace

namespace aicpu {
template <typename T>
static inline T ComputeScales(float scale, int64_t input_size, int64_t output_size) {
  constexpr T zero = 0.;
  if (scale > zero) {
    return static_cast<T>(1.0 / scale);
  } else if (output_size == 0) {
    return T(0);
  } else {
    return (static_cast<T>(input_size) / output_size);
  }
}

template <typename T>
static inline T AreaPixelComputeScale(int64_t input_size, int64_t output_size, bool align_corners, float scale) {
  if (align_corners) {
    if (output_size > 1) {
      return static_cast<T>(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<T>(0);
    }
  } else {
    return ComputeScales<T>(scale, input_size, output_size);
  }
}

template <typename T>
static inline T AreaPixelComputeSourceIndex(T scale, int64_t dst_index, bool align_corners) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    constexpr T zero = 0.;
    T src_idx = scale * (dst_index + 0.5) - 0.5;
    return (src_idx < zero) ? T(0) : src_idx;
  }
}

template <typename T>
static inline void ComputeSourceIndexAndLambda(int64_t *input_index0, int64_t *input_index1, T *lambda0, T *lambda1,
                                               T ratio, int64_t output_index, int64_t input_size, int64_t output_size,
                                               bool align_corners) {
  if (output_size == input_size) {
    // scale_factor = 1
    *input_index0 = output_index;
    *input_index1 = output_index;
    *lambda0 = static_cast<T>(1);
    *lambda1 = static_cast<T>(0);
  } else {
    const T real_input_index = AreaPixelComputeSourceIndex<T>(ratio, output_index, align_corners);
    *input_index0 = static_cast<int64_t>(real_input_index);
    int64_t offset = (*input_index0 < input_size - 1) ? 1 : 0;
    *input_index1 = *input_index0 + offset;
    *lambda1 = real_input_index - *input_index0;
    constexpr T one = 1.0;
    *lambda0 = one - *lambda1;
  }
}

uint32_t UpsampleTrilinear3dCpuKernel::UpsampleTrilinear3dParamCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check params failed.", kUpsampleTrilinear3d);
  auto x_dims = ctx.Input(0)->GetTensorShape()->GetDims();
  KERNEL_CHECK_FALSE(x_dims == kDims5, KERNEL_STATUS_PARAM_INVALID,
                     "For UpsampleTrilinear3d input x should has 5 "
                     "dims. but got %d dim(s).",
                     x_dims);
  auto align_corners_ptr = ctx.GetAttr("align_corners");
  if (align_corners_ptr) {
    align_corners = align_corners_ptr->GetBool();
  }
  auto none_list_ptr = ctx.GetAttr("none_list");
  KERNEL_CHECK_NULLPTR(none_list_ptr, KERNEL_STATUS_PARAM_INVALID, "[%s] get attr 'none_list' failed.",
                       ctx.GetOpType().c_str());
  none_list = none_list_ptr->GetListInt();
  KERNEL_CHECK_FALSE(none_list.size() == 1, KERNEL_STATUS_PARAM_INVALID,
                     "For 'UpsampleTrilinear3D', only one of output_size or scales should be specified.");
  return KERNEL_STATUS_OK;
}

uint32_t UpsampleTrilinear3dCpuKernel::Compute(CpuKernelContext &ctx) {
  if (UpsampleTrilinear3dParamCheck(ctx) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *x = ctx.Input(0);
  DataType data_type = x->GetDataType();
  uint32_t res = KERNEL_STATUS_OK;
  switch (data_type) {
    case DT_FLOAT16:
      res = UpsampleTrilinear3dCompute<Eigen::half, float>(ctx);
      break;
    case DT_FLOAT:
      res = UpsampleTrilinear3dCompute<float, float>(ctx);
      break;
    case DT_DOUBLE:
      res = UpsampleTrilinear3dCompute<double, double>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR(
        "For UpsampleTrilinear3d, input datatype support [float16, float32, "
        "float64], but get invalid input type [%s].",
        DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Failed launching UpsampleTrilinear3d.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename S>
void UpsampleTrilinear3dCpuKernel::InnerCompute(int64_t n, const T *x_ptr, T *y_ptr) {
  const S depth_scale = AreaPixelComputeScale<S>(input_depth, output_depth, align_corners, scales[kIndex0]);
  const S height_scale = AreaPixelComputeScale<S>(input_height, output_height, align_corners, scales[kIndex1]);
  const S width_scale = AreaPixelComputeScale<S>(input_width, output_width, align_corners, scales[kIndex2]);
  auto input_indexr_value = [=](int64_t n, int64_t d, int64_t h, int64_t w) {
    return x_ptr[n * input_slice_size + d * input_height * input_width + h * input_width + w];
  };
  int64_t id0;
  int64_t id1;
  int64_t ih0;
  int64_t ih1;
  int64_t iw0;
  int64_t iw1;
  S d0lambda;
  S d1lambda;
  S h0lambda;
  S h1lambda;
  S w0lambda;
  S w1lambda;

  for (int64_t od = 0; od < output_depth; ++od) {
    ComputeSourceIndexAndLambda(&id0, &id1, &d0lambda, &d1lambda, depth_scale, od, input_depth, output_depth,
                                align_corners);

    for (int64_t oh = 0; oh < output_height; ++oh) {
      ComputeSourceIndexAndLambda(&ih0, &ih1, &h0lambda, &h1lambda, height_scale, oh, input_height, output_height,
                                  align_corners);
      for (int64_t ow = 0; ow < output_width; ++ow) {
        ComputeSourceIndexAndLambda(&iw0, &iw1, &w0lambda, &w1lambda, width_scale, ow, input_width, output_width,
                                    align_corners);
        auto i000 = static_cast<S>(input_indexr_value(n, id0, ih0, iw0));
        auto i001 = static_cast<S>(input_indexr_value(n, id0, ih0, iw1));
        auto i010 = static_cast<S>(input_indexr_value(n, id0, ih1, iw0));
        auto i011 = static_cast<S>(input_indexr_value(n, id0, ih1, iw1));
        auto i100 = static_cast<S>(input_indexr_value(n, id1, ih0, iw0));
        auto i101 = static_cast<S>(input_indexr_value(n, id1, ih0, iw1));
        auto i110 = static_cast<S>(input_indexr_value(n, id1, ih1, iw0));
        auto i111 = static_cast<S>(input_indexr_value(n, id1, ih1, iw1));
        double w000 = static_cast<double>(d0lambda) * h0lambda * w0lambda;
        double w001 = static_cast<double>(d0lambda) * h0lambda * w1lambda;
        double w010 = static_cast<double>(d0lambda) * h1lambda * w0lambda;
        double w011 = static_cast<double>(d0lambda) * h1lambda * w1lambda;
        double w100 = static_cast<double>(d1lambda) * h0lambda * w0lambda;
        double w101 = static_cast<double>(d1lambda) * h0lambda * w1lambda;
        double w110 = static_cast<double>(d1lambda) * h1lambda * w0lambda;
        double w111 = static_cast<double>(d1lambda) * h1lambda * w1lambda;
        y_ptr[n * output_slice_size + od * output_height * output_width + oh * output_width + ow] =
          static_cast<T>(w000 * i000 + w001 * i001 + w010 * i010 + w011 * i011 + w100 * i100 + w101 * i101 +
                         w110 * i110 + w111 * i111);
      }
    }
  }
}

template <typename T, typename S>
uint32_t UpsampleTrilinear3dCpuKernel::UpsampleTrilinear3dCompute(const CpuKernelContext &ctx) {
  // fetch scales
  scales = std::vector<float>{0, 0, 0};
  if (none_list[kIndex0] != static_cast<int64_t>(kIndex2)) {
    auto scales_ptr = reinterpret_cast<float *>(ctx.Input(kIndex1)->GetData());
    for (int i = 0; i < kValue3; ++i) {
      scales[i] = scales_ptr[i];
    }
  }
  //
  Tensor *input_x = ctx.Input(kIndex0);
  Tensor *output_y = ctx.Output(kIndex0);

  std::vector<int64_t> input_shape = input_x->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> output_shape = output_y->GetTensorShape()->GetDimSizes();

  int64_t channels = input_shape[kIndex0] * input_shape[kIndex1];
  input_depth = input_shape[kIndex2];
  input_height = input_shape[kIndex3];
  input_width = input_shape[kIndex4];

  output_depth = output_shape[kIndex2];
  output_height = output_shape[kIndex3];
  output_width = output_shape[kIndex4];

  output_slice_size = output_depth * output_height * output_width;
  input_slice_size = input_depth * input_height * input_width;
  const int64_t SIZE_BOUNDARY = 100000;
  if (channels <= 0 || output_depth <= 0 || output_height <= 0 || output_width <= 0 || output_depth > SIZE_BOUNDARY ||
      output_height > SIZE_BOUNDARY || output_width > SIZE_BOUNDARY) {
    KERNEL_LOG_ERROR(
      "For UpsampleTrilinear3d, output shape can not less than zero or greater than 100000, but get output "
      "shape = (%lld, %lld, %lld, %lld, %lld). ",
      output_shape[kIndex0], output_shape[kIndex1], output_shape[kIndex2], output_shape[kIndex3],
      output_shape[kIndex4]);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto x_ptr = reinterpret_cast<T *>(input_x->GetData());
  auto y_ptr = reinterpret_cast<T *>(output_y->GetData());
  (void)std::fill_n(y_ptr, output_y->NumElements(), T(0));

  if (input_depth == output_depth && input_height == output_height && input_width == output_width) {
    auto cpret = memcpy_s(y_ptr, static_cast<size_t>(channels * input_slice_size * static_cast<int64_t>(sizeof(T))),
                          x_ptr, static_cast<size_t>(channels * input_slice_size * static_cast<int64_t>(sizeof(T))));
    KERNEL_CHECK_FALSE((cpret == EOK), KERNEL_STATUS_INNER_ERROR,
                       "For UpsampleTrilinear3d, memcpy_s to output failed.");
    return KERNEL_STATUS_OK;
  }

  auto loop3d = [&](int64_t begin, int64_t end) {
    for (int64_t n = begin; n < end; ++n) {
      InnerCompute<T, S>(n, x_ptr, y_ptr);
    }
  };
  int64_t total = output_y->NumElements();
  int64_t per_unit_size{total};
  if (total > kParallelDataNum) {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (total <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, kReserveCpu);
    }
    per_unit_size = (total / max_core_num) / output_slice_size;
    per_unit_size = std::max(per_unit_size, 1L);
    per_unit_size = std::min(channels, per_unit_size);
  }
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, channels, per_unit_size, loop3d), "loop3d Compute failed.");
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kUpsampleTrilinear3d, UpsampleTrilinear3dCpuKernel);
}  // namespace aicpu
