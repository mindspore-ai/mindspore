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

#include "upsample_trilinear3d_grad.h"
#include <string>
#include <vector>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_log.h"
#include "securec.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr auto kInputNum = 1;
constexpr auto kOutputNum = 1;
constexpr auto kValue2 = 2;
constexpr auto kValue3 = 3;
constexpr auto kDims5 = 5;
constexpr auto kIndex0 = 0;
constexpr auto kIndex1 = 1;
constexpr int64_t kIndex2 = 2;
constexpr int64_t kIndex3 = 3;
constexpr int64_t kIndex4 = 4;
// when input data size is more than kParallelDataNum, use Parallel func
constexpr int64_t kParallelDataNum = 7 * 1024;
constexpr int64_t kParallelDataNumMid = 35 * 1024;
const int64_t kReserveCpu = 4;
constexpr auto kUpsampleTrilinear3dGrad = "UpsampleTrilinear3dGrad";
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
static inline void ComputeSourceIndexAndLambda(int64_t &input_index0, int64_t &input_index1, T &lambda0, T &lambda1,
                                               T ratio, int64_t output_index, int64_t input_size, int64_t output_size,
                                               bool align_corners) {
  if (output_size == input_size) {
    // scale_factor = 1
    input_index0 = output_index;
    input_index1 = output_index;
    lambda0 = static_cast<T>(1);
    lambda1 = static_cast<T>(0);
  } else {
    const T real_input_index = AreaPixelComputeSourceIndex<T>(ratio, output_index, align_corners);
    input_index0 = static_cast<int64_t>(real_input_index);
    int64_t offset = (input_index0 < input_size - 1) ? 1 : 0;
    input_index1 = input_index0 + offset;
    lambda1 = real_input_index - input_index0;
    constexpr T one = 1.0;
    lambda0 = one - lambda1;
  }
}

/**
 * @brief check positive vector with special size
 * @param args_value input vector with type T
 * @param args_name input vector name
 * @param args_size input vector compare size
 * @param op_name operator name
 * @return status code
 */
template <typename T>
static uint32_t CheckPositiveVectorWithSpecifiedSize(const std::vector<T> &args_value, const std::string &args_name,
                                                     size_t args_size, const std::string &op_name) {
  KERNEL_CHECK_FALSE((args_value.size() == args_size), KERNEL_STATUS_PARAM_INVALID,
                     "For [%s], [%s] should have %d "
                     "positive number but only get %d number.",
                     op_name.c_str(), args_name.c_str(), args_size, args_value.size());
  bool all_positive = true;
  for (auto const &num : args_value) {
    all_positive = (all_positive & (num > T(0)));
  }
  std::string args_str = VectorToString(args_value);
  KERNEL_CHECK_FALSE(all_positive, KERNEL_STATUS_PARAM_INVALID,
                     "For [%s], [%s] should have %d "
                     "positive number. but get %s = (%s).",
                     op_name.c_str(), args_name.c_str(), args_size, args_name.c_str(), args_str.c_str());
  return KERNEL_STATUS_OK;
}
uint32_t UpsampleTrilinear3dGradCpuKernel::UpsampleTrilinear3dGradParamCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check params failed.", kUpsampleTrilinear3dGrad);
  auto grad_output_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  auto grad_input_shape = ctx.Output(kIndex0)->GetTensorShape()->GetDimSizes();
  KERNEL_HANDLE_ERROR(
    CheckPositiveVectorWithSpecifiedSize(grad_output_shape, "grad_output shape", kDims5, kUpsampleTrilinear3dGrad),
    "[%s] check grad_output shape failed.", kUpsampleTrilinear3dGrad);
  auto input_size_ptr = ctx.GetAttr("input_size");
  KERNEL_CHECK_NULLPTR(input_size_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input_size failed.")
  input_size = input_size_ptr->GetListInt();
  KERNEL_HANDLE_ERROR(CheckPositiveVectorWithSpecifiedSize(input_size, "input_size", kDims5, kUpsampleTrilinear3dGrad),
                      "[%s] check attr::input_size failed.", kUpsampleTrilinear3dGrad);
  for (int64_t idx = 0; idx < static_cast<int64_t>(kDims5); ++idx) {
    KERNEL_CHECK_FALSE((input_size[idx] == grad_input_shape[idx]), KERNEL_STATUS_PARAM_INVALID,
                       "For [%s], input_size[%d](get %ld) != "
                       "grad_input_shape[%d](get %ld).",
                       kUpsampleTrilinear3dGrad, idx, input_size[idx], idx, grad_input_shape[idx]);
  }
  auto align_corners_ptr = ctx.GetAttr("align_corners");
  if (align_corners_ptr) {
    align_corners = align_corners_ptr->GetBool();
  }
  auto scales_ptr = ctx.GetAttr("scales");
  if (scales_ptr) {
    scales = scales_ptr->GetListFloat();
  }
  auto output_size_ptr = ctx.GetAttr("output_size");
  if (output_size_ptr) {
    output_size = output_size_ptr->GetListInt();
  }
  if ((output_size.empty() && scales.empty()) || (!output_size.empty() && !scales.empty())) {
    KERNEL_LOG_ERROR(
      "For UpsampleTrilinear3dGrad, only one of 'scales' and 'output_size' "
      "can be specified.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (output_size.empty()) {
    KERNEL_HANDLE_ERROR(CheckPositiveVectorWithSpecifiedSize(scales, "scales", kValue3, kUpsampleTrilinear3dGrad),
                        "[%s] check attr::scales failed.", kUpsampleTrilinear3dGrad);

    for (int64_t idx = 0; idx < kIndex3; ++idx) {
      int64_t idx_dim = int64_t(floor(input_size[idx + kIndex2] * scales[idx]));
      KERNEL_CHECK_FALSE((idx_dim == grad_output_shape[idx + kIndex2]), KERNEL_STATUS_PARAM_INVALID,
                         "For [%s], input_size[%d]*scales[%d](get %ld) != "
                         "grad_output_size[%d](get %ld).",
                         kUpsampleTrilinear3dGrad, idx + kIndex2, idx, idx_dim, idx + kIndex2,
                         grad_output_shape[idx + kIndex2]);
    }
  }
  if (scales.empty()) {
    scales = {0, 0, 0};
    KERNEL_HANDLE_ERROR(
      CheckPositiveVectorWithSpecifiedSize(output_size, "output_size", kValue3, kUpsampleTrilinear3dGrad),
      "[%s] check attr::output_size failed.", kUpsampleTrilinear3dGrad);

    for (int64_t idx = 0; idx < kIndex3; ++idx) {
      KERNEL_CHECK_FALSE((output_size[idx] == grad_output_shape[idx + kIndex2]), KERNEL_STATUS_PARAM_INVALID,
                         "For [%s], attr::output_size[%d](get %ld) != "
                         "input::grad_output_shape[%d](get %ld).",
                         kUpsampleTrilinear3dGrad, idx, output_size[idx], idx + kIndex2,
                         grad_output_shape[idx + kIndex2]);
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t UpsampleTrilinear3dGradCpuKernel::Compute(CpuKernelContext &ctx) {
  if (UpsampleTrilinear3dGradParamCheck(ctx) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  DataType data_type = ctx.Input(0)->GetDataType();
  uint32_t res = KERNEL_STATUS_OK;
  switch (data_type) {
    case DT_FLOAT16:
      res = UpsampleTrilinear3dGradCompute<Eigen::half, float>(ctx);
      break;
    case DT_FLOAT:
      res = UpsampleTrilinear3dGradCompute<float, float>(ctx);
      break;
    case DT_DOUBLE:
      res = UpsampleTrilinear3dGradCompute<double, double>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR(
        "For UpsampleTrilinear3dGrad, input datatype support [float16, "
        "float32, float64], but get invalid input type [%s].",
        DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Failed launching UpsampleTrilinear3dGrad.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename S>
void UpsampleTrilinear3dGradCpuKernel::InnerCompute(int64_t c_idx, const T *grad_output_ptr, S *grad_input_ptr) {
  const auto depth_scale = AreaPixelComputeScale<S>(input_depth, output_depth, align_corners, scales[kIndex0]);
  const auto height_scale = AreaPixelComputeScale<S>(input_height, output_height, align_corners, scales[kIndex1]);
  const auto width_scale = AreaPixelComputeScale<S>(input_width, output_width, align_corners, scales[kIndex2]);
  auto input_index = [=](int64_t c_idx, int64_t d_idx, int64_t h_idx, int64_t w_idx) {
    return c_idx * input_slice_size + d_idx * input_height * input_width + h_idx * input_width + w_idx;
  };
  int64_t id0, id1, ih0, ih1, iw0, iw1;
  S d0lambda, d1lambda, h0lambda, h1lambda, w0lambda, w1lambda;
  for (int64_t od = 0; od < output_depth; ++od) {
    ComputeSourceIndexAndLambda<S>(id0, id1, d0lambda, d1lambda, depth_scale, od, input_depth, output_depth,
                                   align_corners);
    for (int64_t oh = 0; oh < output_height; ++oh) {
      ComputeSourceIndexAndLambda<S>(ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height,
                                     align_corners);
      for (int64_t ow = 0; ow < output_width; ++ow) {
        ComputeSourceIndexAndLambda<S>(iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width,
                                       align_corners);

        auto grad_output_value = static_cast<S>(
          grad_output_ptr[c_idx * output_slice_size + od * output_height * output_width + oh * output_width + ow]);
        double w000 = static_cast<double>(d0lambda) * h0lambda * w0lambda;
        double w001 = static_cast<double>(d0lambda) * h0lambda * w1lambda;
        double w010 = static_cast<double>(d0lambda) * h1lambda * w0lambda;
        double w011 = static_cast<double>(d0lambda) * h1lambda * w1lambda;
        double w100 = static_cast<double>(d1lambda) * h0lambda * w0lambda;
        double w101 = static_cast<double>(d1lambda) * h0lambda * w1lambda;
        double w110 = static_cast<double>(d1lambda) * h1lambda * w0lambda;
        double w111 = static_cast<double>(d1lambda) * h1lambda * w1lambda;
        grad_input_ptr[input_index(c_idx, id0, ih0, iw0)] =
          static_cast<S>(w000 * grad_output_value + grad_input_ptr[input_index(c_idx, id0, ih0, iw0)]);
        grad_input_ptr[input_index(c_idx, id0, ih0, iw1)] =
          static_cast<S>(w001 * grad_output_value + grad_input_ptr[input_index(c_idx, id0, ih0, iw1)]);
        grad_input_ptr[input_index(c_idx, id0, ih1, iw0)] =
          static_cast<S>(w010 * grad_output_value + grad_input_ptr[input_index(c_idx, id0, ih1, iw0)]);
        grad_input_ptr[input_index(c_idx, id0, ih1, iw1)] =
          static_cast<S>(w011 * grad_output_value + grad_input_ptr[input_index(c_idx, id0, ih1, iw1)]);
        grad_input_ptr[input_index(c_idx, id1, ih0, iw0)] =
          static_cast<S>(w100 * grad_output_value + grad_input_ptr[input_index(c_idx, id1, ih0, iw0)]);
        grad_input_ptr[input_index(c_idx, id1, ih0, iw1)] =
          static_cast<S>(w101 * grad_output_value + grad_input_ptr[input_index(c_idx, id1, ih0, iw1)]);
        grad_input_ptr[input_index(c_idx, id1, ih1, iw0)] =
          static_cast<S>(w110 * grad_output_value + grad_input_ptr[input_index(c_idx, id1, ih1, iw0)]);
        grad_input_ptr[input_index(c_idx, id1, ih1, iw1)] =
          static_cast<S>(w111 * grad_output_value + grad_input_ptr[input_index(c_idx, id1, ih1, iw1)]);
      }
    }
  }
}

template <typename T, typename S>
uint32_t UpsampleTrilinear3dGradCpuKernel::UpsampleTrilinear3dGradCompute(const CpuKernelContext &ctx) {
  Tensor *grad_output_ = ctx.Input(kIndex0);
  Tensor *grad_input_ = ctx.Output(kIndex0);

  auto input_shape_ = grad_input_->GetTensorShape()->GetDimSizes();
  auto output_shape_ = grad_output_->GetTensorShape()->GetDimSizes();
  int64_t total = grad_input_->NumElements();
  T *grad_output_ptr = reinterpret_cast<T *>(grad_output_->GetData());
  S *grad_input_ptr = nullptr;
  bool is_fp16 = std::is_same<T, Eigen::half>::value;
  // define for fp16
  std::vector<S> grad_input_copy(1);
  if (is_fp16) {
    grad_input_copy.resize(grad_input_->NumElements(), static_cast<int64_t>(0));
    grad_input_ptr = grad_input_copy.data();
  } else {
    grad_input_ptr = reinterpret_cast<S *>(grad_input_->GetData());
    (void)std::fill_n(grad_input_ptr, total, S(0));
  }

  int64_t channels = input_shape_[kIndex0] * input_shape_[kIndex1];
  input_depth = input_shape_[kIndex2];
  input_height = input_shape_[kIndex3];
  input_width = input_shape_[kIndex4];
  output_depth = output_shape_[kIndex2];
  output_height = output_shape_[kIndex3];
  output_width = output_shape_[kIndex4];
  output_slice_size = output_depth * output_height * output_width;
  input_slice_size = input_depth * input_height * input_width;
  const int64_t SIZE_BOUNDARY = 100000;
  if (channels <= 0 || output_depth <= 0 || output_height <= 0 || output_width <= 0 || output_depth > SIZE_BOUNDARY ||
      output_height > SIZE_BOUNDARY || output_width > SIZE_BOUNDARY) {
    KERNEL_LOG_ERROR(
      "For UpsampleTrilinear3dGrad, output shape can not less than zero or greater than 100000, but get output "
      "shape = (%lld, %lld, %lld, %lld, %lld). ",
      output_shape_[kIndex0], output_shape_[kIndex1], output_shape_[kIndex2], output_shape_[kIndex3],
      output_shape_[kIndex4]);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (input_depth <= 0 || input_height <= 0 || input_width <= 0 || input_depth > SIZE_BOUNDARY ||
      input_height > SIZE_BOUNDARY || input_width > SIZE_BOUNDARY) {
    KERNEL_LOG_ERROR(
      "For UpsampleTrilinear3dGrad, input shape can not less than zero or greater than 100000, but get input "
      "shape = (%lld, %lld, %lld, %lld, %lld). ",
      input_shape_[kIndex0], input_shape_[kIndex1], input_shape_[kIndex2], input_shape_[kIndex3],
      input_shape_[kIndex4]);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // calculate index

  auto loop3d = [&](int64_t begin, int64_t end) {
    for (int64_t c_idx = begin; c_idx < end; ++c_idx) {
      InnerCompute(c_idx, grad_output_ptr, grad_input_ptr);
    }
  };
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
  // memcopy and cast for fp16
  if (is_fp16) {
    T *real_input_ptr = reinterpret_cast<T *>(grad_input_->GetData());
    for (int64_t idx = 0; idx < total; ++idx) {
      real_input_ptr[idx] = static_cast<T>(grad_input_ptr[idx]);
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kUpsampleTrilinear3dGrad, UpsampleTrilinear3dGradCpuKernel);
}  // namespace aicpu
