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
constexpr double kValueZero = 0.;
constexpr auto kInputNum = 3;
constexpr auto kOutputNum = 1;
constexpr int kValue3 = 3;
constexpr auto kDims5 = 5;
constexpr int64_t kIndex0 = 0;
constexpr int64_t kIndex1 = 1;
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
inline T ComputeScales(const double &scale, const size_t &input_size, const size_t &output_size) {
  if (scale > kValueZero) {
    return static_cast<T>(1.0 / scale);
  } else if (output_size > 0) {
    return (static_cast<T>(input_size) / output_size);
  }
  return 0;
}

template <typename T>
inline T AreaPixelComputeScale(int64_t input_size, int64_t output_size, bool align_corners, double scale) {
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
inline T AreaPixelComputeSourceIndex(T scale, int64_t dst_index, bool align_corners) {
  if (align_corners) {
    return scale * static_cast<T>(dst_index);
  } else {
    constexpr T zero = 0.;
    T src_idx = scale * (dst_index + 0.5) - 0.5;
    return src_idx < zero ? zero : src_idx;
  }
}

template <typename T>
inline void ComputeSourceIndexAndLambda(int64_t *const input_index0, int64_t *const input_index1, T *const lambda0,
                                        T *const lambda1, T ratio, int64_t output_index, int64_t input_size,
                                        int64_t output_size, bool align_corners) {
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
    *lambda1 = real_input_index - static_cast<T>(*input_index0);
    constexpr T one = 1.0;
    *lambda0 = one - *lambda1;
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

  auto align_corners_ptr = ctx.GetAttr("align_corners");
  if (align_corners_ptr) {
    align_corners = align_corners_ptr->GetBool();
  }
  auto none_list_ptr = ctx.GetAttr("none_list");
  KERNEL_CHECK_NULLPTR(none_list_ptr, KERNEL_STATUS_PARAM_INVALID, "[%s] get attr 'none_list' failed.",
                       ctx.GetOpType().c_str());
  none_list = none_list_ptr->GetListInt();
  KERNEL_CHECK_FALSE(none_list.size() == 1, KERNEL_STATUS_PARAM_INVALID,
                     "For 'UpsampleNearest3DGrad', only one of output_size or scales should be specified.");

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

template <typename S>
void UpsampleTrilinear3dGradCpuKernel::ComputeWeightsAndIndices(
  UpsampleTrilinear3dGradCpuKernel::WeightsAndIndices<S> &wi, S scale, int64_t out_idx, int64_t input_size,
  int64_t output_size, int64_t stride) {
  (void)ComputeSourceIndexAndLambda<S>(&(wi.id0), &(wi.id1), &(wi.lambda0), &(wi.lambda1), scale, out_idx, input_size,
                                       output_size, align_corners);
  wi.Step(stride);
}

template <typename S>
void UpsampleTrilinear3dGradCpuKernel::ComputeHelper(
  const CpuKernelContext &ctx, std::vector<UpsampleTrilinear3dGradCpuKernel::WeightsAndIndices<S>> &helper, S scale,
  int64_t input_size, int64_t output_size, int64_t stride) {
  for (int64_t out_idx = 0; out_idx < output_size; ++out_idx) {
    (void)ComputeWeightsAndIndices<S>(helper[out_idx], scale, out_idx, input_size, output_size, stride);
  }
}

template <typename T, typename S, typename R>
uint32_t UpsampleTrilinear3dGradCpuKernel::RealCompute(const CpuKernelContext &ctx, T *const grad_output_ptr,
                                                       R *const grad_input_ptr) {
  const S depth_scale = AreaPixelComputeScale<S>(input_depth, output_depth, align_corners, scales[kIndex0]);
  const S height_scale = AreaPixelComputeScale<S>(input_height, output_height, align_corners, scales[kIndex1]);
  const S width_scale = AreaPixelComputeScale<S>(input_width, output_width, align_corners, scales[kIndex2]);

  std::vector<WeightsAndIndices<S>> d_helper(output_depth);
  std::vector<WeightsAndIndices<S>> h_helper(output_height);
  std::vector<WeightsAndIndices<S>> w_helper(output_width);
  (void)ComputeHelper<S>(ctx, d_helper, depth_scale, input_depth, output_depth, input_height * input_width);
  (void)ComputeHelper<S>(ctx, h_helper, height_scale, input_height, output_height, input_width);
  (void)ComputeHelper<S>(ctx, w_helper, width_scale, input_width, output_width, 1);

  auto loop3d = [&](int64_t begin, int64_t end) {
    for (int64_t c_idx = begin, src_c_offset = begin * input_slice_size; c_idx < end;
         ++c_idx, src_c_offset += input_slice_size) {
      int64_t id0{0};
      int64_t id1{0};
      int64_t ih0{0};
      int64_t ih1{0};
      int64_t iw0{0};
      int64_t iw1{0};
      S d0lambda{0};
      S d1lambda{0};
      S h0lambda{0};
      S h1lambda{0};
      S w0lambda{0};
      S w1lambda{0};
      for (int64_t od = 0; od < output_depth; ++od) {
        d_helper[od](&id0, &id1, &d0lambda, &d1lambda);

        for (int64_t oh = 0; oh < output_height; ++oh) {
          h_helper[oh](&ih0, &ih1, &h0lambda, &h1lambda);

          for (int64_t ow = 0; ow < output_width; ++ow) {
            w_helper[ow](&iw0, &iw1, &w0lambda, &w1lambda);

            auto grad_output_value = static_cast<S>(
              grad_output_ptr[c_idx * output_slice_size + od * output_height * output_width + oh * output_width + ow]);
            S w000 = d0lambda * h0lambda * w0lambda;
            S w001 = d0lambda * h0lambda * w1lambda;
            S w010 = d0lambda * h1lambda * w0lambda;
            S w011 = d0lambda * h1lambda * w1lambda;
            S w100 = d1lambda * h0lambda * w0lambda;
            S w101 = d1lambda * h0lambda * w1lambda;
            S w110 = d1lambda * h1lambda * w0lambda;
            S w111 = d1lambda * h1lambda * w1lambda;
            grad_input_ptr[src_c_offset + id0 + ih0 + iw0] += static_cast<R>(w000 * grad_output_value);
            grad_input_ptr[src_c_offset + id0 + ih0 + iw1] += static_cast<R>(w001 * grad_output_value);
            grad_input_ptr[src_c_offset + id0 + ih1 + iw0] += static_cast<R>(w010 * grad_output_value);
            grad_input_ptr[src_c_offset + id0 + ih1 + iw1] += static_cast<R>(w011 * grad_output_value);
            grad_input_ptr[src_c_offset + id1 + ih0 + iw0] += static_cast<R>(w100 * grad_output_value);
            grad_input_ptr[src_c_offset + id1 + ih0 + iw1] += static_cast<R>(w101 * grad_output_value);
            grad_input_ptr[src_c_offset + id1 + ih1 + iw0] += static_cast<R>(w110 * grad_output_value);
            grad_input_ptr[src_c_offset + id1 + ih1 + iw1] += static_cast<R>(w111 * grad_output_value);
          }
        }
      }
    }
  };
  auto block_size = FetchBlockSize(ctx, channels, output_slice_size);
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, channels, block_size, loop3d), "loop3d Compute failed.");
  return KERNEL_STATUS_OK;
}

int64_t UpsampleTrilinear3dGradCpuKernel::FetchBlockSize(const CpuKernelContext &ctx, const int64_t parallel_num,
                                                         const int64_t cost) {
  int64_t block_size{parallel_num};
  int64_t total_cost = cost * parallel_num;
  if (total_cost > kParallelDataNum) {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (total_cost <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, kReserveCpu);
    }
    block_size = (total_cost / max_core_num) / output_slice_size;
    block_size = std::max(block_size, 1L);
    block_size = std::min(parallel_num, block_size);
  }
  return block_size;
}

template <typename T, typename S>
uint32_t UpsampleTrilinear3dGradCpuKernel::UpsampleTrilinear3dGradCompute(const CpuKernelContext &ctx) {
  Tensor *grad_output_ = ctx.Input(kIndex0);
  Tensor *grad_input_ = ctx.Output(kIndex0);

  // fetch scales
  scales = std::vector<float>{0, 0, 0};
  if (none_list[kIndex0] != static_cast<int64_t>(kIndex3)) {
    auto scales_ptr = reinterpret_cast<float *>(ctx.Input(kIndex2)->GetData());
    for (int i = 0; i < kValue3; ++i) {
      scales[i] = scales_ptr[i];
    }
  }

  auto input_shape_ = grad_input_->GetTensorShape()->GetDimSizes();
  auto output_shape_ = grad_output_->GetTensorShape()->GetDimSizes();

  channels = input_shape_[kIndex0] * input_shape_[kIndex1];
  input_depth = input_shape_[kIndex2];
  input_height = input_shape_[kIndex3];
  input_width = input_shape_[kIndex4];
  input_slice_size = input_depth * input_height * input_width;

  output_depth = output_shape_[kIndex2];
  output_height = output_shape_[kIndex3];
  output_width = output_shape_[kIndex4];
  output_slice_size = output_depth * output_height * output_width;

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

  // workspace
  int64_t total = grad_input_->NumElements();
  T *grad_output_ptr = reinterpret_cast<T *>(grad_output_->GetData());
  T *grad_input_ptr = reinterpret_cast<T *>(grad_input_->GetData());
  S *grad_work_ptr = nullptr;

  bool is_fp16 = std::is_same<T, Eigen::half>::value;
  uint32_t paralle_ret = KERNEL_STATUS_OK;
  // define for fp16
  std::vector<S> grad_work_copy(1, 0);
  if (is_fp16) {
    grad_work_copy.resize(total, 0);
    grad_work_ptr = grad_work_copy.data();
    paralle_ret = RealCompute<T, S, S>(ctx, grad_output_ptr, grad_work_ptr);
  } else {
    size_t y_size = total * sizeof(T);
    int ret = memset_s(grad_input_ptr, y_size, 0, y_size);
    KERNEL_CHECK_FALSE(ret == EOK, KERNEL_STATUS_INNER_ERROR,
                       "For 'UpsampleTrilinear3DGrad', memset_s error. Error no: %d.", ret);
    paralle_ret = RealCompute<T, S, T>(ctx, grad_output_ptr, grad_input_ptr);
  }
  if (paralle_ret != KERNEL_STATUS_OK) {
    return paralle_ret;
  }
  // memcopy and cast for fp16
  if (is_fp16) {
    auto memcpy_fp16 = [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        grad_input_ptr[idx] = static_cast<T>(grad_work_ptr[idx]);
      }
    };
    int64_t max_core_num = std::max(1L, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2L);
    int64_t block_size = total / max_core_num;
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, total, block_size, memcpy_fp16),
                        "memcpy_fp16 Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kUpsampleTrilinear3dGrad, UpsampleTrilinear3dGradCpuKernel);
}  // namespace aicpu
