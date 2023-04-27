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

#include "upsample_nearest3d_grad.h"
#include <stdint.h>
#include <cstring>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"
#include "kernel_log.h"
#include "securec.h"
#include "status.h"
#include "utils/eigen_tensor.h"

namespace {
constexpr auto kInputNum = 1;
constexpr auto kOutputNum = 1;
constexpr auto kSize2 = 2;
constexpr auto kSize3 = 3;
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
constexpr auto kUpsampleNearest3dGrad = "UpsampleNearest3dGrad";
}  // namespace

namespace aicpu {
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

template <typename scalar_t>
static inline scalar_t ComputeScalesValue(double scale, int64_t input_size, int64_t output_size) {
  if (scale > 0) {
    return static_cast<scalar_t>(1.0 / scale);
  }
  if (output_size > 0) {
    return (static_cast<scalar_t>(input_size) / output_size);
  }
  return static_cast<scalar_t>(0);
}

static inline int64_t NearestNeighborComputeSourceIndex(const float scale, int64_t dst_index, int64_t input_size) {
  const int64_t src_index = std::min(static_cast<int64_t>(floorf(dst_index * scale)), input_size - 1);
  return src_index;
}

static inline int64_t NearestIdx(int64_t output_index, int64_t input_size, int64_t output_size, double scales) {
  if (output_size == input_size) {
    return output_index;
  } else if (output_size == kSize2 * input_size) {
    return static_cast<uint64_t>(output_index) >> 1;
  } else {
    float scale = ComputeScalesValue<float>(scales, input_size, output_size);
    return NearestNeighborComputeSourceIndex(scale, output_index, input_size);
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
static uint32_t IsSameVector(const std::vector<T> &args_value, const std::string &args_name,
                             const std::vector<T> &target_value, const std::string &target_name,
                             const std::string &op_name) {
  KERNEL_CHECK_FALSE((args_value.size() == target_value.size()), KERNEL_STATUS_PARAM_INVALID,
                     "For [%s], [%s] should have the same size with [%s] "
                     "but get size(%s) = %d, size(%s) = %d.",
                     op_name.c_str(), args_name.c_str(), target_name.c_str(), args_name.c_str(), args_value.size(),
                     target_name.c_str(), target_value.size());
  bool all_same = true;
  for (size_t idx = 0; idx < args_value.size(); ++idx) {
    all_same &= (args_value[idx] == target_value[idx]);
  }
  std::string args_str = VectorToString(args_value);
  std::string target_str = VectorToString(target_value);
  KERNEL_CHECK_FALSE(all_same, KERNEL_STATUS_PARAM_INVALID,
                     "For [%s], [%s] should be same with [%s], but get %s = %s, %s = %s.", op_name.c_str(),
                     args_name.c_str(), target_name.c_str(), args_name.c_str(), args_str.c_str(), target_name.c_str(),
                     target_str.c_str());
  return KERNEL_STATUS_OK;
}

uint32_t UpsampleNearest3dGradCpuKernel::UpsampleNearest3dGradParamCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check params failed.", kUpsampleNearest3dGrad);
  grads_output_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  KERNEL_HANDLE_ERROR(
    CheckPositiveVectorWithSpecifiedSize(grads_output_shape, "grads_output_shape", kDims5, kUpsampleNearest3dGrad),
    "[%s] check grads_output_shape failed.", kUpsampleNearest3dGrad);
  grads_input_shape = ctx.Output(kIndex0)->GetTensorShape()->GetDimSizes();
  KERNEL_HANDLE_ERROR(
    CheckPositiveVectorWithSpecifiedSize(grads_input_shape, "grads_input_shape", kDims5, kUpsampleNearest3dGrad),
    "[%s] check grads_input_shape failed.", kUpsampleNearest3dGrad);
  KERNEL_CHECK_FALSE(grads_output_shape.size() == kDims5, KERNEL_STATUS_PARAM_INVALID,
                     "Upsample with NHWC format supports tensors with 5 "
                     "dims. but got %d dim(s).",
                     grads_output_shape.size());
  auto input_size_ptr = ctx.GetAttr("input_size");
  KERNEL_CHECK_NULLPTR(input_size_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input_size failed.")
  input_size = input_size_ptr->GetListInt();
  KERNEL_HANDLE_ERROR(CheckPositiveVectorWithSpecifiedSize(input_size, "input_size", kDims5, kUpsampleNearest3dGrad),
                      "[%s] check attr::input_size failed.", kUpsampleNearest3dGrad);
  KERNEL_HANDLE_ERROR(
    IsSameVector(input_size, "input_size", grads_input_shape, "grads_input_shape", kUpsampleNearest3dGrad),
    "[%s] check input_size whether same with grads_input_shape failed.", kUpsampleNearest3dGrad);
  auto output_size_ptr = ctx.GetAttr("output_size");
  if (output_size_ptr) {
    output_size = output_size_ptr->GetListInt();
  }
  auto scales_ptr = ctx.GetAttr("scales");
  if (scales_ptr) {
    scales = scales_ptr->GetListFloat();
  }
  if ((output_size.empty() && scales.empty()) || (!output_size.empty() && !scales.empty())) {
    KERNEL_LOG_ERROR(
      "For UpsampleNearest3dGrad, only one of 'scales' and 'output_size' "
      "can be specified, but get both.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (output_size.empty()) {
    KERNEL_HANDLE_ERROR(CheckPositiveVectorWithSpecifiedSize(scales, "scales", kSize3, kUpsampleNearest3dGrad),
                        "[%s] check attr::scales failed.", kUpsampleNearest3dGrad);

    for (int64_t idx = 0; idx < kIndex3; ++idx) {
      int64_t idx_dim = int64_t(floor(input_size[idx + kIndex2] * scales[idx]));
      KERNEL_CHECK_FALSE((idx_dim == grads_output_shape[idx + kIndex2]), KERNEL_STATUS_PARAM_INVALID,
                         "For [%s], input_size[%d]*scales[%d]!= "
                         "grads_output_shape[%d].",
                         kUpsampleNearest3dGrad, idx + kIndex2, idx, idx + kIndex2);
    }
  }
  if (scales.empty()) {
    scales = {0, 0, 0};
    KERNEL_HANDLE_ERROR(
      CheckPositiveVectorWithSpecifiedSize(output_size, "output_size", kSize3, kUpsampleNearest3dGrad),
      "[%s] check attr::output_size failed.", kUpsampleNearest3dGrad);

    for (int64_t idx = 0; idx < kIndex3; ++idx) {
      KERNEL_CHECK_FALSE((output_size[idx] == grads_output_shape[idx + kIndex2]), KERNEL_STATUS_PARAM_INVALID,
                         "For [%s], attr::output_size[%d](get %ld) != "
                         "input::grads_output_shape[%d](get %ld).",
                         kUpsampleNearest3dGrad, idx, output_size[idx], idx + kIndex2,
                         grads_output_shape[idx + kIndex2]);
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t UpsampleNearest3dGradCpuKernel::Compute(CpuKernelContext &ctx) {
  if (UpsampleNearest3dGradParamCheck(ctx) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  DataType data_type = ctx.Input(kIndex0)->GetDataType();
  uint32_t res = KERNEL_STATUS_OK;
  switch (data_type) {
    case DT_FLOAT16:
      res = UpsampleNearest3dGradCompute<Eigen::half, float>(ctx);
      break;
    case DT_FLOAT:
      res = UpsampleNearest3dGradCompute<float, float>(ctx);
      break;
    case DT_DOUBLE:
      res = UpsampleNearest3dGradCompute<double, double>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("UpsampleNearest3dGrad invalid input type [%s].", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Failed launching UpsampleNearest3dGrad.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}
template <typename T, typename S>
void UpsampleNearest3dGradCpuKernel::InnerCompute(int64_t c, const T *grad_output_data, S *grad_input_ptr) {
  for (int64_t od = 0; od < output_depth; ++od) {
    int64_t id = NearestIdx(od, input_depth, output_depth, static_cast<double>(scales[kIndex0]));
    for (int64_t oh = 0; oh < output_height; ++oh) {
      int64_t ih = NearestIdx(oh, input_height, output_height, static_cast<double>(scales[kIndex1]));
      for (int64_t ow = 0; ow < output_width; ++ow) {
        int64_t iw = NearestIdx(ow, input_width, output_width, static_cast<double>(scales[kIndex2]));
        int64_t output_offset = c * output_slice_size + od * output_height * output_width + oh * output_width + ow;
        int64_t input_offset = c * input_slice_size + id * input_height * input_width + ih * input_width + iw;
        grad_input_ptr[input_offset] += static_cast<S>(grad_output_data[output_offset]);
      }
    }
  }
}

template <typename T, typename S>
uint32_t UpsampleNearest3dGradCpuKernel::UpsampleNearest3dGradCompute(const CpuKernelContext &ctx) {
  // the input of backward process is output in forward process
  Tensor *grads_out = ctx.Input(kIndex0);
  Tensor *grads_in = ctx.Output(kIndex0);

  int64_t channels = grads_input_shape[kIndex0] * grads_input_shape[kIndex1];
  input_depth = grads_input_shape[kIndex2];
  input_height = grads_input_shape[kIndex3];
  input_width = grads_input_shape[kIndex4];

  output_depth = grads_output_shape[kIndex2];
  output_height = grads_output_shape[kIndex3];
  output_width = grads_output_shape[kIndex4];

  output_slice_size = output_depth * output_height * output_width;
  input_slice_size = input_depth * input_height * input_width;
  int64_t total = grads_in->NumElements();

  auto grad_output_data = reinterpret_cast<T *>(grads_out->GetData());
  S *grad_input_ptr = nullptr;
  bool is_fp16 = std::is_same<T, Eigen::half>::value;
  // define for fp16
  std::vector<S> grad_input_copy(1);
  if (is_fp16) {
    grad_input_copy.resize(total, static_cast<int64_t>(0));
    grad_input_ptr = grad_input_copy.data();
  } else {
    grad_input_ptr = reinterpret_cast<S *>(grads_in->GetData());
    (void)std::fill_n(grad_input_ptr, total, S(0));
  }
  auto loop3d = [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; ++c) {
      InnerCompute<T, S>(c, grad_output_data, grad_input_ptr);
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
    T *real_input_ptr = reinterpret_cast<T *>(grads_in->GetData());
    for (int64_t idx = 0; idx < total; ++idx) {
      real_input_ptr[idx] = static_cast<T>(grad_input_ptr[idx]);
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kUpsampleNearest3dGrad, UpsampleNearest3dGradCpuKernel);
}  // namespace aicpu
