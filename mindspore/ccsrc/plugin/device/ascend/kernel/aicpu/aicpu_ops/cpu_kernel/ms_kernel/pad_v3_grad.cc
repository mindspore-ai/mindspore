/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "pad_v3_grad.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

#include "securec.h"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kPadV3Grad = "PadV3Grad";
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
constexpr int64_t kParallelNum = 1024 * 64;
const int64_t k3DNum = 6;
const int64_t k2DNum = 4;
const int64_t k1DNum = 2;
constexpr int64_t kpad_l = 0;
constexpr int64_t kpad_t = 2;
constexpr int64_t kpad_f = 4;
constexpr int64_t kwidth = 1;
constexpr int64_t kheight = 2;
constexpr int64_t kchannel = 3;
constexpr int64_t kInput1Dim = 3;
constexpr int64_t kInput2Dim = 4;
constexpr int64_t kInput3Dim = 5;
constexpr int64_t k2Num = 2;
constexpr int64_t k3Num = 3;
constexpr int64_t k4Num = 4;

const std::vector<std::string> mode_list = {"reflect", "edge"};
using float16 = Eigen::half;

#define PAD_V3_GRAD_READ_PADDINGS(DTYPE, TYPE, CTX)                    \
  case (DTYPE): {                                                      \
    uint32_t result1 = PadV3ReadPaddingsAndSetOutputShape1<TYPE>(CTX); \
    uint32_t result2 = PadV3ReadPaddingsAndSetOutputShape2<TYPE>(CTX); \
    if (result1 != KERNEL_STATUS_OK || result2 != KERNEL_STATUS_OK) {  \
      KERNEL_LOG_ERROR("PadV3Grad kernel compute failed.");            \
      return result1 && result2;                                       \
    }                                                                  \
    break;                                                             \
  }

#define PAD_V3_GRAD_COMPUTE_CASE(DTYPE, TYPE, CTX)          \
  case (DTYPE): {                                           \
    uint32_t result = PadV3GradCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                       \
      KERNEL_LOG_ERROR("PadV3Grad kernel compute failed."); \
      return result;                                        \
    }                                                       \
    break;                                                  \
  }
}  // namespace

namespace aicpu {
uint32_t PadV3GradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(PadV3GradCheck(ctx), "PadV3Grad check params failed.");
  auto paddings_type = ctx.Input(1)->GetDataType();
  switch (paddings_type) {
    PAD_V3_GRAD_READ_PADDINGS(DT_INT32, int32_t, ctx)
    PAD_V3_GRAD_READ_PADDINGS(DT_INT64, int64_t, ctx)
    default:
      KERNEL_LOG_ERROR("PadV3Grad paddings data type [%s] not support.", DTypeStr(paddings_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  auto data_type = ctx.Output(0)->GetDataType();
  switch (data_type) {
    PAD_V3_GRAD_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    PAD_V3_GRAD_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    PAD_V3_GRAD_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    PAD_V3_GRAD_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    PAD_V3_GRAD_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    PAD_V3_GRAD_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    PAD_V3_GRAD_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    PAD_V3_GRAD_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    PAD_V3_GRAD_COMPUTE_CASE(DT_FLOAT16, float16, ctx)
    PAD_V3_GRAD_COMPUTE_CASE(DT_FLOAT, float, ctx)
    PAD_V3_GRAD_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    PAD_V3_GRAD_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    PAD_V3_GRAD_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("PadV3Grad kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t PadV3GradCpuKernel::PadV3GradCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "PadV3Grad check failed.");
  if (ctx.GetAttr("paddings_contiguous") == nullptr) {
    padding_contiguous = true;
    KERNEL_LOG_DEBUG("Get attr [paddings_contiguous] failed, use default value [true]");
  } else {
    padding_contiguous = ctx.GetAttr("paddings_contiguous")->GetBool();
  }
  if (ctx.GetAttr("mode") == nullptr) {
    mode = "reflect";
    KERNEL_LOG_DEBUG("Get attr [mode] failed, use default value [reflect]");
  } else {
    mode = ctx.GetAttr("mode")->GetString();
    const bool is_mode_available = std::find(mode_list.begin(), mode_list.end(), mode) != mode_list.end();
    if (is_mode_available == false) {
      KERNEL_LOG_ERROR("Attr [mode] must be included in [reflect, edge], but got [%s]", mode.c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR("Tensor y dtype[%s] must be same with x dtype[%s]", DTypeStr(ctx.Output(0)->GetDataType()).c_str(),
                     DTypeStr(ctx.Input(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  const std::vector<int64_t> paddings_shape = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE(
    paddings_shape.size() == 1 && (paddings_shape[0] == k3DNum + k4Num || paddings_shape[0] == k2DNum + k4Num ||
                                   paddings_shape[0] == k1DNum + k4Num || paddings_shape[0] == 1),
    KERNEL_STATUS_PARAM_INVALID, "Paddings shape is not supported");
  KERNEL_CHECK_FALSE(ctx.Input(0)->GetTensorShape()->GetDims() >= kInput1Dim, KERNEL_STATUS_PARAM_INVALID,
                     "Dims of tensor x should be greater than or equal to 3");
  KERNEL_CHECK_FALSE(ctx.Input(0)->GetTensorShape()->GetDims() <= kInput3Dim, KERNEL_STATUS_PARAM_INVALID,
                     "Only 3D, 4D, 5D padding with non-constant padding are "
                     "supported for now");

  const int64_t input_dim = ctx.Input(0)->GetTensorShape()->GetDims();
  const int64_t num_elem = ctx.Input(1)->NumElements();
  KERNEL_CHECK_FALSE(num_elem % k2Num == 0 || num_elem == 1, KERNEL_STATUS_PARAM_INVALID,
                     "Padding length must be divisible by 2");

  if (input_dim == kInput1Dim) {
    KERNEL_CHECK_FALSE(num_elem == k1DNum + k4Num || num_elem == 1, KERNEL_STATUS_PARAM_INVALID,
                       "3D tensors expect 6 values for padding");
  } else if (input_dim == kInput2Dim) {
    KERNEL_CHECK_FALSE(num_elem == k2DNum + k4Num || num_elem == 1, KERNEL_STATUS_PARAM_INVALID,
                       "4D tensors expect 8 values for padding");
  } else if (input_dim == kInput3Dim) {
    KERNEL_CHECK_FALSE(num_elem == k3DNum + k4Num || num_elem == 1, KERNEL_STATUS_PARAM_INVALID,
                       "5D tensors expect 10 values for padding");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PadV3GradCpuKernel::PadV3ReadPaddingsAndSetOutputShape1(CpuKernelContext &ctx) {
  num_elem = ctx.Input(1)->NumElements();
  input_dim = ctx.Input(0)->GetTensorShape()->GetDims();
  const std::vector<int64_t> input_shape = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  auto paddings_ptr = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  paddings = std::vector<int64_t>(input_dim * k2Num, 0);

  for (int64_t i = 0; i < num_elem; i += k2Num) {
    paddings[i] = static_cast<int64_t>(paddings_ptr[num_elem - i - k2Num]);
    paddings[i + 1] = static_cast<int64_t>(paddings_ptr[num_elem - i - 1]);
  }
  num_elem = num_elem - k4Num;
  if (num_elem == 1) {
    num_elem = k2Num * (input_dim - k2Num);
    for (int64_t i = 0; i < k2Num * (input_dim - k2Num); ++i) {
      paddings[i] = static_cast<int64_t>(paddings_ptr[0]);
    }
  }

  parallelSliceNum = 1;
  for (int64_t i = 0; i < input_dim - num_elem / k2Num; i++) {
    parallelSliceNum *= input_shape[i];
  }

  if (padding_contiguous == false && num_elem == k3DNum) {
    std::vector<int64_t> tmp = paddings;
    paddings[1] = tmp[k3Num];
    paddings[k2Num] = tmp[1];
    paddings[k3Num] = tmp[k4Num];
    paddings[k4Num] = tmp[k2Num];
  }

  if (padding_contiguous == false && num_elem == k2DNum) {
    std::vector<int64_t> tmp = paddings;
    paddings[1] = tmp[k2Num];
    paddings[k2Num] = tmp[1];
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PadV3GradCpuKernel::PadV3ReadPaddingsAndSetOutputShape2(CpuKernelContext &ctx) {
  std::vector<int64_t> output_shape = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  output_shape.end()[-kwidth] -= (paddings[kpad_l] + paddings[kpad_l + 1]);
  output_shape.end()[-kheight] -= (paddings[kpad_t] + paddings[kpad_t + 1]);
  output_shape.end()[-kchannel] -= (paddings[kpad_f] + paddings[kpad_f + 1]);

  KERNEL_CHECK_FALSE(
    output_shape.end()[-kwidth] > 0 && output_shape.end()[-kheight] > 0 && output_shape.end()[-kchannel] > 0,
    KERNEL_STATUS_PARAM_INVALID, "output_shape number must be greater than 0");

  if (output_shape != ctx.Output(0)->GetTensorShape()->GetDimSizes()) {
    ctx.Output(0)->GetTensorShape()->SetDimSizes(output_shape);
    KERNEL_LOG_DEBUG("Set output tensor shape success, num elements:[%llu]",
                     static_cast<uint64_t>(ctx.Output(0)->NumElements()));
  } else {
    KERNEL_LOG_DEBUG("Output tensor is a const tensor, num elements:[%llu]",
                     static_cast<uint64_t>(ctx.Output(0)->NumElements()));
  }
  const std::string padding_contiguous_str = padding_contiguous ? std::string("True") : std::string("False");
  KERNEL_LOG_DEBUG(
    "PadV3GradCpuKernel[%s], x: size[%llu] dtype[%s], "
    "paddings: size[%llu] dtype[%s], y: size[%llu] dtype[%s], mode: [%s], "
    "padding_contiguous: [%s].",
    ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(), DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
    ctx.Input(1)->GetDataSize(), DTypeStr(ctx.Input(1)->GetDataType()).c_str(), ctx.Output(0)->GetDataSize(),
    DTypeStr(ctx.Output(0)->GetDataType()).c_str(), mode.c_str(), padding_contiguous_str.c_str());
  return KERNEL_STATUS_OK;
}

int64_t PadV3GradCpuKernel::IndexCaculate(int64_t pad_value, int64_t now, int64_t output_value, int64_t o_start,
                                          int64_t i_start) {
  int64_t ip = 0;
  if (now < pad_value) {
    if (mode == "reflect") {
      ip = pad_value + pad_value - now;
    } else if (mode == "edge") {
      ip = pad_value;
    }
  } else if (now >= pad_value && now < output_value + pad_value) {
    ip = now;
  } else {
    if (mode == "reflect") {
      ip = (output_value + pad_value - 1) + (output_value + pad_value - 1) - now;
    } else if (mode == "edge") {
      ip = output_value + pad_value - 1;
    }
  }
  ip = ip - o_start + i_start;
  return ip;
}

template <typename T>
uint32_t PadV3GradCpuKernel::PadV3GradCompute1(T *input, T *output, int64_t p) {
  if (num_elem == k1DNum) {
    PadV3GradCompute1D<T>(input, output, p);
  } else if (num_elem == k2DNum) {
    for (int i = 0; i < input_h; i++) {
      PadV3GradCompute2D<T>(input, output, p, i);
    }
  } else if (num_elem == k3DNum) {
    for (int z = 0; z < input_c; z++) {
      PadV3GradCompute3D<T>(input, output, p, z);
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PadV3GradCpuKernel::PadV3GradCompute1D(T *input, T *output, int64_t p) {
  int ip_x;
  for (int j = 0; j < input_w; j++) {
    ip_x = IndexCaculate(pad_l, j, output_w, o_start_x, i_start_x);
    T *src_p = input + p * input_w + j;
    T *dest_p = output + p * output_w + ip_x;
    *dest_p += *src_p;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PadV3GradCpuKernel::PadV3GradCompute2D(T *input, T *output, int64_t p, int64_t i) {
  int ip_x, ip_y;
  for (int j = 0; j < input_w; j++) {
    ip_x = IndexCaculate(pad_l, j, output_w, o_start_x, i_start_x);
    ip_y = IndexCaculate(pad_t, i, output_h, o_start_y, i_start_y);
    T *src_p = input + p * input_w * input_h + i * input_w + j;
    T *dest_p = output + p * output_w * output_h + ip_y * output_w + ip_x;
    *dest_p += *src_p;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PadV3GradCpuKernel::PadV3GradCompute3D(T *input, T *output, int64_t p, int64_t z) {
  int ip_x, ip_y, ip_z;
  for (int i = 0; i < input_h; i++) {
    for (int j = 0; j < input_w; j++) {
      ip_x = IndexCaculate(pad_l, j, output_w, o_start_x, i_start_x);
      ip_y = IndexCaculate(pad_t, i, output_h, o_start_y, i_start_y);
      ip_z = IndexCaculate(pad_f, z, output_c, o_start_z, i_start_z);
      T *src_p = input + p * input_w * input_h * input_c + z * input_w * input_h + i * input_w + j;
      T *dest_p = output + p * output_w * output_h * output_c + ip_z * output_w * output_h + ip_y * output_w + ip_x;
      *dest_p += *src_p;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PadV3GradCpuKernel::PadV3GradCompute(CpuKernelContext &ctx) {
  const std::vector<int64_t> input_shape = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> output_shape = ctx.Output(0)->GetTensorShape()->GetDimSizes();

  T *input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  T *output = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  output_w = output_shape.end()[-kwidth];
  output_h = output_shape.end()[-kheight];
  output_c = output_shape.end()[-kchannel];
  input_w = input_shape.end()[-kwidth];
  input_h = input_shape.end()[-kheight];
  input_c = input_shape.end()[-kchannel];
  i_start_x = std::max(int64_t(0), -paddings[kpad_l]);
  i_start_y = std::max(int64_t(0), -paddings[kpad_t]);
  i_start_z = std::max(int64_t(0), -paddings[kpad_f]);
  o_start_x = std::max(int64_t(0), paddings[kpad_l]);
  o_start_y = std::max(int64_t(0), paddings[kpad_t]);
  o_start_z = std::max(int64_t(0), paddings[kpad_f]);
  pad_l = paddings[kpad_l];
  pad_t = paddings[kpad_t];
  pad_f = paddings[kpad_f];

  int64_t output_num_ = 1;
  for (int64_t i = 0; i < input_dim; i++) {
    output_num_ *= output_shape[i];
  }
  auto ret = memset_s(output, sizeof(T) * output_num_, 0, sizeof(T) * output_num_);
  if (ret != EOK) {
    KERNEL_LOG_ERROR("memset_s error, ret=%d", ret);
    return KERNEL_STATUS_INNER_ERROR;
  }
  auto shard_padv3_grad = [&](int64_t start, int64_t end) {
    for (int p = start; p < end; p++) {
      PadV3GradCompute1<T>(input, output, p);
    }
  };
  const int64_t data_num = parallelSliceNum;
  const bool enable_parallel = parallelSliceNum > kParallelNum;
  if (enable_parallel) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_padv3_grad),
                        "PadV3Grad Compute failed.");
  } else {
    for (int p = 0; p < data_num; p++) {
      PadV3GradCompute1<T>(input, output, p);
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kPadV3Grad, PadV3GradCpuKernel);
}  // namespace aicpu
