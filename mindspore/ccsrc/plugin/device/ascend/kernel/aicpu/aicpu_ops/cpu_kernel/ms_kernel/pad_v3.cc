/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "pad_v3.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kPadV3 = "PadV3";
constexpr int64_t kMinCoreNum = 1;
constexpr int64_t kParallelNum = 1024 * 16;
constexpr int64_t kInput3D = 3;
constexpr int64_t kInput4D = 4;
constexpr int64_t kInput5D = 5;
constexpr int64_t kPadding1D = 2;
constexpr int64_t kPadding2D = 4;
constexpr int64_t kPadding3D = 6;
constexpr int64_t kNum2 = 2;
constexpr int64_t kNum3 = 3;
constexpr int64_t kNum4 = 4;

const std::vector<std::string> mode_list = {"constant", "reflect", "edge"};
using float16 = Eigen::half;

#define PAD_V3_COMPUTE_CASE(DTYPE, TYPE, CTX)           \
  case (DTYPE): {                                       \
    uint32_t result = DoCompute<TYPE>(CTX);             \
    if (result != KERNEL_STATUS_OK) {                   \
      KERNEL_LOG_ERROR("PadV3 kernel compute failed."); \
      return result;                                    \
    }                                                   \
    break;                                              \
  }
}  // namespace

namespace aicpu {
uint32_t PadV3CpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input x failed")
  KERNEL_CHECK_NULLPTR(ctx.Input(1)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input paddings failed")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output y failed")
  KERNEL_HANDLE_ERROR(CheckAndInitParams(ctx), "PadV3 check and init params failed.");
  auto paddings_type = ctx.Input(1)->GetDataType();
  if (paddings_type == DT_INT32) {
    KERNEL_CHECK_FALSE((GetPaddingsAndSetOuputShape<int32_t>(ctx) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                       "Get paddings and set output shape failed.");
  } else if (paddings_type == DT_INT64) {
    KERNEL_CHECK_FALSE((GetPaddingsAndSetOuputShape<int64_t>(ctx) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                       "Get paddings and set output shape failed.");
  } else {
    KERNEL_LOG_ERROR("PadV3 paddings data type [%s] not support.", DTypeStr(paddings_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto data_type_ = ctx.Input(0)->GetDataType();
  switch (data_type_) {
    PAD_V3_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    PAD_V3_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    PAD_V3_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    PAD_V3_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    PAD_V3_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    PAD_V3_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    PAD_V3_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    PAD_V3_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    PAD_V3_COMPUTE_CASE(DT_FLOAT16, float16, ctx)
    PAD_V3_COMPUTE_CASE(DT_FLOAT, float, ctx)
    PAD_V3_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    PAD_V3_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    PAD_V3_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("PadV3 kernel data type [%s] not support.", DTypeStr(data_type_).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

int64_t PadV3CpuKernel::EdgeIndexCaculate(int64_t pad_value, int64_t now, int64_t input_value, int64_t o_start,
                                          int64_t i_start) {
  int64_t ip;
  if (now < pad_value) {
    ip = pad_value;
  } else if (now >= pad_value && now < input_value + pad_value) {
    ip = now;
  } else {
    ip = input_value + pad_value - 1;
  }
  ip = ip - o_start + i_start;
  return ip;
}

template <typename T>
uint32_t PadV3CpuKernel::EdgeCompute1D(T *input, T *output, int64_t p) {
  int64_t nplane = 0;
  int64_t input_w = input_shape[kNum2];
  int64_t output_w = output_shape.end()[-1];
  int64_t pad_l = paddings[0];
  int64_t i_start_x = std::max(int64_t(0), -pad_l);
  int64_t o_start_x = std::max(int64_t(0), pad_l);
  int64_t ip_x;
  for (int64_t j = 0; j < output_w; ++j) {
    ip_x = EdgeIndexCaculate(pad_l, j, input_w, o_start_x, i_start_x);
    T *dest_p = output + p * output_w * (nplane + 1) + j;
    T *src_p = input + +p * input_w * (nplane + 1) + ip_x;
    *dest_p = *src_p;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PadV3CpuKernel::EdgeCompute2D(T *input, T *output, int64_t p) {
  int64_t pad_l = paddings[0];
  int64_t pad_t = paddings[kNum2];
  int64_t nplane = 0;
  int64_t input_h = input_shape[kNum2];
  int64_t input_w = input_shape[kNum3];
  int64_t output_h = input_h + pad_t + paddings[kNum3];
  int64_t output_w = input_w + pad_l + paddings[1];
  int64_t i_start_x = std::max(int64_t(0), -pad_l);
  int64_t i_start_y = std::max(int64_t(0), -pad_t);
  int64_t o_start_x = std::max(int64_t(0), pad_l);
  int64_t o_start_y = std::max(int64_t(0), pad_t);
  int64_t ip_x, ip_y;
  for (int64_t i = 0; i < output_h; ++i) {
    for (int64_t j = 0; j < output_w; ++j) {
      ip_x = EdgeIndexCaculate(pad_l, j, input_w, o_start_x, i_start_x);
      ip_y = EdgeIndexCaculate(pad_t, i, input_h, o_start_y, i_start_y);
      T *dest_p = output + p * output_w * output_h * (nplane + 1) + i * output_w + j;
      T *src_p = input + p * input_w * input_h * (nplane + 1) + ip_y * input_w + ip_x;
      *dest_p = *src_p;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PadV3CpuKernel::EdgeCompute3D(T *input, T *output, int64_t p) {
  int64_t pad_l = paddings[0];
  int64_t pad_t = paddings[kNum2];
  int64_t pad_f = paddings[kNum4];
  int64_t nplane = 0;
  int64_t input_d = input_shape[kNum2];
  int64_t input_h = input_shape[kNum3];
  int64_t input_w = input_shape[kNum4];
  int64_t output_d = output_shape[kNum2];
  int64_t output_h = output_shape[kNum3];
  int64_t output_w = output_shape[kNum4];
  int64_t i_start_x = std::max(int64_t(0), -pad_l);
  int64_t i_start_y = std::max(int64_t(0), -pad_t);
  int64_t i_start_z = std::max(int64_t(0), -pad_f);
  int64_t o_start_x = std::max(int64_t(0), pad_l);
  int64_t o_start_y = std::max(int64_t(0), pad_t);
  int64_t o_start_z = std::max(int64_t(0), pad_f);
  int64_t ip_x, ip_y, ip_z;
  for (int64_t k = 0; k < output_d; ++k) {
    for (int64_t j = 0; j < output_h; ++j) {
      for (int64_t i = 0; i < output_w; ++i) {
        ip_x = EdgeIndexCaculate(pad_l, i, input_w, o_start_x, i_start_x);
        ip_y = EdgeIndexCaculate(pad_t, j, input_h, o_start_y, i_start_y);
        ip_z = EdgeIndexCaculate(pad_f, k, input_d, o_start_z, i_start_z);
        T *dest_p =
          output + p * output_w * output_h * output_d * (nplane + 1) + k * output_w * output_h + j * output_w + i;
        T *src_p =
          input + p * input_w * input_h * input_d * (nplane + 1) + ip_z * input_w * input_h + ip_y * input_w + ip_x;
        *dest_p = *src_p;
      }
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PadV3CpuKernel::EdgeModeCompute(CpuKernelContext &ctx, int64_t p) {
  auto input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  if (paddings_num == kPadding1D) {
    EdgeCompute1D<T>(input, output, p);
  } else if (paddings_num == kPadding2D) {
    EdgeCompute2D<T>(input, output, p);
  } else if (paddings_num == kPadding3D) {
    EdgeCompute3D<T>(input, output, p);
  }
  return KERNEL_STATUS_OK;
}

int64_t PadV3CpuKernel::ReflectIndexCaculate(int64_t pad_value, int64_t now, int64_t input_value, int64_t o_start,
                                             int64_t i_start) {
  int64_t ip;
  if (now < pad_value) {
    ip = pad_value + pad_value - now;
  } else if (now >= pad_value && now < input_value + pad_value) {
    ip = now;
  } else {
    ip = (input_value + pad_value - 1) + (input_value + pad_value - 1) - now;
  }
  ip = ip - o_start + i_start;
  return ip;
}

template <typename T>
uint32_t PadV3CpuKernel::ReflectCompute1D(T *input, T *output, int64_t p) {
  int64_t nplane = 0;
  int64_t input_w = input_shape[kNum2];
  int64_t output_w = output_shape.end()[-1];
  int64_t pad_l = paddings[0];
  int64_t i_start_x = std::max(int64_t(0), -pad_l);
  int64_t o_start_x = std::max(int64_t(0), pad_l);
  int64_t ip_x;
  for (int64_t j = 0; j < output_w; ++j) {
    ip_x = ReflectIndexCaculate(pad_l, j, input_w, o_start_x, i_start_x);
    T *dest_p = output + p * output_w * (nplane + 1) + j;
    T *src_p = input + +p * input_w * (nplane + 1) + ip_x;
    *dest_p = *src_p;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PadV3CpuKernel::ReflectCompute2D(T *input, T *output, int64_t p) {
  int64_t pad_l = paddings[0];
  int64_t pad_t = paddings[kNum2];
  int64_t nplane = 0;
  int64_t input_h = input_shape[kNum2];
  int64_t input_w = input_shape[kNum3];
  int64_t output_h = input_h + pad_t + paddings[kNum3];
  int64_t output_w = input_w + pad_l + paddings[1];
  int64_t i_start_x = std::max(int64_t(0), -pad_l);
  int64_t i_start_y = std::max(int64_t(0), -pad_t);
  int64_t o_start_x = std::max(int64_t(0), pad_l);
  int64_t o_start_y = std::max(int64_t(0), pad_t);
  int64_t ip_x, ip_y;
  for (int64_t i = 0; i < output_h; ++i) {
    for (int64_t j = 0; j < output_w; ++j) {
      ip_x = ReflectIndexCaculate(pad_l, j, input_w, o_start_x, i_start_x);
      ip_y = ReflectIndexCaculate(pad_t, i, input_h, o_start_y, i_start_y);
      T *dest_p = output + p * output_w * output_h * (nplane + 1) + i * output_w + j;
      T *src_p = input + p * input_w * input_h * (nplane + 1) + ip_y * input_w + ip_x;
      *dest_p = *src_p;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PadV3CpuKernel::ReflectCompute3D(T *input, T *output, int64_t p) {
  int64_t pad_l = paddings[0];
  int64_t pad_t = paddings[kNum2];
  int64_t pad_f = paddings[kNum4];
  int64_t nplane = 0;
  int64_t input_d = input_shape[kNum2];
  int64_t input_h = input_shape[kNum3];
  int64_t input_w = input_shape[kNum4];
  int64_t output_d = output_shape[kNum2];
  int64_t output_h = output_shape[kNum3];
  int64_t output_w = output_shape[kNum4];
  int64_t i_start_x = std::max(int64_t(0), -pad_l);
  int64_t i_start_y = std::max(int64_t(0), -pad_t);
  int64_t i_start_z = std::max(int64_t(0), -pad_f);
  int64_t o_start_x = std::max(int64_t(0), pad_l);
  int64_t o_start_y = std::max(int64_t(0), pad_t);
  int64_t o_start_z = std::max(int64_t(0), pad_f);
  int64_t ip_x, ip_y, ip_z;
  for (int64_t k = 0; k < output_d; ++k) {
    for (int64_t j = 0; j < output_h; ++j) {
      for (int64_t i = 0; i < output_w; ++i) {
        ip_x = ReflectIndexCaculate(pad_l, i, input_w, o_start_x, i_start_x);
        ip_y = ReflectIndexCaculate(pad_t, j, input_h, o_start_y, i_start_y);
        ip_z = ReflectIndexCaculate(pad_f, k, input_d, o_start_z, i_start_z);
        T *dest_p =
          output + p * output_w * output_h * output_d * (nplane + 1) + k * output_w * output_h + j * output_w + i;
        T *src_p =
          input + p * input_w * input_h * input_d * (nplane + 1) + ip_z * input_w * input_h + ip_y * input_w + ip_x;
        *dest_p = *src_p;
      }
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PadV3CpuKernel::ReflectModeCompute(CpuKernelContext &ctx, int64_t p) {
  auto input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  if (paddings_num == kPadding1D) {
    ReflectCompute1D<T>(input, output, p);
  } else if (paddings_num == kPadding2D) {
    ReflectCompute2D<T>(input, output, p);
  } else if (paddings_num == kPadding3D) {
    ReflectCompute3D<T>(input, output, p);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PadV3CpuKernel::ConstantModeCompute(CpuKernelContext &ctx, T constant_values) {
  auto input_ptr = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_ptr = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t output_num = ctx.Output(0)->NumElements();
  int64_t input_num = 1;
  std::vector<int64_t> input_strides(input_dims, 0);
  std::vector<int64_t> output_strides(input_dims, 0);
  input_strides[input_dims - 1] = 1;
  output_strides[input_dims - 1] = 1;
  for (int64_t i = input_dims - 1; i >= 1; --i) {
    input_strides[i - 1] = input_strides[i] * input_shape[i];
    output_strides[i - 1] = output_strides[i] * output_shape[i];
  }
  std::vector<int64_t> offsets(input_dims, 0);
  std::vector<int64_t> extents(input_dims, 0);
  for (int64_t i = input_dims - 1; i >= 0; --i) {
    extents[i] = input_shape[i];
    if (paddings[i * kNum2] < 0) {
      extents[i] += paddings[i * kNum2];
      offsets[i] = -paddings[i * kNum2];
      paddings[i * kNum2] = 0;
    }
    if (paddings[i * kNum2 + 1] < 0) {
      extents[i] += paddings[i * kNum2 + 1];
      paddings[i * kNum2 + 1] = 0;
    }
    input_shape[i] = extents[i];
    input_num *= input_shape[i];
  }
  std::vector<T> input_values;
  for (int64_t i = 0; i < input_num; ++i) {
    int64_t k = i;
    int64_t p = 0;
    for (int64_t j = input_dims - 1; j >= 0; --j) {
      p += (offsets[j] + (k % extents[j])) * input_strides[j];
      k /= extents[j];
    }
    input_values.push_back(*(input_ptr + p));
  }
  for (int64_t i = 0; i < output_num; ++i) {
    *(output_ptr + i) = constant_values;
  }
  if (input_dims == 1) {
    for (int64_t i = 0; i < input_num; ++i) {
      *(output_ptr + paddings[0] + i) = input_values[i];
    }
    return KERNEL_STATUS_OK;
  }
  std::vector<int64_t> i_inx_add(input_dims, 0);
  std::vector<int64_t> o_inx_add(input_dims, 0);
  i_inx_add[input_dims - 1] = output_strides[input_dims - 1] * paddings[kNum2 * (input_dims - 1)];
  o_inx_add[input_dims - 1] = output_strides[input_dims - 1] * paddings[kNum2 * (input_dims - 1) + 1];
  for (int64_t i = input_dims - 1; i >= 1; --i) {
    i_inx_add[i - 1] = i_inx_add[i] + output_strides[i - 1] * paddings[kNum2 * (i - 1)];
    o_inx_add[i - 1] = o_inx_add[i] + output_strides[i - 1] * paddings[kNum2 * (i - 1) + 1];
  }
  int64_t i_inx = 0;
  int64_t o_inx = i_inx_add[0];
  std::vector<int64_t> pos(input_dims - 1, 0);
  while (i_inx < input_num) {
    for (int64_t i = 0; i < input_shape[input_dims - 1]; ++i) {
      *(output_ptr + o_inx + i) = input_values[i_inx + i];
    }
    pos[input_dims - kNum2] += 1;
    int64_t dep = input_dims - 1;
    for (int64_t i = input_dims - kNum2; i >= 0; --i) {
      if (i > 0 && pos[i] >= input_shape[i]) {
        pos[i] -= input_shape[i];
        pos[i - 1] += 1;
        dep = i;
      } else {
        break;
      }
    }
    o_inx += i_inx_add[dep] + o_inx_add[dep] + input_shape[input_dims - 1];
    i_inx += input_shape[input_dims - 1];
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PadV3CpuKernel::DoCompute(CpuKernelContext &ctx) {
  if (mode == "constant") {
    T constant_values = static_cast<T>(0);
    if (ctx.Input(kNum2) != nullptr) {
      constant_values = *(reinterpret_cast<T *>(ctx.Input(kNum2)->GetData()));
    } else {
      KERNEL_LOG_DEBUG("Get attr [constant_values] failed, use default value [0]");
    }
    for (int64_t i = 0; i < input_dims / kNum2; ++i) {
      int64_t u = paddings[i * kNum2];
      int64_t v = paddings[i * kNum2 + 1];
      paddings[i * kNum2] = paddings[kNum2 * (input_dims - i - 1)];
      paddings[i * kNum2 + 1] = paddings[kNum2 * (input_dims - i - 1) + 1];
      paddings[kNum2 * (input_dims - i - 1)] = u;
      paddings[kNum2 * (input_dims - i - 1) + 1] = v;
    }
    ConstantModeCompute<T>(ctx, constant_values);
  } else if (mode == "reflect") {
    auto shard_padv3_reflcet = [&](int64_t start, int64_t end) {
      for (int p = start; p < end; p++) {
        ReflectModeCompute<T>(ctx, p);
      }
    };
    const int64_t data_num = parallelSliceNum;
    const bool enable_parallel = data_num > kParallelNum;
    if (enable_parallel) {
      const int64_t max_core_num =
        std::max(static_cast<int64_t>(kMinCoreNum), static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx)));
      const int64_t per_unit_size = data_num / std::min(data_num, max_core_num);
      KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, per_unit_size, shard_padv3_reflcet),
                          "PadV3 Compute failed.");
    } else {
      shard_padv3_reflcet(0, data_num);
    }
  } else if (mode == "edge") {
    auto shard_padv3_edge = [&](int64_t start, int64_t end) {
      for (int p = start; p < end; p++) {
        EdgeModeCompute<T>(ctx, p);
      }
    };
    const int64_t data_num = parallelSliceNum;
    const bool enable_parallel = data_num > kParallelNum;
    if (enable_parallel) {
      const int64_t max_core_num =
        std::max(static_cast<int64_t>(kMinCoreNum), static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx)));
      const int64_t per_unit_size = data_num / std::min(data_num, max_core_num);
      KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, per_unit_size, shard_padv3_edge),
                          "PadV3 Compute failed.");
    } else {
      shard_padv3_edge(0, data_num);
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t PadV3CpuKernel::CheckAndInitParams(CpuKernelContext &ctx) {
  if (ctx.GetAttr("mode") == nullptr) {
    mode = "constant";
    KERNEL_LOG_DEBUG("Get attr [mode] failed, use default value [constant]");
  } else {
    mode = ctx.GetAttr("mode")->GetString();
    const bool is_mode_available = std::find(mode_list.begin(), mode_list.end(), mode) != mode_list.end();
    if (is_mode_available == false) {
      KERNEL_LOG_ERROR(
        "Attr [mode] must be included in [constant, reflect, edge], but got "
        "[%s]",
        mode.c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  if (ctx.GetAttr("paddings_contiguous") != nullptr) {
    paddings_contiguous = ctx.GetAttr("paddings_contiguous")->GetBool();
  } else {
    paddings_contiguous = true;
    KERNEL_LOG_DEBUG("Get attr [paddings_contiguous] failed, use default value [true]");
  }
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR("Tensor y dtype[%s] must be same with x dtype[%s]", DTypeStr(ctx.Output(0)->GetDataType()).c_str(),
                     DTypeStr(ctx.Input(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  input_dims = ctx.Input(0)->GetTensorShape()->GetDims();
  const std::vector<int64_t> paddings_shape = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  paddings_num = ctx.Input(1)->NumElements();
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PadV3CpuKernel::GetPaddingsAndSetOuputShape(CpuKernelContext &ctx) {
  auto paddings_ptr = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  paddings = std::vector<int64_t>(input_dims * kNum2, 0);
  for (int64_t i = 0; i < paddings_num; ++i) {
    paddings[i] = static_cast<int64_t>(paddings_ptr[i]);
  }
  if (paddings_contiguous == false) {
    std::vector<int64_t> tmp = paddings;
    for (int64_t i = 0; i < paddings_num; ++i) {
      if (i % kNum2 == 0) {
        paddings[i] = tmp[i / kNum2];
      } else {
        paddings[i] = tmp[(i + paddings_num) / kNum2];
      }
    }
  }
  input_shape = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  output_shape = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  parallelSliceNum = 1;
  for (int64_t i = 0; i < input_dims - paddings_num / kNum2; ++i) {
    parallelSliceNum *= input_shape[i];
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kPadV3, PadV3CpuKernel);
}  // namespace aicpu
