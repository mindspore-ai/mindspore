/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "cpu_kernel/ms_kernel/adaptive_max_pool_3d.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel/inc/cpu_context.h"

namespace {
constexpr char const *kAdaptiveMaxPool3d = "AdaptiveMaxPool3d";
constexpr uint32_t input_num = 2;
constexpr uint32_t output_num = 2;
constexpr int64_t kParallelDataNums = 512;

#define ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(DTYPE, TYPE, CTX)         \
  case (DTYPE): {                                                   \
    uint32_t result = AdaptiveMaxPool3dCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                               \
      KERNEL_LOG_ERROR("AdaptiveMaxPool3d kernel compute failed."); \
      return result;                                                \
    }                                                               \
    break;                                                          \
  }
}  // namespace

namespace aicpu {
uint32_t AdaptiveMaxPool3dCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(AdaptiveMaxPool3dCheckAndSetShape(ctx), "AdaptiveMaxPool3d check params failed.");
  auto compute_dtype = ctx.Input(0)->GetDataType();
  switch (compute_dtype) {
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(DT_FLOAT, float, ctx)
    ADAPTIVE_MAX_POOL_3D_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("AdaptiveMaxPool3d kernel data type [%s] not support.", DTypeStr(compute_dtype).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t AdaptiveMaxPool3dCpuKernel::AdaptiveMaxPool3dCheckAndSetShape(const CpuKernelContext &ctx) {
  auto x = ctx.Input(0);
  auto output_size = ctx.Input(1);
  auto y = ctx.Output(0);
  auto argmax = ctx.Output(1);
  KERNEL_HANDLE_ERROR(NormalCheck(const_cast<CpuKernelContext &>(ctx), input_num, output_num),
                      "AdaptiveMaxPool3d check params failed.");
  const std::vector<int64_t> input_shape = x->GetTensorShape()->GetDimSizes();
  const size_t input_num_dims = input_shape.size();
  KERNEL_CHECK_FALSE(input_num_dims == 4 || input_num_dims == 5, KERNEL_STATUS_PARAM_INVALID,
                     "Input dimensions must be equal to 4 or 5.");
  KERNEL_CHECK_FALSE(output_size->GetTensorShape()->GetDimSize(0) == 3, KERNEL_STATUS_PARAM_INVALID,
                     "Output size dimensions must be equal to 3.");

  std::vector<int64_t> output_shape = {input_shape[0]};
  if (input_num_dims == 5) {
    output_shape.push_back(input_shape[1]);
  }
  auto output_size_ptr = reinterpret_cast<int32_t *>(output_size->GetData());
  for (size_t i = 0; i < 3; ++i) {
    const int64_t elem = output_size_ptr[i];
    KERNEL_CHECK_FALSE(elem > 0, KERNEL_STATUS_PARAM_INVALID, "Elements of output_size must be greater than 0");
    output_shape.push_back(elem);
  }
  y->GetTensorShape()->SetDimSizes(output_shape);
  const int64_t y_data_size = y->CalcDataSizeByShape();
  y->SetDataSize(y_data_size);
  argmax->GetTensorShape()->SetDimSizes(output_shape);
  const int64_t argmax_data_size = argmax->CalcDataSizeByShape();
  argmax->SetDataSize(argmax_data_size);

  KERNEL_LOG_DEBUG(
    "AdaptiveMaxPool3dCpuKernel[%s], x: size[%llu] dtype[%s]; "
    "output_size: size[%llu] dtype[%s], y: size[%llu] dtype[%s]; "
    "argmax: size[%llu] dtype[%s].",
    ctx.GetOpType().c_str(), x->GetDataSize(), DTypeStr(x->GetDataType()).c_str(), output_size->GetDataSize(),
    DTypeStr(output_size->GetDataType()).c_str(), y->GetDataSize(), DTypeStr(y->GetDataType()).c_str(),
    argmax->GetDataSize(), DTypeStr(argmax->GetDataType()).c_str());

  return KERNEL_STATUS_OK;
}

int64_t AdaptiveMaxPool3dCpuKernel::ComputeStride(const std::vector<int64_t> &shape, size_t index) {
  KERNEL_CHECK_FALSE(index < shape.size(), KERNEL_STATUS_PARAM_INVALID, "Input index must be less than shape dims.");
  int64_t result = 1;
  for (size_t i = index + 1; i < shape.size(); ++i) {
    result *= shape[i];
  }
  return result;
}

template <typename T>
uint32_t AdaptiveMaxPool3dCpuKernel::AdaptiveMaxPool3dCompute(const CpuKernelContext &ctx) {
  auto input_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto indices_data = reinterpret_cast<int32_t *>(ctx.Output(1)->GetData());

  std::vector<int64_t> input_shape = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> output_shape = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  if (input_shape.size() == 4) {
    input_shape.insert(input_shape.begin(), 1);
    output_shape.insert(output_shape.begin(), 1);
  }

  constexpr size_t dimB = 0;
  constexpr size_t dimD = 1;
  constexpr size_t dimT = 2;
  constexpr size_t dimH = 3;
  constexpr size_t dimW = 4;
  const int64_t sizeB = input_shape[dimB];
  const int64_t sizeD = input_shape[dimD];
  const int64_t isizeT = input_shape[dimT];
  const int64_t isizeH = input_shape[dimH];
  const int64_t isizeW = input_shape[dimW];
  const int64_t istrideB = ComputeStride(input_shape, dimB);
  const int64_t istrideD = ComputeStride(input_shape, dimD);
  const int64_t istrideT = ComputeStride(input_shape, dimT);
  const int64_t istrideH = ComputeStride(input_shape, dimH);
  const int64_t istrideW = ComputeStride(input_shape, dimW);
  const int64_t osizeT = output_shape.cend()[-3];
  const int64_t osizeH = output_shape.cend()[-2];
  const int64_t osizeW = output_shape.cend()[-1];

  auto start_index = [=](int64_t dim, int64_t output_range, int64_t input_range) {
    return static_cast<int64_t>(std::floor(static_cast<double>(dim * input_range) / output_range));
  };

  auto end_index = [=](int64_t dim, int64_t output_range, int64_t input_range) {
    return static_cast<int64_t>(std::ceil(static_cast<double>((dim + 1) * input_range) / output_range));
  };

  auto get_per_unit_size = [&](int64_t data_size) -> int64_t {
    KERNEL_CHECK_FALSE(data_size != 0, KERNEL_STATUS_PARAM_INVALID, "data_size can not be 0.");
    const int64_t max_core_num =
      std::max(static_cast<int64_t>(1), static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2));
    return data_size / std::min(max_core_num, data_size);
  };

  int64_t max_index = 0;
  const std::vector<int64_t> data_nums = {sizeB, sizeD, osizeT, osizeH, osizeW};
  for (int64_t i = 0; i < 5; ++i) {
    if (data_nums[i] > data_nums[max_index]) {
      max_index = i;
    }
  }

  auto ComputeKernel = [&](int64_t startB, int64_t endB, int64_t started, int64_t endD, int64_t startT, int64_t endT,
                           int64_t startH, int64_t endH, int64_t startW, int64_t endW) {
    for (int64_t b = startB; b < endB; ++b) {
      auto input_p = input_data + b * istrideB;
      auto output_p = output_data + b * sizeD * osizeT * osizeH * osizeW;
      auto ind_p = indices_data + b * sizeD * osizeT * osizeH * osizeW;
      for (int64_t d = started; d < endD; ++d) {
        // loop over output
        int64_t ot;
        int64_t oh;
        int64_t ow;
        for (ot = startT; ot < endT; ++ot) {
          int64_t istartT = start_index(ot, osizeT, isizeT);
          int64_t iendT = end_index(ot, osizeT, isizeT);
          int64_t kT = iendT - istartT;

          for (oh = startH; oh < endH; ++oh) {
            int64_t istartH = start_index(oh, osizeH, isizeH);
            int64_t iendH = end_index(oh, osizeH, isizeH);
            int64_t kH = iendH - istartH;

            for (ow = startW; ow < endW; ++ow) {
              int64_t istartW = start_index(ow, osizeW, isizeW);
              int64_t iendW = end_index(ow, osizeW, isizeW);
              int64_t kW = iendW - istartW;

              // local pointers
              auto ip = input_p + d * istrideD + istartT * istrideT + istartH * istrideH + istartW * istrideW;
              auto op = output_p + d * osizeT * osizeH * osizeW + ot * osizeH * osizeW + oh * osizeW + ow;
              auto indp = ind_p + d * osizeT * osizeH * osizeW + ot * osizeH * osizeW + oh * osizeW + ow;

              // compute local max:
              int64_t it = 0;
              int64_t ih = 0;
              int64_t iw = 0;
              int64_t maxindex = (it + istartT) * isizeH * isizeW + (ih + istartH) * isizeW + (iw + istartW);
              T maxval = *ip;
              for (it = 0; it < kT; ++it) {
                for (ih = 0; ih < kH; ++ih) {
                  for (iw = 0; iw < kW; ++iw) {
                    T val = *(ip + it * istrideT + ih * istrideH + iw * istrideW);
                    if ((val > maxval) || std::isnan(static_cast<double>(val))) {
                      maxval = val;
                      maxindex = (it + istartT) * isizeH * isizeW + (ih + istartH) * isizeW + (iw + istartW);
                    }
                  }
                }
              }
              // set output to local max
              *op = maxval;
              // store location of max
              *indp = static_cast<int32_t>(maxindex);
            }
          }
        }
      }
    }
  };

  const bool enable_parallel = (sizeB * sizeD * osizeT * osizeH * osizeW) > kParallelDataNums;
  if (enable_parallel == false) {
    ComputeKernel(0, sizeB, 0, sizeD, 0, osizeT, 0, osizeH, 0, osizeW);
  } else {
    switch (max_index) {
      case 0: {
        auto shard_adaptive_max_pool_3d = [&](int64_t start, int64_t end) {
          ComputeKernel(start, end, 0, sizeD, 0, osizeT, 0, osizeH, 0, osizeW);
        };
        KERNEL_HANDLE_ERROR(
          CpuKernelUtils::ParallelFor(ctx, sizeB, get_per_unit_size(sizeB), shard_adaptive_max_pool_3d),
          "AdaptiveMaxPool3d kernel compute failed.");
      } break;

      case 1: {
        auto shard_adaptive_max_pool_3d = [&](int64_t start, int64_t end) {
          ComputeKernel(0, sizeB, start, end, 0, osizeT, 0, osizeH, 0, osizeW);
        };
        KERNEL_HANDLE_ERROR(
          CpuKernelUtils::ParallelFor(ctx, sizeD, get_per_unit_size(sizeD), shard_adaptive_max_pool_3d),
          "AdaptiveMaxPool3d kernel compute failed.");
      } break;

      case 2: {
        auto shard_adaptive_max_pool_3d = [&](int64_t start, int64_t end) {
          ComputeKernel(0, sizeB, 0, sizeD, start, end, 0, osizeH, 0, osizeW);
        };
        KERNEL_HANDLE_ERROR(
          CpuKernelUtils::ParallelFor(ctx, osizeT, get_per_unit_size(osizeT), shard_adaptive_max_pool_3d),
          "AdaptiveMaxPool3d kernel compute failed.");
      } break;

      case 3: {
        auto shard_adaptive_max_pool_3d = [&](int64_t start, int64_t end) {
          ComputeKernel(0, sizeB, 0, sizeD, 0, osizeT, start, end, 0, osizeW);
        };
        KERNEL_HANDLE_ERROR(
          CpuKernelUtils::ParallelFor(ctx, osizeH, get_per_unit_size(osizeH), shard_adaptive_max_pool_3d),
          "AdaptiveMaxPool3d kernel compute failed.");
      } break;

      case 4: {
        auto shard_adaptive_max_pool_3d = [&](int64_t start, int64_t end) {
          ComputeKernel(0, sizeB, 0, sizeD, 0, osizeT, 0, osizeH, start, end);
        };
        KERNEL_HANDLE_ERROR(
          CpuKernelUtils::ParallelFor(ctx, osizeW, get_per_unit_size(osizeW), shard_adaptive_max_pool_3d),
          "AdaptiveMaxPool3d kernel compute failed.");
      } break;
    }
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kAdaptiveMaxPool3d, AdaptiveMaxPool3dCpuKernel);
}  // namespace aicpu
