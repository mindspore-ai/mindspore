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

#include <stdint.h>
#include <algorithm>
#include <tuple>
#include <utility>

#include "tile.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "Eigen/Core"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kTile = "Tile";

#define TILE_COMPUTE_CASE(DTYPE, TYPE1, TYPE2, CTX)    \
  case (DTYPE): {                                      \
    uint32_t result = TileCompute<TYPE1, TYPE2>(CTX);  \
    if (result != KERNEL_STATUS_OK) {                  \
      KERNEL_LOG_ERROR("Tile kernel compute failed."); \
      return result;                                   \
    }                                                  \
    break;                                             \
  }

#define TILE_COMPUTE_CASE_ALL(TYPE, CTX)                            \
  TILE_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, TYPE, CTX)   \
  TILE_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, TYPE, CTX) \
  TILE_COMPUTE_CASE(DT_DOUBLE, double, TYPE, CTX)                   \
  TILE_COMPUTE_CASE(DT_FLOAT, float, TYPE, CTX)                     \
  TILE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, TYPE, CTX)             \
  TILE_COMPUTE_CASE(DT_INT8, int8_t, TYPE, CTX)                     \
  TILE_COMPUTE_CASE(DT_INT16, int16_t, TYPE, CTX)                   \
  TILE_COMPUTE_CASE(DT_INT32, int32_t, TYPE, CTX)                   \
  TILE_COMPUTE_CASE(DT_INT64, int64_t, TYPE, CTX)                   \
  TILE_COMPUTE_CASE(DT_UINT8, uint8_t, TYPE, CTX)                   \
  TILE_COMPUTE_CASE(DT_UINT16, uint16_t, TYPE, CTX)                 \
  TILE_COMPUTE_CASE(DT_UINT32, uint32_t, TYPE, CTX)                 \
  TILE_COMPUTE_CASE(DT_UINT64, uint64_t, TYPE, CTX)
}  // namespace

namespace aicpu {
uint32_t TileCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Tile check input and output number failed.");
  Tensor *input_x0 = ctx.Input(0);
  Tensor *input_x1 = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  auto size_0 = ctx.Input(0)->GetTensorShape()->GetDims();
  auto size_1 = ctx.Input(1)->GetTensorShape()->GetDims();
  KERNEL_CHECK_FALSE((size_0 >= 1), KERNEL_STATUS_PARAM_INVALID, "Dimension of x must be 1 or higher, but got[%zu].",
                     size_0);
  KERNEL_CHECK_FALSE((size_1 == 1), KERNEL_STATUS_PARAM_INVALID, "Dimension of multiples must be 1, but got[%zu].",
                     size_1);
  KERNEL_CHECK_FALSE((size_0 == input_x1->NumElements()), KERNEL_STATUS_PARAM_INVALID,
                     "Multiples length must be the same as the number of dimensions in x.");
  KERNEL_LOG_DEBUG(
    "TileCpuKernel[%s], inputx0: size[%llu];"
    "inputx1: size[%llu], output: size[%llu].",
    ctx.GetOpType().c_str(), input_x0->GetDataSize(), input_x1->GetDataSize(), output->GetDataSize());

  DataType data_type = ctx.Input(0)->GetDataType();
  DataType multiples_type = ctx.Input(1)->GetDataType();
  switch (multiples_type) {
    case DT_INT32:
      switch (data_type) {
        TILE_COMPUTE_CASE_ALL(int32_t, ctx)
        default:
          KERNEL_LOG_ERROR("Input[0] data type[%s] not supported.", DTypeStr(data_type).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    case DT_INT64:
      switch (data_type) {
        TILE_COMPUTE_CASE_ALL(int64_t, ctx)
        default:
          KERNEL_LOG_ERROR("Input[0] data type[%s] not supported.", DTypeStr(data_type).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    default:
      KERNEL_LOG_ERROR("Input[1] data type[%s] not supported.", DTypeStr(multiples_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename M>
void TileCpuKernel::CopyMultipleTimes(const T *in_data, int64_t in_size, M multiplier, T *out_data) {
  for (M i = 0; i < multiplier; ++i) {
    const T *in_end = in_data + in_size;
    T *new_out_data = std::copy(in_data, in_end, out_data);
    in_data = out_data;
    out_data = new_out_data;
  }
}

template <typename T, typename M>
std::pair<int64_t, int64_t> TileCpuKernel::TileOneDimension(const std::vector<int64_t> &in_dimensions, const T *in_data,
                                                            const M *multipliers, T *out_data, int64_t dimension) {
  if (in_dimensions.size() == 0) {
    // If input tensor is a scalar, then just copy it to output (no need to
    // multiply).
    *out_data = *in_data;
    return std::make_pair(0, 0);
  }

  const int64_t dimension_size = in_dimensions[dimension];
  if (dimension == static_cast<int64_t>(in_dimensions.size() - 1)) {
    CopyMultipleTimes(in_data, dimension_size, multipliers[dimension], out_data);
    return std::make_pair(dimension_size, dimension_size * static_cast<int64_t>(multipliers[dimension]));
  }
  int64_t total_stride_size = 0, total_tiled_stride_size = 0;
  const T *copy_from_data = in_data;
  T *copy_to_data = out_data;
  for (int64_t i = 0; i < dimension_size; ++i) {
    int64_t stride_size = 0, tiled_stride_size = 0;
    std::tie(stride_size, tiled_stride_size) =
      TileOneDimension(in_dimensions, copy_from_data, multipliers, copy_to_data, dimension + 1);
    copy_from_data += stride_size;
    copy_to_data += tiled_stride_size;
    total_stride_size += stride_size;
    total_tiled_stride_size += tiled_stride_size;
  }
  CopyMultipleTimes(out_data, total_tiled_stride_size, multipliers[dimension] - 1, out_data + total_tiled_stride_size);
  return std::make_pair(total_stride_size, static_cast<int64_t>(total_tiled_stride_size * multipliers[dimension]));
}

template <typename T, typename M>
uint32_t TileCpuKernel::TileCompute(CpuKernelContext &ctx) {
  auto x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto multiples = reinterpret_cast<M *>(ctx.Input(1)->GetData());
  auto y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  std::vector<int64_t> in_dimensions = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  TileOneDimension(in_dimensions, x, multiples, y, 0);
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kTile, TileCpuKernel);
}  // namespace aicpu
