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

#include "ms_kernel/reversev2.h"
#include <securec.h>
#include <unordered_set>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include "Eigen/Core"

#include "common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/tensor_iterator.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
const char *kReverseV2 = "ReverseV2";

std::vector<int64_t> idx2coord(int idx, const std::vector<int64_t> &accum_dim) {
  std::vector<int64_t> coord(accum_dim.size());
  for (size_t i = 0; i < coord.size(); ++i) {
    coord[i] = idx / accum_dim[i];
    idx -= coord[i] * accum_dim[i];
  }
  return coord;
}

inline int64_t calc_target_idx(const std::vector<int64_t> &coord, const std::unordered_set<int64_t> &dims,
                               const std::vector<int64_t> &shape, const std::vector<int64_t> &accum_dim) {
  int64_t idx = 0;
  for (size_t i = 0; i < coord.size(); ++i) {
    if (dims.count(i) != 0) {
      idx += accum_dim[i] * (shape[i] - coord[i] - 1);
    } else {
      idx += accum_dim[i] * coord[i];
    }
  }
  return idx;
}
}  // namespace

namespace aicpu {
uint32_t ReverseV2CpuKernel::Compute(CpuKernelContext &ctx) {
  int x_max_dim = 8;
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "ReverseV2 check input or output is failed.");
  DataType axis_type = ctx.Input(1)->GetDataType();
  KERNEL_CHECK_FALSE((axis_type == DT_INT32 || axis_type == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of [axis] need be DT_INT32 or DT_INT64.")
  auto x_shape = ctx.Input(0)->GetTensorShape();
  auto axis_shape = ctx.Input(1)->GetTensorShape();
  DataType data_type = DataType(ctx.Input(0)->GetDataType());
  // dims check
  if (x_shape->GetDims() == 0 || axis_shape->GetDims() == 0) {
    return ComputeDiffType(data_type, ctx);
  }
  KERNEL_CHECK_FALSE((x_shape->GetDims() > 0 && x_shape->GetDims() <= x_max_dim), KERNEL_STATUS_PARAM_INVALID,
                     "Shapes of x is not support.")
  KERNEL_CHECK_FALSE((axis_shape->GetDims() == 1), KERNEL_STATUS_PARAM_INVALID, "Shapes of axis is not support.")

  auto input0_datasize = ctx.Input(0)->GetDataSize();
  auto output_datasize = ctx.Output(0)->GetDataSize();
  KERNEL_CHECK_FALSE((input0_datasize == output_datasize), KERNEL_STATUS_PARAM_INVALID,
                     "The data size of input0 [%d] need be same with "
                     "output0 [%d].",
                     input0_datasize, output_datasize)
  int64_t dim = x_shape->GetDims();
  auto input_axis = reinterpret_cast<int64_t *>(ctx.Input(1)->GetData());
  int64_t axis_element = axis_shape->NumElements();
  for (int j = 0; j < axis_element; j++) {
    int64_t realdim = *(input_axis + j) < 0 ? dim + *(input_axis + j) : *(input_axis + j);
    KERNEL_CHECK_FALSE((realdim >= 0 && realdim < dim), KERNEL_STATUS_PARAM_INVALID, "[%d] is invalid", realdim)
  }
  uint32_t ret = ComputeDiffType(data_type, ctx);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  return KERNEL_STATUS_OK;
}

uint32_t ReverseV2CpuKernel::ComputeDiffType(DataType data_type, const CpuKernelContext &ctx) {
  switch (data_type) {
    case DT_FLOAT16:
      return ComputeReverseV2<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeReverseV2<float>(ctx);
    case DT_DOUBLE:
      return ComputeReverseV2<double>(ctx);
    case DT_UINT8:
      return ComputeReverseV2<uint8_t>(ctx);
    case DT_INT8:
      return ComputeReverseV2<int8_t>(ctx);
    case DT_UINT16:
      return ComputeReverseV2<uint16_t>(ctx);
    case DT_INT16:
      return ComputeReverseV2<int16_t>(ctx);
    case DT_INT32:
      return ComputeReverseV2<int32_t>(ctx);
    case DT_INT64:
      return ComputeReverseV2<int64_t>(ctx);
    case DT_BOOL:
      return ComputeReverseV2<bool>(ctx);
    case DT_COMPLEX64:
      return ComputeReverseV2<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return ComputeReverseV2<std::complex<double>>(ctx);
    case DT_STRING:
      return ComputeReverseV2<std::string>(ctx);
    default:
      KERNEL_LOG_ERROR("ReverseV2 invalid input type[%s]", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ReverseV2CpuKernel::ComputeReverseV2(const CpuKernelContext &ctx) {
  auto input = ctx.Input(0);
  auto input_shape = input->GetTensorShape()->GetDimSizes();
  auto input_data = reinterpret_cast<T *>(input->GetData());
  auto num_elem = input->NumElements();
  auto axis = ctx.Input(1);
  auto axis_shape = axis->GetTensorShape()->GetDimSizes();
  auto axis_data = reinterpret_cast<int64_t *>(axis->GetData());
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  if (axis_shape.size() == 0) {
    std::copy(input_data, input_data + num_elem, output_data);
    return KERNEL_STATUS_OK;
  }

  std::unordered_set<int64_t> axes;
  std::transform(axis_data, axis_data + axis->NumElements(), std::inserter(axes, axes.begin()),
                 [&input_shape](int64_t x) { return x >= 0 ? x : input_shape.size() + x; });

  std::vector<int64_t> accum_dim(input_shape.size());
  accum_dim.back() = 1;
  for (size_t i = input_shape.size() - 1; i > 0; --i) {
    accum_dim[i - 1] = accum_dim[i] * input_shape[i];
  }

  auto sharder_reverse = [&](int64_t start, int64_t end) {
    std::vector<int64_t> cur_coord = idx2coord(start, accum_dim);
    auto coord_iter = TensorIterator(input_shape, cur_coord);
    for (int i = start; i < end; ++i) {
      auto target_idx = calc_target_idx(*coord_iter, axes, input_shape, accum_dim);
      output_data[target_idx] = input_data[i];
      ++coord_iter;
    }
  };

  const int64_t kParallelDataNum = 2 * 1024;
  if (num_elem > kParallelDataNum) {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, num_elem, num_elem / max_core_num, sharder_reverse),
                        "ReverseV2 compute failed.");
  } else {
    sharder_reverse(0, num_elem);
  }

  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kReverseV2, ReverseV2CpuKernel);
}  // namespace aicpu
