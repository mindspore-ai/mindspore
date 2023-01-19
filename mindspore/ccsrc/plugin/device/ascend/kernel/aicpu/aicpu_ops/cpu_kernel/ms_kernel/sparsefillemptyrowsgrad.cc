/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "sparsefillemptyrowsgrad.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include "cpu_kernel_utils.h"
#include "utils/allocator_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "kernel_log.h"
#include "status.h"

namespace {
const char *kSparseFillEmptyRowsGrad = "SparseFillEmptyRowsGrad";
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 2;
const int64_t kParallelNum{16384};

bool isVector(const std::shared_ptr<aicpu::TensorShape> shape) { return shape->GetDims() == 1; }
}  // namespace

namespace aicpu {
template <typename T>
uint32_t SparseFillEmptyRowsGradCpuKernel::ComputeSparseFillEmptyRowsGrad(CpuKernelContext &ctx, DataBank &databank) {
  EigenTensor reverse_index_map_e(databank.reverse_index_map, databank.reverse_index_map->GetData());
  EigenTensor grad_values_e(databank.grad_values, databank.grad_values->GetData());
  EigenTensor y_value_e(databank.y_value, databank.y_value->GetData());

  auto reverse_index_map = reverse_index_map_e.vec<int64_t>();
  auto grad_values = grad_values_e.vec<T>();
  auto y_value = y_value_e.vec<T>();

  const int64_t N = databank.reverse_index_map->GetTensorShape()->GetDimSize(0);
  const int64_t N_full = databank.grad_values->GetTensorShape()->GetDimSize(0);

  std::vector<bool> visited(N_full, false);
  T *y_default_value = reinterpret_cast<T *>(databank.y_default_value->GetData());
  *y_default_value = static_cast<T>(0);
  if (N <= kParallelNum) {
    for (int64_t i = 0; i < N; ++i) {
      int64_t reverse_index = reverse_index_map(i);
      KERNEL_CHECK_FALSE(0 <= reverse_index && reverse_index < N_full, KERNEL_STATUS_PARAM_INVALID,
                         "Elements in reverse index must be in [0, [%d]) but got [%d]", N_full, reverse_index)
      y_value(i) = grad_values(reverse_index);
      visited[reverse_index] = true;
    }
  } else {
    int64_t total = N;
    uint32_t cores = CpuKernelUtils::GetCPUNum(ctx);
    int64_t per_unit_size = (total / std::min(std::max(1L, cores - 2L), total));
    uint32_t ret = CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        int64_t reverse_index = reverse_index_map(i);
        KERNEL_CHECK_FALSE_VOID(0 <= reverse_index && reverse_index < N_full,
                                "Elements in reverse index must be in [0, [%d]) but got [%d]", N_full, reverse_index);
        y_value(i) = grad_values(reverse_index);
        visited[reverse_index] = true;
      }
    });
    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_INNER_ERROR, "SparseFillEmptyRowsGrad compute failed.");
  }
  for (int64_t j = 0; j < N_full; ++j) {
    if (!visited[j]) {
      (*y_default_value) += grad_values(j);
    }
  }
  databank.y_default_value->GetTensorShape()->SetDimSizes({});
  databank.y_value->GetTensorShape()->SetDimSizes({N});
  return KERNEL_STATUS_OK;
}

uint32_t SparseFillEmptyRowsGradCpuKernel::NullptrAndMatVecCheck(CpuKernelContext &ctx, DataBank &databank) {
  databank.reverse_index_map = ctx.Input(0);
  databank.grad_values = ctx.Input(1);
  databank.y_value = ctx.Output(0);
  databank.y_default_value = ctx.Output(1);
  KERNEL_CHECK_FALSE(isVector(databank.reverse_index_map->GetTensorShape()), KERNEL_STATUS_PARAM_INVALID,
                     "Inputs reverse_index_map should be vectors")
  KERNEL_CHECK_FALSE(isVector(databank.grad_values->GetTensorShape()), KERNEL_STATUS_PARAM_INVALID,
                     "Inputs grad_values should be vectors")
  return KERNEL_STATUS_OK;
}

uint32_t SparseFillEmptyRowsGradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "SparseFillEmptyRowsGrad check input and output number failed.");
  DataBank databank;
  KERNEL_HANDLE_ERROR(NullptrAndMatVecCheck(ctx, databank), "SparseFillEmptyRowsGrad check params failed.");
  DataType dt = static_cast<DataType>(databank.y_value->GetDataType());

  uint32_t KERNEL_STATUS;
  switch (dt) {
    case DT_INT8:
      KERNEL_STATUS = ComputeSparseFillEmptyRowsGrad<int8_t>(ctx, databank);
      break;
    case DT_UINT8:
      KERNEL_STATUS = ComputeSparseFillEmptyRowsGrad<uint8_t>(ctx, databank);
      break;
    case DT_INT16:
      KERNEL_STATUS = ComputeSparseFillEmptyRowsGrad<int16_t>(ctx, databank);
      break;
    case DT_UINT16:
      KERNEL_STATUS = ComputeSparseFillEmptyRowsGrad<uint16_t>(ctx, databank);
      break;
    case DT_INT32:
      KERNEL_STATUS = ComputeSparseFillEmptyRowsGrad<int32_t>(ctx, databank);
      break;
    case DT_UINT32:
      KERNEL_STATUS = ComputeSparseFillEmptyRowsGrad<uint32_t>(ctx, databank);
      break;
    case DT_INT64:
      KERNEL_STATUS = ComputeSparseFillEmptyRowsGrad<int64_t>(ctx, databank);
      break;
    case DT_UINT64:
      KERNEL_STATUS = ComputeSparseFillEmptyRowsGrad<uint64_t>(ctx, databank);
      break;
    case DT_BOOL:
      KERNEL_STATUS = ComputeSparseFillEmptyRowsGrad<bool>(ctx, databank);
      break;
    case DT_STRING:
      KERNEL_STATUS = ComputeSparseFillEmptyRowsGrad<std::string>(ctx, databank);
      break;
    case DT_FLOAT16:
      KERNEL_STATUS = ComputeSparseFillEmptyRowsGrad<Eigen::half>(ctx, databank);
      break;
    case DT_FLOAT:
      KERNEL_STATUS = ComputeSparseFillEmptyRowsGrad<float>(ctx, databank);
      break;
    case DT_DOUBLE:
      KERNEL_STATUS = ComputeSparseFillEmptyRowsGrad<double>(ctx, databank);
      break;
    case DT_COMPLEX64:
      KERNEL_STATUS = ComputeSparseFillEmptyRowsGrad<std::complex<float>>(ctx, databank);
      break;
    case DT_COMPLEX128:
      KERNEL_STATUS = ComputeSparseFillEmptyRowsGrad<std::complex<double>>(ctx, databank);
      break;
    default:
      KERNEL_LOG_ERROR("SparseFillEmptyRowsGrad can't support this data type [%s].", DTypeStr(dt).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (KERNEL_STATUS != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("SparseFillEmptyRowsGrad failed.");
    return KERNEL_STATUS;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSparseFillEmptyRowsGrad, SparseFillEmptyRowsGradCpuKernel);
}  // namespace aicpu