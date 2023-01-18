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
#include "sparse_matrix_nnz.h"
#include <securec.h>
#include <complex>
#include <numeric>
#include <string>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_log.h"
#include "status.h"
#include "utils/allocator_utils.h"
#include "utils/kernel_util.h"

using namespace std;

namespace aicpu {
const char *SparseMatrixNNZ = "SparseMatrixNNZ";
const int INPUT_PARAMS_NUM = 5;
const int OUTPUT_PARAMS_NUM = 1;
}  // namespace aicpu
namespace aicpu {

uint32_t SparseMatrixNNZCpuKernel::Compute(CpuKernelContext &ctx) {
  if (NormalCheck(ctx, INPUT_PARAMS_NUM, OUTPUT_PARAMS_NUM) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  DataType indice_type = ctx.Input(1)->GetDataType();
  uint32_t status;
  switch (indice_type) {
    case DT_INT32:
      status = DoCompute<int32_t>(ctx);
      break;
    case DT_INT64:
      status = DoCompute<int64_t>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("data type of batch pointers is not int32 or int64");
      status = KERNEL_STATUS_PARAM_INVALID;
  }

  if (status != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("error in do the actual compute!");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename indiceT>
uint32_t SparseMatrixNNZCpuKernel::DoCompute(CpuKernelContext &ctx) {
  const indiceT batch_size = ctx.Input(1)->NumElements() - 1;
  // define some temp arrays to store the output tensor data
  int32_t result_nnz[batch_size];
  // do computer
  indiceT *batch_pointers_x = static_cast<indiceT *>(ctx.Input(1)->GetData());
  indiceT curr = 0;
  for (int i = 1; i < batch_size + 1; i++) {
    result_nnz[i - 1] = batch_pointers_x[i] - curr;
    // update curr
    curr = batch_pointers_x[i];
  }
  // write result
  int32_t *output_y = static_cast<int32_t *>(ctx.Output(0)->GetData());
  std::copy(result_nnz, result_nnz + (int32_t)batch_size, output_y);

  KERNEL_LOG_DEBUG("DoCompute end!!");
  return KERNEL_STATUS_OK;
}

// register the opetaor
REGISTER_CPU_KERNEL(SparseMatrixNNZ, SparseMatrixNNZCpuKernel);
}  // namespace aicpu