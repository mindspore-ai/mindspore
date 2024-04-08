/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "ms_kernel/flatten.h"
#include <vector>
#include <numeric>
#include <securec.h>
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace aicpu {
namespace {
const char *kFlatten = "Flatten";
constexpr size_t kFlattenInputNum = 1;
constexpr size_t kFlattenOutputNum = 1;
struct MatrixInfo {
  std::vector<int64_t> matrix_shape;
  DataType matrix_type;
};
}  // namespace

uint32_t FlattenCpuKernel::Compute(CpuKernelContext &ctx) {
  NormalCheck(ctx, kFlattenInputNum, kFlattenOutputNum);

  auto input = ctx.Input(0);
  auto input_size = input->GetDataSize();
  auto output = ctx.Output(0);
  auto output_size = output->GetDataSize();
  CUST_KERNEL_CHECK_FALSE(ctx, input_size == output_size, KERNEL_STATUS_INNER_ERROR,
                          "Input size [%zu] differs from output size [%zu].", input_size, output_size);
  auto input_data = input->GetData();
  auto output_data = output->GetData();
  auto ret = memcpy_s(output_data, output_size, input_data, input_size);
  CUST_KERNEL_CHECK_FALSE(ctx, ret == EOK, KERNEL_STATUS_INNER_ERROR,
                          "memcpy_s failed, output_addr [%p], output_size [%zu], input_addr[%p].", output_data,
                          output_size, input_data);

  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kFlatten, FlattenCpuKernel);
}  // namespace aicpu
