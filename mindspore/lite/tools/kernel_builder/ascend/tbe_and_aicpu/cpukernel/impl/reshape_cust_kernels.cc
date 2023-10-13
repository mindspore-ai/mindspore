/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of sample
 */

#include "./reshape_cust_kernels.h"
#include <cstring>
#include "./cpu_types.h"

namespace {
const char *RESHAPE_CUST = "ReshapeCust";
}

namespace aicpu {
uint32_t ReshapeCustCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  if (input_tensor == nullptr) {
    return -1;
  }

  Tensor *output_tensor = ctx.Output(0);
  if (output_tensor == nullptr) {
    return -1;
  }
  auto input_data = input_tensor->GetData();
  if (input_data == nullptr) {
    return -1;
  }

  auto output_data = output_tensor->GetData();
  if (output_data == nullptr) {
    return -1;
  }

  uint64_t data_size = input_tensor->GetDataSize();
  return memcpy_s(output_data, data_size, input_data, data_size);
}

REGISTER_CPU_KERNEL(RESHAPE_CUST, ReshapeCustCpuKernel);
}  // namespace aicpu
