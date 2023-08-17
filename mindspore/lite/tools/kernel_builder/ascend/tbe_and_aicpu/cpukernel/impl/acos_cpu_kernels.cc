
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of AcosCpu
 */
#include "./acos_cpu_kernels.h"
#include <cmath>

namespace {
const char *ACOS_CPU = "AcosCpu";
}

namespace aicpu {
uint32_t AcosCpuCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  Tensor *output = ctx.Output(0);

  if (input == nullptr || output == nullptr) {
    return 1;
  }

  auto inputData = static_cast<double *>(input->GetData());
  auto outputData = static_cast<double *>(output->GetData());
  if (inputData == nullptr || outputData == nullptr) {
    return 1;
  }

  DataType inputType = input->GetDataType();
  switch (inputType) {
    case DT_DOUBLE:
      break;
    default:
      return 1;
  }

  auto num = input->NumElements();
  for (int64_t i = 0; i < num; i++) {
    outputData[i] = std::acos(inputData[i]);
  }

  return 0;
}

REGISTER_CPU_KERNEL(ACOS_CPU, AcosCpuCpuKernel);
}  // namespace aicpu
