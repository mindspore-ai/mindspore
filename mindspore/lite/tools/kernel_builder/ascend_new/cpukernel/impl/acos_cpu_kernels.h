
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of AcosCpu
 */

#ifndef _ACOS_CPU_KERNELS_H_
#define _ACOS_CPU_KERNELS_H_

#include "./cpu_kernel.h"

namespace aicpu {
class AcosCpuCpuKernel : public CpuKernel {
 public:
  ~AcosCpuCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;
};
}  // namespace aicpu
#endif
