#ifndef AICPU_KERNELS_NORMALIZED_TRACEGRAD_H_
#define AICPU_KERNELS_NORMALIZED_TRACEGARD_H_

#include "cpu_ops_kernel.h"

namespace aicpu {

class TraceGradCpuKernel : public CpuKernel {
 public:
  TraceGradCpuKernel() = default;
  ~TraceGradCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T1, typename T2>
  static uint32_t TraceGradCompute(CpuKernelContext &ctx);
};
}  // namespace  aicpu
#endif