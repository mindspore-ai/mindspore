#ifndef AICPU_KERNELS_NORMALIZED_CHOLESKYINVERSE_H_
#define AICPU_KERNELS_NORMALIZED_CHOLESKYINVERSE_H_

#include "cpu_ops_kernel.h"

namespace aicpu {

class CholeskyInverseCpuKernel : public CpuKernel {
 public:
  CholeskyInverseCpuKernel() = default;
  ~CholeskyInverseCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  static uint32_t CholeskyInverseCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
