#ifndef AICPU_KERNELS_NORMALIZED_FRACTIONAL_MAX_POOL_H_
#define AICPU_KERNELS_NORMALIZED_FRACTIONAL_MAX_POOL_H_

#include "cpu_ops_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class FractionalMaxPoolCpuKernel : public CpuKernel {
 public:
  FractionalMaxPoolCpuKernel() = default;
  ~FractionalMaxPoolCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);
  uint32_t FractionalMaxPoolParamCheck(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_FRACTIONAL_MAX_POOL_H_
