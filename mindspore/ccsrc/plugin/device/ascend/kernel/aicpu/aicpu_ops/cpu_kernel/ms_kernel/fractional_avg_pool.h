#ifndef AICPU_KERNELS_NORMALIZED_FRACTIONAL_AVG_POOL_H_
#define AICPU_KERNELS_NORMALIZED_FRACTIONAL_AVG_POOL_H_

#include "cpu_ops_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class FractionalAvgPoolCpuKernel : public CpuKernel {
 public:
  FractionalAvgPoolCpuKernel() = default;
  ~FractionalAvgPoolCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);
  uint32_t FractionalAvgPoolParamCheck(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_FRACTIONAL_AVG_POOL_H_
