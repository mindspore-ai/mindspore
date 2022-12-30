#ifndef AICPU_KERNELS_NORMALIZED_FRACTIONAL_AVG_GRAD_POOL_H_
#define AICPU_KERNELS_NORMALIZED_FRACTIONAL_AVG_GRAD_POOL_H_

#include "cpu_ops_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class FractionalAvgPoolGradCpuKernel : public CpuKernel {
 public:
  FractionalAvgPoolGradCpuKernel() = default;
  ~FractionalAvgPoolGradCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);
  uint32_t FractionalAvgPoolGradParamCheck(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_FRACTIONAL_AVG_GRAD_POOL_H_