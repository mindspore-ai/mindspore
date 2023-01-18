#ifndef AICPU_KERNELS_NORMALIZED_FRACTIONAL_MAX_POOL_GRAD_H_
#define AICPU_KERNELS_NORMALIZED_FRACTIONAL_MAX_POOL_GRAD_H_

#include "cpu_ops_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class FractionalMaxPoolGradCpuKernel : public CpuKernel {
 public:
  FractionalMaxPoolGradCpuKernel() = default;
  ~FractionalMaxPoolGradCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);
  uint32_t FractionalMaxPoolGradParamCheck(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_FRACTIONAL_MAX_POOL_GRAD_H_
