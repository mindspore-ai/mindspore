#ifndef AICPU_KERNELS_DEPTHTOSPACE_H_
#define AICPU_KERNELS_DEPTHTOSPACE_H_

#include "cpu_ops_kernel.h"

namespace aicpu {
class DepthToSpaceCpuKernel : public CpuKernel {
 public:
  DepthToSpaceCpuKernel() = default;
  ~DepthToSpaceCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override final;

 private:
  uint32_t STDParamCheck(CpuKernelContext &ctx);

  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

  std::string data_format_;
};  // DepthToSpaceCpuKernel
}  // namespace aicpu
#endif  // AICPU_KERNELS_DEPTHTOSPACE_H_
