#ifndef AICPU_KERNELS_NORMALIZED_LUSOLVE_H_
#define AICPU_KERNELS_NORMALIZED_LUSOLVE_H_

#include "cpu_ops_kernel.h"
#include "utils/bcast.h"
namespace aicpu {

class LuSolveCpuKernel : public CpuKernel {
 public:
  LuSolveCpuKernel() = default;
  ~LuSolveCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T, typename T2>
  static uint32_t LuSolve(CpuKernelContext &ctx, T *b_working_ptr, T *lu_working_ptr, int32_t *pivots_working_ptr,
                          int64_t b_stride, int64_t i);
  template <typename T, typename T2>
  static uint32_t LuSolveCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
