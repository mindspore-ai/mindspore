#ifndef AICPU_KERNELS_NORMALIZED_DCT_H_
#define AICPU_KERNELS_NORMALIZED_DCT_H_

#include "Eigen/Dense"
#include "inc/ms_cpu_kernel.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/FFT"
#include "utils/bcast.h"
namespace aicpu {

class DCTCpuKernel : public CpuKernel {
 public:
  DCTCpuKernel() = default;
  ~DCTCpuKernel() = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t DCTCheck(CpuKernelContext &ctx);

  template <typename T1, typename T2, typename T3, int x_rank>
  static uint32_t DCTCompute(CpuKernelContext &ctx, int n, int axis, int norm_type, const std::vector<int64_t> &x_shape,
                             bool forward, bool grad);
};
}  // namespace aicpu
#endif
