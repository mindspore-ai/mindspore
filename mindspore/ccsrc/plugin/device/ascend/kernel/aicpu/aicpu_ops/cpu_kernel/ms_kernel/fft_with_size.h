#ifndef AICPU_KERNELS_NORMALIZED_FFTWITHSIZE_H_
#define AICPU_KERNELS_NORMALIZED_FFTWITHSIZE_H_

#include "Eigen/Dense"
#include "cpu_ops_kernel.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/FFT"
#include "utils/bcast.h"
namespace aicpu {

class FFTWithSizeCpuKernel : public CpuKernel {
 public:
  FFTWithSizeCpuKernel() = default;
  ~FFTWithSizeCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T1, typename T2, int signal_ndim, bool is_real, bool real_inverse>
  static uint32_t FFTWithSizeCompute(CpuKernelContext &ctx, bool onesided, bool inverse, std::string normalized,
                                     std::vector<int64_t> &checked_signal_size);

  static double Getnormalized(int64_t n, std::string normalized, bool is_reverse);
};
}  // namespace aicpu
#endif
