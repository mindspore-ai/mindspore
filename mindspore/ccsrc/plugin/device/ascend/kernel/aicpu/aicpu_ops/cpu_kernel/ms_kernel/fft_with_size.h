#ifndef AICPU_KERNELS_NORMALIZED_FFTWITHSIZE_H_
#define AICPU_KERNELS_NORMALIZED_FFTWITHSIZE_H_

#include "Eigen/Dense"
#include "inc/ms_cpu_kernel.h"
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

  static double Getnormalized(std::vector<int64_t> shape, std::string normalized, bool is_reverse, uint64_t dim,
                              bool is_real, std::vector<int64_t> end_shape);
};
}  // namespace aicpu
#endif