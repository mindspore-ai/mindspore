/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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