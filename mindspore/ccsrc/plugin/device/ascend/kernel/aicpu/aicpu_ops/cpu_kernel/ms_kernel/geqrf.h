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
#ifndef AICPU_KERNELS_NORMALIZED_GEQRF_H_
#define AICPU_KERNELS_NORMALIZED_GEQRF_H_

#include <complex>
#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class GeqrfCpuKernel : public CpuKernel {
 public:
  GeqrfCpuKernel() = default;
  ~GeqrfCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  void Larfg(int n, int vm, int vn, double **A, T *tau);

  template <typename T>
  void Larf(int m, int n, double **A, T *tau, int cm, int cn);

  template <typename T>
  void Geqrf(int m, int n, double **A, T *tau);

  template <typename T>
  void CLarfg(int n, int vm, int vn, std::complex<T> **A, std::complex<T> *tau);

  template <typename T>
  void CLarf(int m, int n, std::complex<T> **A, std::complex<T> *tau, int cm, int cn);

  template <typename T>
  void CGeqrf(int m, int n, std::complex<T> **A, std::complex<T> *tau);

  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t DoComputeC(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_GEQRF_H_
