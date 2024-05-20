/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_FFTFREQ_H_
#define AICPU_KERNELS_NORMALIZED_FFTFREQ_H_

#include <complex>
#include <utility>
#include <map>
#include <functional>
#include <algorithm>
#include <memory>
#include <vector>
#include <securec.h>
#include "inc/ms_cpu_kernel.h"

namespace aicpu {
const uint32_t kNIndex = 0;
const uint32_t kDIndex = 1;
const uint32_t kIndex0 = 0;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
class FFTFreqCpuKernel : public CpuKernel {
 public:
  ~FFTFreqCpuKernel() = default;

  DataType n_type_;
  DataType d_type_;
  DataType output_type_;
  std::string op_name_;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;
};
}  // namespace aicpu
#endif  //  AICPU_FFTFREQ_H
