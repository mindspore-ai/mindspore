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
#ifndef AICPU_KERNELS_NORMALIZED_DATA_FORMAT_VEC_PERMUTE_H_
#define AICPU_KERNELS_NORMALIZED_DATA_FORMAT_VEC_PERMUTE_H_

#include <string>
#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class DataFormatVecPermute : public CpuKernel {
 public:
  DataFormatVecPermute() = default;
  ~DataFormatVecPermute() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t DataFormatVecPermuteCompute(const int32_t dim, const std::string &src_format_str,
                                       const std::string &dst_format_str, Tensor *x, Tensor *y, CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
