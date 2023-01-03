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
#ifndef AICPU_KERNELS_NORMALIZED_NUCLEARNORM_H_
#define AICPU_KERNELS_NORMALIZED_NUCLEARNORM_H_
#include <memory>
#include <vector>

#include "cpu_ops_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_log.h"
#include "securec.h"
#include "status.h"
#include "utils/bcast.h"

namespace aicpu {
class NuclearNormCpuKernel : public CpuKernel {
 public:
  NuclearNormCpuKernel() = default;
  ~NuclearNormCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t NuclearNormParamCheck(CpuKernelContext &ctx);

  template <typename T>
  uint32_t NuclearNormCompute(CpuKernelContext &ctx);

  template <typename T, int32_t RANK>
  uint32_t ComputeTensorNuclearNorm(const CpuKernelContext &ctx);

  template <typename T>
  std::vector<std::vector<T>> matrix_multiply(std::vector<std::vector<T>> const arrL,
                                              std::vector<std::vector<T>> const arrR);

  template <typename T>
  std::vector<std::vector<T>> transpose(std::vector<std::vector<T>> const arr);

  template <typename T>
  std::vector<size_t> argsort(const std::vector<T> &array);

  template <typename T>
  void get_row_col(std::vector<std::vector<T>> arr, T *max, size_t *row, size_t *col);

  template <typename T>
  void svd(std::vector<std::vector<T>> arr, std::vector<std::vector<T>> &E, std::vector<T> &e);

  template <typename T>
  T matrix_nuclear_norm(T *mat, size_t dim0, size_t dim1);
};
}  // namespace aicpu
#endif
