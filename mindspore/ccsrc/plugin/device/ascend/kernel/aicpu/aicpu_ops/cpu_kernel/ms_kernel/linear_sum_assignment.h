/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef AICPU_KERNELS_NORMALIZED_LINEAR_SUM_ASSIGNMENT_H
#define AICPU_KERNELS_NORMALIZED_LINEAR_SUM_ASSIGNMENT_H

#include <vector>

#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class LinearSumAssignmentCpuKernel : public CpuKernel {
 public:
  LinearSumAssignmentCpuKernel() = default;
  ~LinearSumAssignmentCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  bool maximize = false;
  uint64_t nr = 0;
  uint64_t nc = 0;
  uint64_t raw_nc = 0;
  uint64_t cur_row = 0;
  std::vector<int64_t> path;
  std::vector<int64_t> col4row;
  std::vector<int64_t> row4col;
  std::vector<bool> SR;
  std::vector<bool> SC;
  std::vector<uint64_t> remaining;

  template <typename T>
  uint32_t LinearSumAssignmentCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t SolveProblem(CpuKernelContext &ctx, T *cost, int64_t *a, int64_t *b);

  template <typename T>
  uint32_t Solve(CpuKernelContext &ctx, const T *cost, int64_t *a, int64_t *b);

  template <typename T>
  int64_t AugmentingPath(const T *const cost, std::vector<T> &u, std::vector<T> &v, std::vector<T> &shortestPathCosts,
                         T *p_minVal);

  template <typename T>
  std::vector<uint64_t> ArgSortIter(const std::vector<T> &v);

  template <typename T>
  void ReArrange(std::vector<T> *temp, const T *const cost, bool transpose);

  void AugmentPreviousSolution(int64_t j);

  void PostProcess(int64_t *a, int64_t *b, bool transpose, uint64_t element_num);
};
}  // namespace aicpu

#endif