/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#ifndef AICPU_KERNELS_NORMALIZED_SPARSE_FILL_EMPTY_ROWS_GRAD_H_
#define AICPU_KERNELS_NORMALIZED_SPARSE_FILL_EMPTY_ROWS_GRAD_H_

#include <set>
#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include "utils/sparse_group.h"
#include "utils/sparse_tensor.h"

namespace aicpu {
struct DataBank {
  DataBank() : reverse_index_map(nullptr), grad_values(nullptr), y_value(nullptr), y_default_value(nullptr) {}
  Tensor *reverse_index_map;
  Tensor *grad_values;
  Tensor *y_value;
  Tensor *y_default_value;
};

class SparseFillEmptyRowsGradCpuKernel : public CpuKernel {
 public:
  ~SparseFillEmptyRowsGradCpuKernel() = default;
  SparseFillEmptyRowsGradCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t NullptrAndMatVecCheck(const CpuKernelContext &ctx, DataBank &calc_info);

  template <typename T>
  uint32_t ComputeSparseFillEmptyRowsGrad(const CpuKernelContext &ctx, DataBank &databank);
};
}  // namespace aicpu
#endif
