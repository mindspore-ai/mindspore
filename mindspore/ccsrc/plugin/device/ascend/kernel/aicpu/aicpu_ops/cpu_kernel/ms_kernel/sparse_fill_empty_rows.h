/**
 * Copyright (c) 2022-2022-2023 Huawei Technologies Co., Ltd.  All rights reserved.
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
#ifndef AICPU_KERNELS_NORMALIZED_SPARSE_FILL_EMPTY_ROWS_H_
#define AICPU_KERNELS_NORMALIZED_SPARSE_FILL_EMPTY_ROWS_H_

#include <set>
#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include "utils/sparse_group.h"
#include "utils/sparse_tensor.h"

namespace aicpu {
struct DataBank {
  DataBank()
      : indices(nullptr),
        values(nullptr),
        dense_shape(nullptr),
        default_value(nullptr),
        y_indices(nullptr),
        y_values(nullptr),
        empty_row_indicator(nullptr),
        reverse_index_map(nullptr) {}
  Tensor *indices;
  Tensor *values;
  Tensor *dense_shape;
  Tensor *default_value;
  Tensor *y_indices;
  Tensor *y_values;
  Tensor *empty_row_indicator;
  Tensor *reverse_index_map;
};

class SparseFillEmptyRowsCpuKernel : public CpuKernel {
 public:
  ~SparseFillEmptyRowsCpuKernel() = default;
  SparseFillEmptyRowsCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t NullptrAndMatVecCheck(const CpuKernelContext &ctx, DataBank &calc_info);

  template <typename T>
  uint32_t ComputeSparseFillEmptyRows(DataBank &databank);
};
}  // namespace aicpu
#endif
