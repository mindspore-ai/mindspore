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
#include "cpu_ops_kernel.h"
#include "utils/eigen_tensor.h"

namespace aicpu {
struct DataBank {
  DataBank()
      : a_indices_t(nullptr),
        a_values_t(nullptr),
        a_shape_t(nullptr),
        b_indices_t(nullptr),
        b_values_t(nullptr),
        b_shape_t(nullptr) {}
  Tensor *a_indices_t;
  Tensor *a_values_t;
  Tensor *a_shape_t;
  Tensor *b_indices_t;
  Tensor *b_values_t;
  Tensor *b_shape_t;
  Tensor *output_indices_t;
  Tensor *output_values_t;
};

class SparseMaximumCpuKernel : public CpuKernel {
 public:
  ~SparseMaximumCpuKernel() = default;
  SparseMaximumCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  static void UnionSparseIndicesAndValues(typename TTypes<int64_t>::Matrix a_indices_mat,
                                          typename TTypes<T>::Flat a_values, int64_t a_nnz,
                                          typename TTypes<int64_t>::Matrix b_indices_mat,
                                          typename TTypes<T>::Flat b_values, int64_t b_nnz, int64_t num_dims,
                                          std::vector<T> *a_augmented_values, std::vector<T> *b_augmented_values,
                                          std::vector<std::pair<bool, int64_t>> *entries_to_copy);

  template <typename T>
  uint32_t EigenedSparseMax(DataBank &databank);

  static uint32_t NullptrAndMatVecCheck(CpuKernelContext &ctx, DataBank &calc_info);
};
}  // namespace aicpu
