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
#include <map>
#include <set>
#include <vector>
#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include "utils/sparse_group.h"
#include "utils/sparse_tensor.h"
// 定义命名空间aicpu

namespace aicpu {
enum SetOperation { A_MINUS_B = 0, B_MINUS_A = 1, INTERSECTION = 2, UNION = 3 };
struct DataBank {
  DataBank()
      : set1(nullptr),
        set2_indices(nullptr),
        set2_values(nullptr),
        set2_shape(nullptr),
        result_indices(nullptr),
        result_values(nullptr),
        result_shape(nullptr) {}
  Tensor *set1;
  Tensor *set2_indices;
  Tensor *set2_values;
  Tensor *set2_shape;
  Tensor *result_indices;
  Tensor *result_values;
  Tensor *result_shape;
  SetOperation set_operation_;
  bool validate_indices_;
  CpuKernelContext *ctx;
};

// 算子类继承CpuKernel基类
class DenseToSparseSetOperationCpuKernel : public CpuKernel {
 public:
  ~DenseToSparseSetOperationCpuKernel() = default;
  DenseToSparseSetOperationCpuKernel() = default;
  // 声明函数Compute，且Compute函数需要重写
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t NullptrAndMatVecCheck(const CpuKernelContext &ctx, DataBank &calc_info);

  template <typename T>
  uint32_t ComputeDenseToSparse(DataBank &databank);

  template <typename T>
  uint32_t CheckGroup(const Group &group, const std::vector<int64_t> &sparse_tensor_shape);

  template <typename T>
  uint32_t PopulateFromSparseGroup(const Group &group, const std::vector<int64_t> &sparse_tensor_shape,
                                   std::set<T> &result);
  template <typename T>
  uint32_t PopulateFromDenseGroup(Tensor *input_tensor, const std::vector<int64_t> &input_strides,
                                  const std::vector<int64_t> &group_indices, std::set<T> &result);

  void PopulateGroupIndices(const int64_t flat_group_index, const std::vector<int64_t> &group_shape,
                            std::vector<int64_t> &group_indices);

  template <typename T>
  void ApplySetOperation(const std::set<T> &set1, const std::set<T> &set2, std::set<T> &result,
                         SetOperation set_operation_);

  template <typename T>
  uint32_t OutputSparseTensor(DataBank &databank, const std::vector<int64_t> &output_shape, const int64_t num_values,
                              const std::map<std::vector<int64_t>, std::set<T>> &sets);
};
}  // namespace aicpu
