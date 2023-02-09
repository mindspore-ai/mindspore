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
#include "densetosparsesetoperation.h"
#include <algorithm>
#include <atomic>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include "cpu_kernel_utils.h"
#include "utils/allocator_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "kernel_log.h"
#include "status.h"

namespace {
const char *kDenseToSparseSetOperation = "DenseToSparseSetOperation";
const uint32_t kOutputNum = 3;
const uint32_t kInputNum = 4;
constexpr int64_t kIndex0 = 0;
constexpr int64_t kIndex1 = 1;
constexpr int64_t kIndex2 = 2;
constexpr int64_t kIndex3 = 3;
const int64_t kParallelNum{64};
}  // namespace
// 定义命名空间aicpu
namespace aicpu {
const std::vector<int64_t> Strides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> result(shape.size());
  int64_t product = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    result[i] = product;
    product *= shape[i];
  }
  return result;
}

uint32_t GroupsShape(const std::vector<int64_t> input_shape, std::vector<int64_t> &grouped_shape) {
  if (input_shape.size() < 2) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // grouped_shape is input_shape[:-1]
  grouped_shape.assign(input_shape.begin(), input_shape.end() - 1);
  return KERNEL_STATUS_OK;
}

uint32_t CheckShapesMatch(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2) {
  if (shape1.size() != shape2.size()) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t i = 0; i < shape1.size(); i++) {
    if (shape1[i] != shape2[i]) return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t GroupsShapeFromInputs(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2,
                               std::vector<int64_t> &group_shape) {
  std::vector<int64_t> group_shape_1;
  KERNEL_HANDLE_ERROR(GroupsShape(shape1, group_shape_1), "X1_Shape rank is less than 2.");
  std::vector<int64_t> group_shape_2;
  KERNEL_HANDLE_ERROR(GroupsShape(shape2, group_shape_2), "X2_Shape rank is less than 2.");
  KERNEL_HANDLE_ERROR(CheckShapesMatch(group_shape_1, group_shape_2), "Two shapes mismatch with each other.");
  group_shape.assign(group_shape_1.begin(), group_shape_1.end());
  return KERNEL_STATUS_OK;
}

uint32_t GetsNumElements(const std::vector<int64_t> input_shape, int64_t &res) {
  int64_t result = 1;
  for (uint32_t i = 0; i < input_shape.size(); i++) {
    KERNEL_CHECK_FALSE(MulWithoutOverflow(input_shape[i], result, result), KERNEL_STATUS_PARAM_INVALID,
                       "Overflow when calculate shape size.");
  }
  res = result;
  return KERNEL_STATUS_OK;
}

void DenseToSparseSetOperationCpuKernel::PopulateGroupIndices(const int64_t flat_group_index,
                                                              const std::vector<int64_t> &group_shape,
                                                              std::vector<int64_t> &group_indices) {
  group_indices.clear();
  int64_t running_flat_group_index = flat_group_index;
  for (int64_t group_dim_index = static_cast<int64_t>(group_shape.size()) - 1; group_dim_index >= 0;
       --group_dim_index) {
    const auto group_dim = group_shape[group_dim_index];
    group_indices.insert(group_indices.begin(), running_flat_group_index % group_dim);
    running_flat_group_index /= group_dim;
  }
}

template <typename T>
uint32_t DenseToSparseSetOperationCpuKernel::PopulateFromDenseGroup(Tensor *input_tensor,
                                                                    const std::vector<int64_t> &input_strides,
                                                                    const std::vector<int64_t> &group_indices,
                                                                    std::set<T> &result) {
  result.clear();
  EigenTensor input_tensor_eigen(input_tensor, input_tensor->GetData());
  auto input_flat = input_tensor_eigen.flat<T>();
  const auto start = std::inner_product(group_indices.begin(), group_indices.end(), input_strides.begin(), 0LL);
  auto input_shape = input_tensor->GetTensorShape();
  const auto end = start + input_shape->GetDimSize(input_shape->GetDims() - 1);
  for (int64_t i = start; i < end; ++i) {
    result.insert(input_flat(i));
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t DenseToSparseSetOperationCpuKernel::PopulateFromSparseGroup(const Group &group,
                                                                     const std::vector<int64_t> &sparse_tensor_shape,
                                                                     std::set<T> &result) {
  KERNEL_HANDLE_ERROR(CheckGroup<T>(group, sparse_tensor_shape), "PopulateFromSparseGroup check error.");
  result.clear();
  const auto &group_values = group.values<T>();
  for (int64_t i = 0; i < group_values.size(); ++i) {
    result.insert(group_values(i));
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t DenseToSparseSetOperationCpuKernel::CheckGroup(const Group &group,
                                                        const std::vector<int64_t> &sparse_tensor_shape) {
  const auto &indices = group.indices();
  const auto &values = group.values<T>();
  const auto num_values = values.dimension(0);

  // Sanity check: valid indices.
  const uint32_t expected_rank = sparse_tensor_shape.size();
  for (uint32_t j = 0; j < expected_rank; ++j) {
    const auto dim_size = sparse_tensor_shape[j];
    KERNEL_CHECK_FALSE(dim_size > 0, KERNEL_STATUS_PARAM_INVALID, "Invalid dim_size [%d] for index [%d]", dim_size, j);
    for (int64_t i = 0; i < num_values; ++i) {
      const auto index = indices(i, j);
      KERNEL_CHECK_FALSE(dim_size > index, KERNEL_STATUS_PARAM_INVALID,
                         "indices index ([%d],[%d]) expected < [%d], got [%d].", i, j, dim_size, index);
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void DenseToSparseSetOperationCpuKernel::ApplySetOperation(const std::set<T> &set1, const std::set<T> &set2,
                                                           std::set<T> &result, SetOperation set_operation_) {
  switch (set_operation_) {
    case A_MINUS_B:
      std::set_difference(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(result, result.begin()));
      break;
    case B_MINUS_A:
      std::set_difference(set2.begin(), set2.end(), set1.begin(), set1.end(), std::inserter(result, result.begin()));
      break;
    case INTERSECTION:
      std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(result, result.begin()));
      break;
    case UNION:
      std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(result, result.begin()));
      break;
  }
}

template <typename T>
uint32_t DenseToSparseSetOperationCpuKernel::OutputSparseTensor(
  DataBank &databank, const std::vector<int64_t> &output_shape, const int64_t num_values,
  const std::map<std::vector<int64_t>, std::set<T>> &sets) {
  Tensor *out_indices, *out_values, *out_shape;
  out_indices = databank.result_indices;
  out_values = databank.result_values;
  out_shape = databank.result_shape;

  EigenTensor out_indices_t(out_indices, out_indices->GetData());
  auto out_indices_mat = out_indices_t.matrix<int64_t>();
  EigenTensor out_values_t(out_values, out_values->GetData());
  auto out_values_flat = out_values_t.vec<T>();
  EigenTensor out_shape_t(out_shape, out_shape->GetData());
  auto out_shape_flat = out_shape_t.vec<int64_t>();

  int64_t value_index = 0;
  for (auto it = sets.begin(); it != sets.end(); ++it) {
    const auto &group_indices = it->first;
    KERNEL_CHECK_FALSE(group_indices.size() == output_shape.size() - 1, KERNEL_STATUS_PARAM_INVALID,
                       "Invalid number of indices [%d] expected [%].", group_indices.size(), output_shape.size() - 1)
    const auto &set = it->second;

    // For each set item, write its indices and value to output tensors.
    int64_t group_value_index = 0;
    for (auto value = set.begin(); value != set.end(); ++value, ++value_index, ++group_value_index) {
      // First n-1 dimensions are the group, last dimension is the position in
      // the set.
      for (uint32_t i = 0; i < group_indices.size(); ++i) {
        out_indices_mat(value_index, i) = group_indices[i];
      }
      out_indices_mat(value_index, group_indices.size()) = group_value_index;

      out_values_flat(value_index) = *value;
    }
  }

  for (uint32_t i = 0; i < output_shape.size(); ++i) {
    out_shape_flat(i) = output_shape[i];
  }

  out_indices->GetTensorShape()->SetDimSizes({num_values, static_cast<int64_t>(output_shape.size())});
  out_values->GetTensorShape()->SetDimSizes({num_values});
  out_shape->GetTensorShape()->SetDimSizes({static_cast<int64_t>(output_shape.size())});

  return KERNEL_STATUS_OK;
}

uint32_t DenseToSparseSetOperationCpuKernel::NullptrAndMatVecCheck(CpuKernelContext &ctx, DataBank &databank) {
  databank.set1 = ctx.Input(kIndex0);
  databank.set2_indices = ctx.Input(kIndex1);
  databank.set2_values = ctx.Input(kIndex2);
  databank.set2_shape = ctx.Input(kIndex3);
  databank.result_indices = ctx.Output(kIndex0);
  databank.result_values = ctx.Output(kIndex1);
  databank.result_shape = ctx.Output(kIndex2);
  databank.ctx = &ctx;
  AttrValue *validate_indices = ctx.GetAttr("validate_indices");
  if (validate_indices == nullptr) {
    databank.validate_indices_ = true;
  } else {
    databank.validate_indices_ = validate_indices->GetBool();
  }
  AttrValue *set_operation = ctx.GetAttr("set_operation");
  KERNEL_CHECK_NULLPTR(set_operation, KERNEL_STATUS_PARAM_INVALID, "Missing set_operation.")
  std::string set_operation_str = set_operation->GetString();
  std::transform(set_operation_str.begin(), set_operation_str.end(), set_operation_str.begin(), ::tolower);
  if ("a-b" == set_operation_str) {
    databank.set_operation_ = A_MINUS_B;
  } else if ("b-a" == set_operation_str) {
    databank.set_operation_ = B_MINUS_A;
  } else if ("intersection" == set_operation_str) {
    databank.set_operation_ = INTERSECTION;
  } else if ("union" == set_operation_str) {
    databank.set_operation_ = UNION;
  } else {
    KERNEL_LOG_ERROR("Invalid set_operation.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t DenseToSparseSetOperationCpuKernel::ComputeDenseToSparse(DataBank &databank) {
  EigenTensor set2_shape_e(databank.set2_shape, databank.set2_shape->GetData());
  auto set2_shape = set2_shape_e.vec<int64_t>();
  std::vector<int64_t> shape2(set2_shape.size());
  for (int64_t i = 0; i < set2_shape.size(); ++i) {
    shape2[i] = set2_shape(i);
  }
  const auto rank = shape2.size();
  std::vector<int64_t> order(rank);
  std::iota(order.begin(), order.end(), 0);
  SparseTensor set2;

  Tensor *set1_t = databank.set1;
  SparseTensor *set2_st = &set2;
  KERNEL_HANDLE_ERROR(set2_st->CreateSparseTensor(databank.set2_indices, databank.set2_values, shape2, order),
                      "create sparse tenser fail.");
  if (databank.validate_indices_) {
    KERNEL_HANDLE_ERROR(set2_st->IndicesValid(*databank.ctx), "IndicesValid fail!!");
  }
  std::vector<int64_t> group_shape;
  const auto shape1 = set1_t->GetTensorShape()->GetDimSizes();

  KERNEL_HANDLE_ERROR(GroupsShapeFromInputs(shape1, shape2, group_shape), "GroupsShapeFromInputs error.");
  const std::vector<int64_t> set1_strides = Strides(shape1);
  std::map<std::vector<int64_t>, std::set<T>> group_sets;
  int64_t num_result_values = 0;
  int64_t max_set_size = 0;
  int64_t num_elements;
  KERNEL_HANDLE_ERROR(GetsNumElements(group_shape, num_elements), "NumElements error.");
  if (num_elements <= kParallelNum) {
    std::set<T> set1_group_set;
    std::set<T> set2_group_set;
    const std::vector<int64_t> subspan(order.begin(), order.end() - 1);
    auto set2_grouper = set2_st->group(subspan);
    auto set2_group_it = set2_grouper.begin();
    std::vector<int64_t> group_indices;
    for (int64_t flat_group_index = 0; flat_group_index < num_elements; ++flat_group_index) {
      PopulateGroupIndices(flat_group_index, group_shape, group_indices);

      // Get values from set1.
      PopulateFromDenseGroup<T>(set1_t, set1_strides, group_indices, set1_group_set);
      // Get values from set2, if applicable.
      set2_group_set.clear();
      if (set2_group_it != set2_grouper.end()) {
        const auto &group = *set2_group_it;
        const auto set2_group_indices = group.group();
        bool group_match = true;
        for (uint32_t i = 0; group_match && (i < set2_group_indices.size()); ++i) {
          if (set2_group_indices[i] != group_indices[i]) {
            group_match = false;
          }
        }
        if (group_match) {
          KERNEL_HANDLE_ERROR(PopulateFromSparseGroup<T>(group, shape2, set2_group_set),
                              "PopulateFromSparseGroup error.");
          ++set2_group_it;
        }
      }

      std::set<T> group_set;
      ApplySetOperation(set1_group_set, set2_group_set, group_set, databank.set_operation_);
      if (!group_set.empty()) {
        group_sets[group_indices] = group_set;
        int64_t set_size = group_set.size();
        if (set_size > max_set_size) {
          max_set_size = set_size;
        }
        num_result_values += set_size;
      }
    }
  } else {
    std::mutex mt;
    int64_t total = num_elements;
    uint32_t cores = CpuKernelUtils::GetCPUNum(*databank.ctx);
    int64_t per_unit_size = (total / std::min(std::max(1L, cores - 2L), total));
    uint32_t ret =
      CpuKernelUtils::ParallelFor(*databank.ctx, total, per_unit_size, [&](int64_t begin, int64_t end) -> uint32_t {
        std::set<T> set1_group_set;
        std::set<T> set2_group_set;
        const std::vector<int64_t> subspan(order.begin(), order.end() - 1);
        auto set2_grouper = set2_st->group(subspan);
        auto set2_group_it = set2_grouper.begin();
        std::vector<int64_t> group_indices;
        for (int64_t flat_group_index = begin; flat_group_index < end; ++flat_group_index) {
          PopulateGroupIndices(flat_group_index, group_shape, group_indices);

          // Get values from set1.
          PopulateFromDenseGroup<T>(set1_t, set1_strides, group_indices, set1_group_set);
          // Get values from set2, if applicable.
          set2_group_set.clear();
          if (set2_group_it != set2_grouper.end()) {
            const auto &group = *set2_group_it;
            const auto set2_group_indices = group.group();
            bool group_match = true;
            for (uint32_t i = 0; group_match && (i < set2_group_indices.size()); ++i) {
              if (set2_group_indices[i] != group_indices[i]) {
                group_match = false;
              }
            }
            if (group_match) {
              KERNEL_HANDLE_ERROR(PopulateFromSparseGroup<T>(group, shape2, set2_group_set),
                                  "PopulateFromSparseGroup error.");
              ++set2_group_it;
            }
          }

          std::set<T> group_set;
          ApplySetOperation(set1_group_set, set2_group_set, group_set, databank.set_operation_);
          if (!group_set.empty()) {
            std::lock_guard<std::mutex> lck(mt);
            group_sets[group_indices] = group_set;
            int64_t set_size = group_set.size();
            if (set_size > max_set_size) {
              max_set_size = set_size;
            }
            num_result_values += set_size;
          }
        }
        return KERNEL_STATUS_OK;
      });
    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_INNER_ERROR,
                       "DenseToSparseSetOperation compute failed.");
  }

  group_shape.push_back(max_set_size);
  return OutputSparseTensor<T>(databank, group_shape, num_result_values, group_sets);
}

uint32_t DenseToSparseSetOperationCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "DenseToSparseSetOperation check input and output number failed.");
  DataBank databank;
  KERNEL_HANDLE_ERROR(NullptrAndMatVecCheck(ctx, databank), "DenseToSparseSetOperation check params failed.");
  DataType dt = reinterpret_cast<DataType>(databank.set2_values->GetDataType());

  uint32_t KERNEL_STATUS;
  switch (dt) {
    case DT_INT8:
      KERNEL_STATUS = ComputeDenseToSparse<int8_t>(databank);
      break;
    case DT_UINT8:
      KERNEL_STATUS = ComputeDenseToSparse<uint8_t>(databank);
      break;
    case DT_INT16:
      KERNEL_STATUS = ComputeDenseToSparse<int16_t>(databank);
      break;
    case DT_UINT16:
      KERNEL_STATUS = ComputeDenseToSparse<uint16_t>(databank);
      break;
    case DT_INT32:
      KERNEL_STATUS = ComputeDenseToSparse<int32_t>(databank);
      break;
    case DT_INT64:
      KERNEL_STATUS = ComputeDenseToSparse<int64_t>(databank);
      break;
    case DT_STRING:
      KERNEL_STATUS = ComputeDenseToSparse<std::string>(databank);
      break;
    default:
      KERNEL_LOG_ERROR("DenseToSparseSetOperation can't support this data type [%s].", DTypeStr(dt).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (KERNEL_STATUS != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("DenseToSparseSetOperation failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kDenseToSparseSetOperation, DenseToSparseSetOperationCpuKernel);
}  // namespace aicpu
