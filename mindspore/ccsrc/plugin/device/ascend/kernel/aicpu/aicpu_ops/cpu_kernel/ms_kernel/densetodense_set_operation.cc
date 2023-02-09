/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

#include "densetodense_set_operation.h"

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
const uint32_t kOutputNum = 3;
const uint32_t kInputNum = 2;
const uint32_t kOutputIndex0 = 0;
const uint32_t kOutputIndex1 = 1;
const uint32_t kOutputIndex2 = 2;
const char *kDenseToDenseSetOperation = "DenseToDenseSetOperation";
const int64_t kParallelNum{512};

#define DTOD_SET_OPE_COMPUTE_CASE(DTYPE, TYPE, CTX)                        \
  case (DTYPE): {                                                          \
    uint32_t result = DoCompute<TYPE>(CTX);                                \
    if (result != KERNEL_STATUS_OK) {                                      \
      KERNEL_LOG_ERROR("DenseToDenseSetOperation kernel compute failed."); \
      return result;                                                       \
    }                                                                      \
    break;                                                                 \
  }

const std::vector<int64_t> Strides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> result(shape.size());
  int64_t product = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    result[i] = product;
    product *= shape[i];
  }
  return result;
}
}  // namespace

namespace aicpu {
uint32_t GetNumElements(const std::vector<int64_t> &input_shape, int64_t &res) {
  int64_t result = 1;
  for (size_t i = 0; i < input_shape.size(); i++) {
    KERNEL_CHECK_FALSE(MulWithoutOverflow(input_shape[i], result, result), KERNEL_STATUS_PARAM_INVALID,
                       "Overflow when calculate shape size");
  }
  res = result;
  return KERNEL_STATUS_OK;
}

uint32_t GroupShape(const std::vector<int64_t> &input_shape, std::vector<int64_t> &grouped_shape) {
  if (input_shape.size() < 2) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // grouped_shape is input_shape[:-1]
  grouped_shape.assign(input_shape.begin(), input_shape.end() - 1);
  return KERNEL_STATUS_OK;
}

uint32_t ChecksShapesMatch(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2) {
  if (shape1.size() != shape2.size()) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t i = 0; i < shape1.size(); i++) {
    if (shape1[i] != shape2[i]) return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t GroupShapeFromDenseInputs(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2,
                                   std::vector<int64_t> &group_shape) {
  std::vector<int64_t> group_shape_1;
  KERNEL_HANDLE_ERROR(GroupShape(shape1, group_shape_1), "Shape rank is less than 2");
  std::vector<int64_t> group_shape_2;
  KERNEL_HANDLE_ERROR(GroupShape(shape2, group_shape_2), "Shape rank is less than 2");
  KERNEL_HANDLE_ERROR(ChecksShapesMatch(group_shape_1, group_shape_2), "Two shapes mismatch with each other.");
  group_shape.assign(group_shape_1.begin(), group_shape_1.end());
  return KERNEL_STATUS_OK;
}

// Split `flat_group_index` into separate dimensions based on `group_shape`.
void PopulatesGroupIndices(const int64_t flat_group_index, const std::vector<int64_t> &group_shape,
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
uint32_t PopulateFromDenseGroup(Tensor *input_tensor, const std::vector<int64_t> &input_strides,
                                const std::vector<int64_t> &group_indices, std::set<T> &result) {
  KERNEL_CHECK_FALSE(group_indices.size() == input_strides.size() - 1, KERNEL_STATUS_PARAM_INVALID,
                     "group_indices size is not equal to  input_strides.size-1 ")
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

uint32_t DenseToDenseSetOperationCpuKernel::Check(const CpuKernelContext &ctx) {
  AttrValue *set_operation = ctx.GetAttr("set_operation");
  std::string set_operation_str;
  if (set_operation != nullptr) {
    set_operation_str = set_operation->GetString();
  }
  std::transform(set_operation_str.begin(), set_operation_str.end(), set_operation_str.begin(), ::tolower);
  if ("a-b" == set_operation_str) {
    set_operation_ = A_MINUS_B;
  } else if ("b-a" == set_operation_str) {
    set_operation_ = B_MINUS_A;
  } else if ("intersection" == set_operation_str) {
    set_operation_ = INTERSECTION;
  } else if ("union" == set_operation_str) {
    set_operation_ = UNION;
  } else {
    KERNEL_LOG_ERROR("Invalid set_operation");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  AttrValue *validate_indices = ctx.GetAttr("validate_indices");
  if (validate_indices != nullptr) {
    validate_indices_ = validate_indices->GetBool();
  }
  return KERNEL_STATUS_OK;
}

uint32_t DenseToDenseSetOperationCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "DenseToDenseSetOperation check input and output number failed.");
  KERNEL_HANDLE_ERROR(Check(ctx), "DenseToDenseSetOperation check params failed.");
  auto data_type_x1 = ctx.Input(0)->GetDataType();
  auto data_type_x2 = ctx.Input(1)->GetDataType();
  KERNEL_CHECK_FALSE(data_type_x1 == data_type_x2, KERNEL_STATUS_PARAM_INVALID,
                     "The type of x1 must be the same as x2");
  switch (data_type_x1) {
    DTOD_SET_OPE_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    DTOD_SET_OPE_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    DTOD_SET_OPE_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    DTOD_SET_OPE_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    DTOD_SET_OPE_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    DTOD_SET_OPE_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    default:
      KERNEL_LOG_ERROR("DenseToDenseSetOperation kernel data type [%s] not support.", DTypeStr(data_type_x1).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void DenseToDenseSetOperationCpuKernel::ApplySetOperation(const std::set<T> &set1, const std::set<T> &set2,
                                                          std::set<T> &result) {
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
uint32_t DenseToDenseSetOperationCpuKernel::OutputSparseTensor(
  CpuKernelContext &ctx, const std::vector<int64_t> &output_shape, const int64_t num_values,
  const std::map<std::vector<int64_t>, std::set<T>> &sets) {
  Tensor *out_indices_t, *out_values_t, *out_shape_t;

  out_indices_t = ctx.Output(kOutputIndex0);
  out_values_t = ctx.Output(kOutputIndex1);
  out_shape_t = ctx.Output(kOutputIndex2);

  auto out_indices_shape = out_indices_t->GetTensorShape();
  auto out_values_shape = out_values_t->GetTensorShape();
  auto out_shape_shape = out_shape_t->GetTensorShape();

  int64_t output_shape_size = output_shape.size();

  out_indices_shape->SetDimSizes({num_values, output_shape_size});
  out_values_shape->SetDimSizes({num_values});
  out_shape_shape->SetDimSizes({output_shape_size});

  EigenTensor out_indices_tensor(out_indices_t, out_indices_t->GetData());
  EigenTensor out_values_tensor(out_values_t, out_values_t->GetData());
  EigenTensor out_shape_tensor(out_shape_t, out_shape_t->GetData());

  auto out_indices_mat = out_indices_tensor.matrix<int64_t>();
  auto out_values_flat = out_values_tensor.flat<T>();
  auto out_shape_flat = out_shape_tensor.flat<int64_t>();

  // For each set, write its indices and values to output tensors.
  int64_t value_index = 0;
  for (auto it = sets.begin(); it != sets.end(); ++it) {
    const auto &group_indices = it->first;
    KERNEL_CHECK_FALSE(group_indices.size() == output_shape.size() - 1, KERNEL_STATUS_PARAM_INVALID,
                       "Invalid number of indices .")
    const auto &set = it->second;

    int64_t group_value_index = 0;
    for (auto value = set.begin(); value != set.end(); ++value, ++value_index, ++group_value_index) {
      // First n-1 dimensions are the group, last dimension is the position in
      // the set.
      for (size_t i = 0; i < group_indices.size(); ++i) {
        out_indices_mat(value_index, i) = group_indices[i];
      }
      out_indices_mat(value_index, group_indices.size()) = group_value_index;

      out_values_flat(value_index) = *value;
    }
  }

  for (int64_t i = 0; i < output_shape_size; ++i) {
    out_shape_flat(i) = output_shape[i];
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t DenseToDenseSetOperationCpuKernel::DoCompute(CpuKernelContext &ctx) {
  Tensor *set1_t = ctx.Input(0);
  Tensor *set2_t = ctx.Input(1);
  std::vector<int64_t> group_shape;
  const auto shape1 = set1_t->GetTensorShape()->GetDimSizes();
  const auto shape2 = set2_t->GetTensorShape()->GetDimSizes();
  KERNEL_HANDLE_ERROR(GroupShapeFromDenseInputs(shape1, shape2, group_shape), "Create group shape error.");

  const auto set1_strides = Strides(shape1);
  const auto set2_strides = Strides(shape2);

  std::map<std::vector<int64_t>, std::set<T>> group_sets;
  std::atomic<int64_t> num_result_values(0);
  std::atomic<int64_t> max_set_size(0);

  int64_t num_elements;
  KERNEL_HANDLE_ERROR(GetNumElements(group_shape, num_elements), "Get numelements failed.");

  if (num_elements <= kParallelNum) {
    std::set<T> set1_group_set;
    std::set<T> set2_group_set;
    std::vector<int64_t> group_indices;
    for (int64_t flat_group_index = 0; flat_group_index < num_elements; ++flat_group_index) {
      PopulatesGroupIndices(flat_group_index, group_shape, group_indices);
      KERNEL_HANDLE_ERROR(PopulateFromDenseGroup<T>(set1_t, set1_strides, group_indices, set1_group_set),
                          "PopulateFromDenseGroup set1 compute failed");
      KERNEL_HANDLE_ERROR(PopulateFromDenseGroup<T>(set2_t, set2_strides, group_indices, set2_group_set),
                          "PopulateFromDenseGroup set2 compute failed");

      std::set<T> group_set;
      ApplySetOperation(set1_group_set, set2_group_set, group_set);
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
    uint32_t cores = CpuKernelUtils::GetCPUNum(ctx);
    int64_t per_unit_size = (total / std::min(std::max(1L, cores - 2L), total));
    uint32_t ret = CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, [&](int64_t begin, int64_t end) {
      std::set<T> set1_group_set;
      std::set<T> set2_group_set;
      std::vector<int64_t> group_indices;
      for (int64_t flat_group_index = begin; flat_group_index < end; ++flat_group_index) {
        PopulatesGroupIndices(flat_group_index, group_shape, group_indices);
        KERNEL_HANDLE_ERROR(PopulateFromDenseGroup<T>(set1_t, set1_strides, group_indices, set1_group_set),
                            "PopulateFromDenseGroup set1 compute failed");
        KERNEL_HANDLE_ERROR(PopulateFromDenseGroup<T>(set2_t, set2_strides, group_indices, set2_group_set),
                            "PopulateFromDenseGroup set2 compute failed");
        std::set<T> group_set;
        ApplySetOperation(set1_group_set, set2_group_set, group_set);
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
      return static_cast<uint32_t>(KERNEL_STATUS_OK);
    });
    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_INNER_ERROR, "SparseSplit compute failed.");
  }

  group_shape.push_back(max_set_size);
  return OutputSparseTensor<T>(ctx, group_shape, num_result_values, group_sets);
}

REGISTER_CPU_KERNEL(kDenseToDenseSetOperation, DenseToDenseSetOperationCpuKernel);
}  // namespace aicpu