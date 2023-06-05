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

#include "plugin/device/cpu/kernel/dense_to_sparse_set_operation_cpu_kernel.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <functional>
#include <numeric>
#include <iostream>
#include <string>
#include "utils/ms_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const uint32_t kOutputNum = 3;
const uint32_t kInputNum = 4;
const uint32_t kInputX1 = 0;
const uint32_t kInputX2Indices = 1;
const uint32_t kInputX2Values = 2;
const uint32_t kInputX2Shape = 3;
const uint32_t kOutput1 = 0;
const uint32_t kOutput2 = 1;
const uint32_t kOutput3 = 2;
const char AMinusBStr[] = "a-b";
const char BMinusAStr[] = "b-a";
const char IntersectionStr[] = "intersection";
const char UnionStr[] = "union";

#define DTOD_SET_OPE_COMPUTE_CASE(DTYPE, TYPE)    \
  case (DTYPE): {                                 \
    result = LaunchKernel<TYPE>(inputs, outputs); \
    break;                                        \
  }

#define Int64_matrix Eigen::TensorMap<Eigen::Tensor<int64_t, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
#define T_flat Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
#define Int64_flat Eigen::TensorMap<Eigen::Tensor<int64_t, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
#define Disze_rank2 Eigen::DSizes<Eigen::DenseIndex, 2>
#define Disze_rank1 Eigen::DSizes<Eigen::DenseIndex, 1>

std::vector<int64_t> Strides(ShapeVector shape) {
  std::vector<int64_t> result(shape.size());
  int64_t product = 1;
  for (int64_t i = SizeToLong(shape.size() - 1); i >= 0; --i) {
    result[i] = product;
    product *= shape[i];
  }
  return result;
}

bool GroupShape(ShapeVector input_shape, ShapeVector *const grouped_shape) {
  size_t shape_size = 2;
  if (input_shape.size() < shape_size) {
    return false;
  }
  grouped_shape->assign(input_shape.begin(), input_shape.end() - 1);
  return true;
}

bool CheckShapesMatch(ShapeVector shape1, ShapeVector shape2) {
  if (shape1.size() != shape2.size()) {
    return false;
  }
  for (size_t i = 0; i < shape1.size(); i++) {
    if (shape1[i] != shape2[i]) return false;
  }
  return true;
}

bool GroupShapeFromInputs(ShapeVector shape1, ShapeVector shape2, ShapeVector *const group_shape) {
  ShapeVector group_shape_1;
  if (!GroupShape(shape1, &group_shape_1)) {
    MS_EXCEPTION(ValueError) << "For 'DenseToSparseSetOperation', "
                             << "the rank of the shape of input 'x1' must be at least 2, "
                             << "but got " << shape1.size();
  }
  ShapeVector group_shape_2;
  if (!GroupShape(shape2, &group_shape_2)) {
    MS_EXCEPTION(ValueError) << "For 'DenseToSparseSetOperation', "
                             << "the rank of the shape of input 'x2' must be at least 2, "
                             << "but got " << shape2.size();
  }
  if (!CheckShapesMatch(group_shape_1, group_shape_2)) {
    MS_EXCEPTION(ValueError) << "For 'DenseToDenseSerOperation', "
                             << "'x1.shape[0:-1]' and 'x2.shape[0:-1]' must be equal, "
                             << "but different group shapes were obtained";
  }
  group_shape->assign(group_shape_1.begin(), group_shape_1.end());
  return true;
}

// Split `flat_group_index` into separate dimensions based on `group_shape`.
void PopulateGroupIndices(int64_t flat_group_index, ShapeVector group_shape, ShapeVector *group_indices) {
  group_indices->clear();
  int64_t running_flat_group_index = flat_group_index;
  for (int64_t group_dim_index = SizeToLong(group_shape.size() - 1); group_dim_index >= 0; --group_dim_index) {
    const auto group_dim = group_shape[group_dim_index];
    (void)group_indices->insert(group_indices->begin(), running_flat_group_index % group_dim);
    running_flat_group_index /= group_dim;
  }
}
}  // namespace

template <typename T>
bool PopulateFromDenseGroup(kernel::AddressPtr input, int64_t last_dim, std::vector<int64_t> input_strides,
                            ShapeVector group_indices, std::set<T> *result) {
  if (group_indices.size() != input_strides.size() - 1) {
    MS_EXCEPTION(ValueError) << "For 'DenseToSparseSetOperation', "
                             << "the size of 'group_indices' must be one less than input_strides"
                             << "but got " << group_indices.size() << " and " << input_strides.size() << ".";
  }
  // "for DenseToSparseSetOperation, group_indices size must be equal to input_strides.size-1 ");
  result->clear();
  auto data_ptr = static_cast<T *>(input->addr);
  Disze_rank1 dsize(input->size);
  T_flat input_flat(data_ptr, dsize);
  const auto start = std::inner_product(group_indices.begin(), group_indices.end(), input_strides.begin(), 0L);
  const auto end = start + last_dim;
  for (int64_t i = start; i < end; ++i) {
    result->insert(input_flat(i));
  }
  return true;
}

template <typename T>
bool PopulateFromSparse(kernel::AddressPtr input, int64_t start, int64_t last_dim, std::set<T> *result) {
  result->clear();
  auto data_ptr = static_cast<T *>(input->addr);
  Disze_rank1 dsize(input->size);
  T_flat input_flat(data_ptr, dsize);
  auto end = start + last_dim;
  for (int64_t i = start; i < end; ++i) {
    result->insert(input_flat(i));
  }
  return true;
}

bool pred(const int64_t a, const int64_t b) {
  if (a == b) return true;
  return false;
}

bool SparseStride(kernel::AddressPtr indices, int64_t num_elements, ShapeVector group_shape,
                  std::vector<int64_t> *result, int64_t set2_nums, int64_t set2_dim) {
  ShapeVector group_indices;
  auto indices_ptr = static_cast<int64_t *>(indices->addr);
  Int64_matrix indices_flat(indices_ptr, set2_nums, set2_dim);
  result->clear();
  int64_t i = 0;
  int64_t j = 0;
  int64_t num = 0;
  while (i < num_elements && j < (SizeToLong(indices_flat.size()))) {
    PopulateGroupIndices(i, group_shape, &group_indices);
    if (std::equal(group_indices.begin(), group_indices.end(), &indices_flat(j, 0), pred)) {
      num++;
      j++;
    } else {
      result->push_back(num);
      num = 0;
      i++;
    }
  }
  while (i < num_elements) {
    result->push_back(0);
    i++;
  }
  return true;
}

void ValidateIndices(kernel::AddressPtr indices, ShapeVector shape2, int64_t set2_nums, int64_t set2_dim,
                     int64_t shape2_size) {
  auto indices_ptr = static_cast<int64_t *>(indices->addr);
  Int64_matrix indices_flat(indices_ptr, set2_nums, set2_dim);
  for (int64_t i = 0; i < set2_nums; i++) {
    for (int64_t j = 0; j < shape2_size; j++) {
      if ((indices_flat(i, j) >= shape2[j]) | (indices_flat(i, j) < 0)) {
        MS_EXCEPTION(RuntimeError) << "For 'DenseToSparseSetOperation', "
                                   << "the dimension " << j << " of values in 'x2_indices' must be in range(0, "
                                   << (shape2[j] - 1) << "), but got " << indices_flat(i, j) << ".";
      }
    }
  }

  for (int64_t i = 1; i < set2_nums; i++) {
    for (int64_t j = 0; j < shape2_size; j++) {
      if (indices_flat(i, j) < indices_flat(i - 1, j)) {
        MS_EXCEPTION(RuntimeError) << "For 'DenseToSparseSetOperation', "
                                   << "the indices of set2 must be sequential.";
      } else if (indices_flat(i, j) == indices_flat(i - 1, j)) {
        if (j == (shape2_size - 1)) {
          MS_EXCEPTION(RuntimeError) << "For 'DenseToSparseSetOperation', "
                                     << "the indices of set2 must be sequential.";
        }
      } else if (indices_flat(i, j) > indices_flat(i - 1, j)) {
        break;
      }
    }
  }
}

bool DenseToSparseSetOperationCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &workspace,
                                                   const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);

  bool result = false;
  switch (data_type_) {
    DTOD_SET_OPE_COMPUTE_CASE(kNumberTypeInt8, int8_t)
    DTOD_SET_OPE_COMPUTE_CASE(kNumberTypeInt16, int16_t)
    DTOD_SET_OPE_COMPUTE_CASE(kNumberTypeInt32, int32_t)
    DTOD_SET_OPE_COMPUTE_CASE(kNumberTypeInt64, int64_t)
    DTOD_SET_OPE_COMPUTE_CASE(kNumberTypeUInt8, uint8_t)
    DTOD_SET_OPE_COMPUTE_CASE(kNumberTypeUInt16, uint16_t)
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of input " << TypeIdToType(data_type_)->ToString()
                        << " not support.";
  }
  return result;
}

template <typename T>
void DenseToSparseSetOperationCpuKernelMod::ApplySetOperation(const std::set<T> &set1, const std::set<T> &set2,
                                                              std::set<T> *result) {
  switch (set_operation_) {
    case A_MINUS_B:
      std::set_difference(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(*result, result->begin()));
      break;
    case B_MINUS_A:
      std::set_difference(set2.begin(), set2.end(), set1.begin(), set1.end(), std::inserter(*result, result->begin()));
      break;
    case INTERSECTION:
      std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                            std::inserter(*result, result->begin()));
      break;
    case UNION:
      std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(*result, result->begin()));
      break;
  }
}

void DenseToSparseSetOperationCpuKernelMod::SyncOutputShape() {
  outputs_[kOutput1]->SetShapeVector(infer_shape_[kOutput1]);
  outputs_[kOutput2]->SetShapeVector(infer_shape_[kOutput2]);
  outputs_[kOutput3]->SetShapeVector(infer_shape_[kOutput3]);
}
template <typename T>
bool DenseToSparseSetOperationCpuKernelMod::OutputSparseTensor(const std::vector<kernel::AddressPtr> &inputs,
                                                               const std::vector<kernel::AddressPtr> &outputs,
                                                               ShapeVector *output_shape, const int64_t num_values,
                                                               const std::map<ShapeVector, std::set<T>> &sets) {
  auto out_indices_ptr = reinterpret_cast<int64_t *>(outputs[kOutput1]->addr);
  auto out_values_ptr = reinterpret_cast<T *>(outputs[kOutput2]->addr);
  auto out_shape_ptr = reinterpret_cast<int64_t *>(outputs[kOutput3]->addr);

  int64_t output_shape_size = SizeToLong(output_shape->size());
  infer_shape_ = {{num_values, output_shape_size}, {num_values}, {output_shape_size}};

  Disze_rank2 out_indices_dsize(num_values, output_shape_size);
  Disze_rank1 out_values_dsize(num_values);
  Disze_rank1 out_shape_dsize(output_shape_size);
  Int64_matrix out_indices_mat(out_indices_ptr, out_indices_dsize);
  T_flat out_values_flat(out_values_ptr, out_values_dsize);
  Int64_flat out_shape_flat(out_shape_ptr, out_shape_dsize);

  // For each set, write its indices and values to output tensors.
  int64_t value_index = 0;
  for (auto it = sets.begin(); it != sets.end(); ++it) {
    const auto &group_indices = it->first;
    if (group_indices.size() != output_shape->size() - 1) {
      MS_EXCEPTION(ValueError) << "For 'DenseToSparseSetOperation', "
                               << "the size of 'output_groups' must be one less than 'output_shape'"
                               << "but got " << group_indices.size() << " and " << output_shape->size() << ".";
    }
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
    out_shape_flat(i) = (*output_shape)[i];
  }
  return true;
}

int DenseToSparseSetOperationCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                  const std::vector<KernelTensorPtr> &inputs,
                                                  const std::vector<KernelTensorPtr> &outputs,
                                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret == KRET_UNKNOWN_OUT_SHAPE) {
    shape1_ = inputs.at(kInputX1)->GetShapeVector();
    set2_nums_ = SizeToLong(inputs.at(kInputX2Values)->GetShapeVector()[0]);
    set2_dim_ = SizeToLong(shape1_.size());
  }
  return KRET_OK;
}

template <typename T>
bool DenseToSparseSetOperationCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                         const std::vector<kernel::AddressPtr> &outputs) {
  ShapeVector group_shape;
  auto shape2_ptr = static_cast<int64_t *>(inputs[kInputX2Shape]->addr);
  auto shape2_size = (inputs[kInputX2Shape]->size) / sizeof(int64_t);
  ShapeVector shape2(shape2_ptr, shape2_ptr + shape2_size);

  if (validate_indices_) {
    ValidateIndices(inputs[kInputX2Indices], shape2, set2_nums_, set2_dim_, shape2_size);
  }
  if (!GroupShapeFromInputs(shape1_, shape2, &group_shape)) {
    MS_EXCEPTION(ValueError) << "For 'DenseToDenseSerOperation', "
                             << "Create group shape error.";
  }

  auto set1_strides = Strides(shape1_);
  std::map<ShapeVector, std::set<T>> group_sets;
  int64_t num_result_values = 0;
  int64_t max_set_size = 0;

  int64_t num_elements = std::accumulate(group_shape.begin(), group_shape.end(), 1L, std::multiplies<int64_t>());
  std::vector<int64_t> set2_set_sizes;
  ValidateIndices(inputs[kInputX2Indices], shape2, set2_nums_, set2_dim_, shape2_size);
  if (!SparseStride(inputs[kInputX2Indices], num_elements, group_shape, &set2_set_sizes, set2_nums_, set2_dim_)) {
    MS_EXCEPTION(ValueError) << "For 'DenseToDenseSerOperation', "
                             << "sparseStride compute failed.";
  }
  std::set<T> set1_group_set;
  std::set<T> set2_group_set;
  int64_t set2_start = 0;
  ShapeVector group_indices;
  for (int64_t flat_group_index = 0; flat_group_index < num_elements; ++flat_group_index) {
    PopulateGroupIndices(flat_group_index, group_shape, &group_indices);
    if (!PopulateFromDenseGroup<T>(inputs[kInputX1], shape1_.back(), set1_strides, group_indices, &set1_group_set)) {
      MS_EXCEPTION(ValueError) << "For 'DenseToDenseSerOperation', "
                               << "populateFromDenseGroup set1 compute failed.";
    }
    if (!PopulateFromSparse(inputs[kInputX2Values], set2_start, set2_set_sizes[flat_group_index], &set2_group_set)) {
      MS_EXCEPTION(ValueError) << "For 'DenseToDenseSerOperation', "
                               << "populateFromSparse set2 compute failed.";
    }

    std::set<T> group_set;
    ApplySetOperation(set1_group_set, set2_group_set, &group_set);
    if (!group_set.empty()) {
      group_sets[group_indices] = group_set;
      int64_t set_size = SizeToLong(group_set.size());
      if (set_size > max_set_size) {
        max_set_size = set_size;
      }
      num_result_values += set_size;
    }
    set2_start += set2_set_sizes[flat_group_index];
  }

  group_shape.push_back(max_set_size);
  return OutputSparseTensor<T>(inputs, outputs, &group_shape, num_result_values, group_sets);
}

bool DenseToSparseSetOperationCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                 const std::vector<KernelTensorPtr> &inputs,
                                                 const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  kernel_name_ = base_operator->name();

  std::string set_operation_str = GetValue<std::string>(prim->GetAttr("set_operation"));

  (void)std::transform(set_operation_str.begin(), set_operation_str.end(), set_operation_str.begin(), ::tolower);
  validate_indices_ = GetValue<bool>(prim->GetAttr("validate_indices"));

  if (set_operation_str == AMinusBStr) {
    set_operation_ = A_MINUS_B;
  } else if (set_operation_str == BMinusAStr) {
    set_operation_ = B_MINUS_A;
  } else if (set_operation_str == IntersectionStr) {
    set_operation_ = INTERSECTION;
  } else if (set_operation_str == UnionStr) {
    set_operation_ = UNION;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "," << set_operation_str << " is an invalid 'set_operation'.";
  }
  data_type_ = inputs[kInputX1]->GetDtype();
  is_need_retrieve_output_shape_ = true;
  return true;
}

std::vector<KernelAttr> DenseToSparseSetOperationCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {KernelAttr()
                                                       .AddInputAttr(kNumberTypeInt8)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeInt8)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt8)
                                                       .AddOutputAttr(kNumberTypeInt64),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeInt16)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeInt16)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt16)
                                                       .AddOutputAttr(kNumberTypeInt64),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeInt32)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt32)
                                                       .AddOutputAttr(kNumberTypeInt64),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt64),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeUInt8)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeUInt8)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeUInt8)
                                                       .AddOutputAttr(kNumberTypeInt64),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeUInt16)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddInputAttr(kNumberTypeUInt16)
                                                       .AddInputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeInt64)
                                                       .AddOutputAttr(kNumberTypeUInt16)
                                                       .AddOutputAttr(kNumberTypeInt64)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DenseToSparseSetOperation, DenseToSparseSetOperationCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
