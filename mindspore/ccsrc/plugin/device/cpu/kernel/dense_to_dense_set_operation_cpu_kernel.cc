/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/dense_to_dense_set_operation_cpu_kernel.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <functional>
#include <numeric>
#include <string>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kOutputNum = 3;
const size_t kInputNum = 2;
const size_t kInputX1 = 0;
const size_t kInputX2 = 1;
const size_t kOutput1 = 0;
const size_t kOutput2 = 1;
const size_t kOutput3 = 2;
constexpr size_t kNum2 = 2;

#define DTOD_SET_OPE_COMPUTE_CASE(DTYPE, TYPE)    \
  case (DTYPE): {                                 \
    result = LaunchKernel<TYPE>(inputs, outputs); \
    break;                                        \
  }

#define T_flat Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
#define Int64_matrix Eigen::TensorMap<Eigen::Tensor<int64_t, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
#define Int64_flat Eigen::TensorMap<Eigen::Tensor<int64_t, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>

const std::vector<size_t> GetStrides(const ShapeVector &shape) {
  std::vector<size_t> result(shape.size());
  size_t product = 1;
  for (int64_t i = SizeToLong(shape.size()) - 1; i >= 0; --i) {
    auto idx = LongToSize(i);
    result[idx] = product;
    product *= LongToSize(shape[idx]);
  }
  return result;
}

bool GroupShape(const ShapeVector &input_shape, ShapeVector *grouped_shape) {
  const size_t min_shape_size = 2;
  if (input_shape.size() < min_shape_size) {
    return false;
  }
  grouped_shape->assign(input_shape.begin(), input_shape.end() - 1);
  return true;
}

bool CheckShapesMatch(const ShapeVector &shape1, const ShapeVector &shape2) {
  if (shape1.size() != shape2.size()) {
    return false;
  }
  for (size_t i = 0; i < shape1.size(); i++) {
    if (shape1[i] != shape2[i]) {
      return false;
    }
  }
  return true;
}

void GetCommonShape(const ShapeVector &shape1, const ShapeVector &shape2, ShapeVector *group_shape) {
  ShapeVector group_shape_1;
  if (!GroupShape(shape1, &group_shape_1)) {
    MS_LOG(EXCEPTION) << "For DenseToDenseSerOperation, "
                      << "the shape rank of input x1 must be at least 2, "
                      << "but got " << shape1.size() << ".";
  }
  ShapeVector group_shape_2;
  if (!GroupShape(shape2, &group_shape_2)) {
    MS_LOG(EXCEPTION) << "For DenseToDenseSerOperation, "
                      << "the shape rank of input x2 must be at least 2, "
                      << "but got " << shape2.size() << ".";
  }
  if (!CheckShapesMatch(group_shape_1, group_shape_2)) {
    MS_LOG(EXCEPTION) << "For DenseToDenseSerOperation, "
                      << "the shapes of the first n-1 dimensions of x1 and x2 must be equal, "
                      << "but different group shapes were obtained.";
  }
  group_shape->assign(group_shape_1.begin(), group_shape_1.end());
}

void GetGroupIdx(const int64_t flat_group_index, const ShapeVector &group_shape, std::vector<size_t> *group_indices) {
  group_indices->clear();
  int64_t running_flat_group_index = flat_group_index;
  for (int64_t group_dim_index = SizeToLong(group_shape.size()) - 1; group_dim_index >= 0; --group_dim_index) {
    const auto group_dim = group_shape[LongToSize(group_dim_index)];
    (void)group_indices->insert(group_indices->begin(), running_flat_group_index % group_dim);
    running_flat_group_index /= group_dim;
  }
}
}  // namespace

template <typename T>
void GetGroupSet(const kernel::AddressPtr input, const size_t last_dim, const std::vector<size_t> &input_strides,
                 const std::vector<size_t> &group_indices, std::set<T> *result) {
  if (group_indices.size() != input_strides.size() - 1) {
    MS_LOG(EXCEPTION) << "For DenseToDenseSerOperation, "
                      << "the size of group_indices must be one less than input_strides, "
                      << "but got " << group_indices.size() << " and " << input_strides.size() << ".";
  }
  result->clear();
  auto data_ptr = static_cast<T *>(input->addr);
  const auto start = std::inner_product(group_indices.begin(), group_indices.end(), input_strides.begin(), 0UL);
  const auto end = start + last_dim;
  for (size_t i = start; i < end; ++i) {
    (void)result->insert(data_ptr[i]);
  }
}

bool DenseToDenseSetOperationCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs) {
  std::string kernel_name = base_operator->GetPrim()->name();
  std::string set_operation_str = GetValue<std::string>(base_operator->GetAttr(SET_OPERATION));
  (void)std::transform(set_operation_str.begin(), set_operation_str.end(), set_operation_str.begin(), ::tolower);
  if (set_operation_str == "a-b") {
    set_operation_ = A_MINUS_B;
  } else if (set_operation_str == "b-a") {
    set_operation_ = B_MINUS_A;
  } else if (set_operation_str == "intersection") {
    set_operation_ = INTERSECTION;
  } else if (set_operation_str == "union") {
    set_operation_ = UNION;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << ","
                      << ", the attr set_operation must be any one of "
                         "['a-b','b-a','intersection','union'], "
                      << "but got " << set_operation_str << ".";
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  is_need_retrieve_output_shape_ = true;
  return true;
}

int DenseToDenseSetOperationCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                 const std::vector<KernelTensorPtr> &inputs,
                                                 const std::vector<KernelTensorPtr> &outputs,
                                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK && ret != KRET_UNKNOWN_OUT_SHAPE) {
    return ret;
  }
  outputs_ = outputs;
  x1_shape_ = inputs[kInputX1]->GetDeviceShapeAdaptively();
  x2_shape_ = inputs[kInputX2]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

template <typename T>
void DenseToDenseSetOperationCpuKernelMod::SetCompute(const std::set<T> &set1, const std::set<T> &set2,
                                                      std::set<T> *result) {
  switch (set_operation_) {
    case A_MINUS_B:
      (void)std::set_difference(set1.begin(), set1.end(), set2.begin(), set2.end(),
                                std::inserter(*result, result->begin()));
      break;
    case B_MINUS_A:
      (void)std::set_difference(set2.begin(), set2.end(), set1.begin(), set1.end(),
                                std::inserter(*result, result->begin()));
      break;
    case INTERSECTION:
      (void)std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                                  std::inserter(*result, result->begin()));
      break;
    case UNION:
      (void)std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(*result, result->begin()));
      break;
  }
}

template <typename T>
bool DenseToDenseSetOperationCpuKernelMod::PopulateOutput(const std::vector<kernel::AddressPtr> &inputs,
                                                          const std::vector<kernel::AddressPtr> &outputs,
                                                          const ShapeVector &output_shape, const size_t num_values,
                                                          const std::map<std::vector<size_t>, std::set<T>> *sets) {
  auto out_indices_ptr = static_cast<int64_t *>(outputs[kOutput1]->addr);
  auto out_values_ptr = static_cast<T *>(outputs[kOutput2]->addr);
  auto out_shape_ptr = static_cast<int64_t *>(outputs[kOutput3]->addr);
  size_t output_shape_size = output_shape.size();
  auto num_values_signed = SizeToLong(num_values);
  auto output_shape_size_signed = SizeToLong(output_shape_size);
  real_infer_shape_ = {{num_values_signed, output_shape_size_signed}, {num_values_signed}, {output_shape_size_signed}};
  Eigen::DSizes<Eigen::DenseIndex, kNum2> out_indices_dsize(num_values, output_shape_size);
  Eigen::DSizes<Eigen::DenseIndex, 1> out_values_dsize(num_values);
  Eigen::DSizes<Eigen::DenseIndex, 1> out_shape_dsize(output_shape_size);
  Int64_matrix out_indices_mat(out_indices_ptr, out_indices_dsize);
  T_flat out_values_flat(out_values_ptr, out_values_dsize);
  Int64_flat out_shape_flat(out_shape_ptr, out_shape_dsize);
  int64_t val_idx = 0;
  for (auto it = sets->begin(); it != sets->end(); ++it) {
    const auto &group_indices = it->first;
    if (group_indices.size() != output_shape.size() - 1) {
      MS_LOG(EXCEPTION) << "For DenseToDenseSerOperation, "
                        << "the size of group_indices must be one less than output_shape, "
                        << "but got " << group_indices.size() << " and " << output_shape.size() << ".";
    }
    const auto &set = it->second;
    int64_t group_idx = 0;
    for (auto val = set.begin(); val != set.end(); ++val, ++val_idx, ++group_idx) {
      for (size_t i = 0; i < group_indices.size(); ++i) {
        out_indices_mat(val_idx, i) = SizeToLong(group_indices[i]);
      }
      out_indices_mat(val_idx, group_indices.size()) = SizeToLong(group_idx);
      out_values_flat(val_idx) = *val;
    }
  }
  for (size_t i = 0; i < output_shape_size; ++i) {
    out_shape_flat(i) = SizeToLong(output_shape[i]);
  }
  return true;
}

template <typename T>
bool DenseToDenseSetOperationCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                        const std::vector<kernel::AddressPtr> &outputs) {
  ShapeVector group_shape;
  GetCommonShape(x1_shape_, x2_shape_, &group_shape);
  const auto x1_strides = GetStrides(x1_shape_);
  const auto x2_strides = GetStrides(x2_shape_);
  std::map<std::vector<size_t>, std::set<T>> res_sets_map;
  size_t res_num = 0;
  size_t max_size = 0;
  int64_t num_elements = std::accumulate(group_shape.begin(), group_shape.end(), 1L, std::multiplies<int64_t>());
  std::set<T> x1_set;
  std::set<T> x2_set;
  std::vector<size_t> group_idxs;
  for (int64_t group_idx = 0; group_idx < num_elements; ++group_idx) {
    GetGroupIdx(group_idx, group_shape, &group_idxs);
    GetGroupSet<T>(inputs[kInputX1], LongToSize(x1_shape_.back()), x1_strides, group_idxs, &x1_set);
    GetGroupSet<T>(inputs[kInputX2], LongToSize(x2_shape_.back()), x2_strides, group_idxs, &x2_set);
    std::set<T> res_set;
    SetCompute(x1_set, x2_set, &res_set);
    if (!res_set.empty()) {
      res_sets_map[group_idxs] = res_set;
      size_t set_size = res_set.size();
      if (set_size > max_size) {
        max_size = set_size;
      }
      res_num += set_size;
    }
  }
  group_shape.push_back(max_size);
  return PopulateOutput<T>(inputs, outputs, group_shape, res_num, &res_sets_map);
}

void DenseToDenseSetOperationCpuKernelMod::SyncData() {
  for (uint32_t i = 0; i < real_infer_shape_.size(); i++) {
    outputs_[i]->SetShapeVector(real_infer_shape_[i]);
  }
}

std::vector<std::pair<KernelAttr, DenseToDenseSetOperationCpuKernelMod::DenseSetFunc>>
  DenseToDenseSetOperationCpuKernelMod::func_list_ = {{KernelAttr()
                                                         .AddInputAttr(kNumberTypeInt8)
                                                         .AddInputAttr(kNumberTypeInt8)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt8)
                                                         .AddOutputAttr(kNumberTypeInt64),
                                                       &DenseToDenseSetOperationCpuKernelMod::LaunchKernel<int8_t>},
                                                      {KernelAttr()
                                                         .AddInputAttr(kNumberTypeInt16)
                                                         .AddInputAttr(kNumberTypeInt16)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt16)
                                                         .AddOutputAttr(kNumberTypeInt64),
                                                       &DenseToDenseSetOperationCpuKernelMod::LaunchKernel<int16_t>},
                                                      {KernelAttr()
                                                         .AddInputAttr(kNumberTypeInt32)
                                                         .AddInputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt64),
                                                       &DenseToDenseSetOperationCpuKernelMod::LaunchKernel<int32_t>},
                                                      {KernelAttr()
                                                         .AddInputAttr(kNumberTypeInt64)
                                                         .AddInputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64),
                                                       &DenseToDenseSetOperationCpuKernelMod::LaunchKernel<int64_t>},
                                                      {KernelAttr()
                                                         .AddInputAttr(kNumberTypeUInt8)
                                                         .AddInputAttr(kNumberTypeUInt8)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeUInt8)
                                                         .AddOutputAttr(kNumberTypeInt64),
                                                       &DenseToDenseSetOperationCpuKernelMod::LaunchKernel<uint8_t>},
                                                      {KernelAttr()
                                                         .AddInputAttr(kNumberTypeUInt16)
                                                         .AddInputAttr(kNumberTypeUInt16)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeUInt16)
                                                         .AddOutputAttr(kNumberTypeInt64),
                                                       &DenseToDenseSetOperationCpuKernelMod::LaunchKernel<uint16_t>}};

std::vector<KernelAttr> DenseToDenseSetOperationCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, DenseToDenseSetOperationCpuKernelMod::DenseSetFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DenseToDenseSetOperation, DenseToDenseSetOperationCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
