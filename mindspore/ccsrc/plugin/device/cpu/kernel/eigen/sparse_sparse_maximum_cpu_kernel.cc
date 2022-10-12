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
#include "plugin/device/cpu/kernel/eigen/sparse_sparse_maximum_cpu_kernel.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <utility>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 6;
constexpr size_t kOutputsNum = 2;
constexpr size_t kMatrixDimNum = 2;
constexpr size_t kVectorDimNum = 1;
const uint32_t kInputa_indices = 0;
const uint32_t kInputa_values = 1;
const uint32_t kInputa_shapes = 2;
const uint32_t kInputb_indices = 3;
const uint32_t kInputb_values = 4;
const uint32_t kInputb_shapes = 5;
const uint32_t kOutput_indices = 0;
const uint32_t kOutput_values = 1;

#define SPARSE_SPARSE_MAXIMUM_COMPUTE_CASE(DTYPE, TYPE) \
  case (DTYPE): {                                       \
    ret = LaunchKernel<TYPE>(inputs, outputs);          \
    break;                                              \
  }

#define EIGEN_SHAPE_CAST(INPUT) static_cast<Eigen::DenseIndex>(AnfAlgo::GetInputDeviceShape(node_ptr, INPUT)[0])

inline static int cmp(
  const Eigen::TensorMap<Eigen::Tensor<int64_t, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> &a_idx,
  const Eigen::TensorMap<Eigen::Tensor<int64_t, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> &b_idx,
  const int64_t a_row, const int64_t b_row, const int dims) {
  for (int d = 0; d < dims; ++d) {
    const int64_t a = a_idx(a_row, d);
    const int64_t b = b_idx(b_row, d);
    if (a < b) {
      return -1;
    } else if (a > b) {
      return 1;
    }
  }
  return 0;
}

template <typename T>
void UnionSparseIndicesAndValues(
  const Eigen::TensorMap<Eigen::Tensor<int64_t, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> a_indices_mat,
  const Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> a_values,
  int64_t a_nnz,
  const Eigen::TensorMap<Eigen::Tensor<int64_t, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> b_indices_mat,
  const Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> b_values,
  int64_t b_nnz, int num_dims, std::vector<T> *a_augmented_values, std::vector<T> *b_augmented_values,
  std::vector<std::pair<bool, int64_t>> *entries_to_copy) {
  entries_to_copy->reserve(a_nnz + b_nnz);
  a_augmented_values->reserve(a_nnz);
  b_augmented_values->reserve(b_nnz);
  int64_t i = 0, j = 0;
  const T kZero = static_cast<T>(0);
  while (i < a_nnz && j < b_nnz) {
    switch (cmp(a_indices_mat, b_indices_mat, i, j, num_dims)) {
      case -1:
        (void)entries_to_copy->emplace_back(true, i);
        a_augmented_values->push_back(a_values(i));
        b_augmented_values->push_back(kZero);
        ++i;
        break;
      case 0:
        (void)entries_to_copy->emplace_back(true, i);
        a_augmented_values->push_back(a_values(i));
        b_augmented_values->push_back(b_values(j));
        ++i;
        ++j;
        break;
      case 1:
        (void)entries_to_copy->emplace_back(false, j);
        a_augmented_values->push_back(kZero);
        b_augmented_values->push_back(b_values(j));
        ++j;
        break;
    }
  }
  // Handles leftovers; at most one loop runs.
  while (i < a_nnz) {
    (void)entries_to_copy->emplace_back(true, i);
    a_augmented_values->push_back(a_values(i++));
    b_augmented_values->push_back(kZero);
  }
  while (j < b_nnz) {
    (void)entries_to_copy->emplace_back(false, j);
    a_augmented_values->push_back(kZero);
    b_augmented_values->push_back(b_values(j++));
  }
}
}  // namespace

void SparseSparseMaximumCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  node_ptr = kernel_node;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kInputsNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kOutputsNum, kernel_name_);
}

bool SparseSparseMaximumCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  bool ret = false;
  auto data_type = AnfAlgo::GetInputDeviceDataType(node_ptr, kInputa_values);
  switch (data_type) {
    SPARSE_SPARSE_MAXIMUM_COMPUTE_CASE(kNumberTypeInt8, int8_t)
    SPARSE_SPARSE_MAXIMUM_COMPUTE_CASE(kNumberTypeInt16, int16_t)
    SPARSE_SPARSE_MAXIMUM_COMPUTE_CASE(kNumberTypeInt32, int32_t)
    SPARSE_SPARSE_MAXIMUM_COMPUTE_CASE(kNumberTypeInt64, int64_t)
    SPARSE_SPARSE_MAXIMUM_COMPUTE_CASE(kNumberTypeUInt8, uint8_t)
    SPARSE_SPARSE_MAXIMUM_COMPUTE_CASE(kNumberTypeUInt16, uint16_t)
    SPARSE_SPARSE_MAXIMUM_COMPUTE_CASE(kNumberTypeFloat16, Eigen::half)
    SPARSE_SPARSE_MAXIMUM_COMPUTE_CASE(kNumberTypeFloat32, float)
    SPARSE_SPARSE_MAXIMUM_COMPUTE_CASE(kNumberTypeFloat64, double)
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', Unsupported input data type: " << data_type << ".";
  }
  return ret;
}

void SparseSparseMaximumCpuKernelMod::CheckInputShape(const std::vector<kernel::AddressPtr> &inputs,
                                                      const int64_t a_nnz, const int64_t b_nnz,
                                                      const int64_t num_dims) {
  const int64_t a_values_shape0 = AnfAlgo::GetInputDeviceShape(node_ptr, kInputa_values)[0];
  const int64_t b_values_shape0 = AnfAlgo::GetInputDeviceShape(node_ptr, kInputb_values)[0];
  const int64_t a_shapes_shape0 = AnfAlgo::GetInputDeviceShape(node_ptr, kInputa_shapes)[0];
  const int64_t b_shapes_shape0 = AnfAlgo::GetInputDeviceShape(node_ptr, kInputb_shapes)[0];
  const int64_t b_indices_shape1 = AnfAlgo::GetInputDeviceShape(node_ptr, kInputb_indices)[1];
  auto a_shape_addr = reinterpret_cast<int64_t *>(inputs[kInputa_shapes]->addr);
  auto b_shape_addr = reinterpret_cast<int64_t *>(inputs[kInputb_shapes]->addr);
  auto a_dtype = AnfAlgo::GetInputDeviceDataType(node_ptr, kInputa_values);
  auto b_dtype = AnfAlgo::GetInputDeviceDataType(node_ptr, kInputb_values);
  if (a_values_shape0 != a_nnz) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', x1_values.shape[0] should be same to x1_indices.shape[0], but got values size: "
                      << a_values_shape0 << ", and " << a_nnz;
  }
  if (b_values_shape0 != b_nnz) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', x2_values.shape[0] should be same to x2_indices.shape[0], but got values size: "
                      << b_values_shape0 << ", and " << b_nnz;
  }
  if (num_dims <= 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', Tensors must not be empty.";
  }
  if (b_indices_shape1 != num_dims) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', b_indices.shape[1] and a_indices.shape[1] must match, but got values size: "
                      << b_indices_shape1 << ", and " << num_dims;
  }
  if (a_shapes_shape0 != num_dims) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', a_indices.shape[1] and a_shape.shape[0] must match, but got values size: " << num_dims
                      << ", and " << a_shapes_shape0;
  }
  if (a_shapes_shape0 != b_shapes_shape0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', operands do not have the same ranks, got ranks: " << a_shapes_shape0 << ", and "
                      << b_shapes_shape0;
  }
  for (int64_t i = 0; i < num_dims; ++i) {
    if (a_shape_addr[i] != b_shape_addr[i]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', operand's shapes do not match at index " << i
                        << ", got value: " << a_shapes_shape0 << ", and " << b_shapes_shape0;
    }
  }
  if (a_dtype != b_dtype) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', The type of input a must be the same as b, got ranks: " << a_dtype << ", and " << b_dtype;
  }
}

template <typename T>
bool SparseSparseMaximumCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                   const std::vector<kernel::AddressPtr> &outputs) {
  const auto shape = AnfAlgo::GetInputDeviceShape(node_ptr, kInputa_indices);
  const int64_t a_nnz = shape[0];
  const int64_t num_dims = shape[1];
  const int64_t b_nnz = AnfAlgo::GetInputDeviceShape(node_ptr, kInputb_indices)[0];
  auto output_indices_dtype = AnfAlgo::GetInputDeviceDataType(node_ptr, kInputa_indices);
  auto output_values_dtype = AnfAlgo::GetInputDeviceDataType(node_ptr, kInputa_values);
  CheckInputShape(inputs, a_nnz, b_nnz, num_dims);
  auto a_values_ptr = reinterpret_cast<T *>(inputs[kInputa_values]->addr);
  Eigen::DSizes<Eigen::DenseIndex, 1> a_values_size(EIGEN_SHAPE_CAST(kInputa_values));
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> a_values(a_values_ptr,
                                                                                                     a_values_size);
  auto b_values_ptr = reinterpret_cast<T *>(inputs[kInputb_values]->addr);
  Eigen::DSizes<Eigen::DenseIndex, 1> b_values_size(EIGEN_SHAPE_CAST(kInputb_values));
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> b_values(b_values_ptr,
                                                                                                     b_values_size);
  auto a_indices_ptr = reinterpret_cast<int64_t *>(inputs[kInputa_indices]->addr);
  Eigen::DSizes<Eigen::DenseIndex, kIndex2> a_indices_size(EIGEN_SHAPE_CAST(kInputa_values),
                                                           EIGEN_SHAPE_CAST(kInputa_shapes));
  Eigen::TensorMap<Eigen::Tensor<int64_t, kIndex2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> a_indices_mat(
    a_indices_ptr, a_indices_size);
  auto b_indices_ptr = reinterpret_cast<int64_t *>(inputs[kInputb_indices]->addr);
  Eigen::DSizes<Eigen::DenseIndex, kIndex2> b_indices_size(EIGEN_SHAPE_CAST(kInputb_values),
                                                           EIGEN_SHAPE_CAST(kInputb_shapes));
  Eigen::TensorMap<Eigen::Tensor<int64_t, kIndex2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> b_indices_mat(
    b_indices_ptr, b_indices_size);
  auto a_shape_ptr = reinterpret_cast<int64_t *>(inputs[kInputa_shapes]->addr);
  Eigen::DSizes<Eigen::DenseIndex, 1> a_shape_size(EIGEN_SHAPE_CAST(kInputa_shapes));
  Eigen::TensorMap<Eigen::Tensor<int64_t, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> a_shape(a_shape_ptr,
                                                                                                          a_shape_size);
  auto b_shape_ptr = reinterpret_cast<int64_t *>(inputs[kInputb_shapes]->addr);
  Eigen::DSizes<Eigen::DenseIndex, 1> b_shape_size(EIGEN_SHAPE_CAST(kInputb_shapes));
  Eigen::TensorMap<Eigen::Tensor<int64_t, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> b_shape(b_shape_ptr,
                                                                                                          b_shape_size);

  std::vector<T> a_augmented_values, b_augmented_values;
  std::vector<std::pair<bool, int64_t>> entries_to_copy;  // from_a?, idx
  UnionSparseIndicesAndValues(a_indices_mat, a_values, a_nnz, b_indices_mat, b_values, b_nnz, num_dims,
                              &a_augmented_values, &b_augmented_values, &entries_to_copy);
  const int64_t sum_nnz = SizeToLong(a_augmented_values.size());

  auto output_indices_ptr = reinterpret_cast<int64_t *>(outputs[kOutput_indices]->addr);
  Eigen::DSizes<Eigen::DenseIndex, kIndex2> output_indices_size(static_cast<Eigen::DenseIndex>(sum_nnz),
                                                                static_cast<Eigen::DenseIndex>(num_dims));
  Eigen::TensorMap<Eigen::Tensor<int64_t, kIndex2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    output_indices_mat(output_indices_ptr, output_indices_size);

  for (int64_t i = 0; i < sum_nnz; ++i) {
    const bool from_a = entries_to_copy[i].first;
    const int64_t idx = entries_to_copy[i].second;
    output_indices_mat.chip<0>(i) = from_a ? a_indices_mat.chip<0>(idx) : b_indices_mat.chip<0>(idx);
  }

  using UnalignedTensorMap = Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>, Eigen::Unaligned>;
  auto a_augmented_values_t = UnalignedTensorMap(a_augmented_values.data(), sum_nnz);
  auto b_augmented_values_t = UnalignedTensorMap(b_augmented_values.data(), sum_nnz);
  auto output_values_ptr = reinterpret_cast<T *>(outputs[kOutput_values]->addr);
  Eigen::DSizes<Eigen::DenseIndex, 1> output_values_size(static_cast<Eigen::DenseIndex>(sum_nnz));
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> output_values(
    output_values_ptr, output_values_size);
  // cppcheck-suppress unreadVariable
  output_values = a_augmented_values_t.binaryExpr(b_augmented_values_t, Eigen::internal::scalar_max_op<T, T>());
  ShapeVector out_indcie_shape, out_values_shape;
  out_indcie_shape.push_back(sum_nnz);
  out_indcie_shape.push_back(num_dims);
  out_values_shape.push_back(sum_nnz);
  common::AnfAlgo::SetOutputInferTypeAndShape({output_indices_dtype, output_values_dtype},
                                              {out_indcie_shape, out_values_shape}, cnode_ptr_.lock().get());
  return true;
}

std::vector<KernelAttr> SparseSparseMaximumCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt8)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt8)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt8),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt16)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt16)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt16),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeUInt8)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeUInt8)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeUInt8),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeUInt16)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeUInt16)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeUInt16),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeFloat16),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeFloat32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeFloat64)};

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSparseMaximum, SparseSparseMaximumCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
