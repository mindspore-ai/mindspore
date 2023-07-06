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

#include "plugin/device/cpu/kernel/sparsefillemptyrows_cpu_kernel.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseFillEmptyRowsInputsNum = 4;
constexpr size_t kSparseFillEmptyRowsOutputsNum = 4;
constexpr size_t kIndicesSizeNum = 2;
constexpr size_t kIndicesLastDim = 2;
constexpr size_t kValuesSizeNum = 1;
constexpr size_t kDenseSizeNum = 1;

const uint32_t kInput_indices = 0;
const uint32_t kInput_values = 1;
const uint32_t kInput_dense_shape = 2;
const uint32_t kInput_default_value = 3;
const uint32_t kOutput_y_indices = 0;
const uint32_t kOutput_y_values = 1;
const uint32_t kOutput_empty_row_indicator = 2;
const uint32_t kOutput_reverse_index_map = 3;
constexpr char kKernelName[] = "SparseFillEmptyRows";

#define EIGEN_SHAPE_CAST0(INPUT) static_cast<Eigen::DenseIndex>(AnfAlgo::GetInputDeviceShape(node_ptr, INPUT)[0])
#define EIGEN_SHAPE_CAST1(INPUT) static_cast<Eigen::DenseIndex>(AnfAlgo::GetInputDeviceShape(node_ptr, INPUT)[1])

#define SPARSE_FILL_EMPTY_ROWS_COMPUTE_CASE(DTYPE, TYPE) \
  case (DTYPE): {                                        \
    ret = LaunchKernel<TYPE>(inputs, outputs);           \
    break;                                               \
  }

#define ADD_KERNEL(t1, t2, t3, t4, t5, t6, t7, t8) \
  KernelAttr()                                     \
    .AddInputAttr(kNumberType##t1)                 \
    .AddInputAttr(kNumberType##t2)                 \
    .AddInputAttr(kNumberType##t3)                 \
    .AddInputAttr(kNumberType##t4)                 \
    .AddOutputAttr(kNumberType##t5)                \
    .AddOutputAttr(kNumberType##t6)                \
    .AddOutputAttr(kNumberType##t7)                \
    .AddOutputAttr(kNumberType##t8)
}  // namespace

void SparseFillEmptyRowsCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  node_ptr = kernel_node;
  MS_EXCEPTION_IF_NULL(kernel_node);

  // for DeprecatedNativeCpuKernelMod
  auto output_empty_row_indicator_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, kIndex2);
  if (IsDynamic(output_empty_row_indicator_shape)) {
    return;
  }

  output_indices_type_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  output_values_type_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, 1);
  output_empty_row_indicator_type_ = kNumberTypeBool;
  output_reverse_index_type_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kSparseFillEmptyRowsInputsNum, kKernelName);
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kSparseFillEmptyRowsInputsNum, kKernelName);
  const auto indices_shape = AnfAlgo::GetInputDeviceShape(node_ptr, 0);
  const auto values_shape = AnfAlgo::GetInputDeviceShape(node_ptr, 1);
  const auto dense_shape = AnfAlgo::GetInputDeviceShape(node_ptr, 2);
  if (indices_shape.size() != kIndicesSizeNum && indices_shape[0] != values_shape[0]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', it requires 'indices' must be a 2-D Tensor and the first dimension length "
                         "must be equal to the first dimension length of 'values' "
                      << indices_shape;
  }
  if (indices_shape[1] != kIndicesLastDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the last dim of the indices must be 2, but got "
                      << indices_shape[1];
  }
  if (values_shape.size() != kValuesSizeNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it requires 'dense' must be a 1-D Tensor " << values_shape;
  }
  if (dense_shape.size() != kValuesSizeNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it requires 'dense' must be a 1-D Tensor " << dense_shape;
  }
}

template <typename T>
bool SparseFillEmptyRowsCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                   const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseFillEmptyRowsInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseFillEmptyRowsOutputsNum, kernel_name_);
  auto indices_ptr = reinterpret_cast<int64_t *>(inputs[0]->addr);
  Eigen::DSizes<Eigen::DenseIndex, kIndex2> indices_size(EIGEN_SHAPE_CAST0(kInput_indices),
                                                         EIGEN_SHAPE_CAST1(kInput_indices));
  Eigen::TensorMap<Eigen::Tensor<int64_t, kIndex2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> a_indices(
    indices_ptr, indices_size);
  auto values_ptr = reinterpret_cast<T *>(inputs[1]->addr);
  auto dense_shape_ptr = reinterpret_cast<int64_t *>(inputs[2]->addr);
  const auto *default_value = reinterpret_cast<T *>(inputs[3]->addr);
  const int64_t N = AnfAlgo::GetInputDeviceShape(node_ptr, kInput_indices)[0];
  const int64_t dense_rows = dense_shape_ptr[0];
  int64_t rank = AnfAlgo::GetInputDeviceShape(node_ptr, kInput_indices)[1];
  auto output_empty_row_indicator_ptr = reinterpret_cast<bool *>(outputs[kOutput_empty_row_indicator]->addr);
  auto output_reverse_index_map_ptr = reinterpret_cast<int64_t *>(outputs[kOutput_reverse_index_map]->addr);
  ShapeVector out_indcie_shape;
  ShapeVector out_values_shape;
  ShapeVector out_empty_row_indicator_shape;
  ShapeVector out_reverse_index_shape;
  out_empty_row_indicator_shape.push_back(dense_rows);
  out_reverse_index_shape.push_back(N);
  if (dense_rows == 0) {
    if (N != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', Received SparseTensor with dense_shape[0] = 0, but indices.shape[0] =  " << N;
      return false;
    }
    out_indcie_shape.push_back(0);
    out_indcie_shape.push_back(rank);
    out_values_shape.push_back(0);
    common::AnfAlgo::SetOutputInferTypeAndShape(
      {output_indices_type_, output_values_type_, output_empty_row_indicator_type_, output_reverse_index_type_},
      {out_indcie_shape, out_values_shape, out_empty_row_indicator_shape, out_reverse_index_shape},
      cnode_ptr_.lock().get());
    return true;
  }
  std::vector<int64_t> scratch(dense_rows, 0);
  std::vector<int64_t> filled_count(dense_rows, 0);
  for (int64_t i = 0; i < N; ++i) {
    const int64_t row = a_indices(i, 0);
    if (row < 0 || row >= dense_rows) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', indices([" << i << "], 0) is invalid: [" << row << "] >= [ "
                        << dense_rows << "]";
      return false;
    }
    ++scratch[a_indices(i, 0)];
  }
  for (int64_t row = 0; row < dense_rows; ++row) {
    // Scratch here describes the number of elements in this dense row
    output_empty_row_indicator_ptr[row] = (scratch[row] == 0);
    // In filled version, each row has at least one element.
    scratch[row] = std::max(scratch[row], int64_t{1});
    if (row > 0) {
      scratch[row] += scratch[row - 1];
    }
  }
  out_indcie_shape.push_back(scratch[dense_rows - 1]);
  out_indcie_shape.push_back(rank);
  out_values_shape.push_back(scratch[dense_rows - 1]);
  auto output_y_indices_ptr = reinterpret_cast<int64_t *>(outputs[kOutput_y_indices]->addr);
  auto ret1 = memset_s(output_y_indices_ptr, scratch[dense_rows - 1] * rank * sizeof(int64_t), 0,
                       scratch[dense_rows - 1] * rank * sizeof(int64_t));
  if (ret1 != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output_y_indices failed!";
  }
  Eigen::DSizes<Eigen::DenseIndex, kIndex2> output_y_indices_size(
    static_cast<Eigen::DenseIndex>(scratch[dense_rows - 1]), static_cast<Eigen::DenseIndex>(rank));
  Eigen::TensorMap<Eigen::Tensor<int64_t, kIndex2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    a_output_y_indices(output_y_indices_ptr, output_y_indices_size);
  auto output_y_values_ptr = reinterpret_cast<T *>(outputs[kOutput_y_values]->addr);
  for (int64_t i = 0; i < scratch[dense_rows - 1]; ++i) {
    output_y_values_ptr[i] = (*default_value);
  }
  for (int64_t i = 0; i < N; ++i) {
    const int64_t row = a_indices(i, 0);
    int64_t &offset = filled_count[row];
    const int64_t output_i = ((row == 0) ? 0 : scratch[row - 1]) + offset;
    offset++;  // Increment the filled count for this row.
    (void)std::copy_n(&a_indices(i, 0), rank, &a_output_y_indices(output_i, 0));
    output_y_values_ptr[output_i] = values_ptr[i];
    // We'll need this reverse index map to backprop correctly.
    output_reverse_index_map_ptr[i] = output_i;
  }
  for (int64_t row = 0; row < dense_rows; ++row) {
    const int64_t row_count = filled_count[row];
    if (row_count == 0) {  // We haven't filled this row
      const int64_t starting_index = (row == 0) ? 0 : scratch[row - 1];
      // Remaining index values were set to zero already.
      // The value at this index was set to default_value already.
      // Just need to set the row index in the right location.
      a_output_y_indices(starting_index, 0) = 0;
      a_output_y_indices(starting_index, 0) = row;
    }
  }
  common::AnfAlgo::SetOutputInferTypeAndShape(
    {output_indices_type_, output_values_type_, output_empty_row_indicator_type_, output_reverse_index_type_},
    {out_indcie_shape, out_values_shape, out_empty_row_indicator_shape, out_reverse_index_shape},
    cnode_ptr_.lock().get());
  return true;
}

std::vector<KernelAttr> SparseFillEmptyRowsCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    ADD_KERNEL(Int64, Int8, Int64, Int8, Int64, Int8, Bool, Int64),
    ADD_KERNEL(Int64, UInt8, Int64, UInt8, Int64, UInt8, Bool, Int64),
    ADD_KERNEL(Int64, Int16, Int64, Int16, Int64, Int16, Bool, Int64),
    ADD_KERNEL(Int64, UInt16, Int64, UInt16, Int64, UInt16, Bool, Int64),
    ADD_KERNEL(Int64, Int32, Int64, Int32, Int64, Int32, Bool, Int64),
    ADD_KERNEL(Int64, UInt32, Int64, UInt32, Int64, UInt32, Bool, Int64),
    ADD_KERNEL(Int64, Int64, Int64, Int64, Int64, Int64, Bool, Int64),
    ADD_KERNEL(Int64, UInt64, Int64, UInt64, Int64, UInt64, Bool, Int64),
    ADD_KERNEL(Int64, Float16, Int64, Float16, Int64, Float16, Bool, Int64),
    ADD_KERNEL(Int64, Float32, Int64, Float32, Int64, Float32, Bool, Int64),
    ADD_KERNEL(Int64, Bool, Int64, Bool, Int64, Bool, Bool, Int64),
    ADD_KERNEL(Int64, Float64, Int64, Float64, Int64, Float64, Bool, Int64),
    ADD_KERNEL(Int64, Complex64, Int64, Complex64, Int64, Complex64, Bool, Int64),
    ADD_KERNEL(Int64, Complex128, Int64, Complex128, Int64, Complex128, Bool, Int64)};
  return support_list;
}

bool SparseFillEmptyRowsCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs) {
  bool ret = false;
  auto data_type = AnfAlgo::GetInputDeviceDataType(node_ptr, kInput_values);
  switch (data_type) {
    SPARSE_FILL_EMPTY_ROWS_COMPUTE_CASE(kNumberTypeInt8, int8_t)
    SPARSE_FILL_EMPTY_ROWS_COMPUTE_CASE(kNumberTypeUInt8, uint8_t)
    SPARSE_FILL_EMPTY_ROWS_COMPUTE_CASE(kNumberTypeInt16, int16_t)
    SPARSE_FILL_EMPTY_ROWS_COMPUTE_CASE(kNumberTypeUInt16, uint16_t)
    SPARSE_FILL_EMPTY_ROWS_COMPUTE_CASE(kNumberTypeUInt32, uint32_t)
    SPARSE_FILL_EMPTY_ROWS_COMPUTE_CASE(kNumberTypeUInt64, uint64_t)
    SPARSE_FILL_EMPTY_ROWS_COMPUTE_CASE(kNumberTypeInt32, int32_t)
    SPARSE_FILL_EMPTY_ROWS_COMPUTE_CASE(kNumberTypeInt64, int64_t)
    SPARSE_FILL_EMPTY_ROWS_COMPUTE_CASE(kNumberTypeFloat16, Eigen::half)
    SPARSE_FILL_EMPTY_ROWS_COMPUTE_CASE(kNumberTypeFloat32, float)
    SPARSE_FILL_EMPTY_ROWS_COMPUTE_CASE(kNumberTypeBool, bool)
    SPARSE_FILL_EMPTY_ROWS_COMPUTE_CASE(kNumberTypeFloat64, double)
    SPARSE_FILL_EMPTY_ROWS_COMPUTE_CASE(kNumberTypeComplex64, std::complex<float>)
    SPARSE_FILL_EMPTY_ROWS_COMPUTE_CASE(kNumberTypeComplex128, std::complex<double>)
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', Unsupported input data type: " << data_type << ".";
  }
  return ret;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseFillEmptyRows, SparseFillEmptyRowsCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
