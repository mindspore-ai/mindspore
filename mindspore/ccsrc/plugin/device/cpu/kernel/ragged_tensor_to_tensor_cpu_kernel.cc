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

#include "plugin/device/cpu/kernel/ragged_tensor_to_tensor_cpu_kernel.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <functional>
#include <utility>
#include <complex>
#include <type_traits>
#include <iostream>
#include <numeric>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kRaggedTensorToTensorInputsNum = 4;
constexpr size_t kRaggedTensorToTensorOutputsNum = 1;
constexpr size_t kShapeInputIndex = 0;
constexpr size_t kValueInputIndex = 1;
constexpr size_t kDefaultValueInputIndex = 2;
constexpr size_t kFirstPartitionInputIndex = 3;

#define RAGGEDTENSORTOTENSOR_COMPUTE_CASE(DTYPE, TYPE1, TYPE2) \
  case (DTYPE): {                                              \
    LaunchKernel<TYPE1, TYPE2>(inputs, outputs);               \
    break;                                                     \
  }
}  // namespace

bool RaggedTensorToTensorCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  row_partition_types_ = GetValue<std::vector<std::string>>(base_operator->GetAttr("row_partition_types"));
  ragged_rank_ = GetRaggedRank(row_partition_types_);
  shape_dtype_ = inputs[kShapeInputIndex]->GetDtype();
  values_dtype_ = inputs[kValueInputIndex]->GetDtype();
  size_t output_num = outputs.size();
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kRaggedTensorToTensorOutputsNum, kernel_name_);
  return true;
}

int RaggedTensorToTensorCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  values_shape_ = inputs[kValueInputIndex]->GetShapeVector();
  default_values_shape_ = inputs[kDefaultValueInputIndex]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  if (ragged_rank_ + values_shape_.size() != output_shape_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', row partition size plus 'values' rank should be equal to 'shape' rank: "
                      << output_shape_.size() << ", but got row partition size: " << ragged_rank_
                      << ", 'values' rank: " << values_shape_.size();
  }
  row_partition_shape_list_.clear();
  for (int i = 0; i < ragged_rank_; ++i) {
    row_partition_shape_list_.emplace_back(inputs[kFirstPartitionInputIndex + i]->GetShapeVector());
  }
  return KRET_OK;
}

bool RaggedTensorToTensorCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  switch (shape_dtype_) {
    case kNumberTypeInt32: {
      switch (values_dtype_) {
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeBool, int32_t, bool)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeInt8, int32_t, int8_t)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeInt16, int32_t, int16_t)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeInt32, int32_t, int32_t)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeInt64, int32_t, int64_t)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeUInt8, int32_t, uint8_t)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeUInt16, int32_t, uint16_t)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeFloat16, int32_t, Eigen::half)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeFloat32, int32_t, float)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeFloat64, int32_t, double)
        default:
          MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                                  << "', unsupported values data type: " << TypeIdToType(values_dtype_)->ToString();
      }
      break;
    }
    case kNumberTypeInt64: {
      switch (values_dtype_) {
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeBool, int64_t, bool)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeInt8, int64_t, int8_t)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeInt16, int64_t, int16_t)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeInt32, int64_t, int32_t)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeInt64, int64_t, int64_t)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeUInt8, int64_t, uint8_t)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeUInt16, int64_t, uint16_t)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeFloat16, int64_t, Eigen::half)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeFloat32, int64_t, float)
        RAGGEDTENSORTOTENSOR_COMPUTE_CASE(kNumberTypeFloat64, int64_t, double)
        default:
          MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                                  << "', unsupported values data type: " << TypeIdToType(values_dtype_)->ToString();
      }
      break;
    }
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                              << "', unsupported shape data type: " << TypeIdToType(shape_dtype_)->ToString();
  }
  return true;
}

template <typename TYPE1, typename TYPE2>
void RaggedTensorToTensorCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                    const std::vector<kernel::AddressPtr> &outputs) {
  TYPE1 first_dimension;
  GetFirstDimension<TYPE1>(inputs, &first_dimension);
  std::vector<TYPE1> output_size;
  output_size.reserve(output_shape_.size());
  for (unsigned int dim = 0; dim < output_shape_.size(); dim++) {
    output_size.push_back(output_shape_[dim]);
  }
  if (output_size[0] < 0) {
    output_size[0] = first_dimension;
  }

  std::vector<TYPE1> multiplier;
  multiplier.resize(output_size.size());
  multiplier[multiplier.size() - 1] = 1;
  for (int i = output_size.size() - 2; i >= 0; --i) {
    multiplier[i] = multiplier[i + 1] * output_size[i + 1];
  }
  const TYPE1 full_size = multiplier[0] * output_size[0];
  if (full_size > 0) {
    const TYPE1 min_dimension = std::min(first_dimension, output_size[0]);
    std::vector<TYPE1> output_index;
    output_index.reserve(first_dimension);
    int current_output_index = 0;
    for (TYPE1 i = 0; i < min_dimension; ++i, current_output_index += multiplier[0]) {
      output_index.push_back(current_output_index);
    }
    for (TYPE1 i = min_dimension; i < first_dimension; ++i) {
      output_index.push_back(-1);
    }

    for (int i = 1; i <= ragged_rank_; ++i) {
      // dimension = i - 1
      // get row partition tensor
      std::vector<TYPE1_flat> row_partition_tensor;
      if (row_partition_types_[0] == "FIRST_DIM_SIZE") {
        kernel::AddressPtr row_partition = inputs[i + kFirstPartitionInputIndex];
        auto row_partition_ptr = reinterpret_cast<TYPE1 *>(row_partition->addr);
        auto row_partition_shape = row_partition_shape_list_[i];
        Eigen::DSizes<Eigen::DenseIndex, 1> row_partition_dsize(row_partition_shape[0]);
        TYPE1_flat rowET(row_partition_ptr, row_partition_dsize);
        row_partition_tensor.push_back(rowET);
      } else {
        kernel::AddressPtr row_partition = inputs[i - 1 + kFirstPartitionInputIndex];
        auto row_partition_ptr = reinterpret_cast<TYPE1 *>(row_partition->addr);
        auto row_partition_shape = row_partition_shape_list_[i - 1];
        Eigen::DSizes<Eigen::DenseIndex, 1> row_partition_dsize(row_partition_shape[0]);
        TYPE1_flat rowET(row_partition_ptr, row_partition_dsize);
        row_partition_tensor.push_back(rowET);
      }
      // get row partition type
      std::string partition_type;
      if (row_partition_types_[0] == "FIRST_DIM_SIZE") {
        partition_type = row_partition_types_[i];
      } else {
        partition_type = row_partition_types_[i - 1];
      }

      std::vector<TYPE1> new_output_index;
      if (partition_type == "VALUE_ROWIDS") {
        CalculateOutputIndexValueRowID(row_partition_tensor, output_index, multiplier[i], output_size[i],
                                       &new_output_index);
      } else if (partition_type == "ROW_SPLITS") {
        CalculateOutputIndexRowSplit(row_partition_tensor, output_index, multiplier[i], output_size[i],
                                     &new_output_index);
      } else {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', unknown string for partition type: " << partition_type;
      }
      output_index = new_output_index;
    }
    SetOutput<TYPE1, TYPE2>(inputs, outputs, output_index);
  }
}

int RaggedTensorToTensorCpuKernelMod::GetRaggedRank(std::vector<std::string> types) {
  int rank;
  if (types.empty()) {
    rank = 0;
  } else if (types[0] == "FIRST_DIM_SIZE") {
    rank = types.size() - 1;
  } else {
    rank = types.size();
  }
  return rank;
}

template <typename TYPE1>
void RaggedTensorToTensorCpuKernelMod::GetFirstDimension(const std::vector<kernel::AddressPtr> &inputs,
                                                         TYPE1 *first_dim) {
  auto first_partition_tensor = inputs[kFirstPartitionInputIndex];
  auto firstPartitionShape = row_partition_shape_list_[0];
  std::string first_partition_type = row_partition_types_[0];
  if (first_partition_type == "VALUE_ROWIDS") {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', cannot handle 'VALUE_ROWIDS' in first dimension.";
  } else if (first_partition_type == "FIRST_DIM_SIZE") {
    TYPE1 *first_dim_pt = reinterpret_cast<TYPE1 *>(first_partition_tensor->addr);
    *first_dim = first_dim_pt[0];
  } else if (first_partition_type == "ROW_SPLITS") {
    *first_dim = firstPartitionShape[0] - 1;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', unknown string for partition type: " << first_partition_type;
  }
}

template <typename TYPE1>
bool RaggedTensorToTensorCpuKernelMod::CalculateOutputIndexValueRowID(const std::vector<TYPE1_flat> &value_rowid,
                                                                      const vector<TYPE1> &parent_output_index,
                                                                      TYPE1 output_index_multiplier, TYPE1 output_size,
                                                                      vector<TYPE1> *result) {
  TYPE1_flat value_rowids = value_rowid[0];
  const TYPE1 index_size = value_rowids.size();
  result->reserve(index_size);
  if (index_size == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', value rowids size should not be zero.";
  }
  TYPE1 current_output_column = 0;
  unsigned int current_value_rowid = value_rowids(0);
  if (current_value_rowid >= parent_output_index.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', current value rowid: " << current_value_rowid
                      << ", which should be less than " << parent_output_index.size();
  }
  TYPE1 current_output_index = parent_output_index[current_value_rowid];
  result->push_back(current_output_index);
  for (TYPE1 i = 1; i < index_size; ++i) {
    unsigned int next_value_rowid = value_rowids(i);
    if (next_value_rowid == current_value_rowid && current_output_index >= 0) {
      ++current_output_column;
      if (current_output_column < output_size) {
        current_output_index += output_index_multiplier;
      } else {
        current_output_index = -1;
      }
    }
    if (next_value_rowid != current_value_rowid) {
      current_output_column = 0;
      current_value_rowid = next_value_rowid;
      if (next_value_rowid >= parent_output_index.size()) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', next value rowid: " << next_value_rowid
                          << ", which should be less than " << parent_output_index.size();
      }
      current_output_index = parent_output_index[next_value_rowid];
    }
    result->push_back(current_output_index);
  }
  size_t result_size = result->size();
  size_t value_rowid_size = value_rowids.size();
  if (result_size != value_rowid_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', 'result' index size should be equal to value rowid size: " << value_rowid_size
                      << ", but got " << result_size;
  }
  return true;
}

template <typename TYPE1>
bool RaggedTensorToTensorCpuKernelMod::CalculateOutputIndexRowSplit(const std::vector<TYPE1_flat> &row_splits,
                                                                    const vector<TYPE1> &parent_output_index,
                                                                    TYPE1 output_index_multiplier, TYPE1 output_size,
                                                                    vector<TYPE1> *result) {
  TYPE1_flat row_split = row_splits[0];
  TYPE1 row_split_size = row_split.size();
  if (row_split_size > 0) {
    result->reserve(row_split(row_split_size - 1));
  }
  for (TYPE1 i = 0; i < row_split_size - 1; ++i) {
    TYPE1 row_length = row_split(i + 1) - row_split(i);
    TYPE1 real_length = std::min(output_size, row_length);
    TYPE1 parent_output_index_current = parent_output_index[i];
    if (parent_output_index_current == -1) {
      real_length = 0;
    }
    for (TYPE1 j = 0; j < real_length; ++j) {
      result->push_back(parent_output_index_current);
      parent_output_index_current += output_index_multiplier;
    }
    for (TYPE1 j = 0; j < row_length - real_length; ++j) {
      result->push_back(-1);
    }
  }
  if (row_split_size > 0) {
    unsigned int row_split_size1 = row_split(row_split_size - 1);
    if (result->size() != row_split_size1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', 'result' index size should be equal to row split size: " << row_split_size1
                        << ", but got " << result->size();
    }
  }
  return true;
}

template <typename TYPE1, typename TYPE2>
bool RaggedTensorToTensorCpuKernelMod::SetOutput(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &outputs,
                                                 const vector<TYPE1> &output_index) {
  auto output_tensor_ptr = reinterpret_cast<TYPE2 *>(outputs[0]->addr);
  size_t output_element_sum = SizeToLong(SizeOf(output_shape_));
  auto default_value_tensor = inputs[kDefaultValueInputIndex];
  TYPE2 *default_value_pt = reinterpret_cast<TYPE2 *>(default_value_tensor->addr);
  auto values_tensor = inputs[kValueInputIndex];
  auto values_tensor_ptr = reinterpret_cast<TYPE2 *>(values_tensor->addr);
  if (values_shape_.size() == 1) {
    Eigen::DSizes<Eigen::DenseIndex, 1> output_tensor_dsize(output_element_sum);
    TYPE2_flat outputET(output_tensor_ptr, output_tensor_dsize);
    TYPE2_flat output_flat = outputET;
    TYPE2 *base_output = output_flat.data();
    TYPE2 default_value = default_value_pt[0];
    std::fill(base_output, base_output + output_flat.size(), default_value);
    Eigen::DSizes<Eigen::DenseIndex, 1> values_tensor_dsize(values_shape_[0]);
    TYPE2_flat valuesET(values_tensor_ptr, values_tensor_dsize);
    TYPE2_flat values_flat = valuesET;
    unsigned int values_size = values_flat.size();
    if (values_size != output_index.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', 'values' size must be equal to ragged tensor indices, but got "
                        << "'values' size: " << values_size << ", indices: " << output_index.size();
    }
    for (unsigned int i = 0; i < values_size; ++i) {
      if (output_index[i] >= 0) {
        output_flat(output_index[i]) = values_flat(i);
      }
    }
  } else {
    auto broadcast_shape = CPUKernelUtils::GetBroadcastShape(default_values_shape_, output_shape_);
    if (broadcast_shape != output_shape_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', unable to broadcast default_value of shape "
                        << default_values_shape_ << " to tensor of shape " << output_shape_;
    }
    BroadcastIterator iter(default_values_shape_, output_shape_, broadcast_shape);
    TYPE2 *default_value_addr = reinterpret_cast<TYPE2 *>(inputs[kDefaultValueInputIndex]->addr);
    TYPE2 *output_addr = reinterpret_cast<TYPE2 *>(outputs[0]->addr);

    iter.SetPos(0);
    for (size_t i = 0; i < output_element_sum; ++i) {
      output_addr[i] = default_value_addr[iter.GetInputPosA()];
      iter.GenNextPos();
    }
    Eigen::DSizes<Eigen::DenseIndex, 1> output_tensor_dsize(output_element_sum);
    TYPE2_flat outputET(output_addr, output_tensor_dsize);
    TYPE2_flat output_flat = outputET;
    TYPE2 *base_output = output_flat.data();

    size_t values_tensor_size = SizeToLong(SizeOf(values_shape_));
    Eigen::DSizes<Eigen::DenseIndex, 1> values_tensor_dsize(values_tensor_size);
    TYPE2_flat valuesET(values_tensor_ptr, values_tensor_dsize);
    TYPE2_flat values_flat = valuesET;
    size_t values_size = values_flat.size();
    size_t output_index_size = output_index.size();
    int value_element_size = 0;
    if (output_index_size != 0) {
      value_element_size = values_size / output_index_size;
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', ragged tensor indices should not be zero.";
    }
    int value_element_bytesize = value_element_size * sizeof(TYPE2);
    TYPE2 *values_base = values_flat.data();
    size_t values_dimsize = values_shape_[0];
    if (values_dimsize != output_index_size) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', 'values' shape[0] must be equal to ragged tensor indices: " << output_index_size
                        << ", but got " << values_dimsize;
    }
    if (values_size != output_index_size * value_element_size) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', 'values' size must be equal to ragged tensor indices, but got "
                        << "'values' size: " << values_size << ", indices: " << output_index_size * value_element_size;
    }
    TYPE1 value_index = 0;
    for (unsigned int i = 0; i < output_index_size; ++i, value_index += value_element_size) {
      if (output_index[i] >= 0) {
        TYPE2 *dst = base_output + output_index[i];
        const TYPE2 *src = values_base + value_index;
        memcpy(dst, src, value_element_bytesize);
      }
    }
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RaggedTensorToTensor, RaggedTensorToTensorCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
