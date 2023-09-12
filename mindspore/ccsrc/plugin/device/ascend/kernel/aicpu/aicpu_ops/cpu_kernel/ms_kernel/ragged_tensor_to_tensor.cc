/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
#include <securec.h>
#include "ms_kernel/ragged_tensor_to_tensor.h"
#include <algorithm>

namespace {
constexpr uint32_t kInputNum = 4;
constexpr uint32_t kOutputNum = 1;
const char *kRaggedTensorToTensor = "RaggedTensorToTensor";
}  // namespace

namespace aicpu {
uint32_t RaggedTensorToTensorCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "RaggedTensorToTensor check input and output number failed.");
  DataType type1 = ctx.Input(1)->GetDataType();
  DataType SplitType = ctx.Input(0)->GetDataType();
  switch (SplitType) {
    case DT_INT32:
      switch (type1) {
        case DT_DOUBLE:
          return DoCompute<int32_t, double>(ctx);
        case DT_FLOAT16:
          return DoCompute<int32_t, Eigen::half>(ctx);
        case DT_FLOAT:
          return DoCompute<int32_t, float>(ctx);
        case DT_INT8:
          return DoCompute<int32_t, int8_t>(ctx);
        case DT_INT16:
          return DoCompute<int32_t, int16_t>(ctx);
        case DT_INT32:
          return DoCompute<int32_t, int32_t>(ctx);
        case DT_INT64:
          return DoCompute<int32_t, int64_t>(ctx);
        case DT_UINT8:
          return DoCompute<int32_t, uint8_t>(ctx);
        case DT_UINT16:
          return DoCompute<int32_t, uint16_t>(ctx);
        case DT_BOOL:
          return DoCompute<int32_t, bool>(ctx);
        default: {
          KERNEL_LOG_ERROR("Unsupported datatype [%s]", DTypeStr(type1).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
        }
      }
      break;
    case DT_INT64:
      switch (type1) {
        case DT_DOUBLE:
          return DoCompute<int64_t, double>(ctx);
        case DT_FLOAT16:
          return DoCompute<int64_t, Eigen::half>(ctx);
        case DT_FLOAT:
          return DoCompute<int64_t, float>(ctx);
        case DT_INT8:
          return DoCompute<int64_t, int8_t>(ctx);
        case DT_INT16:
          return DoCompute<int64_t, int16_t>(ctx);
        case DT_INT32:
          return DoCompute<int64_t, int32_t>(ctx);
        case DT_INT64:
          return DoCompute<int64_t, int64_t>(ctx);
        case DT_UINT8:
          return DoCompute<int64_t, uint8_t>(ctx);
        case DT_UINT16:
          return DoCompute<int64_t, uint16_t>(ctx);
        case DT_BOOL:
          return DoCompute<int64_t, bool>(ctx);
        default: {
          KERNEL_LOG_ERROR("Unsupported datatype [%s]", DTypeStr(type1).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
        }
      }
      break;
    default: {
      KERNEL_LOG_ERROR("Unsupported datatype [%s]", DTypeStr(SplitType).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
}

graphStatus RaggedTensorToTensorCpuKernel::GetRowPartitionTypes(const CpuKernelContext &ctx) {
  std::vector<std::string> partition_types;
  AttrValue *row_part = ctx.GetAttr("row_partition_types");
  int64_t N = ctx.Input(0)->GetTensorShape()->GetDims();
  row_partition_types_.reserve(N);
  partition_types.reserve(N);
  if (!row_part) {
    KERNEL_LOG_ERROR("row_partition_types error.");
    return GRAPH_FAILED;
  }
  partition_types = row_part->GetListString();
  const auto string_to_type =
    new std::unordered_map<std::string, RowPartitionType>({{"FIRST_DIM_SIZE", RowPartitionType::FIRST_DIM_SIZE},
                                                           {"VALUE_ROWIDS", RowPartitionType::VALUE_ROWIDS},
                                                           {"ROW_LENGTHS", RowPartitionType::ROW_LENGTHS},
                                                           {"ROW_SPLITS", RowPartitionType::ROW_SPLITS},
                                                           {"ROW_LIMITS", RowPartitionType::ROW_LIMITS},
                                                           {"ROW_STARTS", RowPartitionType::ROW_STARTS}});

  for (const std::string &type_str : partition_types) {
    const auto iter = string_to_type->find(type_str);
    if (iter == string_to_type->end()) {
      delete string_to_type;
      KERNEL_LOG_ERROR("Unknown string for partition info type.");
      return GRAPH_FAILED;
    }
    row_partition_types_.push_back(iter->second);
  }
  delete string_to_type;
  return GRAPH_SUCCESS;
}

int32_t RaggedTensorToTensorCpuKernel::GetRaggedRank(const std::vector<RowPartitionType> &partition_types) {
  if (partition_types.empty()) {
    return 0;
  }
  if (partition_types[0] == RowPartitionType::FIRST_DIM_SIZE) {
    return partition_types.size() - 1;
  }
  return partition_types.size();
}

RowPartitionType RaggedTensorToTensorCpuKernel::GetRowPartitionTypeByDimension(int dimension) {
  if (row_partition_types_[0] == RowPartitionType::FIRST_DIM_SIZE) {
    return row_partition_types_[dimension + 1];
  } else {
    return row_partition_types_[dimension];
  }
}

// Returns the relationship between dimension and dimension + 1.
template <typename INDEX_TYPE>
typename TTypes<INDEX_TYPE>::Flat RaggedTensorToTensorCpuKernel::GetRowPartitionTensor(const CpuKernelContext &c,
                                                                                       int64_t dimension) {
  if (row_partition_types_[0] == RowPartitionType::FIRST_DIM_SIZE) {
    Tensor *row_partition = c.Input(dimension + 1 + kFirstPartitionInputIndex);
    EigenTensor rowET(row_partition, reinterpret_cast<INDEX_TYPE *>(row_partition->GetData()));
    typename TTypes<INDEX_TYPE>::Flat flat_tensor = rowET.flat<INDEX_TYPE>();
    return flat_tensor;
  } else {
    Tensor *row_partition = c.Input(dimension + kFirstPartitionInputIndex);
    EigenTensor rowET(row_partition, reinterpret_cast<INDEX_TYPE *>(row_partition->GetData()));
    typename TTypes<INDEX_TYPE>::Flat flat_tensor = rowET.flat<INDEX_TYPE>();
    return flat_tensor;
  }
}

string RaggedTensorToTensorCpuKernel::RowPartitionTypeToString(RowPartitionType row_partition_type) {
  switch (row_partition_type) {
    case RowPartitionType::FIRST_DIM_SIZE:
      return "FIRST_DIM_SIZE";
    case RowPartitionType::VALUE_ROWIDS:
      return "VALUE_ROWIDS";
    case RowPartitionType::ROW_LENGTHS:
      return "ROW_LENGTHS";
    case RowPartitionType::ROW_SPLITS:
      return "ROW_SPLITS";
    case RowPartitionType::ROW_LIMITS:
      return "ROW_LIMITS";
    case RowPartitionType::ROW_STARTS:
      return "ROW_STARTS";
    default:
      return "UNKNOWN ROW PARTITION TYPE";
  }
}

graphStatus RaggedTensorToTensorCpuKernel::ValidateDefaultValueShape(const TensorShapeProto &default_value_shape,
                                                                     const TensorShapeProto &value_shape,
                                                                     const char *op_name) {
  if (default_value_shape.unknown_rank || value_shape.unknown_rank) {
    return GRAPH_SUCCESS;
  }
  if (default_value_shape.dims.size() > value_shape.dims.size()) {
    KERNEL_LOG_ERROR("default_value must have less dimensions than the values.");
    return GRAPH_FAILED;
  }
  for (size_t i = 0; i < std::min(default_value_shape.dims.size(), value_shape.dims.size() - 1); ++i) {
    if (default_value_shape.dims[i].size >= 0 && value_shape.dims[i + 1].size >= 0 &&
        default_value_shape.dims[i].size != 1 && default_value_shape.dims[i].size != value_shape.dims[i + 1].size) {
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus RaggedTensorToTensorCpuKernel::AsProto(Tensor *tshape, TensorShapeProto *proto, std::string name) const {
  proto->dims.clear();
  if (name == "shape") {
    if (tshape->GetTensorShape()) {
      if ((tshape->GetDataType() == DT_INT32 &&
           static_cast<int64_t *>(tshape->GetData())[0] == static_cast<int32_t>(-1)) ||
          (tshape->GetDataType() == DT_INT64 &&
           static_cast<int64_t *>(tshape->GetData())[0] == static_cast<int64_t>(-1))) {
        proto->unknown_rank = true;
        return KERNEL_STATUS_OK;
      }
    }
    if (tshape->GetDataType() == DT_INT32) {
      int64_t dimsnum = tshape->GetTensorShape()->NumElements();
      Dim tdim;
      proto->dims.reserve(dimsnum);
      auto dd = static_cast<int32_t *>(tshape->GetData());
      for (int64_t i = 0; i < tshape->GetTensorShape()->NumElements(); i++) {
        tdim.size = dd[i];
        proto->dims.push_back(tdim);
        proto->unknown_rank = false;
      }
      return KERNEL_STATUS_OK;
    } else if (tshape->GetDataType() == DT_INT64) {
      int64_t dimsnum = tshape->GetTensorShape()->NumElements();
      Dim tdim;
      proto->dims.reserve(dimsnum);
      for (int64_t i = 0; i < tshape->GetTensorShape()->NumElements(); i++) {
        tdim.size = static_cast<int64_t *>(tshape->GetData())[i];
        proto->dims.push_back(tdim);
        proto->unknown_rank = false;
      }
      return KERNEL_STATUS_OK;
    }
    KERNEL_LOG_ERROR("Expected an int32 or int64 shape tensor.");
    return KERNEL_STATUS_PARAM_INVALID;
  } else {
    if (tshape->GetTensorShape()->GetUnknownRank()) {
      proto->unknown_rank = true;
    } else {
      for (int i = 0; i < tshape->GetTensorShape()->GetDims(); i++) {
        Dim dim;
        dim.size = tshape->GetTensorShape()->GetDimSizes()[i];
        proto->dims.push_back(dim);
      }
    }
    return KERNEL_STATUS_OK;
  }
}

graphStatus RaggedTensorToTensorCpuKernel::CombineRaggedTensorToTensorShapes(int32_t ragged_rank,
                                                                             const TensorShapeProto &shape,
                                                                             const TensorShapeProto &value_shape,
                                                                             TensorShapeProto *output_shape,
                                                                             const char *op_name) {
  if (value_shape.unknown_rank && shape.unknown_rank) {
    output_shape->dims.clear();
    output_shape->unknown_rank = true;
    return GRAPH_SUCCESS;
  }
  if (shape.unknown_rank) {
    while (output_shape->dims.size() < ragged_rank + value_shape.dims.size()) {
      Dim temp_dim;
      temp_dim.size = -1;
      output_shape->dims.emplace_back(temp_dim);
    }
  } else {
    *output_shape = shape;
  }
  if (value_shape.unknown_rank) {
    return GRAPH_SUCCESS;
  }
  if (ragged_rank + value_shape.dims.size() != output_shape->dims.size()) {
    KERNEL_LOG_ERROR(
      "error:ragged_rank plus value_shape dims should be equal to output dim "
      "sizes.");
    return GRAPH_FAILED;
  }

  for (size_t i = 1; i < value_shape.dims.size(); ++i) {
    const Dim value_dim = value_shape.dims[i];
    Dim output_shape_dim = output_shape->dims.at(output_shape->dims.size() - value_shape.dims.size() + i);
    if (value_dim.size >= 0) {
      if (output_shape_dim.size >= 0 && output_shape_dim.size != value_dim.size) {
        KERNEL_LOG_ERROR("Value and shape dimension are inconsistent.");
        return GRAPH_FAILED;
      }
      if (output_shape_dim.size < 0) {
        output_shape_dim.size = value_dim.size;
      }
    }
  }
  return GRAPH_SUCCESS;
}

template <typename INDEX_TYPE>
uint32_t RaggedTensorToTensorCpuKernel::CalculateOutputSize(INDEX_TYPE first_dim, const CpuKernelContext &c,
                                                            vector<INDEX_TYPE> *result) {
  TensorShapeProto value_shape_proto;
  Tensor *value_ptr = c.Input(kValueInputIndex);
  AsProto(value_ptr, &value_shape_proto, "value");
  TensorShapeProto default_value_shape_proto;
  Tensor *default_value_ptr = c.Input(kDefaultValueInputIndex);
  AsProto(default_value_ptr, &default_value_shape_proto, "default_value");
  TensorShapeProto output_shape_proto;
  Tensor *output_ptr = c.Output(0);
  KERNEL_CHECK_NULLPTR(output_ptr, KERNEL_STATUS_PARAM_INVALID, "Output error.");
  KERNEL_CHECK_FALSE(
    (ValidateDefaultValueShape(default_value_shape_proto, value_shape_proto, "RaggedTensorToTensor") != GRAPH_FAILED),
    KERNEL_STATUS_PARAM_INVALID, "ValidateDefaultValueShape error.");
  TensorShapeProto shape_proto;
  {
    Tensor *shape_ptr = c.Input(kShapeInputIndex);
    AsProto(shape_ptr, &shape_proto, "shape");
  }
  KERNEL_CHECK_FALSE((CombineRaggedTensorToTensorShapes(ragged_rank_, shape_proto, value_shape_proto,
                                                        &output_shape_proto, "RaggedTensorToTensor") != GRAPH_FAILED),
                     KERNEL_STATUS_PARAM_INVALID, "CombineRaggedTensorToTensorShapes error.");
  result->reserve(output_shape_proto.dims.size());
  for (unsigned int dim = 0; dim < output_shape_proto.dims.size(); dim++) {
    // Note that this may be -1 (if dimension size is unknown).
    result->push_back(output_shape_proto.dims[dim].size);
  }
  if ((*result)[0] < 0) {
    (*result)[0] = first_dim;
  }
  for (int i = 1; i <= ragged_rank_; ++i) {
    KERNEL_CHECK_FALSE(((*result)[i] >= 0), KERNEL_STATUS_PARAM_INVALID, "Result error.");
  }
  return KERNEL_STATUS_OK;
}

/**
 * The output_index represents the index in the output tensor
 * where the first element of a particular dimension would be written.
 * If it is -1, it indicates that the index is out of scope.
 * Example, given first_dimension = 10, first_dimension_output = 6,
 * and output_index_multiplier = 100:
 * result = [0 100 200 300 400 500 -1 -1 -1 -1]
 * If first_dimension_output = 11 instead, then:
 * result = [0 100 200 300 400 500 600 700 800 900]
 */
template <typename INDEX_TYPE>
vector<INDEX_TYPE> RaggedTensorToTensorCpuKernel::CalculateFirstParentOutputIndex(INDEX_TYPE first_dimension,
                                                                                  INDEX_TYPE output_index_multiplier,
                                                                                  INDEX_TYPE first_dimension_output) {
  const INDEX_TYPE min_dimension = std::min(first_dimension, first_dimension_output);
  vector<INDEX_TYPE> result;
  result.reserve(first_dimension);
  int current_output_index = 0;
  for (INDEX_TYPE i = 0; i < min_dimension; ++i, current_output_index += output_index_multiplier) {
    result.push_back(current_output_index);
  }
  for (INDEX_TYPE i = min_dimension; i < first_dimension; ++i) {
    result.push_back(-1);
  }
  auto fisrt_dim = static_cast<unsigned int>(first_dimension);
  if (result.size() < fisrt_dim) KERNEL_LOG_ERROR("Resize size shou l d be greater equal first dim.");
  return result;
}

template <typename INDEX_TYPE>
uint32_t RaggedTensorToTensorCpuKernel::CalculateOutputIndexRowSplit(const typename TTypes<INDEX_TYPE>::Flat &row_split,
                                                                     const vector<INDEX_TYPE> &parent_output_index,
                                                                     INDEX_TYPE output_index_multiplier,
                                                                     INDEX_TYPE output_size,
                                                                     vector<INDEX_TYPE> *result) {
  INDEX_TYPE row_split_size = row_split.size();
  if (row_split_size > 0) {
    result->reserve(row_split(row_split_size - 1));
  }
  for (INDEX_TYPE i = 0; i < row_split_size - 1; ++i) {
    INDEX_TYPE row_length = row_split(i + 1) - row_split(i);
    INDEX_TYPE real_length = std::min(output_size, row_length);
    INDEX_TYPE parent_output_index_current = parent_output_index[i];
    if (parent_output_index_current == -1) {
      real_length = 0;
    }
    for (INDEX_TYPE j = 0; j < real_length; ++j) {
      result->push_back(parent_output_index_current);
      parent_output_index_current += output_index_multiplier;
    }
    for (INDEX_TYPE j = 0; j < row_length - real_length; ++j) {
      result->push_back(-1);
    }
  }
  if (row_split_size > 0) {
    unsigned int row_split_size1 = row_split(row_split_size - 1);
    KERNEL_CHECK_FALSE((result->size() >= row_split_size1), KERNEL_STATUS_PARAM_INVALID,
                       "Result size should be greater equal row split size.");
  }
  return KERNEL_STATUS_OK;
}

// Calculate the output index of the first element of a list.
// The parent_output_index is the same computation for the previous list.
// -1 indicates an element or list that is out of range.
// The output_index_multiplier is the number of output indices one moves
// forward for each column.
// E.g., given:
// value_rowids:[0 1 2 2 2 3 5 5 6]
// parent_output_index:[1000 1100 2000 2100 -1 3000 4000]
// output_index_multiplier: 10
// output_size: 2
// You get:
// result = [1000 1100 2000 2010 -1 2100 -1 -1 3000]
// result[0] = parent_output_index[value_rowids[0]]
// result[1] = parent_output_index[value_rowids[1]]
// result[2] = parent_output_index[value_rowids[2]]
// result[3] = parent_output_index[value_rowids[2] + 10]
// result[4] = -1 because it is the third element the size is 2.
// result[5] = parent_output_index[value_rowids[3]]
// result[6] = -1 because parent_output_index[value_rowids[6]] == -1
// result[7] = -1 because parent_output_index[value_rowids[6]] == -1
// result[8] = parent_output_index[value_rowids[7]]
template <typename INDEX_TYPE>
uint32_t RaggedTensorToTensorCpuKernel::CalculateOutputIndexValueRowID(
  const typename TTypes<INDEX_TYPE>::Flat &value_rowids, const vector<INDEX_TYPE> &parent_output_index,
  INDEX_TYPE output_index_multiplier, INDEX_TYPE output_size, vector<INDEX_TYPE> *result) {
  const INDEX_TYPE index_size = value_rowids.size();
  result->reserve(index_size);
  KERNEL_CHECK_FALSE((index_size != 0), KERNEL_STATUS_PARAM_INVALID, "Index size should not be zero.");
  INDEX_TYPE current_output_column = 0;
  unsigned int current_value_rowid = value_rowids(0);
  KERNEL_CHECK_FALSE((current_value_rowid < parent_output_index.size()), KERNEL_STATUS_PARAM_INVALID,
                     "Current value rowid should be less than parent output index size.");
  INDEX_TYPE current_output_index = parent_output_index[current_value_rowid];
  result->push_back(current_output_index);
  for (INDEX_TYPE i = 1; i < index_size; ++i) {
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
        KERNEL_LOG_ERROR("Next value rowid should be less than parent output index size.");
        return KERNEL_STATUS_PARAM_INVALID;
      }
      current_output_index = parent_output_index[next_value_rowid];
    }
    result->push_back(current_output_index);
  }
  size_t result_size = result->size();
  size_t value_rowid_size = value_rowids.size();
  KERNEL_CHECK_FALSE((result_size == value_rowid_size), KERNEL_STATUS_PARAM_INVALID, "Invalid row ids.");
  return KERNEL_STATUS_OK;
}

template <typename INDEX_TYPE>
uint32_t RaggedTensorToTensorCpuKernel::CalculateOutputIndex(const CpuKernelContext &ctx, int64_t dimension,
                                                             const vector<INDEX_TYPE> &parent_output_index,
                                                             INDEX_TYPE output_index_multiplier, INDEX_TYPE output_size,
                                                             vector<INDEX_TYPE> *result) {
  const typename TTypes<INDEX_TYPE>::Flat row_partition_tensor = GetRowPartitionTensor<INDEX_TYPE>(ctx, dimension);
  auto partition_type = GetRowPartitionTypeByDimension(dimension);
  switch (partition_type) {
    case RowPartitionType::VALUE_ROWIDS:
      return CalculateOutputIndexValueRowID(row_partition_tensor, parent_output_index, output_index_multiplier,
                                            output_size, result);
    case RowPartitionType::ROW_SPLITS:
      return CalculateOutputIndexRowSplit(row_partition_tensor, parent_output_index, output_index_multiplier,
                                          output_size, result);
    default:
      KERNEL_LOG_ERROR("Unsupported partition type:[%s]", RowPartitionTypeToString(partition_type));
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename INDEX_TYPE>
uint32_t RaggedTensorToTensorCpuKernel::GetFirstDimensionSize(const CpuKernelContext &ctx, INDEX_TYPE *result) {
  const Tensor *first_partition_tensor = ctx.Input(kFirstPartitionInputIndex);
  const RowPartitionType first_partition_type = row_partition_types_[0];

  switch (first_partition_type) {
    case RowPartitionType::FIRST_DIM_SIZE:
      *result = static_cast<INDEX_TYPE *>(first_partition_tensor->GetData())[0];
      return KERNEL_STATUS_OK;
    case RowPartitionType::VALUE_ROWIDS:
      KERNEL_LOG_ERROR("Cannot handle VALUE_ROWIDS in first dimension.");
      return KERNEL_STATUS_PARAM_INVALID;
    case RowPartitionType::ROW_SPLITS:
      *result = first_partition_tensor->GetTensorShape()->GetDimSizes()[0] - 1;
      return KERNEL_STATUS_OK;
    default:
      KERNEL_LOG_ERROR("Cannot handle type [%s]", RowPartitionTypeToString(first_partition_type));
      return KERNEL_STATUS_INNER_ERROR;
  }
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
uint32_t RaggedTensorToTensorCpuKernel::DoCompute(const CpuKernelContext &ctx) {
  KERNEL_CHECK_FALSE((GetRowPartitionTypes(ctx) != GRAPH_FAILED), KERNEL_STATUS_PARAM_INVALID,
                     "GetRowPartitionTypes error");
  ragged_rank_ = GetRaggedRank(row_partition_types_);
  INDEX_TYPE first_dimension;
  KERNEL_CHECK_FALSE((GetFirstDimensionSize(ctx, &first_dimension) == 0), KERNEL_STATUS_PARAM_INVALID,
                     "GetFirstDimensionSize error.");
  vector<INDEX_TYPE> output_size;
  KERNEL_CHECK_FALSE((CalculateOutputSize(first_dimension, ctx, &output_size) == 0), KERNEL_STATUS_PARAM_INVALID,
                     "CalculateOutputSize error.");

  vector<INDEX_TYPE> multiplier;
  multiplier.resize(output_size.size());
  multiplier[multiplier.size() - 1] = 1;
  for (int i = output_size.size() - 2; i >= 0; --i) {
    multiplier[i] = multiplier[i + 1] * output_size[i + 1];
  }

  Tensor *output_tensor = nullptr;
  output_tensor = ctx.Output(0);
  auto output_shape = output_tensor->GetTensorShape();
  auto output_shape_dims = output_shape->GetDimSizes();
  for (unsigned int i = 0; i < output_size.size(); i++) {
    output_shape_dims[i] = output_size[i];
  }

  const INDEX_TYPE full_size = multiplier[0] * output_size[0];
  if (full_size > 0) {
    vector<INDEX_TYPE> output_index = CalculateFirstParentOutputIndex(first_dimension, multiplier[0], output_size[0]);
    for (int i = 1; i <= ragged_rank_; ++i) {
      vector<INDEX_TYPE> new_output_index;
      KERNEL_CHECK_FALSE(
        (CalculateOutputIndex(ctx, i - 1, output_index, multiplier[i], output_size[i], &new_output_index) == 0),
        KERNEL_STATUS_PARAM_INVALID, "CalculateOutputIndex error.");
      output_index = new_output_index;
    }
    return SetOutput<INDEX_TYPE, VALUE_TYPE>(ctx, output_index, output_tensor);
  }
  return KERNEL_STATUS_OK;
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
uint32_t RaggedTensorToTensorCpuKernel::SetOutput(const CpuKernelContext &ctx, const vector<INDEX_TYPE> &output_index,
                                                  Tensor *output_tensor) {
  EigenTensor outputET(output_tensor, reinterpret_cast<INDEX_TYPE *>(output_tensor->GetData()));
  typename aicpu::TTypes<VALUE_TYPE>::Flat output_flat = outputET.flat<VALUE_TYPE>();
  const auto value_tensor = ctx.Input(kValueInputIndex);
  const auto default_value_tensor = ctx.Input(kDefaultValueInputIndex);
  if (value_tensor->GetTensorShape()->GetDims() == 1) {
    // Initialize tensor to default_value.
    VALUE_TYPE *base_output = output_flat.data();
    VALUE_TYPE *default_value_pt = static_cast<VALUE_TYPE *>(default_value_tensor->GetData());
    VALUE_TYPE default_value = default_value_pt[0];
    std::fill(base_output, base_output + output_flat.size(), default_value);
    EigenTensor valuesET(value_tensor, reinterpret_cast<INDEX_TYPE *>(value_tensor->GetData()));
    auto values = valuesET.flat<VALUE_TYPE>();
    unsigned int values_size = values.size();
    KERNEL_CHECK_FALSE((values_size == output_index.size()), KERNEL_STATUS_PARAM_INVALID,
                       "Values and indices must be equal.");
    for (unsigned int i = 0; i < values_size; ++i) {
      if (output_index[i] >= 0) {
        output_flat(output_index[i]) = values(i);
      }
    }
  } else {
    auto output_shape = output_tensor->GetTensorShape()->GetDimSizes();
    auto default_value_shape = default_value_tensor->GetTensorShape()->GetDimSizes();
    int64_t output_element_size = 1;
    for (const int64_t &d : output_shape) {
      output_element_size *= d;
    }
    // Initialize tensor to default_value.
    std::vector<int64_t> broadcast_shape;
    auto ret = GetBroadcastShape(default_value_shape, output_shape, broadcast_shape);
    KERNEL_CHECK_FALSE(ret == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID, "Broadcast failed.");
    KERNEL_CHECK_FALSE(broadcast_shape == output_shape, KERNEL_STATUS_PARAM_INVALID,
                       "Unable to broadcast shape of default_value to result.");
    BroadcastIterator iter(default_value_shape, output_shape, broadcast_shape);
    auto default_value_addr = reinterpret_cast<VALUE_TYPE *>(default_value_tensor->GetData());
    auto output_addr = reinterpret_cast<VALUE_TYPE *>(output_tensor->GetData());
    iter.SetPos(0);
    for (int i = 0; i < output_element_size; ++i) {
      output_addr[i] = default_value_addr[iter.GetInputPosA()];
      iter.GenNextPos();
    }
    VALUE_TYPE *base_output = output_flat.data();
    EigenTensor valuesET(value_tensor, reinterpret_cast<INDEX_TYPE *>(value_tensor->GetData()));
    auto values = valuesET.flat<VALUE_TYPE>();
    size_t values_size = values.size();
    size_t output_index_size = output_index.size();
    //  A value "element" is a group of values that are arranged together.
    // For example, if the value shape is [3,4,5], then 20 values are in a
    // value element.
    unsigned int value_element_size;
    if (output_index_size != 0) {
      value_element_size = values_size / output_index_size;
    } else {
      KERNEL_LOG_DEBUG("Values and indices must be equal");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    unsigned int value_element_bytesize = value_element_size * sizeof(VALUE_TYPE);
    const VALUE_TYPE *values_base = values.data();
    unsigned int values_dimsize = value_tensor->GetTensorShape()->GetDimSizes()[0];
    KERNEL_CHECK_FALSE((values_dimsize == output_index_size), KERNEL_STATUS_PARAM_INVALID,
                       "Values and indices must be equal.");
    KERNEL_CHECK_FALSE((values_size == output_index_size * value_element_size), KERNEL_STATUS_PARAM_INVALID,
                       "Values and indices must be equal.");

    INDEX_TYPE value_index = 0;
    for (unsigned int i = 0; i < output_index_size; ++i, value_index += value_element_size) {
      if (output_index[i] >= 0) {
        VALUE_TYPE *dst = base_output + output_index[i];
        const VALUE_TYPE *src = values_base + value_index;
        auto data_size_max = (output_element_size - output_index[i]) * sizeof(VALUE_TYPE);
        auto ret = memcpy_s(dst, data_size_max, src, value_element_bytesize);
        if (ret != EOK) {
          KERNEL_LOG_ERROR("For 'RaggedTensorToTensor', memcpy_s failed, ret=%d.", ret);
          return KERNEL_STATUS_INNER_ERROR;
        }
      }
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kRaggedTensorToTensor, RaggedTensorToTensorCpuKernel);
}  // namespace aicpu
