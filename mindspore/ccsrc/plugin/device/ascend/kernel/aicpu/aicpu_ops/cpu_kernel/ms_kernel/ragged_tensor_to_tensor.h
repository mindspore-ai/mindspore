/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_RAGGEDTENSORTOTENSOR_H_
#define AICPU_KERNELS_NORMALIZED_RAGGEDTENSORTOTENSOR_H_
#include <securec.h>
#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include "inc/cpu_ops_kernel.h"
#include "common/cpu_kernel_utils.h"
#include "common/kernel_log.h"
#include "common/status.h"
#include "utils/eigen_tensor.h"
#include "utils/broadcast_iterator.h"
#include "utils/kernel_util.h"
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
using std::string;
using std::vector;

namespace aicpu {
struct DimStruct {
  int64_t size = 1;
};
using Dim = DimStruct;

struct TensorShapeProtoStruct {
  std::vector<Dim> dims;
  bool unknown_rank = false;
};
using TensorShapeProto = TensorShapeProtoStruct;

enum class RowPartitionType { FIRST_DIM_SIZE, VALUE_ROWIDS, ROW_LENGTHS, ROW_SPLITS, ROW_LIMITS, ROW_STARTS };
const int kShapeInputIndex = 0;
const int kValueInputIndex = 1;
const int kDefaultValueInputIndex = 2;
const int kFirstPartitionInputIndex = 3;
using graphStatus = uint32_t;
const graphStatus GRAPH_FAILED = 0xFFFFFFFF;
const graphStatus GRAPH_SUCCESS = 0;

class RaggedTensorToTensorCpuKernel : public CpuKernel {
 public:
  graphStatus GetRowPartitionTypes(const CpuKernelContext &ctx);
  int32_t GetRaggedRank(const std::vector<RowPartitionType> &partition_types);
  RowPartitionType GetRowPartitionTypeByDimension(int dimension);

  template <typename INDEX_TYPE>
  typename TTypes<INDEX_TYPE>::Flat GetRowPartitionTensor(const CpuKernelContext &c, int64_t dimension);

  string RowPartitionTypeToString(RowPartitionType row_partition_type);

  graphStatus ValidateDefaultValueShape(const TensorShapeProto &default_value_shape,
                                        const TensorShapeProto &value_shape, const char *op_name);

  graphStatus AsProto(Tensor *tshape, TensorShapeProto *proto, std::string name) const;

  graphStatus CombineRaggedTensorToTensorShapes(int32_t ragged_rank, const TensorShapeProto &shape,
                                                const TensorShapeProto &value_shape, TensorShapeProto *output_shape,
                                                const char *op_name);

  template <typename INDEX_TYPE>
  uint32_t CalculateOutputSize(INDEX_TYPE first_dim, const CpuKernelContext &c, vector<INDEX_TYPE> *result);

  template <typename INDEX_TYPE>
  vector<INDEX_TYPE> CalculateFirstParentOutputIndex(INDEX_TYPE first_dimension, INDEX_TYPE output_index_multiplier,
                                                     INDEX_TYPE first_dimension_output);

  template <typename INDEX_TYPE>
  uint32_t CalculateOutputIndexRowSplit(const typename TTypes<INDEX_TYPE>::Flat &row_split,
                                        const vector<INDEX_TYPE> &parent_output_index,
                                        INDEX_TYPE output_index_multiplier, INDEX_TYPE output_size,
                                        vector<INDEX_TYPE> *result);

  template <typename INDEX_TYPE>
  uint32_t CalculateOutputIndexValueRowID(const typename TTypes<INDEX_TYPE>::Flat &value_rowids,
                                          const vector<INDEX_TYPE> &parent_output_index,
                                          INDEX_TYPE output_index_multiplier, INDEX_TYPE output_size,
                                          vector<INDEX_TYPE> *result);

  template <typename INDEX_TYPE>
  uint32_t CalculateOutputIndex(const CpuKernelContext &context, int64_t dimension,
                                const vector<INDEX_TYPE> &parent_output_index, INDEX_TYPE output_index_multiplier,
                                INDEX_TYPE output_size, vector<INDEX_TYPE> *result);

  template <typename INDEX_TYPE>
  uint32_t GetFirstDimensionSize(const CpuKernelContext &context, INDEX_TYPE *result);

  template <typename INDEX_TYPE, typename VALUE_TYPE>
  uint32_t DoCompute(const CpuKernelContext &context);

  template <typename INDEX_TYPE, typename VALUE_TYPE>
  uint32_t SetOutput(const CpuKernelContext &context, const vector<INDEX_TYPE> &output_index, Tensor *output_tensor);

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  std::vector<RowPartitionType> row_partition_types_;
  int ragged_rank_;
};
};  // namespace aicpu
#endif
