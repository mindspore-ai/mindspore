/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_CONSTRUCT_OPERATOR_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_CONSTRUCT_OPERATOR_H_

#include <string>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/status.h"

namespace mindspore {
namespace parallel {
using Args = std::vector<std::int64_t>;
enum class ReshapeMode : int64_t {
  NO_RESHAPE = 0,
  FROM_ORIGIN_BASE_SLICE_TO_TO_ORIGIN_BASE_SLICE,
  FROM_ORIGIN_SLICE_TO_FROM_LAYOUT_SLICE,
  FROM_ORIGIN_BASE_SLICE_TO_FROM_ORIGIN_SLICE,
  TO_ORIGIN_SLICE_TO_TO_LAYOUT_SLICE,
  TO_ORIGIN_SLICE_TO_TO_ORIGIN_BASE_SLICE,
};

Operator CreateStridedSliceOp(int64_t value, const Shape &begin, const Shape &end, const Shape &strides);
class ConstructOperator {
 public:
  const int64_t DEFAULT = 0;
  ConstructOperator() : dev_size_(0) {}
  ~ConstructOperator() = default;
  Status Init(const RankList &dev_list, const Shape &dev_matrix_shape, bool is_cost_model = false,
              bool is_dynamic_shape = false);
  OperatorVector SkipRedisReshapeOP(const Shape &shape) const;
  Status ReshapeOP(const Shape &shape, bool use_origin_shape = false,
                   enum ReshapeMode reshape_mode = ReshapeMode::NO_RESHAPE);
  Status StridedSliceOP(const Args &args);
  Status ReplaceStridedSliceOpToSplitOp(const Args &args);
  Status AllGatherOP(int64_t dev_dim);
  Status SplitOP(int64_t split_count);
  Status ConcatOP(int64_t concat_dim);
  Status AlltoAllOP(const Args &args);
  Operator GetOperator() const { return op_; }
  void UpdateTensorShape(const Shape &tensor_shape) { tensor_shape_ = tensor_shape; }
  void SetVirtualRank(const int64_t virtual_rank) { virtual_rank_ = virtual_rank; }

 private:
  Operator op_;
  size_t dev_size_;
  Shape tensor_shape_;
  RankList dev_list_;
  Shape dev_matrix_shape_;
  bool is_cost_model_ = false;
  bool is_dynamic_shape_ = false;
  int64_t virtual_rank_ = -1;
  bool check_group() { return virtual_rank_ < 0; }
  Status CreateGroupByDim(size_t axis, std::vector<Group> *group);
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_CONSTRUCT_OPERATOR_H_
