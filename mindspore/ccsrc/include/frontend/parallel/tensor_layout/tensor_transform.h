/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_TRANSFORM_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_TRANSFORM_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <unordered_map>
#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace parallel {
using TransformFunc = std::function<std::pair<std::string, std::vector<int64_t>>(const Operator &)>;
using InferShapeFunc = std::function<Shape(const Shape &, const std::vector<int64_t> &)>;
using ConstructOpFunc = std::function<Operator(const std::vector<int64_t> &)>;
using RedisOpPair = std::pair<std::string, std::vector<int64_t>>;
class FRONTEND_EXPORT TensorTransform {
 public:
  static std::shared_ptr<TensorTransform> GetInstance();
  ~TensorTransform() = default;
  TensorTransform(const TensorTransform &) = delete;
  TensorTransform &operator=(const TensorTransform &) = delete;
  void InitTransforOperator();
  std::vector<RedisOpPair> TransformOperators(const Shapes &from, const Shapes &to, const RankList &dev_list,
                                              const bool redist_opt, int64_t rank_id);
  RedistributionOpListPtr OptimizeTensorRedistributionOperatorList(
    const RedistributionOpListPtr &redistribution_op_list, const Shape &input_shape, int64_t virtual_rank = -1);

 private:
  TensorTransform();
  std::unordered_map<string, TransformFunc> transform_operator_;
  std::unordered_map<string, InferShapeFunc> infer_shape_operator_;
  std::unordered_map<string, ConstructOpFunc> construct_op_operator_;
  bool inited_function_ = false;
  std::pair<std::string, std::vector<int64_t>> ExtractReshapeOp(const Operator &reshape_op_pair) const;
  std::pair<std::string, std::vector<int64_t>> ExtractAllGatherOp(const Operator &allgather_op_pair) const;
  std::pair<std::string, std::vector<int64_t>> ExtractSplitOp(const Operator &split_op_pair) const;
  std::pair<std::string, std::vector<int64_t>> ExtractConcatOp(const Operator &concat_op_pair) const;
  std::pair<std::string, std::vector<int64_t>> ExtractStridedSliceOp(const Operator &slice_op_pair) const;
  std::pair<std::string, std::vector<int64_t>> ExtractAlltoAllOp(const Operator &a2a_op_pair) const;

  Operator ConstructReshapeOp(const std::vector<int64_t> &inputs);
  Operator ConstructAllGatherOp(const std::vector<int64_t> &inputs);
  Operator ConstructSplitOp(const std::vector<int64_t> &inputs);
  Operator ConstructStrideSliceOp(const std::vector<int64_t> &inputs);
  Operator ConstructConcatOp(const std::vector<int64_t> &inputs);
  Operator ConstructAlltoAllOp(const std::vector<int64_t> &inputs);

  Shape InferReshapeOp(const Shape &ori_shape, const std::vector<int64_t> &op) const;
  Shape InferAllGatherOp(const Shape &ori_shape, const std::vector<int64_t> &op) const;
  Shape InferAllConcatOp(const Shape &ori_shape, const std::vector<int64_t> &op) const;
  Shape InferStridedSliceOp(const Shape &ori_shape, const std::vector<int64_t> &op) const;
  Shape InferSliceOp(const Shape &ori_shape, const std::vector<int64_t> &op) const;
  Shape InferAlltoAllOp(const Shape &ori_shape, const std::vector<int64_t> &op) const;

  std::vector<Shape> GetRedistributionOpShape(const Shape &ori_shape,
                                              const std::vector<RedisOpPair> &transform_op_list);
  Status TransAllGatherToAllConcat(std::vector<RedisOpPair> *transform_op_list);
  Status TransAllConcatToAllGather(std::vector<RedisOpPair> *transform_op_list);
  Status TransStridedSliceToSlice(const Shape &input_shape, std::vector<RedisOpPair> *transform_op_list);
  Status TransSliceToStridedSlice(const Shape &input_shape, std::vector<RedisOpPair> *transform_op_list);
  Status ReorderAndMergeRedistributionOp(const Shape &input_shape, std::vector<RedisOpPair> *transform_op_list);
  void ShowRedisOpList(const Shape &input_shape, const std::vector<RedisOpPair> &transform_op_list);
  void EliminateRedundancyReshape(const Shape &input_shape, std::vector<RedisOpPair> *transform_op_list);
  void OptimizeAllConcat(const Shape &input_shape, std::vector<RedisOpPair> *transform_op_list);
  void OptimizeSlice(const Shape &input_shape, std::vector<RedisOpPair> *transform_op_list);
  void MergeAllConcat(std::vector<RedisOpPair> *transform_op_list);
  void MergeSlice(std::vector<RedisOpPair> *transform_op_list);
  RedistributionOpList ConstructRedistributionOpListByRedisOpList(const std::vector<RedisOpPair> &transform_op_list);
  RankList ParseRankListFromGroupName(const std::string &group_name) const;
  TensorRedistribution tensor_redistribution_;

 private:
  int64_t virtual_rank_ = -1;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_TRANSFORM_H_
