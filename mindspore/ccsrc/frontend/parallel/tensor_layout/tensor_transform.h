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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_TRANSFORM_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_TRANSFORM_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <unordered_map>
#include "ir/value.h"
#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
using TransformFunc = std::function<std::pair<std::string, std::vector<int64_t>>(const Operator &)>;
class TensorTransform {
 public:
  static std::shared_ptr<TensorTransform> GetInstance();
  ~TensorTransform() = default;
  TensorTransform(const TensorTransform &) = delete;
  TensorTransform &operator=(const TensorTransform &) = delete;
  void InitTransforOperator();
  std::vector<std::pair<std::string, std::vector<int64_t>>> TransformOperators(const Shapes &from, const Shapes &to,
                                                                               const RankList &dev_list,
                                                                               int64_t rank_id);

 private:
  TensorTransform();
  std::unordered_map<string, TransformFunc> transform_operator_;
  bool inited_function_ = false;
  std::pair<std::string, std::vector<int64_t>> ExtractReshapeOp(const Operator &reshape_op_pair) const;
  std::pair<std::string, std::vector<int64_t>> ExtractAllGatherOp(const Operator &allgather_op_pair) const;
  std::pair<std::string, std::vector<int64_t>> ExtractSplitOp(const Operator &split_op_pair) const;
  std::pair<std::string, std::vector<int64_t>> ExtractConcatOp(const Operator &concat_op_pair) const;
  std::pair<std::string, std::vector<int64_t>> ExtractStridedSliceOp(const Operator &slice_op_pair) const;
  void OptimizeAllConcat(std::vector<std::pair<std::string, std::vector<int64_t>>> *transform_op_list);
  TensorRedistribution tensor_redistribution_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_TRANSFORM_H_
