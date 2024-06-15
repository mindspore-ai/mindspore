/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_REDISTRIBUTION_OPERATOR_INFER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_REDISTRIBUTION_OPERATOR_INFER_H_

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "frontend/parallel/tensor_layout/construct_operator.h"
#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/tensor_layout/redistribution_layout_transfer.h"
#include "include/common/utils/convert_utils.h"

namespace mindspore {
namespace parallel {
using DeviceArrangement = Shape;
using TensorMap = Shape;
using TensorShape = Shape;
using RedistributionOperatorMap = mindspore::HashMap<uint64_t, int64_t>;
using OperatorR = std::pair<OperatorName, Args>;
using OperatorC = std::pair<OperatorR, Shape>;
using OperatorList = std::vector<OperatorC>;

class RedistributionOperatorInfer {
 public:
  const int64_t NONE = -1;
  explicit RedistributionOperatorInfer(bool construct_op_flag = true)
      : construct_op_flag_(construct_op_flag), is_cost_model_(false) {}
  Status Init(const TensorLayout &tensor_layout, const Map &out_tensor_map, RankList dev_list,
              bool is_cost_model = false, bool is_dynamic_shape = false);
  ~RedistributionOperatorInfer() = default;
  OperatorList operator_list() const { return operator_list_; }
  OperatorVector operator_vector() const { return operator_vector_; }
  OutPutInfoVector output_info_vector() const { return output_info_vector_; }
  Status InferRedistributionOperator();
  void SetVirtualRank(const int64_t virtual_rank) { virtual_rank_ = virtual_rank; }
  Status MergePartialToFullForReshapeHasMultiDynamicAxis();
  Status SegmentFullShapeToPartial();

 private:
  Status InferSplitByAxis();
  Status InferPermuteByAxis();
  Status InferConcatByAxis();
  Status TransferSplitByAxis(const Args &args);
  Status TransferPermuteByAxis(const Args &args);
  Status TransferConcatByAxis(const Args &args);
  Status InsertOperator(const OperatorName &name, const Args &args);

  OperatorList operator_list_;
  OperatorVector operator_vector_;
  OutPutInfoVector output_info_vector_;
  Arrangement dev_mat_;
  RedistributionOperatorMap map_;
  Map in_tensor_map_;
  Map out_tensor_map_;
  TensorLayout cur_tensor_layout_;
  ConstructOperator constructor_;
  RankList dev_list_;
  bool construct_op_flag_;
  bool is_cost_model_;
  int64_t virtual_rank_ = -1;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_REDISTRIBUTION_OPERATOR_INFER_H_
