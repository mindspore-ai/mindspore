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

#ifndef MINDSPORE_CCSRC_PARALLEL_TENSOR_LAYOUT_TENSOR_REDISTRIBUTION_H_
#define MINDSPORE_CCSRC_PARALLEL_TENSOR_LAYOUT_TENSOR_REDISTRIBUTION_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "parallel/ops_info/operator_info.h"
#include "parallel/status.h"
#include "parallel/tensor_layout/construct_operator.h"
#include "parallel/tensor_layout/redistribution_operator_infer.h"
#include "parallel/tensor_layout/tensor_layout.h"

namespace mindspore {
namespace parallel {
constexpr double ALLTOALL_SCALE_FACTOR = 2.0;
constexpr double ALLGATHER_REDUCESCATTER_SCALE_FACTOR = 0.5;
class TensorRedistribution {
 public:
  explicit TensorRedistribution(bool construct_op_flag = true, bool keep_reshape = false)
      : reshape_flag_(false),
        comm_cost_(0.0),
        forward_comm_cost_(0.0),
        backward_comm_cost_(0.0),
        computation_cost_(0.0),
        memory_cost_(0.0),
        construct_op_flag_(construct_op_flag),
        keep_reshape_(keep_reshape) {}
  Status Init(const TensorLayout &from, const TensorLayout &to, const RankList &dev_list);
  ~TensorRedistribution() = default;
  RedistributionOpListPtr InferTensorRedistributionOperatorList(bool is_cost_model = false);
  OperatorList operator_list() const { return operator_list_; }
  bool reshape_flag() const { return reshape_flag_; }
  Status ComputeCost();
  double comm_cost() const { return comm_cost_; }
  double computation_cost() const { return computation_cost_; }
  double forward_comm_cost() const { return forward_comm_cost_; }
  double backward_comm_cost() const { return backward_comm_cost_; }
  double memory_cost() const { return memory_cost_; }

 private:
  Status InferReshape(const TensorLayout &from_layout, const TensorLayout &to_layout,
                      OperatorVector *const operator_vector, OutPutInfoVector *const output_info_vector);

  TensorLayout from_origin_;
  TensorLayout to_origin_;
  TensorLayout from_;
  TensorLayout to_;
  RankList dev_list_;
  OperatorList operator_list_;
  bool reshape_flag_;
  // communication cost, which is the sum of forward communication cost and backward communication cost
  double comm_cost_;
  // forward communication cost
  double forward_comm_cost_;
  // backward communication cost
  double backward_comm_cost_;
  // computation_cost models the time spending on computing in this tensor redistribution, which is calculated by the
  // inputs. This is calculated ONLY for forward phase.
  double computation_cost_;
  // memory_cost models the PEAK memory cost in a training iteration contributed by this tensor redistribution, which is
  // calculated by the outputs.
  double memory_cost_;
  bool construct_op_flag_;
  bool keep_reshape_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_TENSOR_LAYOUT_TENSOR_REDISTRIBUTION_H_
