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

#include "frontend/parallel/ops_info/lin_space_info.h"

#include <utility>

#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/graph_util/generate_graph.h"

namespace mindspore {
namespace parallel {
Status LinSpaceInfo::GetAttrs() {
  auto output_0_shape = outputs_shape_.at(0);
  output_size_ = output_0_shape.at(0);
  return SUCCESS;
}

Status LinSpaceInfo::CheckStrategy(const StrategyPtr &strategy) {
  auto strategies = strategy->GetInputDim();
  if (strategies.size() != 1 && strategies[0].size() != 1) {
    MS_LOG(ERROR) << name_ << ": The shape of input_strategy must be [1, 1], but got strategy "
                  << StrategyToString(strategies);
    return FAILED;
  }

  split_num_ = strategies[0][0];
  if (split_num_ <= 0) {
    MS_LOG(ERROR) << name_ << ": Each element in strategy must be a positive integer, but got "
                  << StrategyToString(strategies);
    return FAILED;
  }
  if (outputs_shape_[0][0] > 0 && (outputs_shape_[0][0] % split_num_ != 0)) {
    MS_LOG(ERROR) << name_ << ": The strategy is " << StrategyToString(strategies) << ", output size is  "
                  << output_size_ << " cannot be divisible by strategy value " << split_num_;
    return FAILED;
  }
  if (stage_device_size_ % split_num_ != 0) {
    MS_LOG(ERROR) << name_ << ": The strategy is " << StrategyToString(strategies)
                  << ", the device size in this stage is " << stage_device_size_
                  << " cannot be divisible by the strategy value " << split_num_;
    return FAILED;
  }
  return SUCCESS;
}

Status LinSpaceInfo::CheckStrategyForDynamicShape(const StrategyPtr &strategy) {
  Strategies strategies = strategy->GetInputDim();
  auto x_strategy = strategies[0];
  if (x_strategy[0] != 1) {
    MS_LOG(ERROR) << name_ << ": it can not be split if it's dynamic shape, the strategy is "
                  << ShapesToString(strategies) << ", the output shape: " << ShapeToString(outputs_shape_[0]);
    return FAILED;
  }
  return SUCCESS;
}

Status LinSpaceInfo::InferDevMatrixShape() {
  dev_matrix_shape_.clear();
  auto strategies = strategy_->GetInputDim();
  auto split_strategy = strategies.at(0);
  dev_matrix_shape_.push_back(split_strategy.at(0));
  MS_LOG(INFO) << name_ << ": The device matrix is " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status LinSpaceInfo::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  inputs_tensor_map_.assign(inputs_shape_.size(), TensorMap{});
  (void)outputs_tensor_map_.emplace_back(TensorMap{0});
  return SUCCESS;
}

std::vector<StrategyPtr> LinSpaceInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape split_input(1, 1);
  Shape split_shape(1, output_size_);
  Shapes splittable_inputs = {split_input};
  Shapes splittable_shapes = {split_shape};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, splittable_shapes, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for independent inputs() failed.";
  }
  if (sp_vector.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": No available strategy.";
  }
  return sp_vector;
}

std::shared_ptr<Strategies> LinSpaceInfo::GenerateBatchStrategies() {
  if (InferAttrs() != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Infer attrs failed";
  }

  int64_t dev_num = g_device_manager->stage_device_num();
  Strategies strategies = {Dimensions{dev_num}};
  return std::make_shared<Strategies>(strategies);
}

ReplaceGraphPtr LinSpaceInfo::replace_graph(const CNodePtr &cnode) {
  if (split_num_ > 1 && ComputeReplaceGraph(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": ComputeReplaceGraph failed.";
  }
  return replace_graph_;
}

Status LinSpaceInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  if (split_num_ == 1) {
    MS_LOG(INFO) << name_ << ": split num is 1, no need to replace graph";
    return SUCCESS;
  }

  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed.";
    return FAILED;
  }

  MS_EXCEPTION_IF_ZERO("split_num_", split_num_);
  int64_t slice_output_size = output_size_ / split_num_;
  auto sub = gen_g.PushBack({gen_g.NewOpInst(SUB), gen_g.virtual_input_node(), gen_g.virtual_input_node()});
  auto dtype = gen_g.PushBack({gen_g.NewOpInst(DTYPE), sub});
  auto dtype_id =
    gen_g.PushBack({gen_g.NewOpInst(DTYPETOENUM), CreateStringImm("DtypeToEnum"), CreateStringImm("dtype"), dtype});
  AnfNodePtr interval = nullptr;
  if (output_size_ == 2) {
    interval = sub;
  } else {
    auto interval_divsor = gen_g.PushBack({gen_g.NewOpInst(CAST), CreateInt32Tensor(output_size_ - 2), dtype_id});
    interval = gen_g.PushBack({gen_g.NewOpInst(DIV), sub, interval_divsor});
  }

  // new_start = start + slice_id * slice_output_size * interval
  // new_end = new_start + (slice_output_size - 1) * interval
  // new_x = slice_output_size
  InferSliceId();
  auto offset_size =
    gen_g.PushBack({gen_g.NewOpInst(CAST), CreateInt32Tensor(slice_id_ * slice_output_size), dtype_id});
  auto start_offset = gen_g.PushBack({gen_g.NewOpInst(MUL), interval, offset_size});
  auto new_start = gen_g.PushBack({gen_g.NewOpInst(ADD), gen_g.virtual_input_node(), start_offset});
  auto end_start_offset_size =
    gen_g.PushBack({gen_g.NewOpInst(CAST), CreateInt32Tensor(slice_output_size - 1), dtype_id});
  auto start_end_offset = gen_g.PushBack({gen_g.NewOpInst(MUL), interval, end_start_offset_size});
  auto new_end = gen_g.PushBack({gen_g.NewOpInst(ADD), new_start, start_end_offset});
  auto lin_space = gen_g.PushBack({gen_g.NewOpInst(LIN_SPACE), new_start, new_end, CreatInt64Imm(slice_output_size)});

  std::vector<std::pair<AnfNodePtr, int64_t>> inputs_nodes = {std::make_pair(sub, 1), std::make_pair(sub, 2),
                                                              std::make_pair(new_start, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(inputs_nodes, lin_space));
  return SUCCESS;
}

void LinSpaceInfo::InferSliceId() {
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->rank_index_in_stage();
  slice_id_ = rank;
  if (repeated_calc_num_ > 1) {
    if (repeated_num_in_dev_matrix_right_) {
      slice_id_ /= dev_matrix_shape_.back();
    } else {
      slice_id_ %= dev_matrix_shape_.front();
    }
  }
}

Status LinSpaceInfo::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }
  // No need to insert mirror ops
  if (mirror_ops_.empty()) {
    return SUCCESS;
  }
  if (mirror_ops_.size() == kSizeTwo) {
    // Push empty mirror op for nums
    (void)mirror_ops_.emplace_back(OperatorVector());
  }
  return SUCCESS;
}

REGISTER(LinSpaceInfo);
}  // namespace parallel
}  // namespace mindspore
