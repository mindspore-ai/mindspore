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

#include "frontend/parallel/ops_info/matmul_dds_info.h"

#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
/*
 * MatmulDDS has 4 input
 *  q, k: A 4D float used in transformer model,
 *  the shape is [num_heads * size_per_head // 16, bs * seq_len // 16, 16, 16], num_heads*size_per_head = embedding_size
 *  The shape is reshaped for cube. origin shape is
 *  (bs*seq_len, embedding_size) <=> (bs, num_heads, seq_len, size_per_head)
 *  local_mask: Local mask in sparse attention, the shape is
 *  (seq_len // 16, bs * block_size // 16, 16, 16).
 *  block_num = seq_len // block_size, block_size = 64, always.
 *  global_mask: Global mask in sparse attention,
 *  the shape is (bs * global_size // 16, seq_len // 16, 16, 16)
 *  seq_len = 1024, global_size = 256, always.
 *  Only bs and num_heads can be splited, thus the q[0] should at least be size_per_head,
 *  q[1] should at least be seq_len // 16. The strategy check can use bs/head from attrs.
 */
constexpr size_t kLocalMaskDim2 = 2;
constexpr size_t kLocalMaskDim3 = 3;
Status MatmulDDSInfo::CheckStrategies(const Strategies &stras) {
  if (stras.size() != MATMUL_DDS_INPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy. The strategys size should be 4.";
    return FAILED;
  }
  for (auto stra : stras) {
    if (stra.size() != MATMUL_DDS_STRATEGY_SIZE) {
      MS_LOG(ERROR) << name_
                    << ": Invalid strategy. The strategy size should be 4, but in current dim, "
                       "the strategy is"
                    << stra;
      return FAILED;
    }
  }
  MS_EXCEPTION_IF_ZERO("stras[0][0]", stras[0][0]);
  MS_EXCEPTION_IF_ZERO("stras[0][1]", stras[0][1]);
  if (stras[0][0] != stras[1][0] || num_heads_ % stras[0][0] != 0) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy. The strategys[0][0]:" << stras[0][0]
                  << " should be equal to strategys[1][0]:" << stras[1][0]
                  << " ,and should be divisible by num_heads: " << num_heads_;
    return FAILED;
  }
  if (stras[0][1] != stras[1][1] || stras[0][1] != stras[kLocalMaskDim2][1] ||
      stras[0][1] != stras[kLocalMaskDim3][0]) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy. The strategys[0][1]:" << stras[0][1]
                  << ", strategys[1][1]:" << stras[1][1] << ", strategys[2][1]:" << stras[kLocalMaskDim2][1]
                  << ", strategys[3][0]:" << stras[kLocalMaskDim3][0] << " should be the same.";
    return FAILED;
  }
  if (batch_size_ % stras[0][1] != 0) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy. The strategys[0][1]:" << stras[0][1]
                  << " should be divisible by batch_sizes:" << batch_size_;
    return FAILED;
  }
  for (size_t i = 2; i < stras[0].size(); ++i) {
    if (stras[0][i] != 1) {
      MS_LOG(ERROR) << name_ << ": Invalid strategy. The strategys[0][" << i << "] only support 1";
      return FAILED;
    }
  }
  for (size_t i = 2; i < stras[1].size(); ++i) {
    if (stras[1][i] != 1) {
      MS_LOG(ERROR) << name_ << ": Invalid strategy. The strategys[1][" << i << "] only support 1";
      return FAILED;
    }
  }
  for (size_t i = 0; i < stras[2].size(); ++i) {
    if (i != 1 && stras[2][i] != 1) {
      MS_LOG(ERROR) << name_ << ": Invalid strategy. The strategys[2][" << i << "] only support 1";
      return FAILED;
    }
  }
  for (size_t i = 1; i < stras[3].size(); ++i) {
    if (stras[3][i] != 1) {
      MS_LOG(ERROR) << name_ << ": Invalid strategy. The strategys[3][" << i << "] only support 1";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status MatmulDDSInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy.";
    return FAILED;
  }
  Strategies stras = strategy->GetInputDim();
  if (CheckStrategies(stras) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status MatmulDDSInfo::CheckStrategyForDynamicShape(const StrategyPtr &) {
  MS_LOG(ERROR) << name_
                << ": it does not support dynamic shape now, the inputs' shape: " << ShapesToString(inputs_shape_);
  return FAILED;
}

/*
 * device matrix is extended by the strategy0.
 */
Status MatmulDDSInfo::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  input_strategy_ = input_strategy;
  dev_matrix_shape_ = input_strategy;
  dev_matrix_shape_.push_back(1);
  dev_matrix_shape_.push_back(1);
  dev_matrix_shape_.push_back(1);
  dev_matrix_shape_origin_ = dev_matrix_shape_;
  return SUCCESS;
}

/*
 * q: [num_heads * size_per_head // 16, bs * seq_len // 16, 16, 16]
 * k: [num_heads * size_per_head // 16, bs * seq_len // 16, 16, 16]
 * local_mask: (block_num * block_size // 16, bs * block_size // 16, 16, 16)
 * global_mask: (bs * global_size // 16, seq_len // 16, 16, 16)
 * local_prob: (bs, num_heads, block_num, block_size // 16, block_size // 16, 16, 16)
 * global_prob: (bs, num_heads, block_num, global_size // 16, block_size // 16, 16, 16)
 * device_matrix: [num_heads_stra, bs_stra, 1, 1, 1, 1, 1]
 */
Status MatmulDDSInfo::InferTensorMap() {
  TensorMap input_tensor_map_q;
  // input_tensor_map_q [6, 5, -1, -1]
  for (size_t i = 0; i < inputs_shape_[0].size(); ++i) {
    if (i <= 1) {
      input_tensor_map_q.push_back(static_cast<int64_t>(inputs_shape_[0].size() + kLocalMaskDim3 - i - 1));
    } else {
      input_tensor_map_q.push_back(static_cast<int64_t>(MAP_NONE));
    }
  }
  TensorMap input_tensor_map_k;
  // input_tensor_map_k [6, 5, -1, -1]
  for (size_t i = 0; i < inputs_shape_[1].size(); ++i) {
    if (i <= 1) {
      input_tensor_map_k.push_back(static_cast<int64_t>(inputs_shape_[1].size() + kLocalMaskDim3 - i - 1));
    } else {
      input_tensor_map_k.push_back(static_cast<int64_t>(MAP_NONE));
    }
  }
  TensorMap input_tensor_map_local_mask;
  // input_tensor_map_local_mask [-1, 5, -1, -1]
  for (size_t i = 0; i < inputs_shape_[kLocalMaskDim2].size(); ++i) {
    if (i == 1) {
      input_tensor_map_local_mask.push_back(
        static_cast<int64_t>(inputs_shape_[kLocalMaskDim2].size() + kLocalMaskDim3 - kLocalMaskDim2));
    } else {
      input_tensor_map_local_mask.push_back(static_cast<int64_t>(MAP_NONE));
    }
  }
  TensorMap input_tensor_map_global_mask;
  // input_tensor_map_local_mask [5, -1, -1, -1]
  for (size_t i = 0; i < inputs_shape_[kLocalMaskDim3].size(); ++i) {
    if (i == 0) {
      input_tensor_map_global_mask.push_back(
        static_cast<int64_t>(inputs_shape_[kLocalMaskDim3].size() + kLocalMaskDim3 - kLocalMaskDim2));
    } else {
      input_tensor_map_global_mask.push_back(static_cast<int64_t>(MAP_NONE));
    }
  }
  TensorMap output_tensor_map_local_prob;
  // output_tensor_map_local_prob [5, 6, -1, -1, -1, -1, -1]
  for (size_t i = 0; i < dev_matrix_shape_origin_.size(); ++i) {
    if (i == 0) {
      output_tensor_map_local_prob.push_back(static_cast<int64_t>(dev_matrix_shape_origin_.size() - kLocalMaskDim2));
    } else if (i == 1) {
      output_tensor_map_local_prob.push_back(static_cast<int64_t>(dev_matrix_shape_origin_.size() - 1));
    } else {
      output_tensor_map_local_prob.push_back(static_cast<int64_t>(MAP_NONE));
    }
  }
  TensorMap output_tensor_map_global_prob;
  // output_tensor_map_global_prob [5, 6, -1, -1, -1, -1, -1]
  for (size_t i = 0; i < dev_matrix_shape_origin_.size(); ++i) {
    if (i == 0) {
      output_tensor_map_global_prob.push_back(static_cast<int64_t>(dev_matrix_shape_origin_.size() - kLocalMaskDim2));
    } else if (i == 1) {
      output_tensor_map_global_prob.push_back(static_cast<int64_t>(dev_matrix_shape_origin_.size() - 1));
    } else {
      output_tensor_map_global_prob.push_back(static_cast<int64_t>(MAP_NONE));
    }
  }
  inputs_tensor_map_.push_back(input_tensor_map_q);
  inputs_tensor_map_.push_back(input_tensor_map_k);
  inputs_tensor_map_.push_back(input_tensor_map_local_mask);
  inputs_tensor_map_.push_back(input_tensor_map_global_mask);
  outputs_tensor_map_.push_back(output_tensor_map_local_prob);
  outputs_tensor_map_.push_back(output_tensor_map_global_prob);
  return SUCCESS;
}

Status MatmulDDSInfo::GetAttrs() {
  if ((inputs_shape_.size() != MATMUL_DDS_INPUTS_SIZE) || (outputs_shape_.size() != MATMUL_DDS_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << ": Inputs shape size " << inputs_shape_.size() << " or outputs shape size "
                  << outputs_shape_.size() << " is wrong.";
    return FAILED;
  }
  auto iter = attrs_.find(BS);
  if (iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<Int64Imm>()) {
      batch_size_ = iter->second->cast<Int64ImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << ": The value of axis is not int64_t.";
      return FAILED;
    }
  }
  iter = attrs_.find(HEADS);
  if (iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<Int64Imm>()) {
      num_heads_ = iter->second->cast<Int64ImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << ": The value of axis is not int64_t.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status MatmulDDSInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  int64_t new_batch_size;
  int64_t new_num_heads;
  int64_t batch_shard_num = strategy_->GetInputDim()[0][1];
  int64_t heads_shard_num = strategy_->GetInputDim()[0][0];
  MS_EXCEPTION_IF_ZERO("batch_shard_num", batch_shard_num);
  MS_EXCEPTION_IF_ZERO("heads_shard_num", heads_shard_num);
  new_batch_size = batch_size_ / batch_shard_num;
  new_num_heads = num_heads_ / heads_shard_num;
  ValuePtr new_bs_value = MakeValue(new_batch_size);
  ValuePtr new_heads_value = MakeValue(new_num_heads);
  Attr attr_batch_size = std::make_pair(BS, new_bs_value);
  Attr attr_num_heads = std::make_pair(HEADS, new_heads_value);
  OperatorAttrs attrs = {attr_batch_size, attr_num_heads};
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed";
    return FAILED;
  }
  auto matmul_dds_node =
    gen_g.PushBack({gen_g.NewOpInst(MATMUL_DDS, attrs), gen_g.virtual_input_node(), gen_g.virtual_input_node(),
                    gen_g.virtual_input_node(), gen_g.virtual_input_node()});
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {
    std::make_pair(matmul_dds_node, 1),
    std::make_pair(matmul_dds_node, 2),
    std::make_pair(matmul_dds_node, 3),
    std::make_pair(matmul_dds_node, 4),
  };
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, matmul_dds_node));
  return SUCCESS;
}

ReplaceGraphPtr MatmulDDSInfo::replace_graph(const CNodePtr &cnode) {
  if (ComputeReplaceGraph(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": ComputeReplaceGraph failed.";
  }
  return replace_graph_;
}

std::vector<StrategyPtr> MatmulDDSInfo::GenerateOpStrategies(int64_t stage_id) {
  // to generate the first input's strategy
  Shape input0_split = {1, 1, 0, 0};
  Shapes splittable_input = {input0_split};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies failed";
  }

  // the others strategies are set by the first input's strategy
  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG(EXCEPTION) << name_ << ": The strategy is null or empty";
    }
    Strategies tmp_strategy;
    Dimensions q_strategy = sp->GetInputDim()[0];
    Dimensions k_strategy = q_strategy;
    Dimensions local_mask_strategy = {1, q_strategy[0], 1, 1};
    Dimensions global_mask_strategy = {q_strategy[0], 1, 1, 1};

    tmp_strategy.push_back(q_strategy);            // q
    tmp_strategy.push_back(k_strategy);            // k
    tmp_strategy.push_back(local_mask_strategy);   // local_mask
    tmp_strategy.push_back(global_mask_strategy);  // global_mask
    sp->ResetInputs(tmp_strategy);
  }
  return sp_vector;
}

Status MatmulDDSInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

REGISTER(MatmulDDSInfo);
}  // namespace parallel
}  // namespace mindspore
