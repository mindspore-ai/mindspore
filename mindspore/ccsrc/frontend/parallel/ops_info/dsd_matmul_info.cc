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

#include "frontend/parallel/ops_info/dsd_matmul_info.h"

#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
/*
 * DSDMatmuul has 3 input
 *  input_w1, the shape is (batch_size, head, block_num, head_size // 16, block_size//16, 16, 16)
 *  input_w2, the shape is (batch_size, head, block_num, global_size // 16, head_size//16, 16, 16)
 *  input_v, the shape is (batch_size * seq_len // 16, head * v_embedding//16, 16, 16)
 *  block_num = seq_len // block_size, block_size = 64, always.
 *  Only bs and num_heads can be splited.
 *  output shape is (batch_size, head, v_embedding // 16, seq_len//16, 16, 16)
 *  v_embedding = input_v_shape[1] * 16 // head
 *  batch_size = input_w1_shape[0]
 *  head = input_w1_shape[1], block_num = input_w1_shape[2]
 *  block_size = input_w1_shape[4] * 16
 *  head_size = input_w1_shape[3] * 16
 *  global_size = input_w2_shape[3] * 16
 *  v_embedding = input_v_shape[1] * 16 // head
 */
Status DSDMatmulInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy.";
    return FAILED;
  }
  Strategys stras = strategy->GetInputDim();
  if (stras.size() != DSD_MATMUL_INPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy. The strategys size should be 3.";
    return FAILED;
  }
  if (stras[0].size() != DSD_MATMUL_STRATEGY_W_SIZE || stras[1].size() != DSD_MATMUL_STRATEGY_W_SIZE ||
      stras[2].size() != DSD_MATMUL_STRATEGY_V_SIZE) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy size.";
    return FAILED;
  }
  int64_t batch_size_shard_num = stras[0][0];
  int64_t num_heads_shard_num = stras[0][1];
  for (size_t i = 0; i < stras.size(); ++i) {
    for (size_t j = 0; j < stras[i].size(); ++j) {
      if (j == 0) {
        if (stras[i][j] != batch_size_shard_num) {
          MS_LOG(ERROR) << name_ << ": The strategys[" << i << "][" << j
                        << "] should equal to strategys[0][0]:" << batch_size_shard_num;
          return FAILED;
        }
      } else if (j == 1) {
        if (stras[i][j] != num_heads_shard_num) {
          MS_LOG(ERROR) << name_ << ": The strategys[" << i << "][" << j
                        << "] should equal to strategys[0][1]:" << num_heads_shard_num;
          return FAILED;
        }
      } else {
        if (stras[i][j] != 1) {
          MS_LOG(ERROR) << name_ << ": The strategys[" << i << "][" << j << "] should be 1";
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

/*
 * device matrix use the strategy0.
 */
Status DSDMatmulInfo::InferDevMatrixShape() {
  Strategys stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  input_strategy_ = input_strategy;
  dev_matrix_shape_ = input_strategy;
  return SUCCESS;
}

/*
 *  input_w1, the shape is (batch_size, head, block_num, head_size // 16, block_size//16, 16, 16)
 *  input_w2, the shape is (batch_size, head, block_num, global_size // 16, head_size//16, 16, 16)
 *  input_v, the shape is (batch_size * seq_len // 16, head * v_embedding//16, 16, 16)
 *  output shape is (batch_size, head, v_embedding // 16, seq_len//16, 16, 16)
 *  device_matrix = (batch_size_stra, head_stra, 1, 1, 1, 1, 1)
 */
Status DSDMatmulInfo::InferTensorMap() {
  TensorMap input_tensor_map_w1;
  // input_tensor_map_w1 [6, 5, -1, -1, -1, -1, -1]
  for (size_t i = 0; i < inputs_shape_[0].size(); ++i) {
    if (i <= 1) {
      input_tensor_map_w1.push_back((int64_t)(inputs_shape_[0].size() - i - 1));
    } else {
      input_tensor_map_w1.push_back((int64_t)(MAP_NONE));
    }
  }
  TensorMap input_tensor_map_w2;
  // input_tensor_map_w2 [6, 5, -1, -1, -1, -1, -1]
  for (size_t i = 0; i < inputs_shape_[1].size(); ++i) {
    if (i <= 1) {
      input_tensor_map_w2.push_back((int64_t)(inputs_shape_[1].size() - i - 1));
    } else {
      input_tensor_map_w2.push_back((int64_t)(MAP_NONE));
    }
  }
  TensorMap input_tensor_map_v;
  // input_tensor_map_local_mask [6, 5, -1, -1]
  for (size_t i = 0; i < inputs_shape_[2].size(); ++i) {
    if (i <= 1) {
      input_tensor_map_v.push_back((int64_t)(inputs_shape_[2].size() + 2 - i));
    } else {
      input_tensor_map_v.push_back((int64_t)(MAP_NONE));
    }
  }
  TensorMap output_tensor_map;
  // output_tensor_map [6, 5, -1, -1, -1, -1]
  for (size_t i = 0; i < outputs_shape_[0].size(); ++i) {
    if (i <= 1) {
      output_tensor_map.push_back((int64_t)(outputs_shape_[0].size() - i));
    } else {
      output_tensor_map.push_back((int64_t)(MAP_NONE));
    }
  }
  inputs_tensor_map_.push_back(input_tensor_map_w1);
  inputs_tensor_map_.push_back(input_tensor_map_w2);
  inputs_tensor_map_.push_back(input_tensor_map_v);
  outputs_tensor_map_.push_back(output_tensor_map);
  return SUCCESS;
}

Status DSDMatmulInfo::GetAttrs() {
  if ((inputs_shape_.size() != DSD_MATMUL_INPUTS_SIZE) || (outputs_shape_.size() != DSD_MATMUL_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << ": Inputs shape size " << inputs_shape_.size() << " or outputs shape size "
                  << outputs_shape_.size() << " is wrong.";
    return FAILED;
  }
  return SUCCESS;
}

Status DSDMatmulInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status DSDMatmulInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

std::vector<StrategyPtr> DSDMatmulInfo::GenerateOpStrategies(int64_t stage_id) {
  // to generate the first input's strategy
  Shape input0_split = {1, 1, 0, 0, 0, 0, 0};
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
    Strategys tmp_strategy;
    Dimensions input_w1_strategy = sp->GetInputDim()[0];
    Dimensions input_w2_strategy = input_w1_strategy;
    Dimensions input_v_strategy = {input_w1_strategy[0], input_w1_strategy[1], 1, 1};

    tmp_strategy.push_back(input_w1_strategy);  // input_w1
    tmp_strategy.push_back(input_w2_strategy);  // input_w2
    tmp_strategy.push_back(input_v_strategy);   // input_v
    sp->ResetInputs(tmp_strategy);
  }
  return sp_vector;
}

Status DSDMatmulInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }
}  // namespace parallel
}  // namespace mindspore
