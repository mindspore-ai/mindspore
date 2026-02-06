/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/unique_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/strategy.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
/*
 * unique has one input, two outputs. Currently, unique cannot be split.
 */
Status UniqueInfo::InferTensorMap() {
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  for (auto shp : inputs_shape_) {
    TensorMap out_tensor_map;
    TensorMap in_tensor_map;
    for (size_t i = 0; i < shp.size(); ++i) {
      in_tensor_map.push_back(MAP_NONE);
      out_tensor_map.push_back(MAP_NONE);
    }
    inputs_tensor_map_.push_back(in_tensor_map);
    outputs_tensor_map_.push_back(out_tensor_map);
    outputs_tensor_map_.push_back(out_tensor_map);
  }
  return SUCCESS;
}

Status UniqueInfo::InferDevMatrixShape() {
  dev_matrix_shape_.push_back(stage_device_size_);
  return SUCCESS;
}

Status UniqueInfo::CheckStrategy(const StrategyPtr &strategy) {
  Strategies stras = strategy->GetInputDim();
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  for (Dimensions stra : stras) {
    if (stra.size() != UNIQUE_INPUT_SIZE) {
      MS_LOG(ERROR) << name_ << " : Invalid strategy.";
      return FAILED;
    }
  }

  if (stras[0][0] != 1) {
    MS_LOG(ERROR) << "Currently, unique only support repeat calculate in all devices";
    return FAILED;
  }
  return SUCCESS;
}

Status UniqueInfo::GetAttrs() {
  if ((inputs_shape_.size() != UNIQUE_INPUTS_SIZE) || (outputs_shape_.size() != UNIQUE_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << ": Inputs shape size " << inputs_shape_.size() << " or outputs shape size "
                  << outputs_shape_.size() << " is wrong.";
    return FAILED;
  }
  return SUCCESS;
}

Status UniqueInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> UniqueInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split;
  (void)input0_split.emplace_back(0);
  Shapes splittable_inputs = {input0_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": GenerateStrategiesForIndependentInputs failed";
  }

  return sp_vector;
}
REGISTER(UniqueInfo);
}  // namespace parallel
}  // namespace mindspore
