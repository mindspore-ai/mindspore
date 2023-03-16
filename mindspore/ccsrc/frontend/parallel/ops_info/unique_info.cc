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
#include "mindspore/core/ops/core_ops.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/strategy.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"
#include "ps/ps_context.h"
#include "include/backend/distributed/embedding_cache/embedding_cache_utils.h"
#endif

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
    MS_LOG(EXCEPTION) << name_ << ": GenerateStrategiesForIndependentInputs failed";
  }

  return sp_vector;
}
#if defined(__linux__) && defined(WITH_BACKEND)
Status UniqueInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed";
    return FAILED;
  }

  int64_t bias = 0;
  if (ps::PSContext::instance()->enable_distributed_mindrt()) {
    bias = static_cast<int64_t>(embedding_cache_table_manager.cache_indices_lower_bound());
  }
  auto slice_size = SizeToLong(embedding_cache_table_manager.cache_size());

  auto sub = gen_g.PushBack({gen_g.NewOpInst(SUB), gen_g.virtual_input_node(), CreateInt32Tensor(bias)});
  auto relu = gen_g.PushBack({gen_g.NewOpInst(RELU), sub});
  auto minimum = gen_g.PushBack({gen_g.NewOpInst(MINIMUM), relu, CreateInt32Tensor(slice_size - 1)});
  auto equal = gen_g.PushBack({gen_g.NewOpInst(EQUAL), sub, minimum});
  auto unique = gen_g.PushBack({gen_g.NewOpInst(replace_op_name_), gen_g.virtual_input_node()});
  // Use name of tuple_getitem instance in mindspore.ops.functional, not the Primitive name
  const std::string &tuple_getitem_op = "tuple_getitem";
  auto tuple_getitem_0 = gen_g.PushBack({gen_g.NewOpInst(tuple_getitem_op), unique, CreatInt64Imm(0)});
  auto tuple_getitem_1 = gen_g.PushBack({gen_g.NewOpInst(tuple_getitem_op), unique, CreatInt64Imm(1)});
  auto dtype = gen_g.PushBack({gen_g.NewOpInst(DTYPE), tuple_getitem_1});
  auto cast = gen_g.PushBack({gen_g.NewOpInst(CAST), equal, dtype});
  auto mul = gen_g.PushBack({gen_g.NewOpInst(MUL), tuple_getitem_1, cast});

  Attr attr_op = std::make_pair(OP, MakeValue(REDUCE_OP_SUM));
  OperatorAttrs attrs = {attr_op};
  AnfNodePtr reduce_op = gen_g.PushBack({gen_g.NewOpInst(ALL_REDUCE, attrs), mul});
  // Use name of make_tuple instance in mindspore.ops.functional, not the Primitive name
  const std::string &make_tuple_op = "make_tuple";
  auto make_tuple = gen_g.PushBack({gen_g.NewOpInst(make_tuple_op), tuple_getitem_0, reduce_op});
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(sub, 1), std::make_pair(unique, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, make_tuple));
  return SUCCESS;
}
#endif
ReplaceGraphPtr UniqueInfo::replace_graph(const CNodePtr &cnode) {
#if defined(__linux__) && defined(WITH_BACKEND)
  if (ps::PsDataPrefetch::GetInstance().cache_enable()) {
    auto inputs = cnode->inputs();
    if (inputs.empty()) {
      MS_LOG(EXCEPTION) << "Invalid inputs";
    }
    const auto &primitive = GetValueNode<PrimitivePtr>(inputs[0]);
    const auto &attr = primitive->GetAttr("cache_enable");
    if (attr == nullptr) {
      return nullptr;
    }
    auto need_mask = GetValue<bool>(attr);
    if (!need_mask) {
      return nullptr;
    }
    if (ComputeReplaceGraph(cnode) != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << ": ComputeReplaceGraph failed.";
    }
    return replace_graph_;
  }
#endif
  return nullptr;
}

REGISTER(UniqueInfo);
}  // namespace parallel
}  // namespace mindspore
