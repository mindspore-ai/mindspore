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

#include "frontend/parallel/ops_info/uniform_candidate_sampler_info.h"

#include <string>
#include <memory>
#include <vector>
#include <utility>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "include/utils/parallel_context.h"
#include "frontend/jit/ps/resource.h"

namespace mindspore {
namespace parallel {
Status UniformCandidateSamplerInfo::GetUniformSamplerAttrInt64(const std::string &args, int64_t *value) {
  auto iter = attrs_.find(args);
  if (iter == attrs_.end()) {
    MS_LOG(ERROR) << name_ << ": Can not find the attr for " << args;
    return FAILED;
  }
  MS_EXCEPTION_IF_NULL(iter->second);
  if (!iter->second->isa<Int64Imm>()) {
    MS_LOG(ERROR) << name_ << ": The type of attr is not int, the attr is " << args;
    return FAILED;
  }
  auto val_ptr = iter->second->cast<Int64ImmPtr>();
  MS_EXCEPTION_IF_NULL(val_ptr);
  *value = val_ptr->value();
  return SUCCESS;
}

Status UniformCandidateSamplerInfo::GetUniformSamplerAttrBool(const std::string &args, bool *value) {
  auto iter = attrs_.find(args);
  if (iter == attrs_.end()) {
    MS_LOG(ERROR) << name_ << ": Can not find the attr for " << args;
    return FAILED;
  }
  MS_EXCEPTION_IF_NULL(iter->second);
  if (!iter->second->isa<BoolImm>()) {
    MS_LOG(ERROR) << name_ << ": The type of attr is not bool, the attr is " << args;
    return FAILED;
  }
  auto val_ptr = iter->second->cast<BoolImmPtr>();
  MS_EXCEPTION_IF_NULL(val_ptr);
  *value = val_ptr->value();
  return SUCCESS;
}

Status UniformCandidateSamplerInfo::GetAttrs() {
  if (GetUniformSamplerAttrInt64(NUM_TRUE, &num_true_) != SUCCESS ||
      GetUniformSamplerAttrInt64(NUM_SAMPLED, &num_sampled_) != SUCCESS ||
      GetUniformSamplerAttrBool(UNIQUE_STRING, &unique_) != SUCCESS ||
      GetUniformSamplerAttrInt64(RANGE_MAX, &range_max_) != SUCCESS ||
      GetUniformSamplerAttrInt64(SEED, &seed_) != SUCCESS ||
      GetUniformSamplerAttrBool(REMOVE_ACCIDENTAL_HITS, &remove_accidental_hits_) != SUCCESS) {
    return FAILED;
  } else {
    MS_LOG(INFO) << name_ << ": The num_ture is " << num_true_ << " , the num_sampled is " << num_sampled_
                 << ", the unique is " << unique_ << " , the range max is " << range_max_ << " , the seed is " << seed_
                 << " , the remove_accidental_hits is " << remove_accidental_hits_;
  }
  return SUCCESS;
}

Status UniformCandidateSamplerInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  Dimensions input_strategy = stra.at(0);
  if (remove_accidental_hits_) {
    bool shard = std::any_of(input_strategy.begin(), input_strategy.end(), [](int64_t v) { return v > 1; });
    if (shard) {
      MS_LOG(ERROR) << name_ << ": When remove accidental_hits is true, the operation only supports (1,1) shard.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status UniformCandidateSamplerInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}
// There are three outputs
// sampled_candidates, true_expected_count, sampled_expected_count
// the sampled_candidates and sampled_expected_count is recomputed on each device with tensor map [-1]
// only true_expected_count is shard
Status UniformCandidateSamplerInfo::InferTensorMap() {
  TensorMap tensor_map;

  TensorMap sampled_tensor_map = {-1};
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }

  int32_t size = SizeToInt(inputs_shape_[0].size());
  for (int i = 0; i < size; ++i) {
    tensor_map.push_back(size - i - 1);
  }

  inputs_tensor_map_.push_back(tensor_map);

  // Output 1 sampled_candidates
  outputs_tensor_map_.push_back(sampled_tensor_map);
  // Output 2 true_expected_count
  outputs_tensor_map_.push_back(tensor_map);
  // Output 3 sampled_expected_count
  outputs_tensor_map_.push_back(sampled_tensor_map);

  return SUCCESS;
}

// The UniformCandidateSampler is not supported to be the last op of the net
Status UniformCandidateSamplerInfo::InferAsLossDivisor() {
  as_loss_divisor_ = 1;
  return SUCCESS;
}

Status UniformCandidateSamplerInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

std::vector<StrategyPtr> UniformCandidateSamplerInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input_split = {};
  Shapes splittable_input = {};
  size_t splitable_value = 1;
  if (remove_accidental_hits_) {
    splitable_value = 0;
  }
  for (size_t i = 0; i < inputs_shape_[0].size(); ++i) {
    input_split.push_back(splitable_value);
  }
  splittable_input.push_back(input_split);

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Generate strategies failed";
  }

  return sp_vector;
}

std::shared_ptr<Strategies> UniformCandidateSamplerInfo::GenerateBatchStrategies() {
  if (GetAttrs() != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Get attr failed";
  }
  CheckGlobalDeviceManager();
  Dimensions input_strategy(inputs_shape_[0].size(), 1);
  Strategies strategy_v = {input_strategy};
  return std::make_shared<Strategies>(strategy_v);
}

ReplaceGraphPtr UniformCandidateSamplerInfo::replace_graph(const CNodePtr &cnode) {
  const auto &input_dim = strategy_->GetInputDim();
  auto input_strategy = input_dim.at(0);
  // Only when the axis-1 is sharded, we need to modify the attribute
  if (input_strategy.size() == 2 && input_strategy[1] > 1) {
    if (ComputeReplaceGraph(cnode) != SUCCESS) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << name_ << ": ComputeReplaceGraph failed.";
    }
  }
  return replace_graph_;
}

Status UniformCandidateSamplerInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  GenerateGraph gen_g = GenerateGraph(attrs_);
  const auto &input_dim = strategy_->GetInputDim();
  auto input_strategy = input_dim.at(0);
  if (gen_g.Init(cnode) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateGraph Init failed";
    return FAILED;
  }
  MS_EXCEPTION_IF_ZERO("input_strategy[1]", input_strategy[1]);
  auto slice_num_true = num_true_ / input_strategy[1];
  // Get the attributes of the UnsortedSegmentMin
  Attr attr_num_ture = std::make_pair(NUM_TRUE, MakeValue(slice_num_true));
  Attr attr_num_sampled = std::make_pair(NUM_SAMPLED, MakeValue(num_sampled_));
  Attr attr_unique = std::make_pair(UNIQUE_STRING, MakeValue(unique_));
  Attr attr_range_max = std::make_pair(RANGE_MAX, MakeValue(range_max_));
  Attr attr_seed = std::make_pair(SEED, MakeValue(seed_));
  Attr attr_remove_accidental_hits = std::make_pair(REMOVE_ACCIDENTAL_HITS, MakeValue(remove_accidental_hits_));

  OperatorAttrs attrs = {attr_num_ture,  attr_num_sampled, attr_unique,
                         attr_range_max, attr_seed,        attr_remove_accidental_hits};
  auto new_sampler_op = gen_g.PushBack({gen_g.NewOpInst(UNIFORM_CANDIDATE_SAMPLER, attrs), gen_g.virtual_input_node()});
  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(new_sampler_op, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, new_sampler_op));

  return SUCCESS;
}

REGISTER(UniformCandidateSamplerInfo);
}  // namespace parallel
}  // namespace mindspore
