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

#include "frontend/parallel/ops_info/dropout_do_mask_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "pipeline/jit/resource.h"
#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
static int64_t SEED_NUM = 1;

Status DropoutDoMaskInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (strategy == nullptr) {
    MS_LOG(ERROR) << name_ << ": The strategy is null";
    return FAILED;
  }

  Strategies stra = strategy->GetInputDim();
  if (stra.size() != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy size " << stra.size() << ", it must be 1";
    return FAILED;
  }

  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }

  // only check the input[0]
  Shapes input_shape = {inputs_shape_[0]};
  return CheckStrategyValue(strategy, input_shape);
}

Status DropoutDoMaskInfo::InferDevMatrixShape() {
  if (strategy_ == nullptr) {
    MS_LOG(ERROR) << name_ << ": The strategy is null";
    return FAILED;
  }

  Strategies strategy = strategy_->GetInputDim();
  if (strategy.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = strategy[0];
  return SUCCESS;
}

Status DropoutDoMaskInfo::InferTensorMap() {
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }

  Shape tensor_map_index;
  size_t size = inputs_shape_[0].size();
  // if the dimension of input is 4, and tensor_map_index is [3, 2, 1, 0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back(SizeToLong(size - i - 1));
  }

  // the input[1] do not need tensor map
  inputs_tensor_map_.push_back(tensor_map_index);   // input_0
  outputs_tensor_map_.push_back(tensor_map_index);  // output
  return SUCCESS;
}

Status DropoutDoMaskInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

std::vector<StrategyPtr> DropoutDoMaskInfo::GenerateOpStrategies(int64_t stage_id) {
  if (inputs_shape_.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": The inputs shape is empty";
  }

  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};
  Shapes used_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, used_inputs_shape, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies failed";
  }
  return sp_vector;
}

std::shared_ptr<Strategies> DropoutDoMaskInfo::GenerateBatchStrategies() {
  Dimensions strategy(inputs_shape_[0].size() - 1, 1);
  (void)strategy.insert(strategy.cbegin(), stage_device_size_);
  Strategies strategy_v = {strategy};
  return std::make_shared<Strategies>(strategy_v);
}

size_t GetNonMonadInputSize(const CNodePtr &cnode) {
  size_t cnode_non_monad_size = cnode->size();
  for (auto &input : cnode->inputs()) {
    if (HasAbstractMonad(input)) {
      cnode_non_monad_size--;
    }
  }
  return cnode_non_monad_size;
}

PrimitivePtr GetDropoutGenMaskPrim(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() != DROPOUT_DO_MASK_CNODE_INPUT_SIZE) {
    MS_LOG(EXCEPTION) << "The size of dropout do mask cnode's inputs must be " << DROPOUT_DO_MASK_CNODE_INPUT_SIZE;
  }

  AnfNodePtr dropout_gen_mask = cnode->input(DROPOUT_GEN_MASK_INDEX);
  MS_EXCEPTION_IF_NULL(dropout_gen_mask);
  if (!dropout_gen_mask->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "The dropout do mask cnode's input[" << DROPOUT_GEN_MASK_INDEX << "] must be a cnode";
  }

  auto dropout_gen_mask_cnode = dropout_gen_mask->cast<CNodePtr>();
  size_t cnode_non_monad_size = GetNonMonadInputSize(dropout_gen_mask_cnode);
  if (cnode_non_monad_size != DROPOUT_GEN_MASK_CNODE_INPUT_SIZE) {
    MS_LOG(EXCEPTION) << "The size of dropout gen mask cnode's inputs must be " << DROPOUT_GEN_MASK_CNODE_INPUT_SIZE;
  }
  if (!IsValueNode<Primitive>(dropout_gen_mask_cnode->input(0))) {
    MS_LOG(EXCEPTION) << "The input[0] of dropout gen mask cnode is not primitive";
  }

  ValueNodePtr value_node = dropout_gen_mask_cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  PrimitivePtr prim = value_node->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->name() != DROPOUT_GEN_MASK) {
    MS_LOG(EXCEPTION) << "The primitive name is not DropoutGenMask";
  }
  return prim;
}

void SetGenMaskShape(const CNodePtr &cnode, const Shape &input_slice_shape) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() != DROPOUT_DO_MASK_CNODE_INPUT_SIZE) {
    MS_LOG(EXCEPTION) << "The size of dropout do mask cnode's inputs must be " << DROPOUT_DO_MASK_CNODE_INPUT_SIZE;
  }

  AnfNodePtr dropout_gen_mask = cnode->input(DROPOUT_GEN_MASK_INDEX);
  MS_EXCEPTION_IF_NULL(dropout_gen_mask);
  if (!dropout_gen_mask->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "The dropout do mask cnode's input[" << DROPOUT_GEN_MASK_INDEX << "] must be a cnode.";
  }

  auto dropout_gen_mask_cnode = dropout_gen_mask->cast<CNodePtr>();
  size_t cnode_non_monad_size = GetNonMonadInputSize(dropout_gen_mask_cnode);
  if (cnode_non_monad_size != DROPOUT_GEN_MASK_CNODE_INPUT_SIZE) {
    MS_LOG(EXCEPTION) << "The size of dropout gen mask cnode's inputs must be " << DROPOUT_GEN_MASK_CNODE_INPUT_SIZE;
  }

  if (!IsValueNode<ValueTuple>(dropout_gen_mask_cnode->input(1))) {
    MS_LOG(EXCEPTION) << "The input[1] of dropout gen mask cnode is not ValueTuple.";
  }

  FuncGraphPtr func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(EXCEPTION) << "Failure: AddNode error since manager is nullptr.";
  }
  ValuePtr new_shape = MakeValue(input_slice_shape);
  AnfNodePtr val = NewValueNode(new_shape);
  dropout_gen_mask_cnode->set_input(kIndex1, val);
}

// DropoutDoMask needs to be used together with DropoutGenMask. Only the first input tensor of DropoutGenMask is
// split. Find the DropoutGenMask node in the anf graph according to DropoutDoMask node, and modify the input shape
// of DropoutGenMask according to the strategy of DropoutDoMask. When the DropoutDoMask performs repeated calculation
// and both seeds of DropoutGenMask are 0, two new seeds are automatically generated for DropoutGenMask.
std::vector<Operator> DropoutDoMaskInfo::GetDropoutGenMaskReplaceOp(const CNodePtr &cnode) {
  std::vector<Operator> replace_ops;
  MS_EXCEPTION_IF_NULL(cnode);
  PrimitivePtr prim = GetDropoutGenMaskPrim(cnode);
  MS_EXCEPTION_IF_NULL(prim);

  if (inputs_tensor_info_.empty()) {
    MS_LOG(EXCEPTION) << "The tensor info of dropout do mask is empty";
  }

  if (cnode->inputs().size() != DROPOUT_DO_MASK_CNODE_INPUT_SIZE) {
    MS_LOG(EXCEPTION) << "The size of dropout do mask cnode's inputs must be " << DROPOUT_DO_MASK_CNODE_INPUT_SIZE;
  }

  if (!cnode->input(DROPOUT_DO_MASK_KEEP_PROB_INDEX)->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The keep prob of dropout do mask is not value node";
  }

  ValuePtr keep_prob = GetValueNode(cnode->input(DROPOUT_DO_MASK_KEEP_PROB_INDEX));
  MS_EXCEPTION_IF_NULL(keep_prob);
  auto attr = prim->attrs();
  if ((attr.find(SEED0) == attr.end()) || (attr.find(SEED1) == attr.end())) {
    MS_LOG(EXCEPTION) << "The attrs of dropout gen mask must be have seed0 and seed1";
  }

  Shape input_slice_shape = inputs_tensor_info_[0].slice_shape();
  int64_t seed_0 = GetValue<int64_t>(attr[SEED0]);
  int64_t seed_1 = GetValue<int64_t>(attr[SEED1]);
  if ((seed_0 == 0) && (seed_1 == 0) && (repeated_calc_num_ > 1)) {
    seed_0 = SEED_NUM;
    seed_1 = SEED_NUM;
    SEED_NUM++;
  } else {
    SetGenMaskShape(cnode, input_slice_shape);
    MS_LOG(DEBUG) << "The input slice shape droupout is " << ShapeToString(input_slice_shape);
    return replace_ops;
  }
  ValuePtr new_shape = MakeValue(input_slice_shape);
  Attr attr_0 = std::make_pair(SEED0, MakeValue(seed_0));
  Attr attr_1 = std::make_pair(SEED1, MakeValue(seed_1));
  OperatorAttrs attrs = {attr_0, attr_1};
  Attr param_0 = std::make_pair(SHAPE, new_shape);
  Attr param_1 = std::make_pair(KEEP_PROB, keep_prob);
  OperatorParams params = {std::make_pair(param_0, 1), std::make_pair(param_1, 2)};
  OperatorArgs args = std::make_pair(attrs, params);
  Operator replace_op = {std::make_pair(DROPOUT_GEN_MASK, args)};
  replace_ops.push_back(replace_op);
  return replace_ops;
}

static void ReplaceOneOp(const Operator &replace_op, const CNodePtr &node) {
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:AddNode error since manager is nullptr";
  }
  std::string instance_name = CreateInstanceName(node, 0);
  std::vector<AnfNodePtr> replace_input;
  replace_input = ReplaceOpInput(replace_op, instance_name, node);
  if (node->inputs().size() == DROPOUT_DO_MASK_CNODE_INPUT_SIZE) {
    replace_input.push_back(node->input(3));
  }
  CNodePtr replace_node = func_graph->NewCNode(replace_input);
  MS_EXCEPTION_IF_NULL(replace_node);
  ScopePtr scope = node->scope();
  MS_EXCEPTION_IF_NULL(scope);
  replace_node->set_scope(scope);
  replace_node->set_in_forward_flag(true);
  replace_input[0]->set_scope(scope);
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(replace_node->input(0));
  PrimitivePtr origin_prim = GetValueNode<PrimitivePtr>(node->input(0));
  SetUserAttrs(origin_prim->attrs(), prim);
  (void)manager->Replace(node, replace_node);
}

void DropoutDoMaskInfo::ReplaceNodeInputOrAttrs() {
  for (auto &cnode : cnodes_) {
    MS_EXCEPTION_IF_NULL(cnode);
    std::vector<Operator> replace_op = GetDropoutGenMaskReplaceOp(cnode);
    if (replace_op.empty()) {
      MS_LOG(DEBUG) << name_ << ": No need to replace dropout_gen_mask";
      return;
    }
    if (cnode->inputs().size() != DROPOUT_DO_MASK_CNODE_INPUT_SIZE) {
      MS_LOG(EXCEPTION) << name_ << ": The size of drop out do mask cnode's input is not "
                        << DROPOUT_DO_MASK_CNODE_INPUT_SIZE;
    }
    ReplaceOneOp(replace_op[0], cnode->input(DROPOUT_GEN_MASK_INDEX)->cast<CNodePtr>());
  }
}

REGISTER(DropoutDoMaskInfo);
}  // namespace parallel
}  // namespace mindspore
