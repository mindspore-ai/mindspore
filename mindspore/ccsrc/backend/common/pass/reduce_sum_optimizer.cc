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

#include "backend/common/pass/reduce_sum_optimizer.h"
#include <vector>
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
namespace {
const int axis_input_index = 2;
}  // namespace

AnfNodePtr ReduceSumOptimizer::NewRankOp(const AnfNodePtr &cnode, const KernelGraphPtr &kernel_graph) const {
  std::vector<AnfNodePtr> rank_inputs;
  auto prim = std::make_shared<Primitive>(prim::kPrimRank->name());
  rank_inputs.push_back(NewValueNode(prim));
  auto prev_node = common::AnfAlgo::GetPrevNodeOutput(cnode, 1);
  rank_inputs.push_back(prev_node.first);
  auto rank_op = NewCNode(rank_inputs, kernel_graph);
  MS_EXCEPTION_IF_NULL(rank_op);
  rank_op->set_abstract(prev_node.first->abstract());
  return rank_op;
}

AnfNodePtr ReduceSumOptimizer::NewRangeOp(const AnfNodePtr &rank_op, const KernelGraphPtr &kernel_graph) const {
  std::vector<AnfNodePtr> range_inputs;
  auto prim = std::make_shared<Primitive>(prim::kPrimRange->name());
  range_inputs.push_back(NewValueNode(prim));
  // "start"
  auto start_ = NewValueNode(SizeToLong(0));
  MS_EXCEPTION_IF_NULL(start_);
  auto imm_start = std::make_shared<Int64Imm>(SizeToLong(0));
  start_->set_abstract(std::make_shared<abstract::AbstractScalar>(imm_start));
  range_inputs.push_back(start_);

  // "limit"
  range_inputs.push_back(rank_op);

  // "delta"
  auto delta_ = NewValueNode(SizeToLong(1));
  MS_EXCEPTION_IF_NULL(delta_);
  auto imm_delta = std::make_shared<Int64Imm>(SizeToLong(1));
  delta_->set_abstract(std::make_shared<abstract::AbstractScalar>(imm_delta));
  range_inputs.push_back(delta_);
  // new range op
  auto range_op = NewCNode(range_inputs, kernel_graph);
  MS_EXCEPTION_IF_NULL(range_op);
  range_op->set_abstract(rank_op->abstract());
  return range_op;
}

AnfNodePtr ReduceSumOptimizer::InsertAssistNode(const CNodePtr &cnode, const KernelGraphPtr &) const {
  // the input dim is unknown, need rank + range, don't supported now;
  MS_LOG(EXCEPTION)
    << "Can not support the case that input is dim unknown and axis is empty or axis contain value less 0. node: "
    << trace::DumpSourceLines(cnode);
}

// create a new assist value node to deal with the following two case
// 1: the axis_input is empty, the new tensor of the new value node should be 'range(shape.size())',
// the shape is the first input'shape of ReduceSum;
// 2: the value of axis_input contain the value less 0,
// the new tensor of the new value node should be "shape.size() + the_old_value_less_0",
// the shape is the first input'shape of ReduceSum;
AnfNodePtr ReduceSumOptimizer::NewAssistValueNode(const CNodePtr &cnode, const KernelGraphPtr &kernel_graph) const {
  // axis is a tuple ,maybe empty or contain a value less 0;
  auto axis_input = cnode->input(axis_input_index);
  if (IsValueNode<ValueTuple>(axis_input)) {
    std::vector<AnfNodePtr> new_inputs = {common::AnfAlgo::GetCNodePrimitiveNode(cnode)};
    new_inputs.push_back(cnode->input(1));
    auto value_node = axis_input->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<ValueTuple>()) {
      auto value_tuple = value->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(value_tuple);
      auto x_shape = dyn_cast<abstract::Shape>(cnode->input(1)->Shape());
      MS_EXCEPTION_IF_NULL(x_shape);
      std::vector<int64_t> axes_value;
      ValuePtr valuePtr = nullptr;
      if (value_tuple->value().empty()) {
        // case 1: tensor is empty;
        for (size_t i = 0; i < x_shape->shape().size(); i++) {
          axes_value.emplace_back(SizeToLong(i));
        }
      } else {
        // case 2: contain value less 0;
        for (auto &iter : value_tuple->value()) {
          auto item = GetValue<int64_t>(iter->cast<ScalarPtr>());
          if (item < 0) {
            (void)axes_value.emplace_back(item + static_cast<int64_t>(x_shape->shape().size()));
          } else {
            (void)axes_value.emplace_back(item);
          }
        }
      }
      valuePtr = MakeValue<std::vector<int64_t>>(axes_value);
      auto assist_node = NewValueNode(valuePtr);
      assist_node->set_abstract(std::make_shared<abstract::AbstractTensor>(kInt64, axes_value));
      auto assist_value_node = kernel_graph->NewValueNode(assist_node);
      new_inputs.push_back(assist_value_node);
      auto new_node = NewCNode(cnode, kernel_graph);
      MS_EXCEPTION_IF_NULL(new_node);
      new_node->set_inputs(new_inputs);
      return new_node;
    }
  }
  return nullptr;
}

const AnfNodePtr ReduceSumOptimizer::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  if (op_name != kReduceSumOpName) {
    MS_LOG(DEBUG) << "Current node is not: " << kReduceSumOpName << ", skip!";
    return nullptr;
  }
  if (!common::AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(DEBUG) << "Current node is not dynamic shape, skip!";
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  return NewAssistValueNode(cnode, kernel_graph);
}

const BaseRef ReduceSumOptimizer::DefinePattern() const {
  std::shared_ptr<Var> V = std::make_shared<CondVar>(UnVisited);
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}
}  // namespace opt
}  // namespace mindspore
