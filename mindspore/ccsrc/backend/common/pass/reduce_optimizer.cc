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

#include "backend/common/pass/reduce_optimizer.h"
#include <vector>
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int axis_input_index = 2;
}  // namespace

AnfNodePtr ReduceOptimizer::NewRankOp(const AnfNodePtr &cnode, const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(kernel_graph);
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

AnfNodePtr ReduceOptimizer::NewRangeOp(const AnfNodePtr &rank_op, const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(rank_op);
  MS_EXCEPTION_IF_NULL(kernel_graph);
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

AnfNodePtr ReduceOptimizer::InsertAssistNode(const CNodePtr &cnode, const KernelGraphPtr &) const {
  // the input dim is unknown, need rank + range, don't supported now;
  MS_LOG(EXCEPTION)
    << "Can not support the case that input is dim unknown and axis is empty or axis contain value less 0. node: "
    << trace::DumpSourceLines(cnode);
}

AnfNodePtr ReduceOptimizer::CreateValueNodeWithVector(const CNodePtr &cnode, const KernelGraphPtr &kernel_graph,
                                                      const std::vector<int64_t> &axis) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto new_value_node = NewValueNode(MakeValue<std::vector<int64_t>>(axis));
  MS_EXCEPTION_IF_NULL(new_value_node);
  new_value_node->set_abstract(std::make_shared<abstract::AbstractTensor>(kInt64, axis));
  auto assist_value_node = kernel_graph->NewValueNode(new_value_node);

  std::vector<AnfNodePtr> new_inputs = {common::AnfAlgo::GetCNodePrimitiveNode(cnode), cnode->input(1),
                                        assist_value_node};
  auto new_node = NewCNode(cnode, kernel_graph);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_inputs(new_inputs);
  return new_node;
}

AnfNodePtr ReduceOptimizer::HandleAxisWithEmptyTensor(const CNodePtr &cnode, const KernelGraphPtr &kernel_graph,
                                                      const AnfNodePtr &axis_input) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(axis_input);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto value_node = axis_input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->shape() == ShapeVector({0})) {
      auto x_shape = dyn_cast<abstract::Shape>(cnode->input(1)->Shape());
      MS_EXCEPTION_IF_NULL(x_shape);
      auto shape_vector = x_shape->shape();
      std::vector<int64_t> axis_value;
      for (size_t i = 0; i < shape_vector.size(); ++i) {
        (void)axis_value.emplace_back(SizeToLong(i));
      }
      MS_LOG(INFO) << "Change axis from () to " << axis_value;
      return CreateValueNodeWithVector(cnode, kernel_graph, axis_value);
    }
  }
  return nullptr;
}

// create a new assist value node to deal with the following two case
// 1: the axis_input is empty, the new tensor of the new value node should be 'range(shape.size())',
// the shape is the first input'shape of ReduceSum;
// 2: the value of axis_input contain the value less 0,
// the new tensor of the new value node should be "shape.size() + the_old_value_less_0",
// the shape is the first input'shape of ReduceSum;
AnfNodePtr ReduceOptimizer::NewAssistValueNode(const CNodePtr &cnode, const KernelGraphPtr &kernel_graph) const {
  // axis is a tuple ,maybe empty or contain a value less 0;
  if (cnode->inputs().size() <= axis_input_index) {
    return nullptr;
  }
  auto axis_input = cnode->input(axis_input_index);
  MS_EXCEPTION_IF_NULL(axis_input);
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
      std::vector<int64_t> axis_value;
      if (value_tuple->value().empty()) {
        // case 1: tensor is empty;
        for (size_t i = 0; i < x_shape->shape().size(); i++) {
          axis_value.emplace_back(SizeToLong(i));
        }
      } else {
        // case 2: contain value less 0;
        for (auto &iter : value_tuple->value()) {
          auto item = GetValue<int64_t>(iter->cast<ScalarPtr>());
          if (item < 0 && !(IsDynamicRank(x_shape->shape()))) {
            (void)axis_value.emplace_back(item + static_cast<int64_t>(x_shape->shape().size()));
          } else {
            (void)axis_value.emplace_back(item);
          }
        }
      }

      return CreateValueNodeWithVector(cnode, kernel_graph, axis_value);
    }
  } else if (axis_input->isa<ValueNode>()) {
    return HandleAxisWithEmptyTensor(cnode, kernel_graph, axis_input);
  }
  return nullptr;
}

bool IsReduceOp(const std::string &op_name) {
  const std::set<std::string> kReduceOpSet = {kReduceSumOpName, kReduceMeanOpName, kReduceProdOpName};

  auto iter = kReduceOpSet.find(op_name);
  return iter != kReduceOpSet.end();
}

const AnfNodePtr ReduceOptimizer::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  if (!IsReduceOp(op_name)) {
    MS_LOG(DEBUG) << "Skip ReduceOptimizer for " << op_name;
    return nullptr;
  }
  if (!common::AnfAlgo::IsDynamicShape(cnode) && !common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, cnode)) {
    MS_LOG(DEBUG) << "Current node is not dynamic shape, skip!";
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  return NewAssistValueNode(cnode, kernel_graph);
}

const BaseRef ReduceOptimizer::DefinePattern() const {
  std::shared_ptr<Var> V = std::make_shared<CondVar>(UnVisited);
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}
}  // namespace opt
}  // namespace mindspore
