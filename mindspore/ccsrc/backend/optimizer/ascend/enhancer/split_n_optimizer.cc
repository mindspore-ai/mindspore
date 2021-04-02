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
#include "backend/optimizer/ascend/enhancer/split_n_optimizer.h"
#include <set>
#include <utility>
#include <string>
#include <memory>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "base/core_ops.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;
const std::set<std::string> InvalidOps = {kSplitOpName, kSplitVOpName, kConcatOpName};

void GetSplitOutputs(const FuncGraphPtr &func_graph, const AnfNodePtr &node, std::vector<AnfNodePtr> *out_nodes) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(node) == manager->node_users().end()) {
    return;
  }
  for (const auto &node_index : manager->node_users()[node]) {
    const AnfNodePtr &output = node_index.first;
    MS_EXCEPTION_IF_NULL(output);
    if (!output->isa<CNode>()) {
      continue;
    }
    auto cnode = output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto input0 = cnode->input(0);
    MS_EXCEPTION_IF_NULL(input0);
    if (IsPrimitive(input0, prim::kPrimMakeTuple) || IsPrimitive(input0, prim::kPrimTupleGetItem)) {
      GetSplitOutputs(func_graph, output, out_nodes);
    } else {
      out_nodes->push_back(output);
    }
  }
}

KernelWithIndex VisitSplitKernel(const AnfNodePtr &anf_node, size_t index) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (anf_node->isa<ValueNode>()) {
    return std::make_pair(anf_node, 0);
  } else if (anf_node->isa<Parameter>()) {
    return std::make_pair(anf_node, 0);
  } else if (anf_node->isa<CNode>()) {
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto input0 = cnode->input(0);
    MS_EXCEPTION_IF_NULL(input0);
    if (IsPrimitive(input0, prim::kPrimMakeTuple)) {
      auto node = cnode->input(index + IntToSize(1));
      MS_EXCEPTION_IF_NULL(node);
      return VisitSplitKernel(node, 0);
    } else if (IsPrimitive(input0, prim::kPrimTupleGetItem)) {
      if (cnode->inputs().size() != kTupleGetItemInputSize) {
        MS_LOG(EXCEPTION) << "The node tuple_get_item must have 2 inputs!";
      }
      auto input2 = cnode->input(kInputNodeOutputIndexInTupleGetItem);
      MS_EXCEPTION_IF_NULL(input2);
      auto value_node = input2->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto item_idx = GetValue<int64_t>(value_node->value());
      return VisitSplitKernel(cnode->input(kRealInputNodeIndexInTupleGetItem), LongToSize(item_idx));
    } else {
      return std::make_pair(anf_node, index);
    }
  } else {
    MS_LOG(EXCEPTION) << "The input is invalid";
  }
}

bool InputCheck(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto in_nums = AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < in_nums; i++) {
    auto in_node = VisitSplitKernel(AnfAlgo::GetInputNode(cnode, i), 0).first;
    if (in_node->isa<Parameter>() || in_node->isa<ValueNode>()) {
      MS_LOG(INFO) << "Input is a Parameter or ValueNode, can not optimizer.";
      return false;
    }
    if (in_node->isa<CNode>()) {
      auto in_cnode = in_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(in_cnode);
      auto in_node_name = AnfAlgo::GetCNodeName(in_cnode);
      auto trans_input = AnfAlgo::VisitKernel(in_node, 0).first;
      if (in_node_name == kTransDataOpName && (trans_input->isa<Parameter>() || trans_input->isa<ValueNode>())) {
        MS_LOG(INFO) << "Data->TransData->split, can not optimizer.";
        return false;
      }
      if (in_node_name == prim::kPrimControlDepend->name() || in_node_name == prim::kPrimDepend->name()) {
        return false;
      }
      if ((AnfAlgo::HasNodeAttr("non_task", in_cnode) && AnfAlgo::GetNodeAttr<bool>(in_node, "non_task")) ||
          opt::IsNopNode(in_cnode)) {
        MS_LOG(INFO) << "Input is nop node or has non_task attr, can not optimizer.";
        return false;
      }
    }
  }
  return true;
}

bool OutputCheck(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> outputs;
  GetSplitOutputs(func_graph, node, &outputs);
  if (outputs.empty()) {
    MS_LOG(INFO) << "Next node has no outputs, can not optimizer.";
    return false;
  }
  for (const auto &item : outputs) {
    if (IsPrimitiveCNode(item, prim::kPrimControlDepend) || IsPrimitiveCNode(item, prim::kPrimDepend)) {
      MS_LOG(INFO) << "Split has control edge, can not optimizer.";
      return false;
    }
    if (AnfAlgo::IsRealKernel(item) && (AnfAlgo::GetProcessor(item) != 0)) {
      MS_LOG(INFO) << "Next node is not a AICore node, can not optimizer.";
      return false;
    }
    if (func_graph->output() == item || AnfAlgo::CheckPrimitiveType(item, prim::kPrimReturn)) {
      MS_LOG(INFO) << "Next node is graph output or return, can not optimizer.";
      return false;
    }
    auto op_name = AnfAlgo ::GetCNodeName(item);
    if (InvalidOps.find(op_name) != InvalidOps.end() || AnfAlgo::IsCommunicationOp(item)) {
      MS_LOG(INFO) << "Next node is " << item->fullname_with_scope() << ", not a invalid node, can not optimizer.";
      return false;
    }
    if (!AnfAlgo::GetOutputTensorNum(item)) {
      MS_LOG(INFO) << "Next node has no output, can not optimizer.";
      return false;
    }
  }
  return true;
}

bool NeedSkip(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr func_manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(func_manager);
  int64_t split_dim = -1;
  auto op_name = AnfAlgo::GetCNodeName(node);
  if (op_name == prim::kPrimSplit->name()) {
    split_dim = AnfAlgo::GetNodeAttr<int64_t>(node, kAttrAxis);
  } else if (op_name == prim::kPrimSplitV->name()) {
    split_dim = AnfAlgo::GetNodeAttr<int64_t>(node, kAttrSplitDim);
  }
  if (split_dim != 0) {
    MS_LOG(INFO) << "Split_dim is not 0, can not optimizer.";
    return true;
  }
  if (AnfAlgo::IsDynamicShape(node)) {
    MS_LOG(INFO) << "Split is dynamic shape, can not optimizer.";
    return true;
  }
  if (!(InputCheck(node))) {
    MS_LOG(INFO) << "Split input check failed, can not optimizer.";
    return true;
  }
  if (!(OutputCheck(func_graph, node))) {
    MS_LOG(INFO) << "Split output check failed, can not optimizer.";
    return true;
  }
  return false;
}
}  // namespace

const BaseRef SplitOpOptimizer::DefinePattern() const {
  std::shared_ptr<Var> V = std::make_shared<CondVar>(UnVisited);
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr SplitOpOptimizer::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!AnfAlgo::IsRealCNodeKernel(node)) {
    return nullptr;
  }
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  auto op_name = AnfAlgo::GetCNodeName(node);
  if (op_name != prim::kPrimSplit->name() && op_name != prim::kPrimSplitV->name()) {
    return nullptr;
  }

  if (!NeedSkip(func_graph, node)) {
    AnfAlgo::SetNodeAttr("non_task", MakeValue(true), node);
    return node;
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
