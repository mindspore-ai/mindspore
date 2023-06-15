/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/backend_adapter.h"

#include <memory>
#include <utility>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <deque>
#include <functional>

#include "ir/anf.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "abstract/abstract_value.h"
#include "pipeline/jit/action.h"

namespace mindspore {
namespace backend_adapter {
namespace {
constexpr size_t kPartialFuncGraphPos = 1;
constexpr size_t kPartialInputStartPos = 2;
constexpr size_t kMaxRenormalizeTime = 3;
using AbstractSequence = abstract::AbstractSequence;
using AbstractSequencePtr = abstract::AbstractSequencePtr;
using UnifyDynamicLenFunc = std::function<bool(const AbstractSequencePtr &, const AbstractSequencePtr &)>;
using UnifyDynamicLenFuncPtr =
  std::shared_ptr<std::function<bool(const AbstractSequencePtr &, const AbstractSequencePtr &)>>;

bool CheckUnifyDynamicLen(const AbstractSequencePtr &abs1, const AbstractSequencePtr &abs2) {
  MS_EXCEPTION_IF_NULL(abs1);
  MS_EXCEPTION_IF_NULL(abs2);
  if (abs1->dynamic_len() != abs2->dynamic_len()) {
    MS_LOG(INFO) << "Abstract:" << abs1->ToString() << " and abstract:" << abs2->ToString() << " is inconsistent.";
    return false;
  }
  return true;
}

bool CheckUnifyDynamicLenWithError(const AbstractSequencePtr &abs1, const AbstractSequencePtr &abs2) {
  MS_EXCEPTION_IF_NULL(abs1);
  MS_EXCEPTION_IF_NULL(abs2);
  if (abs1->dynamic_len() != abs2->dynamic_len()) {
    MS_LOG(ERROR) << "Abstract:" << abs1->ToString() << " and abstract:" << abs2->ToString() << " is inconsistent.";
    return false;
  }
  return true;
}

bool ModifyDynamicLenArg(const AbstractSequencePtr &abs1, const AbstractSequencePtr &abs2) {
  MS_EXCEPTION_IF_NULL(abs1);
  MS_EXCEPTION_IF_NULL(abs2);
  if (abs1->dynamic_len() == abs2->dynamic_len()) {
    return true;
  }
  if (!abs1->dynamic_len()) {
    MS_LOG(INFO) << "Set dynamic len args for abstract:" << abs1->ToString();
    abs1->set_dyn_len_arg();
  }
  return true;
}

void AddMutableNodeForDynamicArg(const AnfNodePtr &node, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim::kPrimMutable), node, NewValueNode(true)};
  auto src_func_graph = node->func_graph();
  if (src_func_graph == nullptr) {
    auto &node_users = manager->node_users();
    auto iter = node_users.find(node);
    if (iter == node_users.end() || iter->second.begin()->first == nullptr) {
      MS_LOG(WARNING) << "Failed to get node user from node:" << node->DebugString();
    } else {
      src_func_graph = iter->second.begin()->first->func_graph();
    }
    if (src_func_graph == nullptr) {
      MS_LOG(WARNING) << "Failed to get funcgraph from node:" << node->DebugString();
      src_func_graph = func_graph;
    }
  }
  auto new_cnode = NewCNode(inputs, src_func_graph);
  manager->Replace(node, new_cnode);
  MS_LOG(INFO) << "Add mutable node for node:" << node->DebugString();
}

constexpr size_t kDependOutputIndex = 1;

// Transform a value tuple to make tuple node.
// e.g. ((), 0) =====>  maketuple(mutable(()), 0)
AnfNodePtr TransValueNodeToMakeTuple(const AnfNodePtr &node, size_t index, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<ValueSequence>()) {
    MS_LOG(EXCEPTION) << "Invalid index:" << index << " for value node:" << value_node->DebugString();
  }
  const auto &value_sequence = value->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(value_sequence);
  if (value_sequence->size() <= index) {
    MS_LOG(EXCEPTION) << "Invalid index:" << index << " for value node:" << value_node->DebugString();
  }
  std::vector<AnfNodePtr> tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  for (const auto &sub_value : value_sequence->value()) {
    MS_EXCEPTION_IF_NULL(sub_value);
    tuple_inputs.emplace_back(NewValueNode(sub_value));
    tuple_inputs.back()->set_abstract(sub_value->ToAbstract());
  }

  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto src_func_graph = value_node->func_graph();
  if (src_func_graph == nullptr) {
    auto &node_users = manager->node_users();
    auto iter = node_users.find(value_node);
    if (iter == node_users.end() || iter->second.size() > 1 || iter->second.begin()->first == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get node user from node:" << value_node->DebugString();
    }
    const auto &user_node = iter->second.begin()->first;
    MS_EXCEPTION_IF_NULL(user_node);
    MS_EXCEPTION_IF_NULL(user_node->func_graph());
    src_func_graph = user_node->func_graph();
  }
  auto new_cnode = NewCNode(tuple_inputs, src_func_graph);
  MS_LOG(INFO) << "Replace value node:" << value_node << " by make tuple:" << new_cnode->DebugString();
  manager->Replace(value_node, new_cnode);
  return new_cnode->input(index + 1);
}

// Fetch the real node which should add a mutable node.
AnfNodePtr FetchRealNodeByIndexStack(const AnfNodePtr &node, std::deque<size_t> *index_queue,
                                     const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(index_queue);
  if (index_queue->size() == 0) {
    return node;
  }
  if ((!node->isa<CNode>()) && (!node->isa<ValueNode>())) {
    MS_LOG(EXCEPTION) << "Cannot add mutable for node:" << node->DebugString() << " index size:" << index_queue->size();
  }
  const auto &cnode = node->cast<CNodePtr>();
  if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimDepend)) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->inputs().size() <= kDependOutputIndex) {
      MS_LOG(EXCEPTION) << "Invalid depend node:" << cnode->DebugString();
    }
    return FetchRealNodeByIndexStack(cnode->input(kDependOutputIndex), index_queue, func_graph);
  } else if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimMakeTuple) ||
             common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimMakeList)) {
    MS_EXCEPTION_IF_NULL(cnode);
    size_t index = index_queue->front();
    index_queue->pop_front();
    if (cnode->inputs().size() <= index + 1) {
      MS_LOG(EXCEPTION) << "Invalid make tuple node:" << cnode->DebugString() << " index:" << index;
    }
    MS_LOG(DEBUG) << "Fetch index:" << index << " for tuple node:" << cnode->DebugString();
    return FetchRealNodeByIndexStack(cnode->input(index + 1), index_queue, func_graph);
  } else if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimTupleGetItem) ||
             common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimListGetItem)) {
    MS_EXCEPTION_IF_NULL(cnode);
    size_t index = common::AnfAlgo::GetTupleGetItemOutIndex(cnode);
    index_queue->push_front(index);
    return FetchRealNodeByIndexStack(common::AnfAlgo::GetTupleGetItemRealInput(cnode), index_queue, func_graph);
  } else if (node->isa<ValueNode>()) {
    size_t index = index_queue->front();
    index_queue->pop_front();
    return FetchRealNodeByIndexStack(TransValueNodeToMakeTuple(node, index, func_graph), index_queue, func_graph);
  }
  MS_LOG(EXCEPTION) << "Invalid node:" << node->DebugString() << " for index size:" << index_queue->size();
}

void AddMutableNodeByNode(const AnfNodePtr &node, const abstract::AbstractBasePtr &abs, std::deque<size_t> *index_queue,
                          const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(abs);
  MS_EXCEPTION_IF_NULL(index_queue);
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!abs->isa<AbstractSequence>()) {
    MS_LOG(DEBUG) << "return for abs:" << abs->ToString();
    return;
  }
  const auto &abs_sequence = abs->cast<AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(abs_sequence);
  if (abs_sequence->dyn_len_arg()) {
    auto indexes = *index_queue;
    const auto &real_node = FetchRealNodeByIndexStack(node, &indexes, func_graph);
    AddMutableNodeForDynamicArg(real_node, func_graph);
    return;
  }
  if (abs_sequence->dynamic_len()) {
    return;
  }
  MS_LOG(DEBUG) << "Check tuple abs for node:" << node->DebugString() << " abs:" << abs_sequence->ToString()
                << " start";
  for (size_t i = 0; i < abs_sequence->elements().size(); ++i) {
    MS_EXCEPTION_IF_NULL(abs_sequence->elements()[i]);
    index_queue->emplace_back(i);
    AddMutableNodeByNode(node, abs_sequence->elements()[i], index_queue, func_graph);
    index_queue->pop_back();
  }
  MS_LOG(DEBUG) << "Check tuple abs for node:" << node->DebugString() << " abs:" << abs_sequence->ToString() << " end";
}

void AddMutableNodeByFuncGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "Start add mutable for graph:" << func_graph->ToString();
  auto nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  for (auto &node : nodes) {
    if (node->abstract() == nullptr || common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
      continue;
    }
    std::deque<size_t> index_stack;
    AddMutableNodeByNode(node, node->abstract(), &index_stack, func_graph);
  }
}

void UnifyDynamicLen(const pipeline::ResourcePtr &resource) {
  const auto &func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool enable_save_graphs = context->CanDump(kIntroductory);
  if (enable_save_graphs) {
    DumpIR("unify_dynamic_len_before_graph.ir", func_graph);
  }
#endif
  // 1. Add mutable for node which has dynamic len arg flag.
  MS_LOG(DEBUG) << "Add mutable node for funcgraph:" << func_graph->ToString() << " start";
  AddMutableNodeByFuncGraph(func_graph);
  MS_LOG(DEBUG) << "Add mutable node for funcgraph:" << func_graph->ToString() << " end";
#ifdef ENABLE_DUMP_IR
  if (enable_save_graphs) {
    DumpIR("unify_dynamic_len_add_mutable_graph.ir", func_graph);
  }
#endif
  abstract::AbstractBasePtrList args_abs;
  auto parameters = func_graph->parameters();
  (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_abs),
                       [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
  // 2. Renormalize of abstract.
  MS_LOG(INFO) << "Renormalize start";
  FuncGraphPtr new_func_graph = pipeline::Renormalize(resource, func_graph, args_abs);
  MS_LOG(INFO) << "Renormalize end";
  MS_EXCEPTION_IF_NULL(new_func_graph);
  const auto &manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->KeepRoots({new_func_graph});
#ifdef ENABLE_DUMP_IR
  if (enable_save_graphs) {
    DumpIR("unify_dynamic_len_after_renormalize_graph.ir", new_func_graph);
  }
#endif
  resource->set_func_graph(new_func_graph);
  resource->set_args_abs(args_abs);
  // 3. Eliminate mutable nodes in graph.
  MS_LOG(INFO) << "Remove mutable for graph:" << new_func_graph->ToString();
  auto new_nodes = TopoSort(new_func_graph->get_return(), SuccDeeperSimple);
  for (auto &node : new_nodes) {
    if (!common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimMutable)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->inputs().size() <= 1) {
      MS_LOG(EXCEPTION) << "Invalid input size:" << cnode->inputs().size()
                        << " for mutable cnode:" << cnode->DebugString();
    }
    MS_EXCEPTION_IF_NULL(cnode->input(1));
    cnode->input(1)->set_abstract(cnode->abstract());
    manager->Replace(cnode, cnode->input(1));
    MS_LOG(INFO) << "Add mutable node for node:" << cnode->DebugString();
  }
#ifdef ENABLE_DUMP_IR
  if (enable_save_graphs) {
    DumpIR("unify_dynamic_len_eliminate_mutable_graph.ir", new_func_graph);
  }
#endif
}

bool ProcessAbstract(const abstract::AbstractBasePtr &abs1, const abstract::AbstractBasePtr &abs2,
                     const UnifyDynamicLenFuncPtr &func) {
  if (abs1 == nullptr || abs2 == nullptr) {
    MS_LOG(WARNING) << "Invaliad check as abstract is null.";
    return true;
  }
  if ((!abs1->isa<AbstractSequence>()) && (!abs2->isa<AbstractSequence>())) {
    return true;
  }
  if ((abs1->isa<AbstractSequence>() && (!abs2->isa<AbstractSequence>())) ||
      (abs2->isa<AbstractSequence>() && (!abs1->isa<AbstractSequence>()))) {
    abstract::AbstractJoinedAnyPtr any_abs = nullptr;
    if (abs1->isa<abstract::AbstractJoinedAny>()) {
      any_abs = abs1->cast<abstract::AbstractJoinedAnyPtr>();
    } else if (abs2->isa<abstract::AbstractJoinedAny>()) {
      any_abs = abs2->cast<abstract::AbstractJoinedAnyPtr>();
    }
    if (any_abs != nullptr) {
      any_abs->ThrowException();
    }
    MS_LOG(ERROR) << "Abstract:" << abs1->ToString() << " and abstract:" << abs2->ToString() << " is inconsistent.";
    return false;
  }
  const auto &sequence_abs1 = abs1->cast<AbstractSequencePtr>();
  const auto &sequence_abs2 = abs2->cast<AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(sequence_abs1);
  MS_EXCEPTION_IF_NULL(sequence_abs2);
  if (!(*func)(sequence_abs1, sequence_abs2)) {
    return false;
  }
  if (sequence_abs1->dynamic_len() || sequence_abs2->dynamic_len()) {
    return true;
  }
  if (sequence_abs1->size() != sequence_abs2->size()) {
    MS_LOG(EXCEPTION) << "Invalid sequence size for abstract:" << sequence_abs1->ToString()
                      << " size:" << sequence_abs1->size() << " and abstract:" << sequence_abs2->ToString()
                      << " size:" << sequence_abs2->size();
  }
  for (size_t i = 0; i < sequence_abs1->size(); ++i) {
    if (!ProcessAbstract(sequence_abs1->elements()[i], sequence_abs2->elements()[i], func)) {
      return false;
    }
  }
  return true;
}

bool ProcessTupleDynamicLen(const AnfNodePtr &node, const UnifyDynamicLenFuncPtr &func) {
  MS_EXCEPTION_IF_NULL(node);
  if (common::AnfAlgo::IsCallNode(node)) {
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &func_graphs = abstract::GetFuncGraphsFromCallNode(cnode);
    if (func_graphs.empty()) {
      MS_LOG(EXCEPTION) << "Get func graphs from abstract failed.";
    }
    for (auto func_graph : func_graphs) {
      MS_EXCEPTION_IF_NULL(func_graph);
      // Check the consistency of return outputs and call outputs.
      MS_EXCEPTION_IF_NULL(func_graph->return_node());
      if (!ProcessAbstract(func_graph->return_node()->abstract(), node->abstract(), func)) {
        MS_LOG(INFO) << "Invalid abs:" << func_graph->return_node()->abstract()->ToString()
                     << " for node:" << func_graph->return_node()->DebugString()
                     << " and:" << node->abstract()->ToString() << " for node:" << node->DebugString();
        return false;
      }
      // Check the consistency of arguments and parameters.
      size_t args_num = cnode->inputs().size() - 1;
      size_t para_num = func_graph->parameters().size();
      if (args_num > para_num) {
        MS_LOG(EXCEPTION) << "Invalid args num:" << args_num << " for funcgraph:" << func_graph->ToString()
                          << " parameters num:" << func_graph->parameters().size();
      }
      for (size_t i = 0; i < args_num; ++i) {
        MS_EXCEPTION_IF_NULL(cnode->input(args_num - i));
        MS_EXCEPTION_IF_NULL((func_graph->parameters())[para_num - 1 - i]);
        if (!ProcessAbstract(cnode->input(args_num - i)->abstract(),
                             (func_graph->parameters())[para_num - 1 - i]->abstract(), func)) {
          MS_LOG(INFO) << "Invalid abs:" << cnode->input(args_num - i)->abstract()->ToString()
                       << " for node:" << cnode->DebugString()
                       << " and:" << (func_graph->parameters())[para_num - 1 - i]->ToString()
                       << " for node:" << (func_graph->parameters())[para_num - 1 - i]->DebugString();
          return false;
        }
      }
    }
  } else if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    size_t input_num = cnode->inputs().size();
    if (input_num <= kPartialFuncGraphPos || cnode->input(kPartialFuncGraphPos) == nullptr ||
        (!cnode->input(kPartialFuncGraphPos)->isa<ValueNode>())) {
      MS_LOG(EXCEPTION) << "Invalid partial node:" << node->DebugString();
    }
    const auto &func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kPartialFuncGraphPos));
    MS_EXCEPTION_IF_NULL(func_graph);
    if (func_graph->parameters().size() < input_num - kPartialInputStartPos) {
      MS_LOG(EXCEPTION) << "Invalid args num:" << input_num - kPartialInputStartPos
                        << " in partial node:" << cnode->DebugString() << " for fungraph:" << func_graph->ToString()
                        << " parameter num:" << func_graph->parameters().size();
    }
    for (size_t i = kPartialInputStartPos; i < input_num; ++i) {
      MS_EXCEPTION_IF_NULL(cnode->input(i));
      MS_EXCEPTION_IF_NULL(func_graph->parameters()[i - kPartialInputStartPos]);
      if (!ProcessAbstract(cnode->input(i)->abstract(), func_graph->parameters()[i - kPartialInputStartPos]->abstract(),
                           func)) {
        MS_LOG(INFO) << "Invalid abs:" << cnode->input(i)->abstract()->ToString()
                     << " for node:" << cnode->input(i)->DebugString()
                     << " and:" << func_graph->parameters()[i - kPartialInputStartPos]->ToString()
                     << " for node:" << func_graph->parameters()[i - kPartialInputStartPos]->DebugString();
        return false;
      }
    }
  }
  return true;
}

bool ProcessFuncGraph(const FuncGraphPtr &root_graph, const UnifyDynamicLenFuncPtr &func) {
  MS_EXCEPTION_IF_NULL(root_graph);
  auto nodes = TopoSort(root_graph->get_return(), SuccDeeperSimple);
  for (const auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    // Check consistency of dynamic len flag in partial and call.
    MS_LOG(DEBUG) << "Process node:" << node->DebugString() << " start";
    if ((common::AnfAlgo::IsCallNode(node) || common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) &&
        (!ProcessTupleDynamicLen(node, func))) {
      MS_LOG(INFO) << "Unify check false in node:" << node->DebugString();
      return false;
    }
    MS_LOG(DEBUG) << "Process node:" << node->DebugString() << " end";
  }
  return true;
}

bool HasDynamicArg(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (!abs->isa<AbstractSequence>()) {
    return false;
  }
  const auto &sequence_abs = abs->cast<AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(sequence_abs);
  if (sequence_abs->dyn_len_arg()) {
    return true;
  }
  if (sequence_abs->dynamic_len()) {
    return false;
  }
  for (const auto &sub_abs : sequence_abs->elements()) {
    if (HasDynamicArg(sub_abs)) {
      return true;
    }
  }
  return false;
}

bool HasDynamicArg(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  for (const auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->abstract() == nullptr) {
      continue;
    }
    if (HasDynamicArg(node->abstract())) {
      return true;
    }
  }
  return false;
}
}  // namespace

void CheckDynamicLenUnify(const pipeline::ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  MS_LOG(DEBUG) << "Check dynamic len";
  if (!ProcessFuncGraph(resource->func_graph(), std::make_shared<UnifyDynamicLenFunc>(CheckUnifyDynamicLen))) {
    // 1. Add dynamic len flag for dynamic len tuple.
    MS_LOG(DEBUG) << "Modify dynamic len arg";
    ProcessFuncGraph(resource->func_graph(), std::make_shared<UnifyDynamicLenFunc>(ModifyDynamicLenArg));
    MS_LOG(DEBUG) << "Unify dynamic len";
    size_t renormalize_time = 0;
    // 2. Fix abstract of dynamic tuple.
    do {
      ++renormalize_time;
      MS_LOG(DEBUG) << "Unify dynamic len start time:" << renormalize_time;
      UnifyDynamicLen(resource);
      MS_LOG(DEBUG) << "Unify dynamic len end time:" << renormalize_time;
    } while (renormalize_time < kMaxRenormalizeTime && HasDynamicArg(resource->func_graph()));
    if (renormalize_time != 1) {
      MS_LOG(WARNING) << "For dynamic len unify, renormalize has been run for " << renormalize_time << " time.";
    }
    // 3. Recheck dynamic len flag.
    MS_LOG(DEBUG) << "Recheck dynamic len";
    if (!ProcessFuncGraph(resource->func_graph(),
                          std::make_shared<UnifyDynamicLenFunc>(CheckUnifyDynamicLenWithError))) {
      MS_LOG(EXCEPTION) << "Invalid dynamic len in graph:" << resource->func_graph()->ToString();
    }
  }
  MS_LOG(DEBUG) << "Check dynamic len end";
}
}  // namespace backend_adapter
}  // namespace mindspore
