/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/static_analysis/program_specialize.h"

#include <algorithm>
#include <exception>
#include <unordered_set>
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/do_signature.h"
#include "abstract/abstract_function.h"
#include "abstract/utils.h"
#include "ir/graph_utils.h"
#include "utils/log_adapter.h"
#include "pipeline/jit/debug/trace.h"

namespace mindspore {
namespace abstract {
namespace {
inline AbstractBasePtr GetEvaluatedValue(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  const auto &eval_result = conf->ObtainEvalResult();
  MS_EXCEPTION_IF_NULL(eval_result);
  return eval_result->abstract();
}

AnfNodePtr BuildValueNode(const ValuePtr &v, const AbstractBasePtr &abs_base) {
  MS_EXCEPTION_IF_NULL(abs_base);
  AnfNodePtr value_node = NewValueNode(v);
  value_node->set_abstract(abs_base);
  MS_LOG(DEBUG) << "Create ValueNode: " << value_node->ToString() << ", with abstract: " << abs_base->ToString();
  return value_node;
}

bool IsVisible(FuncGraphPtr fg, const FuncGraphPtr &parent) {
  while (fg != nullptr && fg != parent) {
    fg = fg->parent();
  }
  return fg == parent;
}

bool CanSpecializeValueNode(const AnfNodePtr &node) {
  if (IsValueNode<MetaFuncGraph>(node) || IsValueNode<Primitive>(node)) {
    return true;
  }
  if (IsValueNode<FuncGraph>(node)) {
    if (node->abstract() != nullptr) {
      auto abs_func = node->abstract()->cast_ptr<FuncGraphAbstractClosure>();
      // If this funcgraph had specialized in ProcessCNode of FirstPass,
      // then ignore it.
      if (abs_func != nullptr && abs_func->specialized()) {
        return false;
      }
    }
    return true;
  }
  return false;
}

void PurifyAbstractOfSequence(ProgramSpecializer *const specializer) {
  MS_EXCEPTION_IF_NULL(specializer);
  constexpr int recursive_level = 2;
  for (auto &abstract_and_node : specializer->sequence_abstract_list()) {
    auto &sequence_abs = abstract_and_node.first;
    if (!sequence_abs->PurifyElements()) {
      MS_LOG(ERROR) << "Purify elements failed, abstract: " << sequence_abs->ToString()
                    << ", node: " << abstract_and_node.second->DebugString(recursive_level);
    } else {
      MS_LOG(DEBUG) << "Purify elements, abstract: " << sequence_abs->ToString()
                    << ", node: " << abstract_and_node.second->DebugString(recursive_level);
    }
  }
}

// Second elimination.
// Eliminate the dead node in sequence node, and purify the abstract of sequence node.
void EliminateCollectedSequenceNodes(ProgramSpecializer *const specializer) {
  // Call PurifyElements() to purify tuple/list elements.
  static const auto enable_only_mark_unused_element = (common::GetEnv("MS_DEV_DDE_ONLY_MARK") == "1");
  if (enable_only_mark_unused_element) {
    return;
  }

  // Purify the abstract of tuple/list.
  PurifyAbstractOfSequence(specializer);
  // Eliminate DeadNode in tuple/list.
  for (auto &dead_node_info : specializer->dead_node_list()) {
    auto pos = dead_node_info.second;
    auto node = dead_node_info.first;
    auto flags = GetSequenceNodeElementsUseFlags(node);
    if (flags == nullptr) {
      continue;
    }

    // Handle MakeTuple/MakeList CNode.
    auto cnode = dyn_cast_ptr<CNode>(node);
    if (cnode != nullptr) {
      if (pos + 1 >= cnode->inputs().size()) {
        continue;
      }
      if (!IsDeadNode(cnode->input(pos + 1))) {
        continue;
      }

      constexpr int recursive_level = 2;
      MS_LOG(DEBUG) << "Erase elements[" << pos << "] DeadNode as zero for " << cnode->DebugString(recursive_level);
      // Change the node.
      auto zero_value = NewValueNode(MakeValue(0));
      zero_value->set_abstract(
        std::make_shared<abstract::AbstractScalar>(std::make_shared<Int32Imm>(0), std::make_shared<Problem>()));
      cnode->set_input(pos + 1, zero_value);

      // Change the abstract.
      (*flags)[pos] = false;  // Change the use flag as 0.
      auto sequence_abs = dyn_cast_ptr<AbstractSequence>(node->abstract());
      if (sequence_abs != nullptr && !sequence_abs->PurifyElements()) {
        MS_LOG(ERROR) << "Purify elements failed, abstract: " << sequence_abs->ToString()
                      << ", node: " << node->DebugString(recursive_level);
      }
      continue;
    }
    // Handle ValueTuple/ValueList.
    if (IsValueNode<ValueTuple>(node) || IsValueNode<ValueList>(node)) {
      auto sequence_value = GetValuePtr<ValueSequence>(node);
      MS_EXCEPTION_IF_NULL(sequence_value);
      if (pos >= sequence_value->value().size()) {
        continue;
      }
      ValuePtr element_value = sequence_value->value()[pos];
      auto element_err_value = element_value->cast_ptr<ValueProblem>();
      if (element_err_value == nullptr || !element_err_value->IsDead()) {
        continue;
      }

      MS_LOG(DEBUG) << "Erase elements[" << pos << "] DeadNode as zero for " << node->DebugString();
      // Change the node.
      auto zero = MakeValue(0);
      auto value_list = const_cast<ValuePtrList &>(sequence_value->value());
      value_list[pos] = zero;

      // Change the abstract.
      (*flags)[pos] = false;  // Change the use flag as 0.
      auto sequence_abs = dyn_cast_ptr<AbstractSequence>(node->abstract());
      if (sequence_abs != nullptr && !sequence_abs->PurifyElements()) {
        constexpr int recursive_level = 2;
        MS_LOG(ERROR) << "Purify elements failed, abstract: " << sequence_abs->ToString()
                      << ", node: " << node->DebugString(recursive_level);
      }
    }
  }
}

void BroadenArgs(const AbstractBasePtrList &args_abs_list, AbstractBasePtrList *broaded_args) {
  MS_EXCEPTION_IF_NULL(broaded_args);
  (void)std::transform(args_abs_list.begin(), args_abs_list.end(), std::back_inserter(*broaded_args),
                       [](const AbstractBasePtr &arg) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(arg);
                         if (arg->GetValueTrack() != kValueAny) {
                           return arg->Broaden();
                         }
                         return arg;
                       });
}
}  // namespace

FuncGraphPtr ProgramSpecializer::Run(const FuncGraphPtr &fg, const AnalysisContextPtr &context) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(context);
  MS_LOG(DEBUG) << "Specialize topmost function graph: "
                << (context->func_graph() ? context->func_graph()->ToString() : "FG(Null)");
  if (top_context_ == nullptr) {
    top_context_ = context;
    MS_LOG(INFO) << "Specialize set top func graph context: " << context->ToString();
  }
  auto top_func_graph_spec = NewFuncGraphSpecializer(context, fg);
  PushFuncGraphTodoItem(top_func_graph_spec);
  while (!func_graph_todo_items_.empty()) {
    auto current_fg_spec = func_graph_todo_items_.top();
    if (current_fg_spec->done()) {
      func_graph_todo_items_.pop();
      continue;
    }
    // run current func graph specializer
    current_fg_spec->Run();
  }
  auto res = top_func_graph_spec->specialized_func_graph();
  MS_LOG(DEBUG) << "Specialized top graph:" << res->ToString();
  EliminateCollectedSequenceNodes(this);
  return res;
}

std::shared_ptr<FuncGraphSpecializer> ProgramSpecializer::GetFuncGraphSpecializer(const AnalysisContextPtr &context) {
  MS_EXCEPTION_IF_NULL(context);
  auto iter = specializations_.find(context);
  if (iter != specializations_.end()) {
    return iter->second;
  }
  return nullptr;
}

FuncGraphSpecializerPtr ProgramSpecializer::NewFuncGraphSpecializer(const AnalysisContextPtr &context,
                                                                    const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(context);
  auto result = specializations_.emplace(context, nullptr);
  if (result.second) {
    MS_LOG(DEBUG) << "Make new specializer of context: " << context->ToString() << ", fg: " << fg->ToString();
    auto fg_spec = std::make_shared<FuncGraphSpecializer>(this, fg, context);
    result.first->second = fg_spec;
    return fg_spec;
  }
  MS_LOG(EXCEPTION) << "Specializer exist in cache, can't not create again, context: " << context->ToString();
}

void ProgramSpecializer::PutSpecializedAbstract(const CNodePtr &cnode, const AnfNodePtr &func,
                                                const AbstractFunctionPtr &old_abs_func,
                                                const AbstractFunctionPtr &new_abs_func) {
  MS_EXCEPTION_IF_NULL(old_abs_func);
  auto iter = specialized_abs_map_.find(old_abs_func);
  if (iter == specialized_abs_map_.end()) {
    MS_LOG(DEBUG) << "Emplace cnode: " << cnode->DebugString() << ", func: " << func->ToString()
                  << ", old_abstract: " << old_abs_func->ToString() << ", new_abs_func: " << new_abs_func->ToString();
    (void)specialized_abs_map_.emplace(old_abs_func, std::make_pair(true, new_abs_func));
  } else {
    MS_LOG(DEBUG) << "Duplicate abstract from cnode: " << cnode->DebugString() << ", func: " << func->ToString()
                  << ", old_abstract: " << old_abs_func->ToString() << ", new_abs_func: " << new_abs_func->ToString();
    if (!(*iter->second.second == *new_abs_func)) {
      MS_LOG(DEBUG) << "Duplicate abstract from cnode: " << cnode->DebugString() << ", func: " << func->ToString()
                    << ", old_abstract: " << old_abs_func->ToString() << ", first: " << iter->second.second->ToString()
                    << ", new_abs_func: " << new_abs_func->ToString();
      // Cannot determined which one to use.
      iter->second.first = false;
    }
  }
}

AbstractFunctionPtr ProgramSpecializer::GetSpecializedAbstract(const AbstractFunctionPtr &old_abs_func) {
  MS_EXCEPTION_IF_NULL(old_abs_func);
  auto iter = specialized_abs_map_.find(old_abs_func);
  if (iter != specialized_abs_map_.end()) {
    if (iter->second.first) {
      MS_LOG(DEBUG) << "Find abstract for old_abstract: " << old_abs_func->ToString()
                    << ", new_abs_func: " << iter->second.second->ToString();
      return iter->second.second;
    }
    return nullptr;
  }
  if (old_abs_func->isa<FuncGraphAbstractClosure>()) {
    const auto &old_func_graph_abs = dyn_cast_ptr<FuncGraphAbstractClosure>(old_abs_func);
    auto unique_specialized_abs = GetUniqueFuncGraphAbstract(old_func_graph_abs->func_graph());
    if (unique_specialized_abs != nullptr) {
      MS_LOG(DEBUG) << "Find unique abstract for funcgraph: " << old_func_graph_abs->func_graph()->ToString() << " in "
                    << old_abs_func->ToString() << ", unique_abs: " << unique_specialized_abs->ToString();
      return unique_specialized_abs;
    }
  }
  MS_LOG(DEBUG) << "Cannot find abstract for old_abstract: " << old_abs_func->ToString();
  return nullptr;
}

AbstractFunctionPtr ProgramSpecializer::SpecializeAbstractFuncRecursively(const AbstractFunctionPtr &old_abs_func) {
  MS_EXCEPTION_IF_NULL(old_abs_func);
  AbstractFunctionPtr new_abs = nullptr;
  if (old_abs_func->isa<AbstractFuncUnion>()) {
    AbstractFuncAtomPtrList func_atoms;
    auto build_new_abs = [this, &func_atoms](const AbstractFuncAtomPtr &poss) {
      MS_EXCEPTION_IF_NULL(poss);
      auto resolved_atom = poss;
      if (poss->isa<AsyncAbstractFuncAtom>()) {
        auto async_abs_func = poss->cast_ptr<AsyncAbstractFuncAtom>();
        const auto &resolved_func = async_abs_func->GetUnique();
        resolved_atom = resolved_func->cast<AbstractFuncAtomPtr>();
        MS_EXCEPTION_IF_NULL(resolved_atom);
        MS_LOG(DEBUG) << "Resolved AsyncAbstractFuncAtom is: " << resolved_atom->ToString();
      }
      auto specialized_abs = this->SpecializeAbstractFuncRecursively(resolved_atom);
      AbstractFuncAtomPtr new_abs_atom = nullptr;
      if (specialized_abs == nullptr) {
        MS_LOG(DEBUG) << "Cannot resolve old_abs: " << resolved_atom->ToString()
                      << " to specialized abstract, use old one";
        new_abs_atom = resolved_atom;
      } else if (specialized_abs->isa<AbstractFuncAtom>()) {
        MS_LOG(DEBUG) << "Resolve old_abs: " << resolved_atom->ToString()
                      << " to specialized abstract, specialized abstract: " << specialized_abs->ToString();
        new_abs_atom = specialized_abs->cast<AbstractFuncAtomPtr>();
      } else {
        MS_LOG(DEBUG) << "Cannot resolve old_abs: " << resolved_atom->ToString()
                      << " to AbstractFuncAtom, use old one. Specialized abstract: " << specialized_abs->ToString();
        new_abs_atom = resolved_atom;
      }
      func_atoms.push_back(new_abs_atom);
    };
    old_abs_func->Visit(build_new_abs);
    new_abs = std::make_shared<AbstractFuncUnion>(func_atoms);
  } else if (old_abs_func->isa<FuncGraphAbstractClosure>() || old_abs_func->isa<MetaFuncGraphAbstractClosure>()) {
    new_abs = GetSpecializedAbstract(old_abs_func);
    if (new_abs != nullptr) {
      MS_LOG(DEBUG) << "Find specialized abstract, old_abstract: " << old_abs_func->ToString()
                    << ", specialized_abstract: " << new_abs->ToString();
    } else {
      MS_LOG(DEBUG) << "cannot find specialized abstract, old_abstract: " << old_abs_func->ToString();
    }
  } else if (old_abs_func->isa<PartialAbstractClosure>()) {
    const auto &old_partial_abs = old_abs_func->cast<PartialAbstractClosurePtr>();
    const auto &old_abs_fn = old_partial_abs->fn();
    auto new_abs_fn = GetSpecializedAbstract(old_abs_fn);
    if (new_abs_fn != nullptr && new_abs_fn->isa<AbstractFuncAtom>()) {
      auto new_abs_fn_atom = new_abs_fn->cast<AbstractFuncAtomPtr>();
      new_abs =
        std::make_shared<PartialAbstractClosure>(new_abs_fn_atom, old_partial_abs->args(), old_partial_abs->node());
      MS_LOG(DEBUG) << "Find specialized abstract, old_abstract: " << old_abs_func->ToString()
                    << ", specialized_abstract: " << new_abs->ToString();
    } else {
      MS_LOG(DEBUG) << "cannot find specialized abstract, old_abstract: " << old_abs_func->ToString();
    }
  }
  return new_abs;
}

void ProgramSpecializer::SpecializeCNodeInput0FuncGraph() {
  const auto &all_nodes = mng_->all_nodes();
  for (auto node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto &input0 = node->cast_ptr<CNode>()->input(0);
    MS_EXCEPTION_IF_NULL(input0);
    if (IsValueNode<FuncGraph>(input0)) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(node);
    const auto &old_abs = input0->abstract();
    if (!(old_abs->isa<FuncGraphAbstractClosure>() || old_abs->isa<MetaFuncGraphAbstractClosure>() ||
          old_abs->isa<AbstractFuncUnion>() || old_abs->isa<PartialAbstractClosure>())) {
      continue;
    }
    auto old_abs_func = old_abs->cast<AbstractFunctionPtr>();
    auto new_abs_func = SpecializeAbstractFuncRecursively(old_abs_func);
    if (new_abs_func != nullptr) {
      input0->set_abstract(new_abs_func);
      MS_LOG(DEBUG) << "Find specialized abstract for node: " << input0->DebugString()
                    << ", old_abstract: " << old_abs->ToString()
                    << ", specialized_abstract: " << new_abs_func->ToString();
    } else {
      MS_LOG(DEBUG) << "cannot find specialized abstract for node: " << input0->DebugString()
                    << ", old_abstract: " << old_abs_func->ToString();
    }
  }
}

static int64_t GetNextCounter() {
  static int64_t g_CloneCounter = 1;
  return g_CloneCounter++;
}

FuncGraphSpecializer::FuncGraphSpecializer(ProgramSpecializer *const s, const FuncGraphPtr &fg,
                                           const AnalysisContextPtr &context)
    : specializer_(s), func_graph_(fg), context_(context) {
  parent_ = s->GetFuncGraphSpecializer(context->parent());
  MS_EXCEPTION_IF_NULL(context->parent());
  if (ParentNotSpecialized(context)) {
    MS_LOG(EXCEPTION) << "Parent func graph should be handled in advance, fg: " << fg->ToString()
                      << ", context: " << context->ToString() << ", parent context: " << context->parent()->ToString();
  }
  engine_ = s->engine();
  cloner_ = SpecializerClone(fg, std::make_shared<TraceSpecialize>(GetNextCounter()));
  specialized_func_graph_ = cloner_->cloned_func_graphs().find(fg)->second;
  AddTodoItem(fg->get_return());
  AddTodoItem(fg->parameters());
}

AnfNodePtr FuncGraphSpecializer::ReplicateDisconnectedNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>()) {
    return node;
  }
  std::shared_ptr<FuncGraphSpecializer> specializer = GetTopSpecializer(node);

  // If had replicated, just return that.
  auto iter = specializer->cloned_nodes().find(node);
  if (iter != specializer->cloned_nodes().end()) {
    return iter->second;
  }
  auto new_node = specializer->cloner_->CloneDisconnected(node);
  if (node->isa<CNode>()) {
    if (!new_node->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "new_node must be a CNode, but is " << new_node->DebugString() << ".";
    }
    UpdateNewCNodeInputs(node, new_node);
  }

  iter = specializer->cloned_nodes().find(node);
  if (iter != specializer->cloned_nodes().end()) {
    if (iter->second == node) {
      MS_LOG(EXCEPTION) << "Replicated is same as original node, node: " << node->ToString();
    }
  } else {
    MS_LOG(EXCEPTION) << "Replicate node failed, node: " << node->ToString();
  }
  return new_node;
}

void FuncGraphSpecializer::UpdateNewCNodeInputs(const AnfNodePtr &node, const AnfNodePtr &new_node) {
  MS_EXCEPTION_IF_NULL(node);
  auto c_node = node->cast_ptr<CNode>();
  MS_EXCEPTION_IF_NULL(c_node);
  auto inputs = c_node->inputs();
  std::vector<AnfNodePtr> new_inputs;
  (void)std::transform(
    inputs.begin(), inputs.end(), std::back_inserter(new_inputs), [this](const AnfNodePtr &inp) -> AnfNodePtr {
      auto new_inp = ReplicateDisconnectedNode(inp);
      // Refer the comments in BuildReplacedNode.
      if (inp->isa<CNode>()) {
        auto c_inp = inp->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(c_inp);
        auto c_new_inp = new_inp->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(c_new_inp);
        MS_EXCEPTION_IF_NULL(c_new_inp->func_graph());
        MS_LOG(DEBUG) << "Replace in order, inp node: " << inp->DebugString() << " -> " << new_inp->DebugString();
        c_new_inp->func_graph()->ReplaceInOrder(c_inp, c_new_inp);
      }
      return new_inp;
    });

  auto c_new_node = new_node->cast_ptr<CNode>();
  MS_EXCEPTION_IF_NULL(c_new_node);
  c_new_node->set_inputs(new_inputs);
}

AnfNodePtr FuncGraphSpecializer::GetReplicatedNode(const AnfNodePtr &node) {
  std::shared_ptr<FuncGraphSpecializer> specializer = GetTopSpecializer(node);
  auto iter = specializer->cloned_nodes().find(node);
  if (iter != specializer->cloned_nodes().end()) {
    return iter->second;
  }
  return node;
}

// Return itself if node's ValueNode as top,
// return the top func graph specializer as top if node's forward Parameter,
// or, return the top parent specializer as top.
std::shared_ptr<FuncGraphSpecializer> FuncGraphSpecializer::GetTopSpecializer(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  FuncGraphPtr fg = node->func_graph();
  if (fg == nullptr) {  // If ValueNode, return current specializer.
    MS_LOG(DEBUG) << "Node's a ValueNode, node: " << node->DebugString();
    return shared_from_this();
  }
  std::shared_ptr<FuncGraphSpecializer> specializer = shared_from_this();
  while (fg != specializer->func_graph_) {
    if (specializer->parent_ == nullptr && node->isa<Parameter>()) {
      // If `parent_` is null and forwarded `node` is a Parameter, we'll try to use top func graph as parent.
      auto &top_context = specializer_->top_context();
      MS_EXCEPTION_IF_NULL(top_context);
      if (top_context->func_graph() == fg) {  // `fg` is top func graph.
        MS_LOG(INFO) << "Used top func graph specializer as parent for "
                     << (func_graph_ ? func_graph_->ToString() : "FG(Null)") << ", node: " << node->DebugString()
                     << ", NodeInfo: " << trace::GetDebugInfo(node->debug_info());
        specializer = specializer_->GetFuncGraphSpecializer(top_context);
        if (specializer == nullptr) {
          constexpr auto recursive_level = 2;
          MS_LOG(EXCEPTION) << "Specializer must not be null, node: " << node->DebugString(recursive_level)
                            << ", NodeInfo: " << trace::GetDebugInfo(node->debug_info());
        }
      } else {
        MS_LOG(INFO) << "Used current specializer, fg: " << fg->ToString()
                     << ", current fg: " << specializer->func_graph_->ToString()
                     << ", top fg: " << top_context->func_graph()->ToString();
      }
      break;
    } else {
      specializer = specializer->parent_;
    }
    if (specializer == nullptr) {
      constexpr auto recursive_level = 2;
      MS_LOG(EXCEPTION) << "Specializer should not be null, node: " << node->DebugString(recursive_level)
                        << ", NodeInfo: " << trace::GetDebugInfo(node->debug_info()) << ".\n"
                        << (func_graph_ ? func_graph_->ToString() : "FG(Null)")
                        << " has no parent context? At least not " << fg->ToString();
    }
  }
  return specializer;
}

void FuncGraphSpecializer::Run() {
  MS_LOG(DEBUG) << "Before run, origin func graph name: " << (func_graph_ ? func_graph_->ToString() : "FG(Null)")
                << ", cloned func graph name: "
                << (specialized_func_graph_ ? specialized_func_graph_->ToString() : "FG(Null)") << ", func graph: "
                << (func_graph_ ? func_graph_->get_return() ? func_graph_->get_return()->DebugString() : "return null"
                                : "FG(null)");
  FirstPass();
  SecondPass();
  MS_LOG(DEBUG) << "After run, origin func graph name: " << (func_graph_ ? func_graph_->ToString() : "FG(Null)")
                << ", cloned func graph name: "
                << (specialized_func_graph_ ? specialized_func_graph_->ToString() : "FG(Null)") << ", new func graph: "
                << (specialized_func_graph_ ? specialized_func_graph_->get_return()
                                                ? specialized_func_graph_->get_return()->DebugString()
                                                : "return null"
                                            : "FG(null)");
}

void FuncGraphSpecializer::FirstPass() {
  while (!todo_.empty()) {
    AnfNodePtr node = todo_.back();
    todo_.pop_back();
    if (node->func_graph() == nullptr) {
      // Do nothing for ValueNode
      continue;
    }
    if (node->func_graph() != func_graph_) {
      std::shared_ptr<FuncGraphSpecializer> parent = nullptr;
      if (parent_ != nullptr) {
        parent = parent_;
      } else if (specializer_->top_context()->func_graph() == node->func_graph() && node->isa<Parameter>()) {
        // If `parent_` is null and forwarded `node` is a Parameter, we'll try to use top func graph as parent.
        parent = specializer_->GetFuncGraphSpecializer(specializer_->top_context());
        MS_LOG(INFO) << "Used top func graph specializer as parent for " << func_graph_->ToString()
                     << ", node: " << node->DebugString() << ", NodeInfo: " << trace::GetDebugInfo(node->debug_info());
      }
      if (parent == nullptr) {
        MS_LOG(EXCEPTION) << "Parent must not be null, node: " << node->DebugString()
                          << ", NodeInfo: " << trace::GetDebugInfo(node->debug_info());
      }
      parent->AddTodoItem(node);
      parent->FirstPass();
      AnfNodePtr new_node = parent->GetReplicatedNode(node);
      if (new_node->isa<CNode>()) {
        MS_LOG(DEBUG) << "ProcessCNode in FirstPass for " << func_graph_->ToString()
                      << ", node: " << node->DebugString() << ", new_node: " << new_node->DebugString();
        parent->ProcessCNode(new_node->cast<CNodePtr>());
      }
      continue;
    }
    if (marked_.count(node) > 0) {
      continue;
    }
    (void)marked_.insert(node);
    ProcessNode(node);
  }
}

// Specialize CNode in func graphs
void FuncGraphSpecializer::SecondPass() {
  if (second_pass_todo_.empty()) {
    second_pass_todo_ = BroadFirstSearchGraphCNodes(specialized_func_graph_->return_node());
  }
  MS_LOG(DEBUG) << "Start in index: " << second_pass_todo_inedx_ << ", fg: " << func_graph_->ToString()
                << ", todo list size:" << second_pass_todo_.size();
  while (second_pass_todo_inedx_ < second_pass_todo_.size()) {
    auto success = ProcessCNode(second_pass_todo_[second_pass_todo_inedx_]);
    if (!success) {
      MS_LOG(DEBUG) << "Suspend in index:" << second_pass_todo_inedx_
                    << ", node: " << second_pass_todo_[second_pass_todo_inedx_]->DebugString();
      return;
    }
    ++second_pass_todo_inedx_;
  }
  MS_LOG(DEBUG) << "Set done of fg: " << func_graph_->ToString();
  done_ = true;
}

namespace {
// Update elements use flags for MakeTuple/tuple node,
// and update the node's AbstractSequence 'sequence_nodes' info.
void UpdateSequenceNode(const AnfNodePtr &new_node, const AnfNodePtr &old_node, const AbstractBasePtr &old_abs) {
  if (new_node == old_node) {
    return;
  }
  auto old_sequence_abs = dyn_cast_ptr<AbstractSequence>(old_abs);
  if (old_sequence_abs == nullptr || old_sequence_abs->sequence_nodes() == nullptr ||
      old_sequence_abs->sequence_nodes()->empty()) {
    MS_LOG(DEBUG) << "No sequence node in old abs, " << old_node->DebugString() << " --> " << new_node->DebugString();
    return;
  }

  // Since the 'old_node' may not equal to 'old_abs' sequence node,
  // if the new_node is built by the abstract of 'forward old node',
  // we just set 'new_node' as 'old_abs' sequence node here.
  if (IsValueNode<ValueTuple>(new_node) || IsValueNode<ValueList>(new_node)) {
    // Just find a valid sequence node.
    for (auto &weak_node : *old_sequence_abs->sequence_nodes()) {
      auto sequence_node = weak_node.lock();
      if (sequence_node == nullptr) {
        continue;
      }
      auto flags = GetSequenceNodeElementsUseFlags(sequence_node);
      if (flags == nullptr) {
        continue;
      }
      // Copy the flags to new node, and set new node to sequence abstract.
      // Actually, here we needn't require unique sequence nodes pointer between abstract any more.
      SetSequenceNodeElementsUseFlags(new_node, flags);
      old_sequence_abs->InsertSequenceNode(new_node);
      return;
    }
    MS_LOG(ERROR) << "Not found any valid sequence node, " << old_node->DebugString() << " --> "
                  << new_node->DebugString();
    return;
  }

  for (auto &weak_node : *old_sequence_abs->sequence_nodes()) {
    auto sequence_node = weak_node.lock();
    if (sequence_node == nullptr) {
      MS_LOG(DEBUG) << "The sequence_nodes is free. " << old_node->DebugString() << " --> " << new_node->DebugString();
      continue;
    }
    if (sequence_node != old_node) {
      continue;
    }

    // Update new node's flags with old one, and update old sequence abstract's source node.
    auto flags = GetSequenceNodeElementsUseFlags(old_node);
    MS_LOG(DEBUG) << "Update sequence node, " << old_node->DebugString() << " --> " << new_node->DebugString()
                  << ", elements_use_flags: " << (*flags);
    SetSequenceNodeElementsUseFlags(new_node, flags);
    old_sequence_abs->UpdateSequenceNode(sequence_node, new_node);

    // Update new sequence abstract if it's not equal to old one.
    const AbstractBasePtr &new_abs = new_node->abstract();
    if (old_abs == new_abs) {
      continue;
    }
    MS_LOG(ERROR) << "New abstract, " << old_node->DebugString() << " --> " << new_node->DebugString()
                  << ", elements_use_flags: " << (*flags);
    auto new_sequence_abs = dyn_cast_ptr<AbstractSequence>(new_abs);
    if (new_sequence_abs == nullptr) {
      MS_LOG(EXCEPTION) << "New node should be sequence type as well, but got " << new_abs->ToString();
    }
    if (new_sequence_abs->sequence_nodes() == nullptr || new_sequence_abs->sequence_nodes()->empty()) {
      std::shared_ptr<AnfNodeWeakPtrList> new_sequence_nodes = std::make_shared<AnfNodeWeakPtrList>();
      (void)new_sequence_nodes->emplace_back(AnfNodeWeakPtr(new_node));
      new_sequence_abs->set_sequence_nodes(new_sequence_nodes);
    } else {
      new_sequence_abs->InsertSequenceNode(new_node);
    }
  }
}

// Purify specific input of a CNode.
template <typename T, typename S>
void PurifySequenceValueNode(const CNodePtr &cnode, size_t index, ProgramSpecializer *const specializer) {
  const auto &old_input = cnode->input(index);
  auto sequence_value = GetValuePtr<T>(old_input);
  if (sequence_value == nullptr) {
    return;
  }
  auto flags = GetSequenceNodeElementsUseFlags(old_input);
  if (flags == nullptr) {
    return;
  }
  auto old_input_abs = old_input->abstract();
  MS_EXCEPTION_IF_NULL(old_input_abs);
  auto old_sequence_abs = dyn_cast<AbstractSequence>(old_input_abs);
  MS_EXCEPTION_IF_NULL(old_sequence_abs);

  std::vector<size_t> dead_node_positions;
  ValuePtrList elements;
  AbstractBasePtrList elements_abs{};
  auto sequence_value_size = sequence_value->value().size();
  if (flags->size() < sequence_value_size) {
    MS_LOG(EXCEPTION) << "Inner exception. CNode: " << cnode->ToString() << " input: " << old_input->ToString()
                      << " flags size: " << flags->size() << " values size: " << sequence_value->value().size();
  }
  for (size_t i = 0; i < sequence_value_size; ++i) {
    ValuePtr old_sequence_value = sequence_value->value()[i];
    auto old_sequence_err_value = old_sequence_value->cast_ptr<ValueProblem>();
    if (old_sequence_err_value != nullptr && old_sequence_err_value->IsDead()) {
      MS_LOG(DEBUG) << "Collect for erasing elements[" << i << "] DeadNode as zero for " << old_input->DebugString()
                    << ", which is inputs[" << index << "] of " << cnode->DebugString();
      (void)dead_node_positions.emplace_back(i);
    }
    if (!(*flags)[i]) {
      auto zero = MakeValue(0);
      (void)elements.emplace_back(zero);
      (void)elements_abs.emplace_back(zero->ToAbstract());
      MS_LOG(DEBUG) << "Erase elements[" << i << "] as zero for " << old_input->DebugString() << ", which is inputs["
                    << index << "] of " << cnode->DebugString();
    } else {
      (void)elements.emplace_back(old_sequence_value);
      (void)elements_abs.emplace_back(old_sequence_abs->elements()[i]);
    }
  }

  auto new_sequence_value = std::make_shared<T>(elements);
  auto new_input = NewValueNode(new_sequence_value);
  auto new_sequence_abs = std::make_shared<S>(elements_abs);
  std::shared_ptr<AnfNodeWeakPtrList> sequence_nodes = std::make_shared<AnfNodeWeakPtrList>();
  (void)sequence_nodes->emplace_back(AnfNodeWeakPtr(new_input));
  new_sequence_abs->set_sequence_nodes(sequence_nodes);
  new_input->set_abstract(new_sequence_abs);
  // Always reset tuple value node's use flags as non-use.
  SetSequenceNodeElementsUseFlags(new_input, flags);
  MS_LOG(DEBUG) << "Update ValueTuple/ValueList, " << old_input->DebugString() << " --> " << new_input->DebugString()
                << ", which is inputs[" << index << "] of " << cnode->DebugString() << ", flags: " << (*flags);
  // Keep the node not to release before we purify its abstract.
  (void)specializer->sequence_abstract_list().emplace_back(std::pair(new_sequence_abs, old_input));
  for (size_t pos : dead_node_positions) {
    (void)specializer->dead_node_list().emplace_back(std::pair(new_input, pos));
  }
  cnode->set_input(index, new_input);
}

bool CheckAbstractSequence(const AbstractSequencePtr &abs) {
  return abs != nullptr && abs->sequence_nodes() != nullptr && !abs->sequence_nodes()->empty();
}
}  // namespace

// First elimination.
// Eliminate the unused items of Tuple/List.
// Just adjust the nodes, not change the abstracts and dead nodes.
void FuncGraphSpecializer::EliminateUnusedSequenceItem(const CNodePtr &cnode) const {
  if (cnode == nullptr || cnode->abstract() == nullptr) {
    MS_LOG(EXCEPTION) << "The parameter \'node\' and its abstract should not be null.";
  }
  auto &sequence_abstract_list = specializer_->sequence_abstract_list();

  // Add CNode's inputs if they're sequence abstract, and sequence nodes exist.
  (void)std::for_each(cnode->inputs().begin(), cnode->inputs().end(),
                      [&sequence_abstract_list](const AnfNodePtr &input) {
                        const AbstractBasePtr input_abs = input->abstract();
                        AbstractSequencePtr input_sequence_abs = dyn_cast<AbstractSequence>(input_abs);
                        if (!CheckAbstractSequence(input_sequence_abs)) {
                          return;
                        }
                        // Not call PurifyElements() here, just add to list.
                        (void)sequence_abstract_list.emplace_back(std::pair(input_sequence_abs, input));
                      });

  // Add CNode if it's sequence abstract, and sequence nodes exist.
  const AbstractBasePtr abs = cnode->abstract();
  AbstractSequencePtr sequence_abs = dyn_cast<AbstractSequence>(abs);
  if (!CheckAbstractSequence(sequence_abs)) {
    return;
  }
  // Not call PurifyElements() here, just add to list.
  (void)sequence_abstract_list.emplace_back(std::pair(sequence_abs, cnode));

  // Purify MakeTuple/MakeList CNode.
  if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) || IsPrimitiveCNode(cnode, prim::kPrimMakeList)) {
    auto flags = GetSequenceNodeElementsUseFlags(cnode);
    if (flags != nullptr) {
      std::vector<AnfNodePtr> inputs;
      (void)inputs.emplace_back(cnode->input(0));
      for (size_t i = 0; i < (*flags).size(); ++i) {
        auto old_input = cnode->input(i + 1);
        if (!(*flags)[i]) {
          auto zero_value = NewValueNode(MakeValue(0));
          zero_value->set_abstract(std::make_shared<abstract::AbstractScalar>(std::make_shared<Int32Imm>(0)));
          (void)inputs.emplace_back(zero_value);
          constexpr int recursive_level = 2;
          MS_LOG(DEBUG) << "Erase elements[" << i << "] as zero for " << cnode->DebugString(recursive_level);
        } else if (IsDeadNode(old_input)) {
          constexpr int recursive_level = 2;
          MS_LOG(DEBUG) << "Collect for erasing elements[" << i << "] DeadNode as zero for " << cnode << "/"
                        << cnode->DebugString(recursive_level);
          (void)specializer_->dead_node_list().emplace_back(std::pair(cnode, i));
          (void)inputs.emplace_back(old_input);
        } else {
          (void)inputs.emplace_back(old_input);
        }
      }
      cnode->set_inputs(std::move(inputs));
      cnode->set_abstract(sequence_abs);
    }
  }
  // Purify each Tuple/List ValueNode in CNode.
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    if (IsValueNode<ValueTuple>(cnode->input(i))) {
      PurifySequenceValueNode<ValueTuple, AbstractTuple>(cnode, i, specializer_);
    } else if (IsValueNode<ValueList>(cnode->input(i))) {
      PurifySequenceValueNode<ValueList, AbstractList>(cnode, i, specializer_);
    }
  }
}

void FuncGraphSpecializer::ProcessNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  ScopeGuard scope_guard(node->scope());
  AnfNodeConfigPtr conf = MakeConfig(node);
  MS_EXCEPTION_IF_NULL(conf);
  TraceGuard guard(std::make_shared<TraceCopy>(node->debug_info()));
  AnfNodePtr new_node = GetReplicatedNode(node);
  MS_EXCEPTION_IF_NULL(new_node);
  if (new_node->func_graph() != specialized_func_graph_) {
    MS_LOG(EXCEPTION) << "Found not specialized node, node: " << node->DebugString()
                      << ", new_node: " << new_node->DebugString() << ", new_node->func_graph(): "
                      << (new_node->func_graph() ? new_node->func_graph()->ToString() : "FG(Null)")
                      << ", specialized_func_graph_: " << specialized_func_graph_->ToString();
  }
  EvalResultPtr conf_eval_result = nullptr;
  try {
    conf_eval_result = conf->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(conf_eval_result);
    new_node->set_abstract(conf_eval_result->abstract());
  } catch (const std::exception &) {
    MS_LOG(EXCEPTION) << "Fail to get abstract value with " << conf->ToString() << ", for " << new_node->DebugString();
  }
  if (new_node->isa<CNode>() && new_node->abstract()->isa<PartialAbstractClosure>()) {
    auto partial_abstract = dyn_cast_ptr<PartialAbstractClosure>(new_node->abstract());
    if (partial_abstract->node() == node) {
      partial_abstract->set_node(new_node);
    }
  }
  MS_LOG(DEBUG) << "Set new_node: " << new_node->DebugString() << ", abstract as: " << new_node->abstract()->ToString()
                << ", func_graph_: " << func_graph_->ToString()
                << ", specialized_func_graph_: " << specialized_func_graph_->ToString();

  if (!node->isa<CNode>()) {
    return;
  }
  static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
  auto attrs = conf_eval_result->attribute();
  auto c_old = node->cast_ptr<CNode>();
  auto c_new = new_node->cast_ptr<CNode>();
  MS_EXCEPTION_IF_NULL(c_new);
  auto new_inputs = c_new->inputs();
  auto old_inputs = c_old->inputs();
  for (size_t i = 0; i < old_inputs.size(); ++i) {
    auto node_input = old_inputs[i];
    AnfNodeConfigPtr input_conf = MakeConfig(node_input);
    AbstractBasePtr abs;
    try {
      abs = GetEvaluatedValue(input_conf);
    } catch (const std::exception &) {
      MS_LOG(EXCEPTION) << "Fail to get input's abstract value, with input config: " << input_conf->ToString()
                        << ", in old node: " << c_old->DebugString();
    }
    bool ignore_build_value = false;
    AnfNodePtr replace_node = nullptr;
    if (specializer_->engine()->check_side_effect()) {
      auto cnode_input = dyn_cast_ptr<CNode>(node_input);
      ignore_build_value = (cnode_input != nullptr && cnode_input->has_side_effect_node());
      if (ignore_build_value) {
        MS_LOG(INFO) << "Don't build value node for CNode which contains isolated side-effect inputs, node: "
                     << cnode_input->DebugString() << ", flag: " << cnode_input->has_side_effect_node();
      }
    }
    if (!ignore_build_value) {
      // First try to check if node_input can be replaced by a ValueNode. If cannot, then try to check if
      // can be replaced by another CNode from anfnode_config_map, otherwise use the replicated node.
      replace_node = BuildPossibleValueNode(node_input, abs, attrs, node);
    }
    if (replace_node == nullptr) {
      replace_node = BuildReplacedNode(input_conf);
      replace_node->set_abstract(abs);
      MS_LOG(DEBUG) << "Set replaced input[" << i << "]: " << replace_node->DebugString()
                    << ", NodeConfig: " << input_conf->ToString() << ", result: " << abs.get() << "/"
                    << abs->ToString();
    } else {
      MS_LOG(DEBUG) << "Build possible value node for node: " << node_input->DebugString()
                    << ", abs: " << abs->ToString() << ", replace_node: " << replace_node->DebugString();
    }
    if (enable_eliminate_unused_element) {
      UpdateSequenceNode(replace_node, node_input, abs);
    }
    if (new_inputs[i] != replace_node) {
      new_inputs[i] = replace_node;
      MS_LOG(DEBUG) << "Set new_input[" << i << "] = " << replace_node->DebugString();
    }
  }
  c_new->set_inputs(new_inputs);
}

AnfNodePtr FuncGraphSpecializer::BuildReplacedNode(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  auto conf_iter = engine_->anfnode_config_map().find(conf);
  AnfNodeConfigPtr new_conf = conf;
  while (conf_iter != engine_->anfnode_config_map().end()) {
    MS_LOG(DEBUG) << "Origin conf: node(" << (new_conf->node() ? new_conf->node()->DebugString() : "Node(Null)") << ")";
    new_conf = conf_iter->second;
    MS_EXCEPTION_IF_NULL(new_conf);
    const auto &forward_node = new_conf->node();
    MS_LOG(DEBUG) << "Replaced conf: node(" << forward_node->DebugString() << ")";
    const auto &replicated_forward_node = ReplicateDisconnectedNode(forward_node);
    if (replicated_forward_node && replicated_forward_node->isa<CNode>()) {
      // The AnfNode in order_list can be:
      // case 1: also in FuncGraphManager, so it can be got from nodes API of func_graph. it will
      //         be replaced in CloneOrderList in Cloner.
      // case 2: AnfNode is not in FuncGraphManager which generated in Analyze phase, so it will not
      //         be cloned by normal clone API.
      //    2.1: A forward node , the original node is in FuncGraphManager. The original node will
      //         be cloned in CloneOrderList in Cloner, and the replicated forward node will replace
      //         the replicated original node here.
      //    2.2: an input of a forward node, such as Cast CNode generated in DoCast. It is also another
      //         original node to fowrad.
      //    2.3: an input of an input of a forward node, but it's not an original node. Like the Cast CNode
      //         in MixedPrecisionCastHelper.
      // For 2.2 and 2.3, we will put a placeholder in order list of replicated func_graph, refer to
      // CloneOrderlist, and it will be replaced inside ReplicateDisconnectedNode.
      // For 2.1 the following code will do the job, replace replicated origin cnode with the replicated
      // forward one in the replicated func_graph.
      MS_EXCEPTION_IF_NULL(conf_iter->first);
      const auto &origin_node = conf_iter->first->node();
      const auto &replicated_origin_node = GetReplicatedNode(origin_node);
      if (replicated_origin_node != origin_node) {
        MS_LOG(DEBUG) << "Replace replicated origin node in order list: " << replicated_origin_node->DebugString()
                      << ", with replicated forwarded node: " << replicated_forward_node->DebugString();
        MS_EXCEPTION_IF_NULL(replicated_forward_node->func_graph());
        replicated_forward_node->func_graph()->ReplaceInOrder(replicated_origin_node, replicated_forward_node);
      } else {
        MS_LOG(EXCEPTION) << "Origin node is not replicated in specialized func_graph, origin node: "
                          << (origin_node ? origin_node->DebugString() : "Node(Null)");
      }
    }
    conf_iter = engine_->anfnode_config_map().find(new_conf);
  }
  AddTodoItem(new_conf->node());
  auto repl = GetReplicatedNode(new_conf->node());
  if (repl->func_graph()) {
    MS_LOG(DEBUG) << "Set repl: graph(" << repl->func_graph()->ToString() << "), node: " << repl->DebugString()
                  << ") to replace origin: " << new_conf->node()->DebugString();
  } else {
    MS_LOG(DEBUG) << "Set repl: graph(nullptr), node(" << repl->DebugString()
                  << ") to replace origin: " << new_conf->node()->DebugString();
  }
  return repl;
}

AnfNodePtr FuncGraphSpecializer::BuildSpecializedNode(const CNodePtr &cnode, const AnfNodePtr &func,
                                                      const AbstractBasePtr &abs, const AbstractBasePtrList &argvals) {
  MS_EXCEPTION_IF_NULL(abs);
  MS_EXCEPTION_IF_NULL(func);
  auto real_a = dyn_cast_ptr<AbstractFunction>(abs);
  MS_EXCEPTION_IF_NULL(real_a);

  AbstractFunctionPtr func_abs = real_a->GetUnique();
  SpecializeStatusCode errcode;
  ScopeGuard scope_guard(func->scope());
  AnfNodePtr repl = BuildSpecializedNodeInner(cnode, func, abs, func_abs, argvals, &errcode);
  if (repl == nullptr) {
    // If errcode is success, it means child graph specialize.
    if (errcode == kSpecializeSuccess) {
      return nullptr;
    }
    if (errcode == kSpecializeDead) {
      const auto err_dead_value = std::make_shared<ValueProblem>(ValueProblemType::kDead);
      const auto err_dead_abstract = std::make_shared<AbstractProblem>(err_dead_value, func);
      repl = BuildValueNode(err_dead_value, err_dead_abstract);
      constexpr auto recursive_level = 2;
      MS_LOG(DEBUG) << "DEAD for func: " << func->DebugString(recursive_level) << ", abstract: " << abs->ToString();
    } else if (errcode == kSpecializePoly) {
      const auto error_poly_value = std::make_shared<ValueProblem>(ValueProblemType::kPoly);
      const auto error_poly_abstract = std::make_shared<AbstractProblem>(error_poly_value, func);
      repl = BuildValueNode(error_poly_value, error_poly_abstract);
      constexpr auto recursive_level = 2;
      MS_LOG(DEBUG) << "POLY for func: " << func->DebugString(recursive_level) << ", abstract: " << abs->ToString();
    } else {
      MS_LOG(EXCEPTION) << "Failed to build specialized func, func: " << func->DebugString()
                        << ", abstract: " << abs->ToString();
    }
  }

  // Set the flag, so this MetaFuncGraph will be Re-AutoMonaded.
  MS_EXCEPTION_IF_NULL(func_abs);
  if (func_abs->isa<MetaFuncGraphAbstractClosure>()) {
    auto specialized_fg = GetValuePtr<FuncGraph>(repl);
    if (specialized_fg != nullptr && (argvals.size() > 1) && argvals.back() != nullptr &&
        argvals.back()->isa<AbstractUMonad>()) {
      specialized_fg->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
    }
  }
  return repl;
}

AnfNodePtr FuncGraphSpecializer::BuildSpecializedNodeInner(const CNodePtr &cnode, const AnfNodePtr &func,
                                                           const AbstractBasePtr &abs,
                                                           const AbstractFunctionPtr &func_abs,
                                                           const AbstractBasePtrList &args,
                                                           SpecializeStatusCode *errcode) {
  MS_EXCEPTION_IF_NULL(abs);
  MS_EXCEPTION_IF_NULL(func_abs);
  MS_EXCEPTION_IF_NULL(errcode);
  *errcode = kSpecializeSuccess;
  auto real_func = dyn_cast_ptr<TypedPrimitiveAbstractClosure>(func_abs);
  if (real_func != nullptr) {
    return BuildValueNode(real_func->prim(), abs);
  }

  EvaluatorPtr eval = engine_->GetEvaluatorFor(func_abs);
  MS_EXCEPTION_IF_NULL(eval);
  eval->set_bound_node(cnode);
  AbstractBasePtrList argvals = eval->NormalizeArgs(args);
  std::pair<AbstractBasePtrList, AbstractBasePtr> result;
  SpecializeStatusCode status = AcquireUniqueEvalVal(func_abs, eval, argvals, &result);
  if (status != kSpecializeSuccess) {
    *errcode = status;
    return nullptr;
  }
  argvals = result.first;
  AbstractBasePtr unique_output = result.second;

  auto prim_func = dyn_cast_ptr<PrimitiveAbstractClosure>(func_abs);
  if (prim_func != nullptr) {
    auto type_func = std::make_shared<TypedPrimitiveAbstractClosure>(prim_func->prim(), argvals, unique_output);
    return BuildValueNode(prim_func->prim(), type_func);
  }

  if (!eval->isa<BaseFuncGraphEvaluator>()) {
    MS_LOG(EXCEPTION) << "Expect the eval is a BaseGraphEvaluator, but got " << eval->ToString()
                      << ", func: " << func->DebugString() << ", abs: " << func_abs->ToString() << ", args: " << args;
  }
  auto real_eval = dyn_cast<BaseFuncGraphEvaluator>(eval);

  if (func_abs->context() == nullptr) {
    MS_LOG(EXCEPTION) << "Func context is nullptr NodeInfo: " << trace::GetDebugInfo(func_graph_->debug_info());
  }
  auto context = GetAnalysisContext(engine_, real_eval, argvals);
  if (context == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to get context from static analysis cache, call node:" << cnode->DebugString()
                      << ", args: " << mindspore::ToString(args);
  }

  constexpr auto recursive_level = 2;
  MS_LOG(DEBUG) << "Specialize function graph: " << context->func_graph()->ToString() << ", args: " << argvals
                << ", func: " << func->DebugString(recursive_level) << ", context:" << context.get() << ", "
                << context->ToString();
  MS_EXCEPTION_IF_NULL(context->func_graph());
  if (context->func_graph()->stub()) {
    MS_LOG(DEBUG) << "Specialize stub function graph, return the original node: " << context->func_graph()->ToString()
                  << ", args: " << argvals.size() << ", graph: " << context->func_graph()->get_return()->DebugString()
                  << ", " << func->ToString();
    return func;
  }
  // Get the upper most func graph of which parent has been specialized.
  while (ParentNotSpecialized(context)) {
    context = context->parent();
  }
  auto fg_spec = specializer_->GetFuncGraphSpecializer(context);
  // If func graph specializer dose not exist before, make a new specializer and push to stack, and return nullptr.
  if (fg_spec == nullptr) {
    fg_spec = specializer_->NewFuncGraphSpecializer(context, context->func_graph());
    specializer_->PushFuncGraphTodoItem(fg_spec);
    return nullptr;
  }

  FuncGraphPtr func_graph = fg_spec->specialized_func_graph();
  MS_LOG(DEBUG) << "Get spec fg of func graph:" << context->func_graph()->ToString()
                << ", specialized fg:" << func_graph->ToString();
  MS_EXCEPTION_IF_NULL(func_graph);
  func_graph->set_flag(kFuncGraphFlagUndetermined, false);
  static auto dummy_context = AnalysisContext::DummyContext();
  MS_EXCEPTION_IF_NULL(dummy_context);
  // Build a map that map unspecialized abstract function to specialized function, later it can be used
  // for specialize input0 of CNode in specialized func graph if input0 is not FuncGraph.
  auto new_abs_func = std::make_shared<FuncGraphAbstractClosure>(func_graph, dummy_context, nullptr, true);
  specializer_->PutSpecializedAbstract(cnode, func, func_abs, new_abs_func);
  if (func_abs->isa<FuncGraphAbstractClosure>()) {
    const auto &func_graph_abs = dyn_cast_ptr<FuncGraphAbstractClosure>(func_abs);
    specializer_->PutSpecializedFuncGraphToAbstract(func_graph_abs->func_graph(), new_abs_func);
  }
  return BuildValueNode(func_graph, new_abs_func);
}

AnfNodePtr FuncGraphSpecializer::BuildSpecializedParameterNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto new_inputs = cnode->inputs();
  if (new_inputs.empty()) {
    MS_LOG(EXCEPTION) << "inputs can't be empty.";
  }
  AnfNodePtr func = new_inputs[0];
  MS_EXCEPTION_IF_NULL(func);
  AbstractBasePtr fnval = func->abstract();

  AbstractBasePtrList args;
  auto backed_fnval = fnval;
  if (fnval->isa<PartialAbstractClosure>()) {
    auto partial_closure = dyn_cast_ptr<PartialAbstractClosure>(fnval);
    backed_fnval = partial_closure->fn();
    args = partial_closure->args();
  }
  std::transform(new_inputs.cbegin() + 1, new_inputs.cend(), std::back_inserter(args),
                 [](const AnfNodePtr &inp) { return inp->abstract(); });

  ScopeGuard scope_guard(cnode->scope());
  auto specialized_node = BuildSpecializedNode(cnode, func, backed_fnval, args);
  if (specialized_node == nullptr) {
    return nullptr;
  }
  auto wrapped_node = specialized_node;
  if (fnval->isa<PartialAbstractClosure>()) {
    auto partial_closure = dyn_cast<PartialAbstractClosure>(fnval);
    AnfNodePtrList partial_node_list = {BuildValueNode(prim::kPrimPartial, FromValueInside(prim::kPrimPartial)),
                                        specialized_node};
    auto anf_node = partial_closure->node();
    if (!anf_node->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Must be cnode, but " << anf_node->DebugString();
    }
    auto partial_cnode = anf_node->cast<CNodePtr>();
    constexpr auto extra_args_size = 2;
    if (partial_cnode->size() != partial_closure->args().size() + extra_args_size) {
      MS_LOG(EXCEPTION) << "Size of cnode: " << partial_cnode->DebugString()
                        << " is not equal to 2 added to size of args: " << mindspore::ToString(partial_closure->args());
    }
    auto attrs = std::make_shared<AttrValueMap>();
    for (size_t i = 0; i < partial_closure->args().size(); i++) {
      auto old_node = partial_cnode->input(i + 2);
      auto possibile_value_node = BuildPossibleValueNode(old_node, partial_closure->args()[i], attrs);
      if (possibile_value_node != nullptr) {
        partial_node_list.push_back(possibile_value_node);
      } else {
        if (!(old_node->isa<CNode>() || old_node->isa<Parameter>())) {
          MS_LOG(EXCEPTION) << "Old node should be CNode or Parameter, but " << old_node->ToString();
        }
        partial_node_list.push_back(old_node);
      }
    }
    MS_EXCEPTION_IF_NULL(cnode->func_graph());
    wrapped_node = cnode->func_graph()->NewCNode(std::move(partial_node_list));
    wrapped_node->set_abstract(partial_closure);
  }
  return wrapped_node;
}

const EvaluatorCacheMgrPtr FuncGraphSpecializer::GetEvalCache(const EvaluatorPtr &eval) {
  MS_EXCEPTION_IF_NULL(eval);
  auto cache_iter = eval_cache_.find(eval);
  if (cache_iter == eval_cache_.end()) {
    eval_cache_[eval] = eval->evaluator_cache_mgr();
    return eval->evaluator_cache_mgr();
  }
  return cache_iter->second;
}

std::pair<AbstractBasePtrList, AbstractBasePtr> FuncGraphSpecializer::BuildFromBroadedArgsVal(
  const EvaluatorPtr &eval) {
  MS_EXCEPTION_IF_NULL(eval);
  std::unordered_set<AbstractBasePtrList, AbstractBasePtrListHasher, AbstractBasePtrListEqual> choices;
  EvalResultPtr res = nullptr;
  AbstractBasePtrList broaded_argvals;
  std::vector<AbstractBasePtrList> args_vector;
  auto eval_cache_iter = eval_cache_.find(eval);
  if (eval_cache_iter == eval_cache_.end()) {
    MS_LOG(EXCEPTION) << "Evaluator: " << eval->ToString() << " not exist in cache.";
  }
  auto &origin_eval_cache = eval_cache_iter->second->GetCache();
  for (auto &argvals_map : origin_eval_cache) {
    auto argvals = argvals_map.first;
    args_vector.push_back(argvals);
  }
  // If joinable, maybe choices size is 1 or dynamic shape.
  constexpr auto args_size = 2;
  if (args_vector.size() < args_size) {
    MS_LOG(EXCEPTION) << "Should have " << args_size << " or more choices, but: " << args_vector.size();
  }
  AbstractBasePtrList joined_argvals = args_vector[0];
  for (size_t i = 1; i < args_vector.size(); ++i) {
    // The args may be not joinable (AbstractScalar join with AbstractTensor), just ignore that case.
    try {
      MS_LOG_TRY_CATCH_SCOPE;
      joined_argvals = abstract::AbstractJoin(joined_argvals, args_vector[i]);
    } catch (const std::exception &e) {
      MS_LOG(DEBUG) << "Cannot join, args1: " << ::mindspore::ToString(joined_argvals)
                    << ", args2: " << ::mindspore::ToString(args_vector[i]);
      return std::make_pair(AbstractBasePtrList(), nullptr);
    }
  }
  MS_LOG(DEBUG) << "Joined argvals: " << joined_argvals.size() << ", " << ::mindspore::ToString(joined_argvals);

  EvaluatorCacheMgrPtr real = std::make_shared<EvaluatorCacheMgr>();
  const auto joined_eval_result = origin_eval_cache.get(joined_argvals);
  if (joined_eval_result != nullptr) {
    MS_LOG(DEBUG) << "Find unique choice in original eval cache for joined argvals: " << joined_eval_result->ToString();
    real->SetValue(joined_argvals, joined_eval_result);
    eval_cache_[eval] = real;
    return std::make_pair(joined_argvals, joined_eval_result->abstract());
  }
  for (const auto &argvals : args_vector) {
    broaded_argvals.clear();
    BroadenArgs(argvals, &broaded_argvals);
    (void)choices.insert(broaded_argvals);
    MS_LOG(DEBUG) << "Broaded_argvals: " << broaded_argvals.size() << ", " << ::mindspore::ToString(broaded_argvals);
  }
  if (choices.size() == 1) {
    ConfigPtrList args_conf_list;
    (void)std::transform(broaded_argvals.cbegin(), broaded_argvals.cend(), std ::back_inserter(args_conf_list),
                         [](const AbstractBasePtr &v) -> ConfigPtr { return std::make_shared<VirtualConfig>(v); });
    MS_LOG(DEBUG) << "Cannot find joined argvals in cache, run with broaded argsvals: " << broaded_argvals.size()
                  << ", " << ::mindspore::ToString(broaded_argvals);
    res = eval->SingleRun(engine_, args_conf_list, nullptr);
    MS_EXCEPTION_IF_NULL(res);
    real->SetValue(broaded_argvals, res);
    eval_cache_[eval] = real;
    return std::make_pair(broaded_argvals, res->abstract());
  }
  MS_LOG(DEBUG) << "Choices.size: " << choices.size();
  return std::make_pair(AbstractBasePtrList(), nullptr);
}

bool IsHighOrderCall(const AnfNodePtr &func) {
  return !func->isa<ValueNode>() && func->abstract()->isa<AbstractFunction>() &&
         !func->abstract()->isa<AbstractFuncUnion>();
}

bool FuncGraphSpecializer::ProcessCNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (specializer_->seen().count(cnode) > 0) {
    return true;
  }
  auto new_inputs = cnode->inputs();
  if (new_inputs.empty()) {
    MS_LOG(EXCEPTION) << "Inputs of CNode is empty";
  }
  AnfNodePtr func = new_inputs[0];
  MS_EXCEPTION_IF_NULL(func);
  constexpr auto recursive_level = 2;
  MS_LOG(DEBUG) << "Handle CNode: " << cnode->DebugString(recursive_level);

  // First element is func so arg start from 1
  std::vector<AnfNodePtr> args(new_inputs.begin() + 1, new_inputs.end());
  // CNode(CNode(Partial, f, arg1), arg2, ...) --> CNode(f, arg1, arg2, ...)
  const size_t arg_start_index = 2;
  while (IsPrimitiveCNode(func, prim::kPrimPartial)) {
    std::vector<AnfNodePtr> inputs = func->cast_ptr<CNode>()->inputs();
    // First element is partial, second is func so arg is start from 2
    (void)args.insert(args.cbegin(), inputs.cbegin() + SizeToInt(arg_start_index), inputs.cend());
    func = inputs[1];
  }
  new_inputs = args;
  (void)new_inputs.insert(new_inputs.cbegin(), func);

  // Deal with the CNode|Parameter function call including Partial closure ahead.
  if (IsHighOrderCall(func)) {
    auto func_abs = func->abstract()->cast<AbstractFunctionPtr>();
    EvaluatorPtr eval = engine_->GetEvaluatorFor(func_abs);
    std::pair<AbstractBasePtrList, AbstractBasePtr> result;
    AbstractBasePtrList empty_args;
    auto status = AcquireUniqueEvalVal(func_abs, eval, empty_args, &result);
    MS_LOG(DEBUG) << "Poly: " << (status == kSpecializePoly) << ", func: " << func->ToString() << ", "
                  << ", abstract: " << func_abs->ToString() << ", "
                  << func->func_graph()->has_flag(FUNC_GRAPH_FLAG_SPECIALIZE_PARAMETER);
    // If a node is a poly node, or an input parameter is a PartialAbstractClosure, expand it early
    MS_EXCEPTION_IF_NULL(func->func_graph());
    if (status == kSpecializePoly ||
        (func->isa<Parameter>() && func->func_graph()->has_flag(FUNC_GRAPH_FLAG_SPECIALIZE_PARAMETER))) {
      auto wrapped_node = BuildSpecializedParameterNode(cnode);
      if (wrapped_node == nullptr) {
        return false;
      }
      MS_LOG(DEBUG) << "Partial closure is handled, wrapped_node: " << wrapped_node->DebugString(recursive_level);
      new_inputs[0] = wrapped_node;
    }
  }

  // Specialize the function, aka inputs[0], if input0 is a ValueNode<FuncGraph> or ValueNode<Primitive>,
  // CanSpecializeValueNode return true, otherwise false.
  if (CanSpecializeValueNode(func)) {
    // For primitive node, we build the primitive node with inferred attributes in the first pass,
    // so we do not build replaced node again here in second pass.
    if (IsValueNode<Primitive>(func)) {
      new_inputs[0] = func;
    } else {
      AbstractBasePtrList argvals;
      AbstractBasePtr fnval = new_inputs[0]->abstract();
      // First element is function, so the arguments start from 1.
      for (size_t i = 1; i < new_inputs.size(); ++i) {
        argvals.push_back(new_inputs[i]->abstract());
      }
      auto specialized_func_node = BuildSpecializedNode(cnode, func, fnval, argvals);
      if (specialized_func_node == nullptr) {
        return false;
      }
      new_inputs[0] = specialized_func_node;
      MS_LOG(DEBUG) << "Specalize func: " << func->type_name() << "/" << func->DebugString(recursive_level)
                    << ", new_func: " << new_inputs[0]->DebugString(recursive_level) << ", argvals: " << argvals;
    }
  }

  // Specialize the arguments, except inputs[0].
  for (size_t i = 1; i < new_inputs.size(); ++i) {
    auto &old_node = new_inputs[i];
    if (CanSpecializeValueNode(old_node)) {
      auto new_node = BuildSpecializedNode(cnode, old_node, old_node->abstract(), std::vector<AbstractBasePtr>{});
      if (new_node == nullptr) {
        return false;
      }
      MS_LOG(DEBUG) << "Specalize arg[" << i << "]: " << old_node->DebugString(recursive_level)
                    << ", new_node: " << new_node->DebugString(recursive_level);
      new_inputs[i] = new_node;
    }
  }

  // Set the updated inputs.
  cnode->set_inputs(new_inputs);

  // Eliminate the unused elements in the tuple/list.
  static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
  static const auto enable_only_mark_unused_element = (common::GetEnv("MS_DEV_DDE_ONLY_MARK") == "1");
  if (enable_eliminate_unused_element && !enable_only_mark_unused_element) {
    EliminateUnusedSequenceItem(cnode);
  }
  // Only success processed node can be added to seen.
  specializer_->AddSeen(cnode);
  return true;
}

bool FuncGraphSpecializer::ParentNotSpecialized(const AnalysisContextPtr &context) {
  auto parent_context = context->parent();
  auto parent_specializer = specializer_->GetFuncGraphSpecializer(parent_context);
  // If can't get specializer of parent and parent is not DummyContext, it means parent not specialized.
  auto parent_not_specialized = parent_specializer == nullptr && parent_context->func_graph() != nullptr;
  return parent_not_specialized;
}

namespace {
void DumpEvaluatorCache(const EvaluatorCacheMgrPtr &evaluator_cache_mgr, const AbstractBasePtrList &argvals) {
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr);
  MS_LOG(DEBUG) << "Find unique argvals failed: " << argvals.size() << ", " << argvals << ". Check cache all items.";
  int64_t i = 0;
  const EvalResultCache &map = evaluator_cache_mgr->GetCache();
  for (const auto &item : map) {
    MS_LOG(DEBUG) << "\tevaluator_cache[" << i++ << "]: " << item.first;
  }
}

bool IsPolyFunc(const AbstractFunctionPtr &func, const AbstractBasePtrList &argvals) {
  MS_EXCEPTION_IF_NULL(func);
  if (func->isa<PrimitiveAbstractClosure>() && argvals.empty()) {
    MS_LOG(DEBUG) << "High order primitive return POLY.";
    return true;
  }
  if (func->isa<MetaFuncGraphAbstractClosure>() && argvals.empty()) {
    auto meta_func_graph_wrapper = dyn_cast_ptr<MetaFuncGraphAbstractClosure>(func);
    auto meta_func_graph = meta_func_graph_wrapper->meta_func_graph();
    if (meta_func_graph != nullptr && meta_func_graph->isa<prim::DoSignatureMetaFuncGraph>()) {
      auto do_signature = dyn_cast_ptr<prim::DoSignatureMetaFuncGraph>(meta_func_graph);
      if (do_signature != nullptr && do_signature->function()->isa<Primitive>()) {
        MS_LOG(DEBUG) << "High order primitive " << do_signature->function()->ToString() << " return POLY.";
        return true;
      }
    }
  }
  return false;
}
}  // namespace

SpecializeStatusCode FuncGraphSpecializer::AcquireUniqueEvalVal(const AbstractFunctionPtr &func,
                                                                const EvaluatorPtr &eval,
                                                                const AbstractBasePtrList &argvals,
                                                                std::pair<AbstractBasePtrList, AbstractBasePtr> *res) {
  MS_EXCEPTION_IF_NULL(func);
  MS_EXCEPTION_IF_NULL(eval);
  MS_EXCEPTION_IF_NULL(res);

  EvaluatorCacheMgrPtr evaluator_cache_mgr = eval->evaluator_cache_mgr();
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr);
  auto data = evaluator_cache_mgr->GetValue(argvals);
  if (data != nullptr) {
    *res = std::make_pair(argvals, data->abstract());
    return kSpecializeSuccess;
  }
  DumpEvaluatorCache(evaluator_cache_mgr, argvals);

  auto cache = GetEvalCache(eval);
  MS_EXCEPTION_IF_NULL(cache);
  const EvalResultCache &choices = cache->GetCache();
  auto eval_result = choices.get(argvals);
  if (eval_result != nullptr) {
    *res = std::make_pair(argvals, eval_result->abstract());
    return kSpecializeSuccess;
  } else if (choices.size() == 1) {
    MS_LOG(DEBUG) << "Evaluator cache has a single item, just use it.";
    MS_EXCEPTION_IF_NULL(choices.begin()->second);
    *res = std::make_pair(choices.begin()->first, choices.begin()->second->abstract());
    return kSpecializeSuccess;
  } else if (choices.empty()) {
    MS_LOG(DEBUG) << "Find DEAD code, it may be optimized in later phase " << func->ToString() << " | "
                  << func->type_name() << ", evaluator:" << eval->ToString() << ",ptr:" << eval.get();
    return kSpecializeDead;
  } else {
    if (IsPolyFunc(func, argvals)) {
      return kSpecializePoly;
    }
    *res = BuildFromBroadedArgsVal(eval);
    if (!res->first.empty()) {
      MS_LOG(DEBUG) << "Build for generalized argvals successfully.";
      // Synchronize the new evaluated abstract with the abstract from common evaluating routine.
      MS_EXCEPTION_IF_NULL(res->second);
      auto new_sequence_abs = dyn_cast<abstract::AbstractSequence>(res->second);
      for (auto &choice : choices) {
        MS_EXCEPTION_IF_NULL(choice.second);
        MS_EXCEPTION_IF_NULL(choice.second->abstract());
        auto abs = choice.second->abstract()->cast<AbstractSequencePtr>();
        if (abs != nullptr) {
          SynchronizeSequenceElementsUseFlagsRecursively(abs, new_sequence_abs);
        }
      }
      return kSpecializeSuccess;
    }
    MS_LOG(DEBUG) << "Find POLY code, it may be unused code or unresolved polymorphism, "
                  << "func: " << func->ToString() << ", choices.size: " << choices.size()
                  << ", argvals.size: " << argvals.size();
    return kSpecializePoly;
  }
}

static PrimitivePtr BuildPrimtiveValueWithAttributes(const PrimitivePtr &prim, const AttrValueMapPtr &attrs) {
  MS_EXCEPTION_IF_NULL(prim);
  auto &prim_attrs = prim->attrs();
  bool is_attr_same = true;
  for (auto &item : *attrs) {
    auto itr = prim_attrs.find(item.first);
    if (itr != prim_attrs.end()) {
      MS_EXCEPTION_IF_NULL(itr->second);
      MS_EXCEPTION_IF_NULL(item.second);
      if (!(*(itr->second) == *(item.second))) {
        is_attr_same = false;
        break;
      }
    } else {
      is_attr_same = false;
      break;
    }
  }
  if (!is_attr_same) {
    auto cloned_prim = prim->Clone();
    for (auto &item : *attrs) {
      cloned_prim->AddAttr(item.first, item.second);
    }
    return cloned_prim;
  }
  return prim;
}

AnfNodePtr FuncGraphSpecializer::BuildValueNodeForAbstractFunction(const AnfNodePtr &origin_node,
                                                                   const AbstractBasePtr &ival,
                                                                   const AttrValueMapPtr &attrs,
                                                                   const AnfNodePtr &cnode,
                                                                   const AbstractFunctionPtr &abs) {
  ValuePtr value = nullptr;
  if (abs->isa<PrimitiveAbstractClosure>()) {
    auto real_fn = dyn_cast_ptr<PrimitiveAbstractClosure>(abs);
    MS_EXCEPTION_IF_NULL(real_fn);
    // For primitive, check if the attribute is the same with cnode inferred attribute, if not, clone a new one
    if (attrs != nullptr) {
      value = BuildPrimtiveValueWithAttributes(real_fn->prim(), attrs);
    } else {
      value = real_fn->prim();
    }
  } else if (abs->isa<MetaFuncGraphAbstractClosure>()) {
    auto real_fn = dyn_cast_ptr<MetaFuncGraphAbstractClosure>(abs);
    value = real_fn->meta_func_graph();
  } else if (abs->isa<FuncGraphAbstractClosure>()) {
    auto real_fn = dyn_cast_ptr<FuncGraphAbstractClosure>(abs);
    value = real_fn->func_graph();
  } else {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<FuncGraph>() || value->cast_ptr<FuncGraph>()->parent() == nullptr ||
      (IsValueNode<FuncGraph>(origin_node) && IsVisible(func_graph_, value->cast_ptr<FuncGraph>()->parent()))) {
    return BuildValueNode(value, ival);
  } else if (IsPrimitiveCNode(cnode, prim::kPrimJ) && origin_node->isa<Parameter>() &&
             !value->cast_ptr<FuncGraph>()->has_flag(FUNC_GRAPH_FLAG_K_GRAPH)) {
    // Only if J(Parameter=func_graph) and func_graph(aka 'value') is not K graph.
    MS_LOG(DEBUG) << "Specialize the parameter used by J CNode, cnode: " << cnode->DebugString();
    return BuildValueNode(value, ival);
  }
  return nullptr;
}

AnfNodePtr FuncGraphSpecializer::BuildPossibleValueNode(const AnfNodePtr &origin_node, const AbstractBasePtr &ival,
                                                        const AttrValueMapPtr &attrs, const AnfNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(origin_node);
  MS_EXCEPTION_IF_NULL(ival);

  AbstractFunctionPtr abs = dyn_cast<AbstractFunction>(ival);
  if (abs != nullptr) {
    // Cannot build a deterministic ValueNode if there are multiple possible AbstractFunction.
    if (abs->isa<AbstractFuncUnion>()) {
      return nullptr;
    }
    return BuildValueNodeForAbstractFunction(origin_node, ival, attrs, cnode, abs);
  } else {
    ValuePtr val = ival->BuildValue();
    if (val->isa<ValueAny>()) {
      return nullptr;
    }
    // If node is an AutoMonad node, don't convert the node to value node `U` or `IO` to avoid side-effect op miss.
    if (val->isa<Monad>()) {
      return nullptr;
    }
    // Keep primitive 'depend' not to be optimized
    if (IsPrimitiveCNode(origin_node, prim::kPrimDepend)) {
      return nullptr;
    }
    return BuildValueNode(val, ival);
  }
}

inline AnalysisContextPtr FuncGraphSpecializer::GetAnalysisContext(const AnalysisEnginePtr &engine,
                                                                   const BaseFuncGraphEvaluatorPtr &evaluator,
                                                                   const AbstractBasePtrList &args_spec_list) const {
  // If it is common calling header, try to use the context generated by the infer process of body calling header, so
  // need broaden the args to keep context of common calling header same with context of body calling header.
  AbstractBasePtrList normalized_args_spec_list = evaluator->NormalizeArgs(args_spec_list);
  FuncGraphPtr fg = evaluator->GetFuncGraph(engine, normalized_args_spec_list);
  auto parent_context = evaluator->parent_context();
  MS_EXCEPTION_IF_NULL(parent_context);
  auto cached_context = parent_context->GetCachedContext(fg, normalized_args_spec_list);
  if (cached_context != nullptr) {
    return cached_context;
  }
  // If can't get context by broadened args, try to get context by not broadened args.
  cached_context = parent_context->GetCachedContext(fg, args_spec_list);
  if (cached_context != nullptr) {
    return cached_context;
  }
  // if it is a bprop meta func graph, need to make a new context and do static analysis in ProcessNode.
  return parent_context->NewContext(fg, normalized_args_spec_list);
}
}  // namespace abstract
}  // namespace mindspore
