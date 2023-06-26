/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "load_mindir/infer_mindir.h"
#include <deque>
#include <set>
#include <map>
#include <memory>
#include <algorithm>
#include <string>
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "ir/func_graph.h"
#include "abstract/abstract_function.h"
#include "abstract/abstract_value.h"
#include "utils/ms_context.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace {
class MindIREngine {
 public:
  explicit MindIREngine(const FuncGraphPtr &root) : func_graph_(root), nodeuser_map_(root->manager()->node_users()) {}
  ~MindIREngine() = default;
  MindIREngine(const MindIREngine &) = delete;
  MindIREngine &operator=(const MindIREngine &) = delete;

  bool InferShape(const AbstractBasePtrList &args);

  void SetException(bool flag) { raise_exception_ = flag; }

 private:
  using AbstractBasePtrListPtr = std::shared_ptr<AbstractBasePtrList>;

  void Init(const AbstractBasePtrList &args);
  AbstractBasePtr InferPrimitiveShape(const PrimitivePtr &prim, const AbstractBasePtrList &args_abs_list) const;
  void EvalCommonPrimitive(const PrimitivePtr &prim, const CNodePtr &node, const AbstractBasePtrListPtr &args);
  void EvalPartialPrimitive(const CNodePtr &node, const AbstractBasePtrListPtr &args);
  void EvalReturnPrimitive(const CNodePtr &node);
  void InferParameter(const AnfNodePtr &node);
  void InferValueNode(const AnfNodePtr &node);
  void InferCNode(const AnfNodePtr &node);
  void EvalAbstractFunction(const abstract::AbstractFuncAtomPtr &func, const CNodePtr &node,
                            const AbstractBasePtrListPtr &args);
  void EvalPrimitiveAbastract(const abstract::PrimitiveAbstractClosurePtr &func, const CNodePtr &node,
                              const AbstractBasePtrListPtr &args);
  void EvalFuncGraphAbastract(const abstract::FuncGraphAbstractClosurePtr &func, const CNodePtr &node,
                              const AbstractBasePtrListPtr &args);
  void EvalPartialAbastract(const abstract::PartialAbstractClosurePtr &func, const CNodePtr &node,
                            const AbstractBasePtrListPtr &args);
  bool CheckCNodeNotReady(const CNodePtr &node);
  void UpdateReady(const AnfNodePtr &node);
  void SaveNodeInferResult(const AnfNodePtr &node, const AbstractBasePtr &result);
  AbstractBasePtr GetCNodeOperatorAbstract(const AnfNodePtr &node);

  FuncGraphPtr func_graph_;
  std::map<AnfNodePtr, int> node_input_depends_;
  std::map<AnfNodePtr, AbstractBasePtr> infer_result_;
  std::map<std::string, AbstractBasePtr> func_graph_result_;
  std::map<std::string, std::set<AnfNodePtr>> func_graph_visited_;
  std::deque<AnfNodePtr> ready_;
  std::set<AnfNodePtr> todo_;
  NodeUsersMap nodeuser_map_;
  bool raise_exception_ = false;
};

// Infer the root function graph.
bool MindIREngine::InferShape(const AbstractBasePtrList &args) {
  Init(args);
  while (!ready_.empty()) {
    auto current = ready_.front();
    MS_EXCEPTION_IF_NULL(current);
    ready_.pop_front();
    if (current->isa<CNode>()) {
      InferCNode(current);
    } else if (current->isa<ValueNode>()) {
      InferValueNode(current);
    } else if (current->isa<Parameter>()) {
      InferParameter(current);
    } else {
      MS_LOG(WARNING) << " There is something changed. Please check the code.";
    }
  }

  // Set abstract of node.
  for (const auto &item : infer_result_) {
    item.first->set_abstract(item.second);
  }

  if (todo_.empty()) {
    MS_LOG(DEBUG) << "Finish to Infere.";
    return true;
  }
  MS_LOG(INFO) << "Not finished to infer: " << todo_.size();
  for (const auto &node : todo_) {
    MS_LOG(DEBUG) << "Node uninfered: " << node->DebugString();
  }
  return false;
}

void MindIREngine::Init(const AbstractBasePtrList &args) {
  MS_EXCEPTION_IF_NULL(func_graph_);
  auto manager = func_graph_->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (const auto &node : manager->all_nodes()) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      (void)todo_.insert(node);
      node_input_depends_[node] = SizeToInt(cnode->inputs().size());
    } else if (node->isa<Parameter>()) {
      auto param = node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param);
      if (param->has_default()) {
        node_input_depends_[node] = 0;
        auto default_param = param->default_param();
        MS_EXCEPTION_IF_NULL(default_param);
        infer_result_[node] = default_param->ToAbstract();
        ready_.push_back(node);
      } else {
        node_input_depends_[node] = 1;
        (void)todo_.insert(node);
      }
    } else {
      // Value Node
      node_input_depends_[node] = 0;
      ready_.push_back(node);
    }
  }

  auto inputs = func_graph_->get_inputs();
  if (inputs.size() != args.size()) {
    MS_LOG(EXCEPTION) << "The input number of parameters is not Compatible.\n"
                      << "Mindir:" << inputs.size() << " inputs: " << args.size()
                      << " FuncGraph:" << func_graph_->ToString() << "\n"
                      << "For more details, please refer to the FAQ at https://www.mindspore.cn.";
  }
  // Root Func Parameters
  for (size_t i = 0; i < args.size(); ++i) {
    this->SaveNodeInferResult(inputs[i], args[i]);
  }
  MS_LOG(DEBUG) << "Finish init. Size of nodes:" << manager->all_nodes().size();
}

// Infer primitive using C++ implement.
AbstractBasePtr MindIREngine::InferPrimitiveShape(const PrimitivePtr &prim,
                                                  const AbstractBasePtrList &args_abs_list) const {
  MS_EXCEPTION_IF_NULL(prim);
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    auto found = abstract::GetPrimitiveInferImpl(prim);
    if (found.has_value()) {
      auto infer = found.value();
      if (infer.IsImplInferShapeAndType()) {
        return infer.InferShapeAndType(nullptr, prim, args_abs_list);
      }
    }
    if (raise_exception_) {
      MS_LOG(INTERNAL_EXCEPTION) << "Get infer shape function failed, primitive name:" << prim->name()
                                 << " primitive type:" << prim->type_name()
                                 << " It will keep the previous value with danger.";
    } else {
      MS_LOG(INFO) << "Get infer shape function failed, primitive name:" << prim->name()
                   << " primitive type:" << prim->type_name() << " It will keep the previous value with danger.";
    }
  } catch (const std::exception &ex) {
    if (raise_exception_) {
      MS_LOG(INTERNAL_EXCEPTION) << "Catch primitive:" << prim->ToString()
                                 << " InferPrimitiveShape exception:" << ex.what()
                                 << " It will keep the previous value with danger.";
    } else {
      MS_LOG(INFO) << "Catch primitive:" << prim->ToString() << " InferPrimitiveShape exception:" << ex.what()
                   << " It will keep the previous value with danger.";
    }
  }
  return nullptr;
}

void MindIREngine::EvalCommonPrimitive(const PrimitivePtr &prim, const CNodePtr &node,
                                       const AbstractBasePtrListPtr &args) {
  // Save MakeTuple cnode abstract by its own abstract when MakeTuple have an abstract of
  // AbstractCSRTensor/AbstractCOOTensor that can not be inferred by its Infer Functions.
  if (prim->name() == prim::kPrimMakeTuple->name()) {
    if (node->abstract() != nullptr && (node->abstract()->isa<abstract::AbstractSparseTensor>())) {
      MS_LOG(INFO) << "Save MakeTuple cnode abstract by its own abstract : " << node->abstract()->ToString();
      SaveNodeInferResult(node, node->abstract());
      return;
    }
  }

  AbstractBasePtrList args_abs_list;
  // Args has been resolved by partial
  if (args != nullptr) {
    (void)args_abs_list.insert(args_abs_list.end(), args->begin(), args->end());
  } else {
    (void)std::transform(node->inputs().begin() + 1, node->inputs().end(), std::back_inserter(args_abs_list),
                         [this](const AnfNodePtr &arg) { return infer_result_[arg]; });
  }

  // Call C++ infer
  auto result = InferPrimitiveShape(prim, args_abs_list);
  if (result == nullptr) {
    MS_LOG(INFO) << node->ToString()
                 << " can't be inferred shape. It will keep the previous value with danger. Prim: " << prim->ToString();
    result = node->abstract();
  }
  SaveNodeInferResult(node, result);
}

void MindIREngine::EvalReturnPrimitive(const CNodePtr &node) {
  if (node->inputs().size() < 2) {
    MS_LOG(INTERNAL_EXCEPTION) << node->DebugString() << " input size < 2";
  }
  auto result = infer_result_[node->inputs()[1]];
  auto funcName = node->func_graph()->ToString();
  auto it = func_graph_result_.find(funcName);
  if (it != func_graph_result_.end()) {
    try {
      MS_LOG_TRY_CATCH_SCOPE;
      result = result->Join(it->second);
    } catch (const std::exception &e) {
      MS_LOG(INFO) << "Join abstract for return node " << node->DebugString() << " failed, exception: " << e.what();
    }
  }
  this->func_graph_result_[funcName] = result;
  SaveNodeInferResult(node, result);
  MS_LOG(DEBUG) << funcName << " result: " << result->ToString();

  // Set the result of the node whose Operator is this funcGraph
  for (const auto &item : func_graph_visited_[funcName]) {
    SaveNodeInferResult(item, result);
  }
}

void MindIREngine::EvalPartialPrimitive(const CNodePtr &node, const AbstractBasePtrListPtr &args) {
  // Args has  been resolved
  if (args != nullptr) {
    if (args->size() < 2) {
      MS_LOG(INTERNAL_EXCEPTION) << node->DebugString() << " input size < 2";
    }
    auto real_func = (*args)[0]->cast<abstract::AbstractFuncAtomPtr>();
    if (real_func == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << (*args)[0]->ToString() << " is not a function abstract.";
    }
    AbstractBasePtrList partial_args_list;
    (void)partial_args_list.insert(partial_args_list.end(), args->begin() + 1, args->end());
    auto partial_func = std::make_shared<abstract::PartialAbstractClosure>(real_func, partial_args_list, node);
    SaveNodeInferResult(node, partial_func);
    return;
  }
  // Not Resolved.
  constexpr size_t kSizeTwo = 2;
  if (node->inputs().size() < kSizeTwo) {
    MS_LOG(INTERNAL_EXCEPTION) << node->DebugString() << " input size < " << kSizeTwo;
  }
  auto &func = infer_result_[node->inputs()[1]];
  auto real_func = func->cast<abstract::AbstractFuncAtomPtr>();
  if (real_func == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << func->ToString() << " is not a function abstract.";
  }
  AbstractBasePtrList partial_args_list;
  (void)std::transform(node->inputs().begin() + 2, node->inputs().end(), std::back_inserter(partial_args_list),
                       [this](const AnfNodePtr &arg) { return infer_result_[arg]; });
  auto partial_func = std::make_shared<abstract::PartialAbstractClosure>(real_func, partial_args_list, node);
  SaveNodeInferResult(node, partial_func);
}

void MindIREngine::EvalPartialAbastract(const abstract::PartialAbstractClosurePtr &func, const CNodePtr &node,
                                        const AbstractBasePtrListPtr &args) {
  AbstractBasePtrListPtr partial_args_list = std::make_shared<AbstractBasePtrList>();
  // Join arguments in partial and the rest arguments from args_conf_list.
  auto func_args = func->args();
  (void)partial_args_list->insert(partial_args_list->end(), func_args.begin(), func_args.end());
  if (args == nullptr) {
    // Not Recursive
    (void)std::transform(node->inputs().begin() + 1, node->inputs().end(), std::back_inserter(*partial_args_list),
                         [this](const AnfNodePtr &arg) { return infer_result_[arg]; });
  } else {
    // Recursive
    (void)partial_args_list->insert(partial_args_list->end(), args->begin(), args->end());
  }

  // Get real function
  abstract::AbstractFuncAtomPtrList abstractFuncList;
  auto build_fuction = [&abstractFuncList](const abstract::AbstractFuncAtomPtr &poss) {
    abstractFuncList.push_back(poss);
  };
  func->fn()->Visit(build_fuction);
  for (const auto &abstractFunc : abstractFuncList) {
    EvalAbstractFunction(abstractFunc, node, partial_args_list);
  }
}

void MindIREngine::SaveNodeInferResult(const AnfNodePtr &node, const AbstractBasePtr &result) {
  auto answer = result;
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    auto it = infer_result_.find(node);
    if (it != infer_result_.end()) {
      MS_LOG(DEBUG) << node->ToString() << " result: " << it->second->ToString();
      answer = result->Join(it->second);
      if (*answer == *(it->second)) {
        MS_LOG(DEBUG) << node->ToString() << " The value is not changed.";
        return;
      }
    }
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "Join abstract for node " << node->DebugString() << " failed, exception: " << e.what();
    return;
  }

  MS_LOG(DEBUG) << node->ToString() << " result: " << answer->ToString();
  infer_result_[node] = answer;
  UpdateReady(node);
}

void MindIREngine::EvalPrimitiveAbastract(const abstract::PrimitiveAbstractClosurePtr &func, const CNodePtr &node,
                                          const AbstractBasePtrListPtr &args) {
  auto prim = func->prim();
  // Return Primitive
  if (prim->name() == prim::kPrimReturn->name()) {
    EvalReturnPrimitive(node);
    return;
  }
  // Partial Primitive
  if (prim->name() == prim::kPrimPartial->name()) {
    EvalPartialPrimitive(node, args);
    return;
  }
  // common Primitive
  EvalCommonPrimitive(prim, node, args);
}

bool MindIREngine::CheckCNodeNotReady(const CNodePtr &node) {
  int depend = 0;
  for (const auto &input : node->inputs()) {
    depend += infer_result_.find(input) != infer_result_.end() ? 0 : 1;
  }
  this->node_input_depends_[node] = depend;
  return depend != 0;
}

void MindIREngine::EvalFuncGraphAbastract(const abstract::FuncGraphAbstractClosurePtr &func, const CNodePtr &node,
                                          const AbstractBasePtrListPtr &args) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func);
  MS_EXCEPTION_IF_NULL(func->func_graph());
  // Has Processd
  MS_LOG(DEBUG) << node->ToString() << " FuncGraph: " << func->ToString();
  auto funcName = func->func_graph()->ToString();
  auto it = func_graph_result_.find(funcName);
  if (it != func_graph_result_.end()) {
    MS_LOG(DEBUG) << "The abstract of " << node->ToString() << " = abstract of " << func->ToString();
    SaveNodeInferResult(node, it->second);

    // Process only one return valueNode function graph
    auto func_inputs = func->func_graph()->parameters();
    // args has been resolved in partial.
    if (args != nullptr) {
      if (func_inputs.size() != args->size()) {
        MS_LOG(INTERNAL_EXCEPTION) << func->func_graph()->ToString() << " input size:" << func_inputs.size()
                                   << " CNode:" << node->DebugString() << " input size:" << args->size();
      }
      for (size_t i = 0; i < func_inputs.size(); ++i) {
        infer_result_[func_inputs[i]] =
          (*args)[i];  // Not use SaveNodeInferResult because this function has been evaluated.
        (void)todo_.erase(func_inputs[i]);
      }
      return;
    }
    // args is not resolved.
    auto &cnode_inputs = node->inputs();
    if (func_inputs.size() != cnode_inputs.size() - 1) {
      MS_LOG(INTERNAL_EXCEPTION) << func->func_graph()->ToString() << " input size:" << func_inputs.size()
                                 << " CNode:" << node->DebugString() << " input size:" << cnode_inputs.size();
    }
    for (size_t i = 0; i < func_inputs.size(); ++i) {
      infer_result_[func_inputs[i]] = infer_result_[cnode_inputs[i + 1]];
      (void)todo_.erase(func_inputs[i]);
    }
    return;
  }

  // Be handling
  auto visitIt = func_graph_visited_.find(funcName);
  if (visitIt != func_graph_visited_.end()) {
    (void)visitIt->second.insert(node);
    return;
  }
  func_graph_visited_[funcName] = std::set<AnfNodePtr>({node});

  // Call the funcGraph
  auto func_inputs = func->func_graph()->parameters();

  // args has been resolved in partial.
  if (args != nullptr) {
    if (func_inputs.size() != args->size()) {
      MS_LOG(INTERNAL_EXCEPTION) << func->func_graph()->ToString() << " input size:" << func_inputs.size()
                                 << " CNode:" << node->DebugString() << " input size:" << args->size()
                                 << " may have unsupported parameters.";
    }
    for (size_t i = 0; i < func_inputs.size(); ++i) {
      SaveNodeInferResult(func_inputs[i], (*args)[i]);
    }
    return;
  }
  // args is not resolved.
  auto &cnode_inputs = node->inputs();
  if (func_inputs.size() != cnode_inputs.size() - 1) {
    MS_LOG(INTERNAL_EXCEPTION) << func->func_graph()->ToString() << " input size:" << func_inputs.size()
                               << " CNode:" << node->DebugString() << " input size:" << cnode_inputs.size()
                               << " may have unsupported parameters.";
  }

  for (size_t i = 0; i < func_inputs.size(); ++i) {
    SaveNodeInferResult(func_inputs[i], infer_result_[cnode_inputs[i + 1]]);
  }
}

void MindIREngine::InferParameter(const AnfNodePtr &node) { UpdateReady(node); }

void MindIREngine::InferValueNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = GetValueNode(node);
  MS_EXCEPTION_IF_NULL(value);
  AbstractBasePtr result;
  if (value->isa<FuncGraph>()) {
    auto func_graph = value->cast<FuncGraphPtr>();
    auto temp_context = abstract::AnalysisContext::DummyContext();
    result = std::make_shared<abstract::FuncGraphAbstractClosure>(func_graph, temp_context, node);
  } else if (value->isa<Primitive>()) {
    auto prim = value->cast<PrimitivePtr>();
    result = std::make_shared<abstract::PrimitiveAbstractClosure>(prim, node);
  } else {
    result = value->ToAbstract();
  }

  SaveNodeInferResult(node, result);
}

AbstractBasePtr MindIREngine::GetCNodeOperatorAbstract(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op = cnode->inputs()[0];
  auto it = infer_result_.find(op);
  if (it != infer_result_.end()) {
    return it->second;
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Can't get the abstract of Node:" << op->DebugString();
}

// If args is nullPtr, it is called by InferCNode, else it is called recursively by EvalPartialAbastract.
void MindIREngine::EvalAbstractFunction(const abstract::AbstractFuncAtomPtr &func, const CNodePtr &node,
                                        const AbstractBasePtrListPtr &args) {
  MS_EXCEPTION_IF_NULL(func);
  if (func->isa<abstract::PrimitiveAbstractClosure>()) {
    // C++ Primitive
    auto prim = func->cast<abstract::PrimitiveAbstractClosurePtr>();
    EvalPrimitiveAbastract(prim, node, args);
  } else if (func->isa<abstract::FuncGraphAbstractClosure>()) {
    // FuncGraph
    auto funcGraph = func->cast<abstract::FuncGraphAbstractClosurePtr>();
    EvalFuncGraphAbastract(funcGraph, node, args);
  } else if (func->isa<abstract::PartialAbstractClosure>()) {
    // Partial
    auto partialPrim = func->cast<abstract::PartialAbstractClosurePtr>();
    EvalPartialAbastract(partialPrim, node, args);
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "MindIR can't process the abstractFunc: " << func->DumpText();
  }
}

void MindIREngine::UpdateReady(const AnfNodePtr &node) {
  (void)todo_.erase(node);
  auto it = nodeuser_map_.find(node);
  if (it == nodeuser_map_.end()) {
    return;
  }
  const auto &users = it->second;
  MS_LOG(DEBUG) << node->ToString() << " has users: " << users.size();
  for (const auto &user : users) {
    int count = node_input_depends_[user.first];
    node_input_depends_[user.first] = count - 1;
    if (count <= 1) {
      ready_.push_back(user.first);
      MS_LOG(DEBUG) << "Node:" << user.first->ToString() << " is ready.";
      if (count < 1) {
        MS_LOG(INFO) << " There is something to do. Node:" << node->ToString() << " user:" << user.first->DebugString();
      }
    }
  }
}

void MindIREngine::InferCNode(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (CheckCNodeNotReady(cnode)) {
    MS_LOG(INFO) << "The node is not ready: " << cnode->DebugString();
    return;
  }
  AbstractBasePtr possible_func = GetCNodeOperatorAbstract(cnode);
  MS_EXCEPTION_IF_NULL(possible_func);
  auto type = possible_func->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  if (type->type_id() == kObjectTypeUndeterminedType) {
    MS_LOG(INTERNAL_EXCEPTION) << "EvalCNode eval Undetermined";
  }
  abstract::AbstractFunctionPtr func = dyn_cast<abstract::AbstractFunction>(possible_func);
  if (func == nullptr) {
    MS_LOG(ERROR) << "Can not cast to a AbstractFunction: " << possible_func->ToString() << ".";
    MS_EXCEPTION(ValueError) << "This may be not defined, and it can't be a operator. Please check code.";
  }
  abstract::AbstractFuncAtomPtrList abstractFuncList;
  auto build_fuction = [&abstractFuncList](const abstract::AbstractFuncAtomPtr &poss) {
    abstractFuncList.push_back(poss);
  };
  func->Visit(build_fuction);
  for (const auto &abstractFunc : abstractFuncList) {
    EvalAbstractFunction(abstractFunc, cnode, nullptr);
  }
}
}  // namespace
bool InferMindir(const FuncGraphPtr &root, const AbstractBasePtrList &args, bool raise_exception) {
  auto engine = std::make_shared<MindIREngine>(root);
  engine->SetException(raise_exception);
  return engine->InferShape(args);
}

bool ValidMindir(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto manager = root->manager();
  if (manager == nullptr) {
    manager = MakeManager();
    manager->AddFuncGraph(root, true);
  }
  abstract::AbstractBasePtrList func_args;
  const auto inputs = root->get_inputs();
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(func_args),
                       [](const AnfNodePtr &arg) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(arg);
                         if (arg->abstract() == nullptr) {
                           MS_LOG(ERROR) << "The parameter's abstract is null:" << arg->DebugString();
                         }
                         MS_EXCEPTION_IF_NULL(arg->abstract());
                         return arg->abstract();
                       });
  auto valid = InferMindir(root, func_args);
  if (!valid) {
    MS_LOG(ERROR) << "There is some wrong in the mindir. " << root->ToString() << " : " << root.get();
    return false;
  }
  MS_LOG(DEBUG) << "Success to valid the mindir. " << root->ToString() << " : " << root.get();
  return true;
}
}  // namespace mindspore
