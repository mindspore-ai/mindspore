/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/parse/function_block.h"

#include <string>
#include <memory>
#include <algorithm>

#include "pybind11/pybind11.h"
#include "pipeline/jit/parse/resolve.h"
#include "pipeline/jit/parse/parse.h"
#include "pipeline/jit/parse/data_converter.h"
#include "frontend/operator/ops.h"
#include "utils/info.h"
#include "debug/trace.h"
#include "utils/utils.h"

namespace mindspore {
namespace py = pybind11;

namespace parse {
FunctionBlock::FunctionBlock(const Parser &parser) : parser_(parser) {
  func_graph_ = std::make_shared<FuncGraph>();
  matured_ = false;
}

void FunctionBlock::AddPrevBlock(const FunctionBlockPtr &block) { prev_blocks_.push_back(block.get()); }

static bool CanBeIsolatedNode(const std::string &var_name, const AnfNodePtr &node) {
  auto cnode = dyn_cast<CNode>(node);
  if (cnode == nullptr || cnode->inputs().empty()) {
    // Not a valid cnode, can not be isolate node.
    return false;
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->inputs().at(0));
  if (prim == nullptr) {
    // Not a primitive cnode, it may have side effects or not,
    // We add it as an isolate node if its name is not '_' or empty.
    // this means that code like:
    //    _ = func_call()
    // will be ignored even if func_call() has side effects.
    return !var_name.empty() && var_name != "_";
  }
  // Primitive cnode with side effects can be isolate nodes.
  auto effect_info = GetPrimEffectInfo(prim);
  bool has_effects = (effect_info.memory || effect_info.io);
  if (has_effects) {
    return true;
  }
  // Primitive cnode with 'no_eliminate' flag can be isolate nodes.
  return GetPrimitiveFlag(prim, ATTR_NO_ELIMINATE);
}

// Write variable records the variable name to corresponding node
void FunctionBlock::WriteVariable(const std::string &var_name, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << (func_graph_ ? func_graph_->ToString() : "FG(Null)") << " write var `" << var_name << "` with node "
                << node->DebugString();
  auto [iter, is_new_name] = assigned_vars_.emplace(var_name, std::make_pair(node, false));
  if (!is_new_name) {
    // If a cnode variable with same name already existed but not used,
    // add it as an isolate node. for example:
    //   a = print(x)
    //   a = print(y)
    // When we write variable 'a = print(y)',
    // the cnode 'print(x)' should added as an isolate node.
    auto is_used = iter->second.second;
    auto hidden_node = iter->second.first;
    auto is_isolated = CanBeIsolatedNode(var_name, hidden_node);
    if (!is_used && is_isolated) {
      MS_EXCEPTION_IF_NULL(hidden_node);
      MS_LOG(INFO) << "Isolated node found(Hidden), hidden_node: " << hidden_node->DebugString(2) << " is hidden by "
                   << node->DebugString(2) << " with the same name, var_name: " << var_name << ", block: " << this
                   << "/" << (func_graph_ ? func_graph_->ToString() : "FG(Null)")
                   << ", Line: " << trace::GetDebugInfo(hidden_node->debug_info(), "", kSourceLineTipDiscard);
      AddIsolatedNode(hidden_node);
    }
    iter->second = std::make_pair(node, false);
  }
}

// Read variable from predecessors
AnfNodePtr FunctionBlock::ReadVariable(const std::string &var) {
  MS_LOG(DEBUG) << "Read begin, var: " << var << ", block: " << ToString();
  // Get var node if it is found
  auto found = assigned_vars_.find(var);
  if (found != assigned_vars_.end()) {
    auto &node = found->second.first;
    MS_EXCEPTION_IF_NULL(node);
    // Mark the variable as used.
    found->second.second = true;
    auto iter = resolve_to_removable_phis_.find(node);
    if (iter != resolve_to_removable_phis_.end()) {
      return iter->second;
    }
    return node;
  }
  // Get var from predecessor block, if can't get then make a resolve node to it
  if (matured_) {
    // If only one predecessor block, read the definition of var from it.
    if (prev_blocks_.size() == 1) {
      auto block = prev_blocks_[0];
      MS_EXCEPTION_IF_NULL(block);
      auto res = block->ReadVariable(var);
      MS_LOG(INFO) << "Update global params of block: " << ToString() << ", with previous block: " << block->ToString()
                   << ",\nCurrent: " << py::str(global_py_params())
                   << "\nInsert: " << py::str(block->global_py_params());
      CopyGlobalPyParam(block->global_py_params());
      return res;
    } else if (prev_blocks_.empty()) {
      // Get namespace and make Resolve
      auto it = var_to_resolve_.find(var);
      if (it != var_to_resolve_.end()) {
        return it->second;
      }
      MS_LOG(DEBUG) << "var: " << var;
      auto tmp_node = MakeResolveSymbol(var);
      var_to_resolve_[var] = tmp_node;
      return tmp_node;
    }
  }
  // If have more than one predecessor blocks then build a phi node.
  auto debug_info = std::make_shared<NodeDebugInfo>();
  debug_info->set_name(var);
  TraceGuard guard(std::make_shared<TracePhi>(debug_info));
  ParameterPtr phi_param = std::make_shared<Parameter>(func_graph());
  MS_LOG(DEBUG) << (func_graph_ ? func_graph_->ToString() : "FG(Null)") << " generate phi node "
                << phi_param->ToString() << " for " << var;
  func_graph()->add_parameter(phi_param);
  phi_nodes_[phi_param] = var;
  WriteVariable(var, phi_param);
  if (matured_) {
    SetPhiArgument(phi_param);
  }
  return phi_param;
}

// Resolve Ast operator node
AnfNodePtr FunctionBlock::MakeResolveAstOp(const py::object &op) {
  auto ast = parser_.ast();
  MS_EXCEPTION_IF_NULL(ast);
  TraceGuard trace_guard(parser_.GetLocation(op));
  py::tuple namespace_var = ast->CallParseModFunction(PYTHON_PARSE_GET_AST_NAMESPACE_SYMBOL, op);
  if (namespace_var.size() != 2) {
    MS_LOG(EXCEPTION) << "Resolve ast op failed, get namespace tuple size=" << namespace_var.size();
  }
  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_AST, namespace_var[0]);
  SymbolPtr symbol = std::make_shared<Symbol>(namespace_var[1].cast<std::string>());
  MS_LOG(DEBUG) << "name_space: " << name_space->ToString() << ", symbol: " << symbol->ToString();
  return MakeResolve(name_space, symbol);
}

// Resolve class member, two possible: method, member variable
AnfNodePtr FunctionBlock::MakeResolveClassMember(const std::string &attr) {
  auto ast = parser_.ast();
  MS_EXCEPTION_IF_NULL(ast);
  py::object namespace_var = ast->CallParseModFunction(PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL, ast->obj());
  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, namespace_var);
  SymbolPtr symbol = std::make_shared<Symbol>(attr);
  MS_LOG(DEBUG) << "name_space: " << name_space->ToString() << ", symbol: " << symbol->ToString();
  return MakeResolve(name_space, symbol);
}

AnfNodePtr FunctionBlock::HandleNamespaceInfo(const py::tuple &namespace_info) {
  const size_t namespace_info_size = 2;
  const size_t namespace_more_info_size = 3;
  if (namespace_info.size() != namespace_info_size && namespace_info.size() != namespace_more_info_size) {
    MS_EXCEPTION(NameError) << "namespace info size should be 2 or 3, but got " << namespace_info.size();
  }
  bool unsupported = false;
  py::object py_obj;
  if (namespace_info.size() == namespace_more_info_size) {
    if (namespace_info[0].is_none()) {  // If namespace is None, the symbol is an undefined name.
      MS_EXCEPTION(NameError) << namespace_info[namespace_more_info_size - 1].cast<std::string>();
    } else {  // Or, the symbol is an unsupported builtin symbol in Graph mode.
      unsupported = true;
      py_obj = namespace_info[namespace_more_info_size - 1];
    }
  }
  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_SYMBOL_STR, namespace_info[0]);
  SymbolPtr symbol = std::make_shared<Symbol>(namespace_info[1].cast<std::string>());
  MS_LOG(DEBUG) << "[" << func_graph()->ToString() << "] name_space: " << name_space->ToString()
                << ", symbol: " << symbol->ToString() << ", unsupported: " << unsupported;
  auto resolved_node = MakeResolve(name_space, symbol);
  if (unsupported) {
    resolved_node->set_interpret(true);
    AddGlobalPyParam(symbol->name(), py_obj);
    MS_LOG(INFO) << "[" << func_graph()->ToString() << "] Added global python symblol: {" << symbol->name() << " : "
                 << py::str(py_obj) << "}";
  }
  return resolved_node;
}

// Make a resolve node for symbol string
AnfNodePtr FunctionBlock::MakeResolveSymbol(const std::string &value) {
  MS_LOG(DEBUG) << "value: " << value;
  if (value.compare(0, strlen("self"), "self") == 0) {
    auto start = value.find_first_of('.') + 1;
    if (start >= value.size()) {
      MS_LOG(ERROR) << "Find invalid resolve symbol str: " << value;
      return nullptr;
    }
    auto bits_str = value.substr(start);
    return MakeResolveClassMember(bits_str);
  }
  auto ast = parser_.ast();
  MS_EXCEPTION_IF_NULL(ast);

  // The fallback feature is enabled in default.
  // Not support change the flag during the process is alive.
  static const auto use_fallback = (parser_.support_fallback() == "1");
  if (!use_fallback) {
    py::tuple namespace_info = ast->CallParserObjMethod(PYTHON_PARSE_GET_NAMESPACE_SYMBOL, value);
    return HandleNamespaceInfo(namespace_info);
  } else {
    py::tuple namespace_info = ast->CallParserObjMethod(PYTHON_PARSE_GET_BUILTIN_NAMESPACE_SYMBOL, value);
    return HandleNamespaceInfo(namespace_info);
  }
}

AnfNodePtr FunctionBlock::MakeResolveOperation(const std::string &value) {
  auto ast = parser_.ast();
  MS_EXCEPTION_IF_NULL(ast);
  py::tuple namespace_var = ast->CallParseModFunction(PYTHON_PARSE_GET_OPERATION_NAMESPACE_SYMBOL, value);
  const size_t namespace_var_size = 2;
  if (namespace_var.size() < namespace_var_size) {
    MS_EXCEPTION(NameError) << "namespace_var is less than 2";
  }
  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_COMMON_OPS, namespace_var[0]);
  SymbolPtr symbol = std::make_shared<Symbol>(namespace_var[1].cast<std::string>());
  MS_LOG(DEBUG) << "name_space: " << name_space->ToString() << ", symbol: " << symbol->ToString();
  return MakeResolve(name_space, symbol);
}

AnfNodePtr FunctionBlock::MakeResolve(const NameSpacePtr &name_space, const SymbolPtr &resolve_symbol) {
  MS_LOG(DEBUG) << "MakeResolve for " << (name_space ? (std::string)py::str(name_space->obj()) : "null namespace")
                << " , " << (resolve_symbol ? (std::string)resolve_symbol->symbol() : "null resoleve symbol.");
  ValueNodePtr module_node = NewValueNode(name_space);
  ValueNodePtr symbol_node = NewValueNode(resolve_symbol);
  auto node = func_graph_->NewCNodeInOrder({NewValueNode(prim::kPrimResolve), module_node, symbol_node});
  return node;
}

AnfNodePtr FunctionBlock::MakeInterpret(const std::string &script_text, const AnfNodePtr &global_dict_node,
                                        const AnfNodePtr &local_dict_node, const AnfNodePtr &orig_node) {
  MS_LOG(DEBUG) << "MakeInterpret for " << script_text;
  ScriptPtr script = std::make_shared<Script>(script_text);
  auto script_node = NewValueNode(script);
  auto node = func_graph_->NewCNodeInOrder(
    {NewValueNode(prim::kPrimPyInterpret), script_node, global_dict_node, local_dict_node});
  node->set_interpreted_node(orig_node);
  return node;
}

// Add input for the block's phi parameter
void FunctionBlock::SetPhiArgument(const ParameterPtr &phi) {
  MS_EXCEPTION_IF_NULL(phi);
  TraceGuard trace_guard(std::make_shared<TraceResolve>(phi->debug_info()));
  std::string var = phi_nodes_[phi];
  MS_LOG(DEBUG) << "graph " << (func_graph_ ? func_graph_->ToString() : "FG(Null)") << " set phi " << phi->ToString()
                << " for var `" << var << "`";
  auto removable = CollectRemovablePhi(phi);
  // If the phi node is not necessary, not need to add to jumps_ of the prev blocks.
  if (removable) {
    MS_LOG(DEBUG) << "remove the phi when call graph " << (func_graph_ ? func_graph_->ToString() : "FG(Null)")
                  << " var `" << var << "`";
    return;
  }
  for (auto &pred : prev_blocks_) {
    MS_EXCEPTION_IF_NULL(pred);
    MS_LOG(DEBUG) << "graph " << (func_graph_ ? func_graph_->ToString() : "FG(Null)") << " pred_blocks_ "
                  << (pred->func_graph_ ? pred->func_graph_->ToString() : "FG(Null)");
    AnfNodePtr arg_node = pred->ReadVariable(var);
    CNodePtr jump = pred->jumps_[this];
    MS_EXCEPTION_IF_NULL(jump);
    jump->add_input(arg_node);
  }
}

AnfNodePtr FunctionBlock::SearchReplaceNode(const std::string &var, const ParameterPtr &phi) {
  AnfNodePtr arg_node = nullptr;
  MS_LOG(DEBUG) << "Prev_blocks size: " << prev_blocks_.size();
  for (auto &prev : prev_blocks_) {
    MS_EXCEPTION_IF_NULL(prev);
    AnfNodePtr temp_node = prev->ReadVariable(var);
    MS_EXCEPTION_IF_NULL(temp_node);
    if (temp_node != phi) {
      if (arg_node == nullptr) {
        arg_node = temp_node;
        MS_LOG(DEBUG) << "graph " << (prev->func_graph_ ? prev->func_graph_->ToString() : "FG(Null)") << " phi "
                      << (phi ? phi->ToString() : "null") << " may be replaced by node " << arg_node->DebugString();
      } else if (temp_node == arg_node) {
        MS_LOG(DEBUG) << "graph " << (prev->func_graph_ ? prev->func_graph_->ToString() : "FG(Null)") << " phi "
                      << (phi ? phi->ToString() : "null") << " is same as node " << arg_node->DebugString();
      } else {
        MS_LOG(DEBUG) << "phi " << (phi ? phi->ToString() : "null")
                      << " cannot be removed as it assigns to different node. node1: " << arg_node->DebugString()
                      << ", node2: " << temp_node->DebugString();
        return nullptr;
      }
    }
  }
  return arg_node;
}

// Check if there is removable unnecessary phi node in this graph.
// As per the FIRM TR 3.2, a phi node can be remove if:
// <Quote>
//    If all arguments of a φ-function are the same value s or the φfunction itself,
//    then we remove the φ-function and let all users directly uses. We call such a
//    φ-function obviously unnecessary.
//    When we removed a φ-function p, then we recursively try to apply this simpliﬁcation
//    rule with all (former) users of p, because they may have become obviously unnecessary
//    due to the removal of p
// <Quote>
// phi node in graph will be removed after the whole function is parsed in a DFS visit
// of that graph.The reason is :
// 1. when this function is called, not all usage of this phi node had bound to the
// graph of this function block, some may stay in vars_ in other blocks.
// 2. it's costly to iterate the graph to replace the phi for each phi.
// Args: phi: This parameter node is functioning as a phi node.
bool FunctionBlock::CollectRemovablePhi(const ParameterPtr &phi) {
  MS_EXCEPTION_IF_NULL(phi);
  std::string var = phi_nodes_[phi];
  MS_LOG(DEBUG) << "check phi " << phi->DebugString() << " for " << var;
  if (prev_blocks_.empty()) {
    MS_LOG(DEBUG) << "no phi " << phi->DebugString() << " for var " << var;
    return false;
  }
  AnfNodePtr arg_node = SearchReplaceNode(var, phi);
  if (arg_node != nullptr) {
    arg_node->set_debug_info(phi->debug_info());
    MS_LOG(DEBUG) << "graph " << (func_graph_ ? func_graph_->ToString() : "FG(Null)") << " phi " << phi->ToString()
                  << " can be replaced with " << arg_node->DebugString();
    // Replace var with new one. This equal to statement in TR "v0 is immediately replaced by v1."
    WriteVariable(var, arg_node);
    removable_phis_[phi] = arg_node;
    resolve_to_removable_phis_[arg_node] = phi;
    // The following equal to statement "The φ-function defining v1, which now reads φ(v2, v1), is optimized
    // recursively". check if phi1 is assigned with this phi before, then phi1 can be replaced with arg_node.
    for (auto &prev : prev_blocks_) {
      MS_EXCEPTION_IF_NULL(prev);
      if (!prev->matured_) {
        continue;
      }
      for (auto &phi_iter : prev->removable_phis_) {
        MS_EXCEPTION_IF_NULL(phi_iter.second);
        if (phi_iter.second->isa<Parameter>()) {
          const auto &param = phi_iter.second->cast<ParameterPtr>();
          if (param == phi) {
            MS_LOG(DEBUG) << "graph " << (prev->func_graph_ ? prev->func_graph_->ToString() : "FG(Null)") << " var "
                          << phi_iter.first->DebugString() << " can be replaced from " << param->DebugString()
                          << " with " << arg_node->DebugString() << " in graph "
                          << (arg_node->func_graph() ? arg_node->func_graph()->ToString() : "FG(Null)");
            prev->removable_phis_[phi_iter.first] = arg_node;
          }
        }
      }
    }
    return true;
  }
  return false;
}

// A block should be marked matured if its predecessor blocks have been processed
void FunctionBlock::Mature() {
  const auto &graph_params = func_graph_->parameters();
  for (auto &param_itr : graph_params) {
    MS_EXCEPTION_IF_NULL(param_itr);
    auto param = param_itr->cast<ParameterPtr>();
    if (phi_nodes_.find(param) != phi_nodes_.cend()) {
      SetPhiArgument(param);
    }
  }
  matured_ = true;
}

// Force the condition node to bool using bool operation
CNodePtr FunctionBlock::ForceToBoolNode(const AnfNodePtr &cond) {
  MS_EXCEPTION_IF_NULL(cond);
  TraceGuard trace_guard(std::make_shared<TraceForceBool>(cond->debug_info()));
  CNodePtr op_apply_node = func_graph_->NewCNodeInOrder({MakeResolveOperation(NAMED_PRIMITIVE_BOOL), cond});
  return op_apply_node;
}

CNodePtr FunctionBlock::ForceToWhileCond(const AnfNodePtr &cond) {
  MS_EXCEPTION_IF_NULL(cond);
  TraceGuard trace_guard(std::make_shared<TraceForceWhileCond>(cond->debug_info()));
  CNodePtr op_apply_node = func_graph_->NewCNodeInOrder({MakeResolveOperation("while_cond"), cond});
  return op_apply_node;
}

// Perform a jump from this block to target block
void FunctionBlock::Jump(const FunctionBlockPtr &target_block, const std::vector<AnfNodePtr> &args) {
  MS_LOG(DEBUG) << "Jump from block: " << ToString() << " to block: " << target_block->ToString();
  MS_EXCEPTION_IF_NULL(target_block);
  if (is_dead_block_) {
    MS_LOG(DEBUG) << "Dead code block should not jump to other block! block: " << ToString();
    return;
  }
  if (func_graph_->get_return() != nullptr) {
    MS_LOG(EXCEPTION) << "Failure: have return node! NodeInfo: "
                      << trace::GetDebugInfo(func_graph_->get_return()->debug_info());
  }
  std::vector<AnfNodePtr> input_nodes;
  input_nodes.emplace_back(NewValueNode(target_block->func_graph()));
  (void)std::copy(args.begin(), args.end(), std::back_inserter(input_nodes));

  CNodePtr jump = func_graph_->NewCNodeInOrder(input_nodes);
  jumps_[target_block.get()] = jump;
  target_block->AddPrevBlock(shared_from_this());
  func_graph_->set_output(jump);
}

// Perform a conditional jump using switch operation.
// The first CNode select graph with condition, and than execute this graph
void FunctionBlock::ConditionalJump(AnfNodePtr condNode, const FunctionBlockPtr &true_block,
                                    const FunctionBlockPtr &false_block, bool) {
  MS_EXCEPTION_IF_NULL(true_block);
  MS_EXCEPTION_IF_NULL(false_block);
  if (func_graph_->get_return() != nullptr) {
    MS_LOG(EXCEPTION) << "Failure: have return node! NodeInfo: "
                      << trace::GetDebugInfo(func_graph_->get_return()->debug_info());
  }
  CNodePtr switch_app =
    func_graph_->NewCNodeInOrder({NewValueNode(prim::kPrimSwitch), condNode, NewValueNode(true_block->func_graph()),
                                  NewValueNode(false_block->func_graph())});
  CNodePtr switch_app_new = func_graph_->NewCNodeInOrder({switch_app});
  func_graph_->set_output(switch_app_new);
}

// Create cnode for the assign statement like 'self.target = source'.
// convert it to 'P.Assign(self.target, source)' and then add the cnode as isolate node.
void FunctionBlock::SetStateAssign(const AnfNodePtr &target, const AnfNodePtr &source) {
  const std::string primitive_name("assign");
  const std::string module_name("mindspore.ops.functional");
  ValueNodePtr assign_op = NewValueNode(prim::GetPythonOps(primitive_name, module_name, true));
  auto assign_node = func_graph_->NewCNodeInOrder({assign_op, target, source});
  MS_LOG(DEBUG) << "Isolated node found(Assign), assign_node: " << assign_node->DebugString(2) << ", block: " << this
                << "/" << func_graph_->ToString()
                << ", Line: " << trace::GetDebugInfo(assign_node->debug_info(), "", kSourceLineTipDiscard);
  AddIsolatedNode(assign_node);
}

void FunctionBlock::FindIsolatedNodes() {
  //
  // Search isolate nodes from variables, for example,
  // variable 'a' is an isolate node in below code:
  //
  //    def construct(self, x, y):
  //        a = print(x) # isolate node
  //        return x + y
  //
  std::set<AnfNodePtr> used;
  // Find used variables.
  for (const auto &var : assigned_vars_) {
    auto &node = var.second.first;
    if (node == nullptr) {
      continue;
    }
    bool is_used = var.second.second;
    if (is_used) {
      used.emplace(node);
    }
  }
  // Add isolated nodes which is unused var but not found in used set.
  for (const auto &var : assigned_vars_) {
    auto &node = var.second.first;
    bool is_used = var.second.second;
    if (node == nullptr || is_used) {
      continue;
    }
    auto &var_name = var.first;
    if (used.find(node) == used.end() && CanBeIsolatedNode(var_name, node)) {
      MS_LOG(INFO) << "Isolated node found(NoUse), node: " << node->DebugString(2) << ", var_name: " << var_name
                   << ", block: " << this << "/" << (func_graph() ? func_graph()->ToString() : "FG(Null)")
                   << ", Line: " << trace::GetDebugInfo(node->debug_info(), "", kSourceLineTipDiscard);
      AddIsolatedNode(node);
    }
  }
}

void FunctionBlock::AddIsolatedNode(const AnfNodePtr &target) { isolated_nodes_.add(target); }

void FunctionBlock::AttachIsolatedNodesBeforeReturn() {
  if (isolated_nodes_.empty()) {
    return;
  }
  std::vector<AnfNodePtr> states;
  states.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  constexpr int recursive_level = 2;
  for (auto &node : isolated_nodes_) {
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Adding dependency, node: " << node->DebugString(recursive_level) << " in "
                  << func_graph_->ToString();
    if (node->func_graph() == func_graph_) {
      states.emplace_back(node);
    } else {
      MS_LOG(INFO) << "Ignored FV dependency, node: " << node->DebugString(recursive_level) << " in "
                   << func_graph_->ToString();
    }
  }
  isolated_nodes_.clear();

  AnfNodePtr state = nullptr;
  if (states.size() == 1) {
    // Only MakeTuple, no state left.
    return;
  } else if (states.size() == 2) {
    // If there are only MakeTuple and another node in states(the states size is 2),
    // do not need to MakeTuple, just use the node.
    state = states[1];
  } else {
    state = func_graph_->NewCNode(states);
  }

  AnfNodePtr old_output = nullptr;
  auto return_node = func_graph_->get_return();
  if (return_node) {
    const size_t return_input_size = 2;
    if (return_node->inputs().size() < return_input_size) {
      MS_LOG(EXCEPTION) << "Length of inputs of output node is less than 2";
    }
    old_output = return_node->input(1);
  } else {
    old_output = NewValueNode(kNone);
  }
  AnfNodePtr stop_grad_node = func_graph_->NewCNode({NewValueNode(prim::kPrimStopGradient), state});
  CNodePtr depend_node = func_graph_->NewCNode({NewValueNode(prim::kPrimDepend), old_output, stop_grad_node});
  // We add this attribute for @constexpr use scene, since we must infer them before other nodes.
  // That means isolated nodes will be evaluated first. It's not complete, but works in most scenes.
  depend_node->AddAttr(kAttrTopoSortRhsFirst, MakeValue(true));
  MS_EXCEPTION_IF_NULL(state);
  MS_LOG(INFO) << "Attached for side-effect nodes, depend_node: " << depend_node->DebugString()
               << ", state: " << state->DebugString(2);
  func_graph_->set_output(depend_node, true);
}

void FunctionBlock::SetAsDeadBlock() { is_dead_block_ = true; }
}  // namespace parse
}  // namespace mindspore
