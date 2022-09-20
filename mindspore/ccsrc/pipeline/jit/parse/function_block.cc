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

#include "pipeline/jit/parse/function_block.h"

#include <algorithm>
#include <queue>

#include "pipeline/jit/parse/resolve.h"
#include "pipeline/jit/parse/parse.h"
#include "pipeline/jit/parse/data_converter.h"
#include "frontend/operator/ops.h"
#include "utils/info.h"
#include "utils/hash_set.h"
#include "pipeline/jit/debug/trace.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace py = pybind11;

namespace parse {
FunctionBlock::FunctionBlock(const Parser &parser)
    : func_graph_(std::make_shared<FuncGraph>()), parser_(parser), matured_(false) {}

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
  constexpr auto kRecursiveLevel = 2;
  // The fallback feature is enabled in default.
  // Not support change the flag during the process is alive.
  static const auto use_fallback = (parser_.support_fallback() != "0");
  // a[::][::] = b will be translated to c = a[::] c[::] = b and the c is a no named variable.
  if (var_name.empty()) {
    MS_LOG(DEBUG) << "The node is " << node->DebugString(kRecursiveLevel)
                  << "added in the isolated list.\nBlock: " << this << "/"
                  << (func_graph_ ? func_graph_->ToString() : "FG(Null)")
                  << ", Line: " << trace::GetDebugInfo(node->debug_info(), "", kSourceLineTipDiscard);
    AddIsolatedNode(node);
    return;
  }
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
      MS_LOG(INFO) << "Isolated node found(Hidden), hidden_node: " << hidden_node->DebugString(kRecursiveLevel)
                   << " is hidden by " << node->DebugString(kRecursiveLevel)
                   << " with the same name, var_name: " << var_name << ", block: " << this << "/"
                   << (func_graph_ ? func_graph_->ToString() : "FG(Null)")
                   << ", Line: " << trace::GetDebugInfo(hidden_node->debug_info(), "", kSourceLineTipDiscard);
      AddIsolatedNode(hidden_node);
    }
    iter->second = std::make_pair(node, false);
  }
  if (use_fallback) {
    UpdateLocalPyParam(var_name, node);
  }
}

AnfNodePtr FunctionBlock::ReadLocalVariable(const std::string &var_name) {
  auto found = assigned_vars_.find(var_name);
  if (found != assigned_vars_.end()) {
    auto &node = found->second.first;
    MS_EXCEPTION_IF_NULL(node);
    // Mark the variable as used.
    found->second.second = true;
    MS_LOG(DEBUG) << "Found var: " << var_name << ", as: " << node->DebugString();
    return node;
  }
  return nullptr;
}

std::pair<AnfNodePtr, bool> FunctionBlock::FindPredInterpretNode(const std::string &var_name) {
  // Search the predecessors of the current block for the local parameter. If one of the local parameter of the
  // predecessors is interpret node, the phi_param needs to set the interpret true.
  mindspore::HashSet<FunctionBlock *> visited_block;
  std::queue<FunctionBlock *> block_queue;
  block_queue.push(this);
  bool has_found = false;
  while (!block_queue.empty()) {
    const auto cur_block = block_queue.front();
    MS_EXCEPTION_IF_NULL(cur_block);
    block_queue.pop();
    (void)visited_block.insert(cur_block);
    auto pred_node = cur_block->ReadLocalVariable(var_name);
    if (pred_node != nullptr) {
      has_found = true;
      bool interpret_without_internal =
        IsPrimitiveCNode(pred_node, prim::kPrimPyInterpret) && !pred_node->interpret_internal_type();
      if (pred_node->interpret() || interpret_without_internal) {
        return std::make_pair(pred_node, has_found);
      }
    } else {
      for (const auto &cur_pred_block : cur_block->prev_blocks()) {
        if (visited_block.count(cur_pred_block) == 0) {
          block_queue.push(cur_pred_block);
        }
      }
    }
  }
  return std::make_pair(nullptr, has_found);
}

// Read variable from predecessors
AnfNodePtr FunctionBlock::ReadVariable(const std::string &var_name) {
  MS_LOG(DEBUG) << "Read begin, var: " << var_name << ", block: " << ToString();
  // The fallback feature is enabled in default.
  // Not support change the flag during the process is alive.
  static const auto use_fallback = (parser_.support_fallback() != "0");
  // Get var node if it is found
  auto node = ReadLocalVariable(var_name);
  if (node != nullptr) {
    if (use_fallback) {
      UpdateLocalPyParam(var_name, node);
    }
    return node;
  }

  // Get var from predecessor block, if can't get then make a resolve node to it
  if (matured_) {
    // If only one predecessor block, read the definition of var from it.
    if (prev_blocks_.size() == 1) {
      auto block = prev_blocks_[0];
      MS_EXCEPTION_IF_NULL(block);
      auto res = block->ReadVariable(var_name);
      if (use_fallback) {
        MS_LOG(DEBUG) << "Update global params of block: " << ToString()
                      << ", with previous block: " << block->ToString()
                      << ",\nCurrent: " << py::str(const_cast<py::dict &>(global_py_params()))
                      << "\nInsert: " << py::str(const_cast<py::dict &>(block->global_py_params()));
        UpdateGlobalPyParam(block->global_py_params());
        UpdateLocalPyParam(var_name, res);
      }
      return res;
    } else if (prev_blocks_.empty()) {
      // Get namespace and make Resolve
      auto it = var_to_resolve_.find(var_name);
      if (it != var_to_resolve_.end()) {
        return it->second;
      }
      MS_LOG(DEBUG) << "var: " << var_name;
      auto tmp_node = MakeResolveSymbol(var_name);
      var_to_resolve_[var_name] = tmp_node;
      return tmp_node;
    }
  }
  // If have more than one predecessor blocks then build a phi node.
  auto debug_info = std::make_shared<NodeDebugInfo>();
  debug_info->set_name(var_name);
  TraceGuard guard(std::make_shared<TracePhi>(debug_info));
  ParameterPtr phi_param = std::make_shared<Parameter>(func_graph());
  MS_LOG(DEBUG) << (func_graph_ ? func_graph_->ToString() : "FG(Null)") << " generate phi node "
                << phi_param->ToString() << " for " << var_name;

  if (use_fallback) {
    auto [pred_node, has_found] = FindPredInterpretNode(var_name);
    if (pred_node != nullptr) {
      phi_param->set_interpret(true);
    } else if (!has_found) {
      // If the current node is created as a phi node at the first time.(the var_name has not be found in pre blocks)
      // need resolve to determine whether it needs to be marked with interpret.
      auto resolve_node = MakeResolveSymbol(var_name);
      MS_EXCEPTION_IF_NULL(resolve_node);
      phi_param->set_interpret(resolve_node->interpret());
      phi_param->set_interpret_internal_type(resolve_node->interpret_internal_type());
    }
  }

  func_graph()->add_parameter(phi_param);
  phi_nodes_[phi_param] = var_name;
  WriteVariable(var_name, phi_param);
  if (matured_) {
    SetPhiArgument(phi_param);
  }
  // In SetPhiArgument/CollectRemovablePhi, this phi may be set as removable and set it as
  // real node, so check it again.
  MS_LOG(DEBUG) << "Read again, var: " << var_name << ", block: " << ToString();
  node = ReadLocalVariable(var_name);
  if (node != nullptr) {
    return node;
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
  // The fallback feature is enabled in default.
  // Not support change the flag during the process is alive.
  static const auto use_fallback = (parser_.support_fallback() != "0");
  if (use_fallback && !global_py_params().contains("self")) {
    py::object self_namespace = ast->CallParseModFunction(PYTHON_MOD_GET_ATTR_NAMESPACE_SYMBOL, ast->obj());
    AddGlobalPyParam("self", self_namespace);
  }

  py::object namespace_var = ast->CallParseModFunction(PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL, ast->obj());
  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, namespace_var);
  SymbolPtr symbol = std::make_shared<Symbol>(attr);
  MS_LOG(DEBUG) << "name_space: " << name_space->ToString() << ", symbol: " << symbol->ToString();
  return MakeResolve(name_space, symbol);
}

AnfNodePtr FunctionBlock::GetResolveNode(const py::tuple &info) {
  constexpr size_t namespace_index = 0;
  constexpr size_t symbol_index = 1;
  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_SYMBOL_STR, info[namespace_index]);
  SymbolPtr symbol = std::make_shared<Symbol>(info[symbol_index].cast<std::string>());
  return MakeResolve(name_space, symbol);
}

AnfNodePtr FunctionBlock::HandleNamespaceInfo(const py::tuple &info) {
  constexpr size_t namespace_index = 0;
  constexpr size_t symbol_index = 1;
  constexpr size_t namespace_info_size = 2;
  if (info.size() != namespace_info_size) {
    MS_EXCEPTION(NameError) << "namespace info size should be 2, but got " << info.size();
  }

  // If namespace is None, the symbol is an undefined name.
  if (info[namespace_index].is_none()) {
    MS_EXCEPTION(NameError) << info[symbol_index].cast<std::string>();
  }
  return GetResolveNode(info);
}

AnfNodePtr FunctionBlock::HandleBuiltinNamespaceInfo(const py::tuple &info) {
  constexpr size_t closure_info_size = 2;
  constexpr size_t namespace_info_size = 4;
  constexpr size_t namespace_index = 0;
  constexpr size_t symbol_index = 1;
  constexpr size_t value_index = 2;
  constexpr size_t flag_index = 3;
  if (info.size() != closure_info_size && info.size() != namespace_info_size) {
    MS_EXCEPTION(NameError) << "namespace info size should be 2 or 4, but got " << info.size();
  }

  // Handle closure namespace info.
  if (info.size() == closure_info_size) {
    // If namespace is None, the symbol is an undefined name.
    if (info[namespace_index].is_none()) {
      MS_EXCEPTION(NameError) << info[symbol_index].cast<std::string>();
    }
    return GetResolveNode(info);
  }

  // Handle global namespace info.
  auto resolved_node = GetResolveNode(info);
  auto syntax_support = info[flag_index].cast<int32_t>();
  if (syntax_support != SYNTAX_SUPPORTED && syntax_support != SYNTAX_HYBRID_TYPE) {
    resolved_node->set_interpret(true);
    if (syntax_support == SYNTAX_UNSUPPORTED_INTERNAL_TYPE) {
      resolved_node->set_interpret_internal_type(true);
    }
  }
  // The value may not be supported to do ConvertData such as api `mutable`,
  // and we get its converted object from python.
  auto ast = parser_.ast();
  MS_EXCEPTION_IF_NULL(ast);
  py::object py_obj = ast->CallParserObjMethod(PYTHON_PARSE_GET_CONVERT_OBJECT_FOR_UNSUPPORTED_TYPE, info[value_index]);

  auto symbol_name = info[symbol_index].cast<std::string>();
  AddGlobalPyParam(symbol_name, py_obj);
  MS_LOG(INFO) << "[" << func_graph()->ToString() << "] Added global python symbol: {" << symbol_name << " : "
               << py::str(py_obj) << "}";
  return resolved_node;
}

// Make a resolve node for symbol string
AnfNodePtr FunctionBlock::MakeResolveSymbol(const std::string &value) {
  MS_LOG(DEBUG) << "value: " << value;
  // The prefix of value is "self.".
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
  static const auto use_fallback = (parser_.support_fallback() != "0");
  if (!use_fallback) {
    py::tuple namespace_info = ast->CallParserObjMethod(PYTHON_PARSE_GET_NAMESPACE_SYMBOL, value);
    return HandleNamespaceInfo(namespace_info);
  } else {
    py::tuple namespace_info = ast->CallParserObjMethod(PYTHON_PARSE_GET_BUILTIN_NAMESPACE_SYMBOL, value);
    return HandleBuiltinNamespaceInfo(namespace_info);
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
                << " , " << (resolve_symbol ? (std::string)resolve_symbol->symbol() : "null resolve symbol.");
  ValueNodePtr module_node = NewValueNode(name_space);
  ValueNodePtr symbol_node = NewValueNode(resolve_symbol);
  auto node = func_graph_->NewCNodeInOrder({NewValueNode(prim::kPrimResolve), module_node, symbol_node});
  return node;
}

AnfNodePtr FunctionBlock::MakeInterpret(const std::string &script_text, const AnfNodePtr &global_dict_node,
                                        const AnfNodePtr &local_dict_node, const AnfNodePtr &orig_node) {
  MS_LOG(DEBUG) << "MakeInterpret for " << script_text;
  MS_EXCEPTION_IF_NULL(orig_node);
  ScriptPtr script = std::make_shared<Script>(script_text);
  auto script_node = NewValueNode(script);
  auto node = func_graph_->NewCNodeInOrder(
    {NewValueNode(prim::kPrimPyInterpret), script_node, global_dict_node, local_dict_node});
  MS_EXCEPTION_IF_NULL(node);
  node->set_interpret_internal_type(orig_node->interpret_internal_type());
  return node;
}

// Add input for the block's phi parameter
void FunctionBlock::SetPhiArgument(const ParameterPtr &phi) {
  MS_EXCEPTION_IF_NULL(phi);
  TraceGuard trace_guard(std::make_shared<TraceResolve>(phi->debug_info()));
  std::string var = phi_nodes_[phi];
  MS_LOG(DEBUG) << "graph " << (func_graph_ ? func_graph_->ToString() : "FG(Null)") << " set phi " << phi->ToString()
                << " for var `" << var << "`";
  CollectRemovablePhi(phi);
  for (auto &pred : prev_blocks_) {
    MS_EXCEPTION_IF_NULL(pred);
    MS_LOG(DEBUG) << "graph " << (func_graph_ ? func_graph_->ToString() : "FG(Null)") << " pred_blocks_ "
                  << (pred->func_graph_ ? pred->func_graph_->ToString() : "FG(Null)");
    AnfNodePtr arg_node = pred->ReadVariable(var);
    auto jump = pred->GetJumpNode(this);
    if (jump == nullptr) {
      // If prev block is a switch call's prev block, no jumps here.
      continue;
    }
    jump->add_input(arg_node);
  }
}

std::set<AnfNodePtr> FunctionBlock::SearchAllArgsOfPhiNode(const std::string &var, const ParameterPtr &phi) {
  std::set<AnfNodePtr> all_arg_nodes;
  MS_LOG(DEBUG) << "Search block:" << ToString() << "Prev_blocks size: " << prev_blocks_.size();
  for (auto &prev : prev_blocks_) {
    MS_EXCEPTION_IF_NULL(prev);
    AnfNodePtr temp_node = prev->ReadVariable(var);
    MS_LOG(DEBUG) << "Read from prev block:" << prev->ToString() << ", temp_node: " << temp_node->DebugString();
    (void)all_arg_nodes.insert(temp_node);
  }
  if (all_arg_nodes.size() == 1) {
    auto arg_node = *(all_arg_nodes.begin());
    MS_EXCEPTION_IF_NULL(arg_node);
    MS_LOG(DEBUG) << "graph " << (func_graph_ ? func_graph_->ToString() : "FG(Null)") << " phi "
                  << (phi ? phi->ToString() : "null") << " may be replaced by node " << arg_node->DebugString();
  }
  return all_arg_nodes;
}

// Check if there is removable unnecessary phi node in this graph.
// As per the FIRM TR 3.2, a phi node can be remove if:
// <Quote>
//    If all arguments of a φ-function are the same value s or the φfunction itself,
//    then we remove the φ-function and let all users directly uses. We call such a
//    φ-function obviously unnecessary.
//    When we removed a φ-function p, then we recursively try to apply this simplification
//    rule with all (former) users of p, because they may have become obviously unnecessary
//    due to the removal of p
// <Quote>
// phi node in graph will be removed after the whole function is parsed in a DFS visit
// of that graph.The reason is :
// 1. when this function is called, not all usage of this phi node had bound to the
// graph of this function block, some may stay in vars_ in other blocks.
// 2. it's costly to iterate the graph to replace the phi for each phi.
// Args: phi: This parameter node is functioning as a phi node.
void FunctionBlock::CollectRemovablePhi(const ParameterPtr &phi) {
  MS_EXCEPTION_IF_NULL(phi);
  const auto &var_name = phi_nodes_[phi];
  MS_LOG(DEBUG) << "check phi " << phi->DebugString() << " for " << var_name;
  if (prev_blocks_.empty()) {
    MS_LOG(DEBUG) << "no phi " << phi->DebugString() << " for var " << var_name;
    return;
  }
  auto arg_nodes = SearchAllArgsOfPhiNode(var_name, phi);
  phi_args_[phi] = arg_nodes;
  if (arg_nodes.size() == 1) {
    auto arg_node = *arg_nodes.begin();
    arg_node->set_debug_info(phi->debug_info());
    MS_LOG(DEBUG) << "graph " << (func_graph_ ? func_graph_->ToString() : "FG(Null)") << " phi " << phi->ToString()
                  << " can be replaced with " << arg_node->DebugString();
    // Replace var with new one. This equal to statement in TR "v0 is immediately replaced by v1."
    WriteVariable(var_name, arg_node);
    static const auto use_fallback = (parser_.support_fallback() != "0");
    if (use_fallback) {
      bool interpret_without_internal =
        IsPrimitiveCNode(arg_node, prim::kPrimPyInterpret) && !arg_node->interpret_internal_type();
      if (arg_node->interpret() || interpret_without_internal) {
        phi->set_interpret(true);
        if (arg_node->interpret_internal_type()) {
          phi->set_interpret_internal_type(true);
        }
      }
    }
  }
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

  CNodePtr jump = func_graph_->NewCNodeInOrder(std::move(input_nodes));
  jumps_[target_block.get()] = jump;
  target_block->AddPrevBlock(shared_from_this());
  func_graph_->set_output(jump);
}

// Perform a conditional jump using switch operation.
// The first CNode select graph with condition, and than execute this graph
CNodePtr FunctionBlock::ConditionalJump(const AnfNodePtr &cond_node, const AnfNodePtr &true_block_call,
                                        const AnfNodePtr &false_block_call) {
  MS_EXCEPTION_IF_NULL(true_block_call);
  MS_EXCEPTION_IF_NULL(false_block_call);
  if (func_graph_->get_return() != nullptr) {
    MS_LOG(EXCEPTION) << "Failure: have return node! NodeInfo: "
                      << trace::GetDebugInfo(func_graph_->get_return()->debug_info());
  }
  CNodePtr switch_app =
    func_graph_->NewCNodeInOrder({NewValueNode(prim::kPrimSwitch), cond_node, true_block_call, false_block_call});
  CNodePtr switch_app_new = func_graph_->NewCNodeInOrder({switch_app});
  func_graph_->set_output(switch_app_new);
  return switch_app_new;
}

CNodePtr FunctionBlock::ConditionalJump(const AnfNodePtr &cond_node, const FunctionBlockPtr &true_block,
                                        const FunctionBlockPtr &false_block) {
  MS_EXCEPTION_IF_NULL(true_block);
  MS_EXCEPTION_IF_NULL(false_block);
  return ConditionalJump(cond_node, NewValueNode(true_block->func_graph()), NewValueNode(false_block->func_graph()));
}

// Create cnode for the assign statement like 'self.target = source'.
// convert it to 'P.Assign(self.target, source)' and then add the cnode as isolate node.
void FunctionBlock::SetStateAssign(const AnfNodePtr &target, const AnfNodePtr &source) {
  const std::string primitive_name("assign");
  const std::string module_name("mindspore.ops.functional");
  ValueNodePtr assign_op = NewValueNode(prim::GetPythonOps(primitive_name, module_name, true));
  auto assign_node = func_graph_->NewCNodeInOrder({assign_op, target, source});
  const int recursive_level = 2;
  MS_LOG(DEBUG) << "Isolated node found(Assign), assign_node: " << assign_node->DebugString(recursive_level)
                << ", block: " << this << "/" << func_graph_->ToString()
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
      const int recursive_level = 2;
      MS_LOG(INFO) << "Isolated node found(NoUse), node: " << node->DebugString(recursive_level)
                   << ", var_name: " << var_name << ", block: " << this << "/"
                   << (func_graph() ? func_graph()->ToString() : "FG(Null)")
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
  for (const auto &node : isolated_nodes_) {
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
  constexpr size_t no_state_size = 1;
  constexpr size_t only_one_state_size = 2;
  if (states.size() == no_state_size) {
    // Only MakeTuple, no state left.
    return;
  } else if (states.size() == only_one_state_size) {
    // If there are only MakeTuple and another node in states(the states size is 2),
    // do not need to MakeTuple, just use the node.
    state = states[1];
  } else {
    state = func_graph_->NewCNode(std::move(states));
    if (state != nullptr && state->debug_info() != nullptr) {
      state->debug_info()->set_location(nullptr);
    }
  }

  AnfNodePtr old_output = nullptr;
  auto return_node = func_graph_->get_return();
  if (return_node != nullptr) {
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
  if (stop_grad_node->debug_info()) {
    stop_grad_node->debug_info()->set_location(nullptr);
  }
  if (depend_node->debug_info()) {
    depend_node->debug_info()->set_location(nullptr);
  }
  // We add this attribute for @constexpr use scene, since we must infer them before other nodes.
  // That means isolated nodes will be evaluated first. It's not complete, but works in most scenes.
  depend_node->AddAttr(kAttrTopoSortRhsFirst, MakeValue(true));
  MS_EXCEPTION_IF_NULL(state);
  MS_LOG(INFO) << "Attached for side-effect nodes, depend_node: " << depend_node->DebugString()
               << ", state: " << state->DebugString(recursive_level);
  func_graph_->set_output(depend_node, true);
  // Update new return node's debug_info with old one.
  if (return_node != nullptr && return_node->debug_info()) {
    auto new_return = func_graph_->get_return();
    MS_EXCEPTION_IF_NULL(new_return);
    new_return->set_debug_info(return_node->debug_info());
  }
}

void FunctionBlock::SetAsDeadBlock() { is_dead_block_ = true; }

CNodePtr FunctionBlock::GetJumpNode(FunctionBlock *target_block) {
  auto it = jumps_.find(target_block);
  if (it == jumps_.end()) {
    MS_LOG(DEBUG) << "Can't find jump node from block:" << ToString() << " to block:" << target_block->ToString();
    return nullptr;
  }
  return it->second;
}

void FunctionBlock::SetReturnStatementInside() { is_return_statement_inside_ = true; }
void FunctionBlock::SetBreakContinueStatementInside() { is_break_continue_statement_inside_ = true; }
}  // namespace parse
}  // namespace mindspore
