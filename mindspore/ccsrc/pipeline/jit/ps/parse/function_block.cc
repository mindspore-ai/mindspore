/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/ps/parse/function_block.h"

#include <algorithm>
#include <queue>

#include "frontend/operator/ops.h"
#include "include/common/utils/python_adapter.h"
#include "include/common/utils/utils.h"
#include "ir/cell.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/structure_ops.h"
#include "pipeline/jit/ps/debug/trace.h"
#include "pipeline/jit/ps/fallback.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/jit/ps/parse/parse.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "utils/hash_set.h"
#include "utils/info.h"

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
  // a[::][::] = b will be translated to c = a[::] c[::] = b and the c is a no named variable.
  if (var_name.empty()) {
    MS_LOG(DEBUG) << "The node is " << node->DebugString(kRecursiveLevel)
                  << "added in the isolated list.\nBlock: " << this << "/"
                  << (func_graph_ ? func_graph_->ToString() : "FG(Null)")
                  << ", Line: " << trace::GetDebugInfoStr(node->debug_info(), "", kSourceLineTipDiscard);
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
                   << ", Line: " << trace::GetDebugInfoStr(hidden_node->debug_info(), "", kSourceLineTipDiscard);
      AddIsolatedNode(hidden_node);
    }
    MS_LOG(INFO) << (func_graph_ ? func_graph_->ToString() : "FG(Null)") << " update var `" << var_name
                 << "` with node " << node->DebugString();
    iter->second = std::make_pair(node, false);
  }
  if (!HasGlobalPyParam(var_name)) {
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

bool FunctionBlock::CheckHasVariable(const std::string &var_name) {
  auto node = ReadLocalVariable(var_name);
  if (node != nullptr) {
    return true;
  }
  if (!prev_blocks_.empty()) {
    auto block = prev_blocks_[0];
    MS_EXCEPTION_IF_NULL(block);
    return block->CheckHasVariable(var_name);
  }
  return false;
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
  MS_LOG(DEBUG) << "Read begin, var_name: " << var_name << ", block: " << ToString();
  // Get var node if it is found
  auto node = ReadLocalVariable(var_name);
  if (node != nullptr) {
    if (!HasGlobalPyParam(var_name)) {
      UpdateLocalPyParam(var_name, node);
    }
    return node;
  }

  MS_LOG(DEBUG) << "matured_: " << matured_ << ", prev_blocks_.size: " << prev_blocks_.size();
  // Get var from predecessor block, if can't get then make a resolve node to it
  if (matured_) {
    // If only one predecessor block, read the definition of var from it.
    if (prev_blocks_.size() == 1) {
      auto block = prev_blocks_[0];
      MS_EXCEPTION_IF_NULL(block);
      auto res = block->ReadVariable(var_name);
      MS_LOG(DEBUG) << "Update global params of block: " << ToString() << ", with previous block: " << block->ToString()
                    << ",\nCurrent: " << py::str(const_cast<py::dict &>(global_py_params()))
                    << "\nInsert: " << py::str(const_cast<py::dict &>(block->global_py_params()));
      UpdateGlobalPyParam(block->global_py_params());
      if (!HasGlobalPyParam(var_name)) {
        UpdateLocalPyParam(var_name, res);
      }
      return res;
    } else if (prev_blocks_.empty()) {
      // Get namespace and make Resolve
      auto it = var_to_resolve_.find(var_name);
      if (it != var_to_resolve_.end()) {
        return it->second;
      }
      MS_LOG(DEBUG) << "var_name: " << var_name;
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

  auto [pred_node, has_found] = FindPredInterpretNode(var_name);
  if (pred_node != nullptr) {
    phi_param->set_interpret(true);
  } else if (!has_found) {
    // If the current node is created as a phi node at the first time.(the var_name has not be found in pre blocks)
    // need resolve to determine whether it needs to be marked with interpret.
    auto resolve_node = MakeResolveSymbol(var_name);
    MS_EXCEPTION_IF_NULL(resolve_node);
    CheckUndefinedSymbol(var_name, resolve_node);
    phi_param->set_interpret(resolve_node->interpret());
    phi_param->set_interpret_internal_type(resolve_node->interpret_internal_type());
    if (resolve_node->isa<Parameter>()) {
      phi_param->set_debug_info(resolve_node->debug_info());
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
  MS_LOG(DEBUG) << "Read again, var_name: " << var_name << ", block: " << ToString();
  node = ReadLocalVariable(var_name);
  if (node != nullptr) {
    return node;
  }
  return phi_param;
}

// Resolve Ast operator node
py::tuple FunctionBlock::GetAstOpNameSpace(const py::object &op) {
  auto ast = parser_.ast();
  MS_EXCEPTION_IF_NULL(ast);
  TraceGuard trace_guard(parser_.GetLocation(op));
  py::tuple namespace_var = ast->CallParseModFunction(PYTHON_PARSE_GET_AST_NAMESPACE_SYMBOL, op);
  constexpr size_t namespace_size = 3;
  if (namespace_var.size() != namespace_size) {
    MS_LOG(INTERNAL_EXCEPTION) << "Resolve ast op failed, get namespace tuple size=" << namespace_var.size();
  }
  return namespace_var;
}

// Resolve Ast operator node
AnfNodePtr FunctionBlock::MakeResolveAstOpNameSpace(const py::tuple &namespace_var) {
  constexpr size_t namespace_index = 0;
  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_AST, namespace_var[namespace_index]);
  constexpr size_t symbol_index = 1;
  SymbolPtr symbol = std::make_shared<Symbol>(namespace_var[symbol_index].cast<std::string>());
  MS_LOG(DEBUG) << "name_space: " << name_space->ToString() << ", symbol: " << symbol->ToString();
  return MakeResolve(name_space, symbol);
}

// Resolve class object self.
AnfNodePtr FunctionBlock::MakeResolveClassObject() {
  auto ast = parser_.ast();
  MS_EXCEPTION_IF_NULL(ast);
  py::object namespace_var = ast->CallParseModFunction(PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL, ast->obj());
  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_CLASS_OBJECT, namespace_var);
  constexpr auto self_name = "self";
  SymbolPtr symbol = std::make_shared<Symbol>(self_name);  // Must be 'self'.
  MS_LOG(DEBUG) << "name_space: " << name_space->ToString() << ", symbol: " << symbol->ToString();
  return MakeResolve(name_space, symbol);
}

// Resolve class member: method, member variable.
AnfNodePtr FunctionBlock::MakeResolveClassMember(const std::string &attr_or_self) {
  auto ast = parser_.ast();
  MS_EXCEPTION_IF_NULL(ast);
  py::object namespace_var = ast->CallParseModFunction(PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL, ast->obj());
  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, namespace_var);
  SymbolPtr symbol = std::make_shared<Symbol>(attr_or_self);
  MS_LOG(DEBUG) << "name_space: " << name_space->ToString() << ", symbol: " << symbol->ToString();
  return MakeResolve(name_space, symbol);
}

void FunctionBlock::CheckUndefinedSymbol(const std::string &var, const AnfNodePtr &node) const {
  if (node->isa<ValueNode>()) {
    auto value = GetValuePtr<ValueProblem>(node->cast<ValueNodePtr>());
    if (!is_dead_block() && value != nullptr && value->IsUndefined()) {
      MS_EXCEPTION(NameError) << "The name '" << var << "' is not defined, or not supported in graph mode.";
    }
  }
}

AnfNodePtr FunctionBlock::HandleNamespaceSymbol(const std::string &var_name) {
  auto ast = parser_.ast();
  MS_EXCEPTION_IF_NULL(ast);
  const py::tuple &info = ast->CallParserObjMethod(PYTHON_PARSE_GET_NAMESPACE_SYMBOL, var_name);

  constexpr size_t closure_info_size = 3;
  constexpr size_t global_info_size = 4;
  constexpr size_t namespace_index = 0;
  constexpr size_t symbol_index = 1;
  constexpr size_t value_index = 2;
  constexpr size_t flag_index = 3;
  if (info.size() != closure_info_size && info.size() != global_info_size) {
    MS_INTERNAL_EXCEPTION(NameError) << "The namespace info size should be 3 or 4, but got " << info.size();
  }
  // If namespace is None, the symbol is an undefined name.
  if (info[namespace_index].is_none()) {
    const auto undefined_symbol = std::make_shared<ValueProblem>(ValueProblemType::kUndefined);
    MS_LOG(WARNING) << "Undefined symbol: " << var_name << ", during parsing " << py::str(ast->function()) << " of "
                    << py::str(ast->obj());
    return NewValueNode(undefined_symbol);
  }

  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_SYMBOL_STR, info[namespace_index]);
  SymbolPtr symbol = std::make_shared<Symbol>(info[symbol_index].cast<std::string>());
  auto resolved_node = MakeResolve(name_space, symbol);

  // Handle closure namespace info.
  if (info.size() == closure_info_size) {
    return resolved_node;
  }

  // Handle global namespace info.
  auto syntax_support = info[flag_index].cast<int32_t>();
  if (syntax_support != SYNTAX_SUPPORTED && syntax_support != SYNTAX_HYBRID_TYPE) {
    resolved_node->set_interpret(true);
    if (syntax_support == SYNTAX_UNSUPPORTED_INTERNAL_TYPE) {
      resolved_node->set_interpret_internal_type(true);
    }
  }

  auto symbol_name = info[symbol_index].cast<std::string>();
  py::object py_obj = info[value_index];
  AddGlobalPyParam(symbol_name, py_obj);
  MS_LOG(INFO) << "[" << func_graph()->ToString() << "] Added global python symbol: {" << symbol_name << " : "
               << py::str(py_obj) << "}";
  fallback::SetPyObjectToNode(resolved_node, py_obj);
  return resolved_node;
}

// Make a resolve node for symbol string
AnfNodePtr FunctionBlock::MakeResolveSymbol(const std::string &var_name) {
  MS_LOG(DEBUG) << "var_name: " << var_name << ", ast object type: " << parser_.ast()->target_type();

  // Handle self. The prefix of var_name is "self".
  constexpr auto self_name = "self";
  const auto self_name_len = strlen(self_name);
  // For PARSE_TARGET_METHOD or PARSE_TARGET_OBJECT_INSTANCE, should deal with self here, exclude PARSE_TARGET_FUNCTION.
  if ((parser_.ast()->target_type() == PARSE_TARGET_METHOD ||
       parser_.ast()->target_type() == PARSE_TARGET_OBJECT_INSTANCE) &&
      var_name.compare(0, self_name_len, self_name) == 0) {
    auto start = var_name.find_first_of('.');
    if (start != std::string::npos) {  // 'self.xxx'
      ++start;
      if (start >= var_name.size()) {
        MS_LOG(ERROR) << "Find invalid resolve symbol str: " << var_name;
        return nullptr;
      }
      auto bits_str = var_name.substr(start);
      auto resolve_node = MakeResolveClassMember(bits_str);
      if (!HasGlobalPyParam(var_name)) {
        UpdateLocalPyParam(var_name, resolve_node);
      }
      return resolve_node;
    } else if (var_name.size() == self_name_len) {  // 'self'
      auto resolve_node = MakeResolveClassObject();
      if (!HasGlobalPyParam(var_name)) {
        UpdateLocalPyParam(var_name, resolve_node);
      }
      return resolve_node;
    }
  }

  // Handle non-self.
  return HandleNamespaceSymbol(var_name);
}

AnfNodePtr FunctionBlock::MakeResolveOperation(const std::string &value) {
  auto ast = parser_.ast();
  MS_EXCEPTION_IF_NULL(ast);
  py::tuple namespace_var = ast->CallParseModFunction(PYTHON_PARSE_GET_OPERATION_NAMESPACE_SYMBOL, value);
  const size_t namespace_var_size = 2;
  if (namespace_var.size() < namespace_var_size) {
    MS_INTERNAL_EXCEPTION(NameError) << "namespace_var is less than 2";
  }
  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_COMMON_OPS, namespace_var[0]);
  SymbolPtr symbol = std::make_shared<Symbol>(namespace_var[1].cast<std::string>());
  MS_LOG(DEBUG) << "name_space: " << name_space->ToString() << ", symbol: " << symbol->ToString();
  return MakeResolve(name_space, symbol);
}

namespace {
// The same as TransformVectorFuncValueNode() in mindspore/ccsrc/pipeline/jit/parse/resolve.cc, but not add to manager.
bool TransformVectorFuncValueNode(const FuncGraphPtr &func_graph, const ValuePtr &value,
                                  AnfNodePtr *const transformed) {
  MS_EXCEPTION_IF_NULL(value);
  const auto &value_vec = GetValue<ValuePtrList>(value);
  if (value_vec.empty()) {
    return false;
  }
  std::vector<AnfNodePtr> nodes;
  (void)nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  bool is_all_func = true;
  for (auto &elem : value_vec) {
    MS_EXCEPTION_IF_NULL(elem);
    AnfNodePtr node = nullptr;
    if (elem->isa<ValueTuple>() || elem->isa<ValueList>()) {
      is_all_func = is_all_func && TransformVectorFuncValueNode(func_graph, elem, &node);
    } else if (elem->isa<FuncGraph>()) {
      FuncGraphPtr new_fg = elem->cast<FuncGraphPtr>();
      node = NewValueNode(new_fg);
    } else if (elem->isa<Primitive>()) {
      node = NewValueNode(elem);
    } else {
      is_all_func = false;
    }
    (void)nodes.emplace_back(node);
  }
  if (is_all_func) {
    // (1) The celllist or ordered_cell will be parsed as valuetuple of const graph in it,
    // So if has graph in list, try to replace the node with make tuple of graph value node.
    // We do this because the graph manager won't investigate the graph inside valuetuple,
    // change the vector of graph to be make_tuple of graph value node.
    // (2) the primitive valuetuple or valuelist may encounter to abstract error, make it all
    // independent nodes.
    *transformed = func_graph->NewCNode(std::move(nodes));
  }
  return is_all_func;
}
}  // namespace

AnfNodePtr FunctionBlock::MakeResolve(const NameSpacePtr &name_space, const SymbolPtr &resolve_symbol) {
  MS_LOG(DEBUG) << "MakeResolve for "
                << (name_space ? (std::string)py::str(name_space->namespace_obj()) : "null namespace") << " , "
                << (resolve_symbol ? (std::string)resolve_symbol->symbol() : "null resolve symbol.");
  ValueNodePtr module_node = NewValueNode(name_space);
  ValueNodePtr symbol_node = NewValueNode(resolve_symbol);
  auto node = func_graph_->NewCNodeInOrder({NewValueNode(prim::kPrimResolve), module_node, symbol_node});

  // Directly resolve the symbol.
  return DoResolve(node, name_space, resolve_symbol);
}

AnfNodePtr FunctionBlock::DoResolve(const AnfNodePtr &node, const std::shared_ptr<NameSpace> &name_space,
                                    const std::shared_ptr<Symbol> &resolve_symbol) {
  static const auto boost_parse = common::GetEnv("MS_DEV_GREED_PARSE");
  if (Parser::defer_resolve() || boost_parse != "1") {
    return node;
  }
  // Directly resolve the symbol.
  const auto &obj = GetSymbolObject(name_space, resolve_symbol, node);
  // Avoid recursively resolving Cell.
  if (py::isinstance<Cell>(obj) && resolve_symbol->symbol() == "self") {
    MS_LOG(ERROR) << "Not direct resolve Cell self. node: " << node->DebugString() << ", ns: " << name_space->ToString()
                  << ", sym: " << resolve_symbol->ToString();
    return node;
  }
  AnfNodePtr resolved_node = nullptr;
  bool success = ResolveObjectToNode(node, obj, &resolved_node);
  if (!success || resolved_node == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Parse Resolve covert failed." << node->DebugString()
                               << ", ns: " << name_space->ToString() << ", sym: " << resolve_symbol->ToString();
  }
  // If the constant node is constant of vector of graph, add graph to manager.
  if (IsValueNode<ValueTuple>(resolved_node) || IsValueNode<ValueList>(resolved_node)) {
    auto value = resolved_node->cast<ValueNodePtr>()->value();
    if (!TransformVectorFuncValueNode(func_graph_, value, &resolved_node)) {
      MS_LOG(INFO) << "Fail to convert value tuple/list to CNode, " << resolved_node->DebugString();
    }
  }
  MS_LOG(DEBUG) << "node: " << node->DebugString() << ", ns: " << name_space->ToString()
                << ", sym: " << resolve_symbol->ToString() << ", resolved_node: " << resolved_node->DebugString();
  return resolved_node;
}

AnfNodePtr FunctionBlock::MakeInterpret(const std::string &script_text, const AnfNodePtr &global_dict_node,
                                        const AnfNodePtr &local_dict_node, const AnfNodePtr &orig_node) {
  MS_LOG(DEBUG) << "MakeInterpret for " << script_text;
  MS_EXCEPTION_IF_NULL(orig_node);
  auto script = std::make_shared<parse::Script>(script_text);
  auto script_node = NewValueNode(script);
  auto node = func_graph_->NewCNodeInOrder(
    {NewValueNode(prim::kPrimPyInterpret), script_node, global_dict_node, local_dict_node});
  MS_EXCEPTION_IF_NULL(node);
  node->set_debug_info(orig_node->debug_info());
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

namespace {
std::string GetVariableDefinedLocation(const FunctionBlock *block, const std::string &var, int start_line) {
  MS_EXCEPTION_IF_NULL(block);
  HashSet<FunctionBlock *> visited;
  std::vector<FunctionBlock *> todo_list = {};
  (void)std::copy(block->prev_blocks().cbegin(), block->prev_blocks().cend(), std::back_inserter(todo_list));
  while (!todo_list.empty()) {
    auto cur_block = todo_list.back();
    todo_list.pop_back();
    if (visited.find(cur_block) != visited.cend()) {
      continue;
    }
    (void)visited.insert(cur_block);
    (void)std::copy(cur_block->prev_blocks().cbegin(), cur_block->prev_blocks().cend(), std::back_inserter(todo_list));
    auto node = cur_block->ReadLocalVariable(var);
    if (node != nullptr && !node->isa<Parameter>()) {
      const auto &debug_info = trace::GetSourceCodeDebugInfo(node->debug_info());
      const auto &location = debug_info->location();
      return location->ToString(kSourceSectionTipNextLineHere, start_line);
    }
  }
  return "";
}
}  // namespace

void FunctionBlock::CheckVariableNotDefined(const std::pair<std::string, AnfNodePtr> &not_defined_branch,
                                            const std::string &var) {
  std::ostringstream oss;
  std::string not_defined_branch_name = not_defined_branch.first;
  const auto &debug_info = trace::GetSourceCodeDebugInfo(this->func_graph()->debug_info());
  const auto &location = debug_info->location();
  int start_line = location->line();
  if ((not_defined_branch_name == "while") || (not_defined_branch_name == "for")) {
    oss << "The local variable '" << var << "' defined in the '" << not_defined_branch_name
        << "' loop body cannot be used outside of the loop body. "
        << "Please define variable '" << var << "' before '" << not_defined_branch_name << "'.\n";
  }
  if ((not_defined_branch_name == "true branch") || (not_defined_branch_name == "false branch")) {
    oss << "The local variable '" << var << "' is not defined in " << not_defined_branch_name << ", but defined in "
        << (not_defined_branch_name == "true branch" ? "false branch" : "true branch") << ".\n";
  }
  oss << GetVariableDefinedLocation(this, var, start_line);
  MS_EXCEPTION(UnboundLocalError) << oss.str();
}

std::set<AnfNodePtr> FunctionBlock::SearchAllArgsOfPhiNode(const std::string &var, const ParameterPtr &phi) {
  std::vector<std::pair<std::string, AnfNodePtr>> defined_branch;
  std::pair<std::string, AnfNodePtr> not_defined_branch;
  MS_LOG(DEBUG) << "Search block:" << ToString() << "Prev_blocks size: " << prev_blocks_.size();
  for (auto &prev : prev_blocks_) {
    MS_EXCEPTION_IF_NULL(prev);
    AnfNodePtr temp_node = prev->ReadVariable(var);
    MS_EXCEPTION_IF_NULL(temp_node);
    MS_LOG(DEBUG) << "Read from prev block:" << prev->ToString() << "Found var: " << var
                  << ", as: " << temp_node->DebugString();
    bool undefined_symbol_flag = false;
    if (temp_node->isa<ValueNode>()) {
      auto value = GetValuePtr<ValueProblem>(temp_node->cast<ValueNodePtr>());
      if ((value != nullptr) && (value->IsUndefined())) {
        undefined_symbol_flag = true;
      }
    }
    if (undefined_symbol_flag) {
      not_defined_branch = std::make_pair(prev->block_name(), temp_node);
    } else {
      defined_branch.push_back(std::make_pair(prev->block_name(), temp_node));
    }
  }
  if (defined_branch.size() == 1) {
    auto arg_node = defined_branch.front().second;
    MS_EXCEPTION_IF_NULL(arg_node);
    MS_LOG(DEBUG) << "graph " << (func_graph_ ? func_graph_->ToString() : "FG(Null)") << " phi "
                  << (phi ? phi->ToString() : "null") << " may be replaced by node " << arg_node->DebugString();
  }

  if (not_defined_branch.second != nullptr) {
    if (!defined_branch.empty()) {
      TraceGuard trace_guard(phi->debug_info()->location());
      CheckVariableNotDefined(not_defined_branch, var);
    }
    MS_EXCEPTION(NameError) << "The name '" << var << "' is not defined, or not supported in graph mode.";
  }

  std::set<AnfNodePtr> all_arg_nodes;
  for (auto &item : defined_branch) {
    (void)all_arg_nodes.insert(item.second);
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
    if (arg_node->debug_info() == nullptr) {
      arg_node->set_debug_info(phi->debug_info());
    }
    MS_LOG(DEBUG) << "graph " << (func_graph_ ? func_graph_->ToString() : "FG(Null)") << " phi " << phi->ToString()
                  << " can be replaced with " << arg_node->DebugString();
    // Replace var with new one. This equal to statement in TR "v0 is immediately replaced by v1."
    WriteVariable(var_name, arg_node);
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

// Get the truth value testing for cond node.
CNodePtr FunctionBlock::ForceToCondNode(const AnfNodePtr &cond, bool is_while_cond) {
  MS_EXCEPTION_IF_NULL(cond);
  CNodePtr op_apply_node =
    func_graph_->NewCNodeInOrder({NewValueNode(prim::kPrimCond), cond, NewValueNode(MakeValue(is_while_cond))});
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
    MS_LOG(INTERNAL_EXCEPTION) << "Failure: have return node! NodeInfo: "
                               << trace::GetDebugInfoStr(func_graph_->get_return()->debug_info());
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
    MS_LOG(INTERNAL_EXCEPTION) << "Failure: have return node! fg: " << func_graph_->ToString()
                               << "\nNodeInfo: " << trace::GetDebugInfoStr(func_graph_->get_return()->debug_info())
                               << "\ncond_node: " << cond_node->DebugString()
                               << "\nNodeInfo: " << trace::GetDebugInfoStr(cond_node->debug_info());
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
                << ", Line: " << trace::GetDebugInfoStr(assign_node->debug_info(), "", kSourceLineTipDiscard);
  AddIsolatedNode(assign_node);
}

void FunctionBlock::ConvertUnusedNodesToIsolated(const std::pair<std::string, std::pair<AnfNodePtr, bool>> var) {
  auto &node = var.second.first;
  bool is_used = var.second.second;
  if (node == nullptr || is_used) {
    return;
  }
  auto &var_name = var.first;
  if (CanBeIsolatedNode(var_name, node)) {
    const int recursive_level = 2;
    MS_LOG(INFO) << "Isolated node found(NoUse), node: " << node->DebugString(recursive_level)
                 << ", var_name: " << var_name << ", block: " << this << "/"
                 << (func_graph() ? func_graph()->ToString() : "FG(Null)")
                 << ", Line: " << trace::GetDebugInfoStr(node->debug_info(), "", kSourceLineTipDiscard);
    AddIsolatedNode(node);
  }
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
  // Add isolated nodes which is unused var but not found in used set.
  for (const auto &var : assigned_vars_) {
    ConvertUnusedNodesToIsolated(var);
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
      MS_LOG(INTERNAL_EXCEPTION) << "Length of inputs of output node is less than 2";
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
    depend_node->debug_info()->set_location(old_output->debug_info()->location());
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

void FunctionBlock::set_is_return_statement_inside() { is_return_statement_inside_ = true; }
void FunctionBlock::set_break_continue_statement_inside() { is_break_continue_statement_inside_ = true; }
}  // namespace parse
}  // namespace mindspore
