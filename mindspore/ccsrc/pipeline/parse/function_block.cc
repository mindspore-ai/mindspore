/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "pipeline/parse/function_block.h"
#include <string>
#include <memory>
#include <fstream>
#include "pipeline/parse/resolve.h"
#include "pipeline/parse/parse.h"
#include "operator/ops.h"
#include "debug/info.h"
#include "debug/trace.h"

namespace mindspore {
namespace parse {
FunctionBlock::FunctionBlock(const Parser& parser) : parser_(parser) {
  func_graph_ = std::make_shared<FuncGraph>();
  matured_ = false;
}

void FunctionBlock::AddPrevBlock(const FunctionBlockPtr& block) { prev_blocks_.push_back(block.get()); }

// write variable records the variable name to corresponding node
void FunctionBlock::WriteVariable(const std::string& var_name, const AnfNodePtr& node) {
  MS_LOG(DEBUG) << "" << func_graph_->ToString() << " write var " << var_name << " with node " << node->DebugString();
  vars_[var_name] = node;
}

// read variable from predecessors
AnfNodePtr FunctionBlock::ReadVariable(const std::string& var) {
  // get var node if it is found
  if (vars_.count(var)) {
    AnfNodePtr node = vars_[var];
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<ValueNode>()) {
      return NewValueNode(GetValueNode(node));
    } else {
      return node;
    }
  }
  // get var from predecessor block ,if can't get the make a resolve node to it
  if (matured_) {
    // If only one predecessor block, read the definition of var from it.
    if (prev_blocks_.size() == 1) {
      auto block = prev_blocks_[0];
      MS_EXCEPTION_IF_NULL(block);
      return block->ReadVariable(var);
    } else if (prev_blocks_.empty()) {
      // get namespace and make Reslove
      return MakeResolveSymbol(var);
    }
  }
  // If have more than one predecessor blocks then build a phi node.
  auto debug_info = std::make_shared<NodeDebugInfo>();
  debug_info->set_name(var);
  TraceManager::DebugTrace(std::make_shared<TracePhi>(debug_info));
  ParameterPtr phi_param = std::make_shared<Parameter>(func_graph());
  TraceManager::EndTrace();
  MS_LOG(DEBUG) << "" << func_graph_->ToString() << " generate phi node " << phi_param->ToString() << " for " << var;
  func_graph()->add_parameter(phi_param);
  phi_nodes_[phi_param] = var;
  WriteVariable(var, phi_param);
  if (matured_) {
    SetPhiArgument(phi_param);
  }
  return phi_param;
}

// Resolve Ast operator node
AnfNodePtr FunctionBlock::MakeResolveAstOp(const py::object& op) {
  auto ast = parser_.ast();
  MS_EXCEPTION_IF_NULL(ast);
  TraceGuard trace_guard(parser_.GetLocation(op));
  py::tuple namespace_var = ast->CallParserObjMethod(PYTHON_PARSE_GET_AST_NAMESPACE_SYMBOL, op);
  if (namespace_var.size() != 2) {
    MS_LOG(EXCEPTION) << "Resolve ast op failed, get namespace tuple size=" << namespace_var.size();
  }
  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_AST, namespace_var[0]);
  SymbolPtr symbol = std::make_shared<Symbol>(namespace_var[1].cast<std::string>());
  return MakeResolve(name_space, symbol);
}

// Resolve class member, two possible: method, member variable
AnfNodePtr FunctionBlock::MakeResolveClassMember(std::string attr) {
  py::object namespace_var =
    parser_.ast()->CallParseModFunction(PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL, parser_.ast()->obj());
  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, namespace_var);
  SymbolPtr symbol = std::make_shared<Symbol>(attr);
  return MakeResolve(name_space, symbol);
}

// Make a resolve node for symbol string
AnfNodePtr FunctionBlock::MakeResolveSymbol(const std::string& value) {
  if (value.compare(0, strlen("self."), "self.") == 0) {
    auto start = value.find_first_of('.') + 1;
    if (start >= value.size()) {
      MS_LOG(ERROR) << "Find invalid resolve symbol str: " << value;
      return nullptr;
    }
    auto bits_str = value.substr(start);
    return MakeResolveClassMember(bits_str);
  }
  py::tuple namespace_var = parser_.ast()->CallParserObjMethod(PYTHON_PARSE_GET_NAMESPACE_SYMBOL, value);

  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_SYMBOL_STR, namespace_var[0]);
  SymbolPtr symbol = std::make_shared<Symbol>(namespace_var[1].cast<std::string>());
  return MakeResolve(name_space, symbol);
}

AnfNodePtr FunctionBlock::MakeResolveOperation(const std::string& value) {
  py::tuple namespace_var = parser_.ast()->CallParserObjMethod(PYTHON_PARSE_GET_OPERATION_NAMESPACE_SYMBOL, value);
  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_COMMON_OPS, namespace_var[0]);
  SymbolPtr symbol = std::make_shared<Symbol>(namespace_var[1].cast<std::string>());
  return MakeResolve(name_space, symbol);
}

AnfNodePtr FunctionBlock::MakeResolve(const NameSpacePtr& name_space, const SymbolPtr& resolve_symbol) {
  MS_LOG(DEBUG) << "MakeResolve for " << ((std::string)py::str(name_space->obj())) << " , "
                << ((std::string)resolve_symbol->symbol());
  ValueNodePtr module_node = NewValueNode(name_space);
  ValueNodePtr symbol_node = NewValueNode(resolve_symbol);
  auto node = func_graph()->NewCNode({NewValueNode(prim::kPrimResolve), module_node, symbol_node});
  return node;
}

// add input for the block's phi parameter
void FunctionBlock::SetPhiArgument(const ParameterPtr& phi) {
  std::string var = phi_nodes_[phi];
  MS_LOG(DEBUG) << "graph " << func_graph_->ToString() << " set phi " << phi->ToString() << " for var " << var;
  for (auto& pred : prev_blocks_) {
    MS_EXCEPTION_IF_NULL(pred);
    MS_LOG(DEBUG) << "graph " << func_graph_->ToString() << " pred_blocks_ " << pred->func_graph_->ToString();
    AnfNodePtr arg_node = pred->ReadVariable(var);
    CNodePtr jump = pred->jumps_[this];
    jump->add_input(arg_node);
  }
  // If the phi node in the body part of a for/while loop is being removed,
  // then the closure convert phase will generate a cycle in graph if the
  // loop is kept after specialization. This should be investigate further.
  // Just now user has to set a flag on a function to indicate the for loop
  // will definitely can be unroll as the sequence in for statement is fixed
  // size in compile time.
  if (parser_.func_graph()->has_flag(GRAPH_FLAG_LOOP_CAN_UNROLL) ||
      parser_.func_graph()->has_flag(GRAPH_FLAG_HAS_EFFECT)) {
    CollectRemovablePhi(phi);
  }
}

AnfNodePtr FunctionBlock::SearchReplaceNode(const std::string& var, const ParameterPtr& phi) {
  AnfNodePtr arg_node = nullptr;
  for (auto& prev : prev_blocks_) {
    MS_EXCEPTION_IF_NULL(prev);
    AnfNodePtr temp_node = prev->ReadVariable(var);
    MS_LOG(DEBUG) << "graph " << prev->func_graph_->ToString() << " phi " << phi->ToString() << " for var " << var
                  << " is " << temp_node->DebugString();
    if (temp_node != phi) {
      if (arg_node == nullptr) {
        arg_node = temp_node;
        MS_LOG(DEBUG) << "graph " << prev->func_graph_->ToString() << " phi " << phi->ToString()
                      << " may be replaced by node " << arg_node->DebugString();
      } else if (temp_node == arg_node) {
        MS_LOG(DEBUG) << "graph " << prev->func_graph_->ToString() << " phi " << phi->ToString() << " is same as node "
                      << arg_node->DebugString();
      } else {
        MS_LOG(DEBUG) << "phi " << phi->ToString()
                      << " cannot be removed as it assigns to different node. node1: " << arg_node->DebugString()
                      << ", node2: " << temp_node->DebugString();
        return nullptr;
      }
    }
  }
  return arg_node;
}

// Check if there is removable unnecessary phi node in this graph.
// as per the FIRM TR 3.2, a phi node can be remove if:
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
// Args :
// phi  : This parameter node is functioning as a phi node.
void FunctionBlock::CollectRemovablePhi(const ParameterPtr& phi) {
  MS_EXCEPTION_IF_NULL(phi);
  std::string var = phi_nodes_[phi];
  MS_LOG(DEBUG) << "check phi " << phi->ToString() << " for " << var << " in graph " << func_graph_->ToString();
  if (prev_blocks_.size() == 0) {
    MS_LOG(DEBUG) << "no phi " << phi->ToString() << " for var " << var << " in graph " << func_graph_->ToString();
    return;
  }
  AnfNodePtr arg_node = SearchReplaceNode(var, phi);
  if (arg_node != nullptr) {
    MS_LOG(DEBUG) << "graph " << func_graph_->ToString() << " phi " << phi->ToString() << " can be replaced with "
                  << arg_node->DebugString();
    // replace var with new one. This equal to statement in TR "v0 is immediately replaced by v1."
    WriteVariable(var, arg_node);
    removable_phis_[phi] = arg_node;
    // The following equal to statement "The φ-function defining v1, which now reads φ(v2, v1), is optimized
    // recursively". check if phi1 is assigned with this phi before, then phi1 can be replaced with arg_node.
    for (auto& prev : prev_blocks_) {
      MS_EXCEPTION_IF_NULL(prev);
      if (!prev->matured_) {
        continue;
      }
      for (auto& phi_iter : prev->removable_phis_) {
        MS_EXCEPTION_IF_NULL(phi_iter.second);
        if (phi_iter.second->isa<Parameter>()) {
          const auto& param = phi_iter.second->cast<ParameterPtr>();
          if (param == phi) {
            MS_LOG(DEBUG) << "graph " << prev->func_graph_->ToString() << " var " << phi_iter.first->DebugString()
                          << " can be replaced from " << param->DebugString() << " with " << arg_node->DebugString();
            prev->removable_phis_[phi_iter.first] = arg_node;
          }
        }
      }
    }
  }
}

// A block should be marked matured if its predecessor blocks have been processed
void FunctionBlock::Mature() {
  const auto& graphParamVec = func_graph_->parameters();
  for (auto& paramItr : graphParamVec) {
    MS_EXCEPTION_IF_NULL(paramItr);
    ParameterPtr param = paramItr->cast<ParameterPtr>();
    if (phi_nodes_.find(param) != phi_nodes_.cend()) {
      SetPhiArgument(param);
    }
  }
  matured_ = true;
}

// Force the conditon node to bool using bool operation
CNodePtr FunctionBlock::ForceToBoolNode(const AnfNodePtr& cond) {
  TraceManager::DebugTrace(std::make_shared<TraceForceBool>(cond->debug_info()));
  CNodePtr op_apply_node = func_graph()->NewCNode({MakeResolveOperation(NAMED_PRIMITIVE_BOOL), cond});
  TraceManager::EndTrace();
  return op_apply_node;
}

// Perform a jump from this block to target block
void FunctionBlock::Jump(const FunctionBlockPtr& target_block, AnfNodePtr node) {
  if (func_graph()->get_return() != nullptr) {
    MS_LOG(EXCEPTION) << "Failure: have return node! NodeInfo: "
                      << trace::GetDebugInfo(func_graph()->get_return()->debug_info());
  }
  std::vector<AnfNodePtr> input_nodes;
  input_nodes.emplace_back(NewValueNode(target_block->func_graph()));
  if (node != nullptr) {
    input_nodes.emplace_back(node);
  }

  CNodePtr jump = func_graph()->NewCNode(input_nodes);
  jumps_[target_block.get()] = jump;
  target_block->AddPrevBlock(shared_from_this());
  func_graph()->set_output(jump);
  InsertDependItemsBeforeReturn();
}

// Perform a conditional jump using switch operation.
// The first CNode select graph with condition, and than execute this graph
void FunctionBlock::ConditionalJump(AnfNodePtr condNode, const FunctionBlockPtr& true_block,
                                    const FunctionBlockPtr& false_block) {
  if (func_graph()->get_return() != nullptr) {
    MS_LOG(EXCEPTION) << "Failure: have return node! NodeInfo: "
                      << trace::GetDebugInfo(func_graph()->get_return()->debug_info());
  }
  CNodePtr switch_app =
    func_graph()->NewCNode({NewValueNode(prim::kPrimSwitch), condNode, NewValueNode(true_block->func_graph()),
                            NewValueNode(false_block->func_graph())});
  CNodePtr switch_app_new = func_graph()->NewCNode({switch_app});
  func_graph()->set_output(switch_app_new);
  InsertDependItemsBeforeReturn();
}

void FunctionBlock::SetStateAssgin(const AnfNodePtr& target, const std::string& readid) {
  state_assign_[target] = readid;
}

void FunctionBlock::AddAutoDepend(const AnfNodePtr& target) { auto_depends_.push_back(target); }

void FunctionBlock::InsertDependItemsBeforeReturn() {
  if (!prev_blocks_.empty()) {
    for (auto& prev_block : prev_blocks_) {
      MS_LOG(DEBUG) << "Has prev_block " << prev_block->func_graph()->debug_info().get();
    }
  }

  ValueNodePtr make_tuple_op = NewValueNode(prim::kPrimMakeTuple);
  ValueNodePtr depend_op = NewValueNode(prim::kPrimDepend);
  ValueNodePtr get_refkey_op = NewValueNode(prim::kPrimGetRefKey);
  ValueNodePtr stop_gradient_op = NewValueNode(prim::kPrimStopGradient);
  const std::string primitive_name("assign");
  const std::string module_name("mindspore.ops.functional");
  ValueNodePtr assign_op = NewValueNode(prim::GetPythonOps(primitive_name, module_name));

  if (state_assign_.size() == 0 && auto_depends_.size() == 0) {
    return;
  }
  AnfNodePtr state = nullptr;
  std::vector<AnfNodePtr> vec_states;
  vec_states.emplace_back(make_tuple_op);
  for (auto& item : state_assign_) {
    auto source = ReadVariable(item.second);
    auto refkey = func_graph()->NewCNode({get_refkey_op, item.first});
    auto assign = func_graph()->NewCNode({assign_op, refkey, source});
    MS_LOG(INFO) << "SetState read " << item.first->ToString() << ", " << item.second;
    vec_states.emplace_back(assign);
  }
  for (auto& item : auto_depends_) {
    MS_LOG(DEBUG) << "auto_depends " << item->ToString();
    vec_states.emplace_back(item);
  }
  // if there are only make_tuple_op and another node in vec_states(the vec_states size is 2)
  // do not need to make_tuple, just use the node.
  if (vec_states.size() == 2) {
    state = vec_states[1];
  } else {
    state = func_graph()->NewCNode(vec_states);
  }

  AnfNodePtr old_ret = nullptr;
  auto return_node = func_graph()->get_return();
  if (return_node) {
    if (return_node->inputs().size() < 1) {
      MS_LOG(EXCEPTION) << "length of inputs of output node is less than 2";
    }
    old_ret = return_node->input(1);
  } else {
    old_ret = NewValueNode(kNone);
  }
  AnfNodePtr stopped = func_graph()->NewCNode({stop_gradient_op, state});
  AnfNodePtr ret = func_graph()->NewCNode({depend_op, old_ret, stopped});
  func_graph()->set_output(ret, true);
  state_assign_.clear();
}
}  // namespace parse
}  // namespace mindspore
