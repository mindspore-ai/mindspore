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

#include "pipeline/jit/parse/parse.h"

#include <utility>
#include <string>
#include <memory>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include "pipeline/jit/parse/resolve.h"
#include "frontend/operator/ops.h"
#include "pipeline/jit/parse/data_converter.h"
#include "frontend/operator/composite/composite.h"
#include "utils/ms_context.h"
#include "debug/trace.h"

namespace mindspore {
namespace parse {

FuncGraphPtr ParsePythonCode(const py::object &obj, const std::string &python_mod_get_parse_method) {
  (void)python_adapter::set_python_scoped();

  if (!obj || py::isinstance<py::none>(obj)) {
    MS_LOG(ERROR) << "Parse the python code failed, obj is nullptr or none";
    return nullptr;
  }

  auto ast = std::make_shared<ParseAst>(obj);
  bool success = ast->InitParseAstInfo(python_mod_get_parse_method);
  if (!success) {
    MS_LOG(ERROR) << "Parse code to ast tree failed.";
    return nullptr;
  }

  auto parser = std::make_shared<Parser>(ast);

  FuncGraphPtr func_graph = parser->ParseFuncGraph();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Parse python code failed, errcode = " << parser->errcode();
    return nullptr;
  }

  return func_graph;
}

TypePtr GetMixedPrecisionTargetType(const FuncGraphPtr &func_graph) {
  if (func_graph->has_flag(GRAPH_FLAG_MIX_PRECISION_FP32)) {
    return kFloat32;
  } else if (func_graph->has_flag(GRAPH_FLAG_MIX_PRECISION_FP16)) {
    return kFloat16;
  } else {
    return nullptr;
  }
}

// If any mixed precision flag add a cast node after the parameter node.
AnfNodePtr GetMixedPrecisionCastHelp(const FuncGraphPtr &func_graph, const AnfNodePtr &param) {
  TypePtr dst_type;
  if (func_graph->has_flag(GRAPH_FLAG_MIX_PRECISION_FP32)) {
    dst_type = kFloat32;
  } else if (func_graph->has_flag(GRAPH_FLAG_MIX_PRECISION_FP16)) {
    dst_type = kFloat16;
  } else {
    return param;
  }
  auto cast_helper = prim::kPrimMixedPrecisionCast;
  auto cast = func_graph->NewCNodeAfter(param, {NewValueNode(cast_helper), NewValueNode(dst_type), param});
  return cast;
}

FuncGraphWeakPtr Parser::top_func_graph_ = FuncGraphWeakPtr();

Parser::Parser(const std::shared_ptr<ParseAst> &ast) : ast_(ast) {
  errcode_ = PARSE_SUCCESS;
  BuildMethodMap();
}

void Parser::BuildMethodMap() {
  stmt_method_map_["Return"] = &Parser::ParseReturn;
  stmt_method_map_["Expr"] = &Parser::ParseExpr;
  stmt_method_map_["If"] = &Parser::ParseIf;
  stmt_method_map_["Assign"] = &Parser::ParseAssign;
  stmt_method_map_["While"] = &Parser::ParseWhile;
  stmt_method_map_["For"] = &Parser::ParseFor;
  stmt_method_map_["FunctionDef"] = &Parser::ParseFunctionDef;
  stmt_method_map_["AugAssign"] = &Parser::ParseAugAssign;
  stmt_method_map_["Global"] = &Parser::ParseGlobal;
  stmt_method_map_["Break"] = &Parser::ParseBreak;
  stmt_method_map_["Continue"] = &Parser::ParseContinue;
  stmt_method_map_["Pass"] = &Parser::ParsePass;
  expr_method_map_["NoneType"] = &Parser::ParseNone;
  expr_method_map_["BinOp"] = &Parser::ParseBinOp;
  expr_method_map_["Name"] = &Parser::ParseName;
  expr_method_map_["Num"] = &Parser::ParseNum;
  expr_method_map_["Str"] = &Parser::ParseStr;
  expr_method_map_["Constant"] = &Parser::ParseConstant;
  expr_method_map_["NameConstant"] = &Parser::ParseNameConstant;
  expr_method_map_["Call"] = &Parser::ParseCall;
  expr_method_map_["IfExp"] = &Parser::ParseIfExp;
  expr_method_map_["Attribute"] = &Parser::ParseAttribute;
  expr_method_map_["Compare"] = &Parser::ParseCompare;
  expr_method_map_["BoolOp"] = &Parser::ParseBoolOp;
  expr_method_map_["Lambda"] = &Parser::ParseLambda;
  expr_method_map_["Tuple"] = &Parser::ParseTuple;
  expr_method_map_["List"] = &Parser::ParseList;
  expr_method_map_["Subscript"] = &Parser::ParseSubscript;
  expr_method_map_["Slice"] = &Parser::ParseSlice;
  expr_method_map_["ExtSlice"] = &Parser::ParseExtSlice;
  expr_method_map_["Index"] = &Parser::ParseIndex;
  expr_method_map_["UnaryOp"] = &Parser::ParseUnaryOp;
  expr_method_map_["Dict"] = &Parser::ParseDict;
  expr_method_map_["Ellipsis"] = &Parser::ParseEllipsis;
}

void Parser::UpdateTopFuncGraph(const FuncGraphPtr &func_graph) { top_func_graph_ = FuncGraphWeakPtr(func_graph); }

void Parser::InitParserEnvironment(const py::object &obj) {
  Parser::top_func_graph_ = FuncGraphWeakPtr();
  ScopeManager::GetInstance().ClearScope();
  (void)python_adapter::CallPyFn(PYTHON_MOD_PARSE_MODULE, PYTHON_PARSE_GENERATE_SCOPE, obj);
}

void Parser::CleanParserResource() {
  Parser::top_func_graph_ = FuncGraphWeakPtr();
  ScopeManager::GetInstance().ClearScope();
}

AnfNodePtr AppendParameterObj(const FuncGraphPtr &func_graph, const py::object &obj) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto value = py::cast<tensor::MetaTensorPtr>(obj);
  // Parameter object should not be none
  if (value == nullptr || !value->is_parameter()) {
    MS_LOG(EXCEPTION) << "Parameter error: because obj is not Parameter object.";
  }

  // Get the parameter name from parameter object
  auto param_name = value->param_info()->name();

  auto top_graph = func_graph;
  // If the parameter node has been created , return it
  AnfNodePtr para_node = nullptr;
  for (const auto &param : top_graph->parameters()) {
    auto param_node = dyn_cast<Parameter>(param);
    if (param_node != nullptr && param_node->name() == param_name) {
      para_node = param;
      break;
    }
  }
  if (para_node == nullptr) {
    auto node = top_graph->AddWeightParameter(param_name);

    node->set_default_param(value);
    // set_abstract for parameter
    auto abs = value->ToAbstract();
    // Boarden value
    abs = abs->Broaden();
    node->set_abstract(abs);
    para_node = node;
  }
  return para_node;
}

void UpdataParam(const FuncGraphPtr &top_graph, const py::object &cell) {
  auto params = py::list(cell.attr("get_parameters")()).cast<std::vector<py::object>>();
  for (const auto &param : params) {
    (void)AppendParameterObj(top_graph, param);
  }
}

void CheckFuncReturn(const FuncGraphPtr &fn, const std::shared_ptr<ParseAst> &ast) {
  // Check whether the functions referred by this function and itself are missing 'return' statement
  auto mng = Manage(fn, false);
  for (const auto &func_graph : mng->func_graphs()) {
    if (func_graph->get_return() != nullptr) {
      continue;
    }
    py::object node = ast->GetAstNode();
    py::list ret = ast->CallParserObjMethod(PYTHON_PARSE_GET_LOCATION, node);
    py::str desc =
      python_adapter::CallPyModFn(ast->module(), PYTHON_MOD_GET_OBJECT_DESCRIPTION, ast->function(), ret[0], ret[1]);
    MS_EXCEPTION(TypeError) << "Missing return statement in " << desc.cast<std::string>() << ".";
  }
}

FuncGraphPtr Parser::ParseFuncGraph() {
  // Get ast FunctionDef node
  py::object node = ast_->GetAstNode();
  FunctionBlockPtr pFnBlock = ParseFunction(node);
  if (errcode() != PARSE_SUCCESS) {
    MS_LOG(ERROR) << "Parse function error, code is " << errcode();
    return nullptr;
  }

  RemoveUnnecessaryPhis();

  MS_EXCEPTION_IF_NULL(pFnBlock);
  CheckFuncReturn(pFnBlock->func_graph(), ast_);

  return pFnBlock->func_graph();
}

void Parser::GenerateArgsNodeForFunction(const FunctionBlockPtr &block, const py::object &fn_node) {
  py::object func_args = python_adapter::GetPyObjAttr(fn_node, "args");
  py::object var_arg_node = python_adapter::GetPyObjAttr(func_args, "vararg");
  block->func_graph()->set_has_vararg(!py::isinstance<py::none>(var_arg_node));

  py::object kw_arg_node = python_adapter::GetPyObjAttr(func_args, "kwarg");
  block->func_graph()->set_has_kwarg(!py::isinstance<py::none>(kw_arg_node));

  py::list kwonly_args = python_adapter::GetPyObjAttr(func_args, "kwonlyargs");
  block->func_graph()->set_kwonlyargs_count(SizeToLong(kwonly_args.size()));

  MS_EXCEPTION_IF_NULL(ast_);
  py::list args = ast_->GetArgs(fn_node);
  for (std::size_t i = 0; i < args.size(); i++) {
    std::string arg_name = py::cast<std::string>(args[i].attr("arg"));
    if (ast()->target_type() == PARSE_TARGET_OBJECT_INSTANCE) {
      if (arg_name == "self") {
        continue;
      }
    }
    TraceGuard guard(GetLocation(args[i]));
    auto para_node = std::make_shared<Parameter>(block->func_graph());
    MS_EXCEPTION_IF_NULL(para_node);
    para_node->set_name(arg_name);
    para_node->debug_info()->set_name(arg_name);
    block->func_graph()->add_parameter(para_node);
    AnfNodePtr para_after_cast = GetMixedPrecisionCastHelp(block->func_graph(), para_node);
    block->WriteVariable(arg_name, para_after_cast);
    MS_LOG(DEBUG) << "The arg[" << i << "] is " << arg_name;
  }
}

void Parser::GenerateArgsDefaultValueForFunction(const FunctionBlockPtr &block, const py::object &fn_node) {
  py::list defaults = ast_->GetArgsDefaultValues(fn_node);
  py::list args = ast_->GetArgs(fn_node);
  std::vector<std::string> namelist_for_default_value;
  std::vector<AnfNodePtr> default_values;
  for (std::size_t i = 0; i < args.size(); i++) {
    std::string arg_name = py::cast<std::string>(args[i].attr("arg"));
    if (ast()->target_type() == PARSE_TARGET_OBJECT_INSTANCE) {
      if (arg_name == "self") {
        continue;
      }
    }

    namelist_for_default_value.push_back(arg_name);
    if (py::isinstance<py::none>(defaults[i])) {
      default_values.push_back(NewValueNode(kNull));
    } else {
      default_values.push_back(ParseExprNode(block, defaults[i]));
    }
  }
  block->func_graph()->SetDefaultValues(namelist_for_default_value, default_values);
}

ScopePtr Parser::GetScopeForParseFunction() {
  ScopePtr scope = ScopeManager::GetInstance().GetCurrentScope();
  if (ast()->target_type() == PARSE_TARGET_OBJECT_INSTANCE) {
    py::object scope_str = python_adapter::CallPyFn(PYTHON_MOD_PARSE_MODULE, PYTHON_PARSE_GET_SCOPE_NAME, ast_->obj());
    if (!py::isinstance<py::none>(scope_str)) {
      auto scope_name = py::cast<std::string>(scope_str);
      scope = std::make_shared<Scope>(scope_name);
    }
  }
  return scope;
}

FunctionBlockPtr Parser::ParseFunction(const py::object &node, const FunctionBlockPtr &block) {
  ScopePtr scope = GetScopeForParseFunction();
  // The node created in the parsefunction context, will inherit the scope created using scope_guard
  ScopeGuard scope_guard(scope);
  TraceGuard trace_guard(data_converter::GetObjKey(ast()->obj())[0], GetLocation(node));
  FunctionBlockPtr pFunBlock = MakeFunctionBlock(*this);
  if (block != nullptr) {
    pFunBlock->AddPrevBlock(block);
  } else {
    func_graph_ = pFunBlock->func_graph();
  }
  pFunBlock->Mature();
  auto current_fg = pFunBlock->func_graph();
  auto function_name = py::cast<std::string>(python_adapter::GetPyObjAttr(node, "name"));
  MS_LOG(DEBUG) << "The function name is " << function_name;

  current_fg->debug_info()->set_name(function_name);
  MS_EXCEPTION_IF_NULL(ast_);
  py::list deco_list = node.attr("decorator_list");
  if (!deco_list.empty()) {
    current_fg->debug_info()->set_deco_location(GetLocation(deco_list));
  }

  bool set_flag = UpdateFuncGraphFlags(ast_->function(), current_fg);
  if (!ast_->obj().is(ast_->function())) {
    set_flag = set_flag && UpdateFuncGraphFlags(ast_->obj(), current_fg);
  }

  if (!set_flag) {
    MS_LOG(ERROR) << "Set flags failed";
    return nullptr;
  }
  GenerateArgsNodeForFunction(pFunBlock, node);

  // When parsing the top graph of construct, save the top graph
  if (GetTopFuncGraph() == nullptr) {
    UpdateTopFuncGraph(pFunBlock->func_graph());
  }

  // Save the function node to block
  pFunBlock->WriteVariable(function_name, NewValueNode(current_fg));

  py::object funcObj = python_adapter::GetPyObjAttr(node, "body");
  (void)ParseStatements(pFunBlock, funcObj);

  // Add unused variables as isolate nodes.
  for (auto &func_block : func_block_list_) {
    if (func_block->func_graph()->get_return() != nullptr) {
      // Find unused variables.
      func_block->FindIsolatedNodes();
      // Attach all isolated nodes.
      func_block->AttachIsolatedNodesBeforeReturn();
    }
  }

  if (current_fg->get_return() == nullptr) {
    py::list ret = ast_->CallParserObjMethod(PYTHON_PARSE_GET_LOCATION, node);
    py::str desc = python_adapter::CallPyModFn(ast_->module(), PYTHON_MOD_GET_OBJECT_DESCRIPTION, node, ret[0], ret[1]);
    MS_EXCEPTION(TypeError) << "Missing return statement in " << desc.cast<std::string>() << ".";
  }
  GenerateArgsDefaultValueForFunction(pFunBlock, node);
  return pFunBlock;
}

FunctionBlockPtr Parser::ParseStatements(FunctionBlockPtr block, const py::object &nodes) {
  auto node_list = py::cast<py::list>(nodes);
  size_t count = py::len(node_list);
  MS_LOG(DEBUG) << "The nodes count is " << count;
  for (size_t i = 0; i < count; ++i) {
    auto node = node_list[i];
    block = ParseStatement(block, node);
    // Insert appropriate depended items for the function block if it has a return node
    if (block->func_graph()->get_return() != nullptr) {
      // Skip statements after 'return' (or 'break', 'continue').
      break;
    }
  }
  return block;
}

FunctionBlockPtr Parser::ParseStatement(const FunctionBlockPtr &block, const py::object &node) {
  TraceGuard trace_guard(GetLocation(node));
  auto node_type = ast_->GetNodeType(node);

  // Check the node type
  AstMainType nodeType = node_type->main_type();
  if (nodeType != AST_MAIN_TYPE_STMT) {
    MS_LOG(INFO) << "Node type is error : " << nodeType;
    return block;
  }
  // Call the process function
  std::string node_name = node_type->node_name();
  MS_LOG(DEBUG) << "Ast node is " << node_name;
  if (stmt_method_map_.count(node_name)) {
    auto stmt_block = (this->*stmt_method_map_[node_name])(block, node);
    TraceManager::ClearParseOrResolveDebugInfo();
    return stmt_block;
  } else {
    errcode_ = PARSE_NODE_METHOD_UNSUPPORTED;
    MS_LOG(EXCEPTION) << "Unsupported syntax '" << node_name << "'.";
  }
}

AnfNodePtr Parser::ParseExprNode(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast expr";
  TraceGuard trace_guard(GetLocation(node));
  auto node_type = ast_->GetNodeType(node);
  // Check the node type
  AstMainType node_main_type = node_type->main_type();
  if (node_main_type != AST_MAIN_TYPE_EXPR) {
    MS_LOG(ERROR) << "Node type is error : " << node_main_type;
    errcode_ = PARSE_NODE_TYPE_NO_MATCH;
    return nullptr;
  }
  // Call the process function
  std::string node_name = node_type->node_name();
  MS_LOG(DEBUG) << "Ast node is " << node_name;
  if (expr_method_map_.count(node_name)) {
    auto expr_node = (this->*expr_method_map_[node_name])(block, node);
    TraceManager::ClearParseOrResolveDebugInfo();
    return expr_node;
  } else {
    errcode_ = PARSE_NODE_METHOD_UNSUPPORTED;
    MS_LOG(EXCEPTION) << "Unsupported syntax '" << node_name << "'.";
  }
}

// Process the expr statement and expand it
FunctionBlockPtr Parser::ParseExpr(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Expr";
  // Expr only have value, no target
  py::tuple expand_info = ast_->CallParseModFunction(PYTHON_PARSE_EXPAND_EXPR_STATEMENT, node);

  // Refer python function expand_expr_statement, expand_info is one of the following:
  // True, expr.value, x
  // True, expr.value
  // False, None, None
  //
  // Check the expand info result
  auto is_expand = py::cast<bool>(expand_info[0]);
  if (is_expand) {
    // Process the expr statement
    py::object value_object = expand_info[1];
    // Make a Expr CNode.
    AnfNodePtr call_node = ParseExprNode(block, value_object);
    if (py::len(expand_info) == 2) {
      // Expression that not assigned to any variable.
      // This is usually a call with side effects.
      // e.g.: print(x)
      // We save it as an isolated node.
      auto &no_return_node = call_node;
      MS_LOG(INFO) << "Isolated node found(NoReturn), no_return_node: " << no_return_node->DebugString(2)
                   << ", block: " << block << "/"
                   << (block->func_graph() ? block->func_graph()->ToString() : "FG(Null)")
                   << ", Line: " << trace::GetDebugInfo(no_return_node->debug_info(), "", kSourceLineTipDiscard);
      block->AddIsolatedNode(no_return_node);
    } else {
      // Expand the assign statement,
      // e.g.: x.append(y)  -> x = x.append(y)
      py::object target_node = expand_info[2];
      WriteAssignVars(block, target_node, call_node);
    }
  }
  return block;
}

LocationPtr Parser::GetLocation(const py::object &node) const {
  MS_EXCEPTION_IF_NULL(ast_);
  py::list ret = ast_->CallParserObjMethod(PYTHON_PARSE_GET_LOCATION, node);
  if (ret.size() < 5) {
    MS_LOG(EXCEPTION) << "List size should not be less than 5.";
  }
  // Refer to Location::Location() for each member of ret: line, column, line_end, column_end.
  auto location = std::make_shared<Location>(ret[0].cast<std::string>(), ret[1].cast<int64_t>(), ret[2].cast<int64_t>(),
                                             ret[3].cast<int64_t>(), ret[4].cast<int64_t>());
  return location;
}

void Parser::MakeConditionBlocks(const FunctionBlockPtr &pre_block, const FunctionBlockPtr &true_block,
                                 const FunctionBlockPtr &false_block) {
  true_block->AddPrevBlock(pre_block);
  true_block->Mature();

  false_block->AddPrevBlock(pre_block);
  false_block->Mature();
}

FunctionBlockPtr Parser::ParseReturn(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast return";
  MS_EXCEPTION_IF_NULL(block);
  // Create return valuenode
  AnfNodePtr pReturnValueNode = NewValueNode(prim::kPrimReturn);
  // Parse the return Statements value
  py::object value = python_adapter::GetPyObjAttr(node, "value");
  AnfNodePtr pReturnStatementNode = ParseExprNode(block, value);
  // Create the cnode
  CNodePtr pReturnCNode = block->func_graph()->NewCNodeInOrder({pReturnValueNode, pReturnStatementNode});

  block->func_graph()->set_return(pReturnCNode);

  return block;
}

// Process binary operators,eg: `a + b`, `a | b`, etc.
AnfNodePtr Parser::ParseBinOp(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast BinOP";

  py::object left = python_adapter::GetPyObjAttr(node, "left");
  py::object right = python_adapter::GetPyObjAttr(node, "right");
  py::object op = python_adapter::GetPyObjAttr(node, "op");
  // Create left and right ANF node
  AnfNodePtr left_node = ParseExprNode(block, left);
  if (left_node == nullptr) {
    MS_LOG(WARNING) << "DoBinOp process left node failed: " << errcode();
    return nullptr;
  }
  AnfNodePtr right_node = ParseExprNode(block, right);
  if (right_node == nullptr) {
    MS_LOG(WARNING) << "DoBinOp process right node failed:" << errcode();
    return nullptr;
  }
  // Resolve the op
  AnfNodePtr op_node = block->MakeResolveAstOp(op);
  // Create apply node
  return block->func_graph()->NewCNodeInOrder({op_node, left_node, right_node});
}

AnfNodePtr Parser::ParseName(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Name";
  auto name_id = py::cast<std::string>(python_adapter::GetPyObjAttr(node, "id"));
  MS_LOG(DEBUG) << "The Name id is " << name_id;
  if (block->IsGlobalVar(name_id)) {
    return block->MakeResolveSymbol(name_id);
  }
  return block->ReadVariable(name_id);
}

AnfNodePtr Parser::ParseNone(const FunctionBlockPtr &, const py::object &) {
  MS_LOG(DEBUG) << "Process ast NoneType";
  return NewValueNode(kNone);
}

AnfNodePtr Parser::ParseEllipsis(const FunctionBlockPtr &, const py::object &) {
  MS_LOG(DEBUG) << "Process ast Ellipsis";
  return NewValueNode(kEllipsis);
}

AnfNodePtr Parser::ParseNum(const FunctionBlockPtr &, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Num";
  py::object obj = python_adapter::GetPyObjAttr(node, "n");
  if (py::isinstance<py::int_>(obj)) {
    MS_LOG(INFO) << "The Num is int64_t:" << (std::string)py::str(obj);
    auto data = py::cast<int64_t>(obj);
    return NewValueNode(data);
  } else if (py::isinstance<py::float_>(obj)) {
    MS_LOG(INFO) << "The Num is float:" << (std::string)py::str(obj);
    auto data = py::cast<float>(obj);
    return NewValueNode(data);
  } else {
    // no else actually
    MS_LOG(ERROR) << "Unsupported Num type : " << (std::string)py::str(obj);
    errcode_ = PARSE_NODE_TYPE_UNKNOWN;
    return nullptr;
  }
}

AnfNodePtr Parser::ParseStr(const FunctionBlockPtr &, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Str";
  auto str_s = py::cast<std::string>(python_adapter::GetPyObjAttr(node, "s"));
  return NewValueNode(str_s);
}

AnfNodePtr Parser::ParseConstant(const FunctionBlockPtr &, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Constant";
  py::object obj = python_adapter::GetPyObjAttr(node, "value");
  if (py::isinstance<py::bool_>(obj)) {
    MS_LOG(INFO) << "The Constant is bool:" << (std::string)py::str(obj);
    auto data = py::cast<bool>(obj);
    return NewValueNode(data);
  } else if (py::isinstance<py::int_>(obj)) {
    MS_LOG(INFO) << "The Constant is int64_t:" << (std::string)py::str(obj);
    auto data = py::cast<int64_t>(obj);
    return NewValueNode(data);
  } else if (py::isinstance<py::float_>(obj)) {
    MS_LOG(INFO) << "The Constant is float:" << (std::string)py::str(obj);
    auto data = py::cast<float>(obj);
    return NewValueNode(data);
  } else if (py::isinstance<py::str>(obj)) {
    MS_LOG(INFO) << "The Constant is string:" << (std::string)py::str(obj);
    auto data = py::cast<std::string>(obj);
    return NewValueNode(data);
  } else if (py::isinstance<py::none>(obj)) {
    MS_LOG(INFO) << "The Constant is none:" << (std::string)py::str(obj);
    return NewValueNode(kNone);
  } else if (py::isinstance<py::ellipsis>(obj)) {
    MS_LOG(INFO) << "The Constance is ellipsis:" << (std::string)py::str(obj);
    return NewValueNode(kEllipsis);
  } else {
    // no else actually
    MS_EXCEPTION(TypeError) << "Unsupported Constant type : " << (std::string)py::str(obj);
  }
}

AnfNodePtr Parser::ParseNameConstant(const FunctionBlockPtr &, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast NameConstant";
  py::object obj = python_adapter::GetPyObjAttr(node, "value");
  if (py::isinstance<py::bool_>(obj)) {
    MS_LOG(INFO) << "The NameConstant is bool:" << (std::string)py::str(obj);
    auto data = py::cast<bool>(obj);
    return NewValueNode(data);
  } else if (py::isinstance<py::none>(obj)) {
    MS_LOG(INFO) << "The NameConstant is none:" << (std::string)py::str(obj);
    return NewValueNode(kNone);
  } else {
    // no else actually
    MS_LOG(ERROR) << "Unsupported NameConstant type: " << (std::string)py::str(obj);
    errcode_ = PARSE_NODE_TYPE_UNKNOWN;
    return nullptr;
  }
}
AnfNodePtr Parser::GenerateMakeTuple(const FunctionBlockPtr &block, const std::vector<AnfNodePtr> &element_nodes) {
  AnfNodePtr make_tuple_op = block->MakeResolveOperation(NAMED_PRIMITIVE_MAKETUPLE);
  std::vector<AnfNodePtr> make_tuple_nodes;
  make_tuple_nodes.push_back(make_tuple_op);
  (void)std::transform(element_nodes.begin(), element_nodes.end(), std::back_inserter(make_tuple_nodes),
                       [](AnfNodePtr arg) -> AnfNodePtr { return arg; });
  return block->func_graph()->NewCNodeInOrder(make_tuple_nodes);
}

AnfNodePtr Parser::ParseSuper(const FunctionBlockPtr &block, const py::list &args) {
  py::object father_class;
  if (args.empty()) {
    father_class = py::none();
  } else if (args.size() == 2) {
    father_class = args[0];
    auto arg_type = AstSubType(py::cast<int32_t>(ast_->CallParseModFunction(PYTHON_PARSE_GET_AST_TYPE, args[1])));
    if (arg_type != AST_SUB_TYPE_NAME || py::cast<std::string>(python_adapter::GetPyObjAttr(args[1], "id")) != "self") {
      MS_EXCEPTION(ArgumentError) << "When call 'super', the second arg should be 'self'.";
    }
  } else {
    MS_EXCEPTION(ArgumentError) << "When call 'super', the args number should be 0 or 2, but got" << args.size() << ".";
  }
  py::object target_class_instance = ast()->CallParserObjMethod(PYTHON_PARSE_ANALYZE_SUPER, father_class, ast()->obj());
  py::object namespace_var = ast_->CallParseModFunction(PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL, target_class_instance);
  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, namespace_var);
  SymbolPtr symbol = std::make_shared<Symbol>("namespace");
  return block->MakeResolve(name_space, symbol);
}

// Process function call, eg : f1(x, y) ...
AnfNodePtr Parser::ParseCall(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Call";
  // Process function call
  py::object function_ast_node = python_adapter::GetPyObjAttr(node, "func");
  py::list args = python_adapter::GetPyObjAttr(node, "args");

  auto arg_type =
    AstSubType(py::cast<int32_t>(ast_->CallParseModFunction(PYTHON_PARSE_GET_AST_TYPE, function_ast_node)));
  if (arg_type == AST_SUB_TYPE_NAME) {
    auto name_id = py::cast<std::string>(python_adapter::GetPyObjAttr(function_ast_node, "id"));
    if (name_id == "super") {
      return ParseSuper(block, args);
    }
  }

  AnfNodePtr call_function_anf_node = ParseExprNode(block, function_ast_node);
  // Function call arguments should be passed in as groups and unpacked later using unpack call
  std::vector<AnfNodePtr> packed_arguments;
  std::vector<AnfNodePtr> group_arguments;

  bool need_unpack_args = ParseArgsInCall(block, args, &packed_arguments, &group_arguments);
  bool need_unpack_keywords = ParseKeywordsInCall(block, node, &packed_arguments);
  // If there is stared or keyword argument, unpack may be needed
  bool need_unpack = need_unpack_args || need_unpack_keywords;

  return GenerateAnfNodeForCall(block, call_function_anf_node, packed_arguments, group_arguments, need_unpack);
}

CNodePtr MakeUnpackCall(const FuncGraphPtr &func_graph, const AnfNodePtr &call_function_anf_node,
                        const std::vector<AnfNodePtr> &packed_arguments) {
  std::vector<AnfNodePtr> unpack_call_nodes;
  auto unpack_call_op = NewValueNode(std::make_shared<prim::UnpackCall>(NAMED_METAGRAPH_UNPACKCALL));
  unpack_call_nodes.push_back(unpack_call_op);
  unpack_call_nodes.push_back(call_function_anf_node);
  (void)std::transform(packed_arguments.begin(), packed_arguments.end(), std::back_inserter(unpack_call_nodes),
                       [](AnfNodePtr node) -> AnfNodePtr { return node; });
  CNodePtr unpack_call = func_graph->NewCNodeInOrder(unpack_call_nodes);
  return unpack_call;
}

AnfNodePtr Parser::GenerateAnfNodeForCall(const FunctionBlockPtr &block, const AnfNodePtr &call_function_anf_node,
                                          const std::vector<AnfNodePtr> &packed_arguments,
                                          const std::vector<AnfNodePtr> &group_arguments, bool need_unpack) const {
  // If there is keyword arguments or starred, using an unpack_call op to unpack the argument
  if (need_unpack) {
    return MakeUnpackCall(block->func_graph(), call_function_anf_node, packed_arguments);
  }
  // else there is no keyword arguments and starred, parsed as normal arguments without unpack
  std::vector<AnfNodePtr> func_call_nodes;
  func_call_nodes.push_back(call_function_anf_node);
  (void)std::transform(group_arguments.begin(), group_arguments.end(), std::back_inserter(func_call_nodes),
                       [](AnfNodePtr node) -> AnfNodePtr { return node; });
  CNodePtr call_anf_node = block->func_graph()->NewCNodeInOrder(func_call_nodes);
  return call_anf_node;
}

bool Parser::ParseArgsInCall(const FunctionBlockPtr &block, const py::list &args,
                             std::vector<AnfNodePtr> *packed_arguments, std::vector<AnfNodePtr> *group_arguments) {
  bool need_unpack = false;
  for (size_t i = 0; i < args.size(); i++) {
    auto arg_node = AstSubType(py::cast<int32_t>(ast_->CallParseModFunction(PYTHON_PARSE_GET_AST_TYPE, args[i])));
    if (arg_node == AST_SUB_TYPE_STARRED) {
      if (!group_arguments->empty()) {
        packed_arguments->push_back(GenerateMakeTuple(block, *group_arguments));
      }
      packed_arguments->push_back(ParseExprNode(block, python_adapter::GetPyObjAttr(args[i], "value")));
      group_arguments->clear();
      need_unpack = true;
    } else {
      group_arguments->push_back(ParseExprNode(block, args[i]));
    }
  }
  if (!group_arguments->empty()) {
    packed_arguments->push_back(GenerateMakeTuple(block, *group_arguments));
  }
  return need_unpack;
}

bool Parser::ParseKeywordsInCall(const FunctionBlockPtr &block, const py::object &node,
                                 std::vector<AnfNodePtr> *packed_arguments) {
  bool need_unpack = false;
  py::list keywords = python_adapter::GetPyObjAttr(node, "keywords");
  if (!keywords.empty()) {
    need_unpack = true;
    std::vector<AnfNodePtr> keys;
    std::vector<AnfNodePtr> values;
    for (size_t index = 0; index < keywords.size(); index++) {
      auto kw_key = python_adapter::GetPyObjAttr(keywords[index], "arg");
      auto kw_value = python_adapter::GetPyObjAttr(keywords[index], "value");
      if (py::isinstance<py::none>(kw_key)) {
        packed_arguments->push_back(ParseExprNode(block, kw_value));
      } else {
        auto kw_key_c = kw_key.cast<std::string>();
        keys.push_back(NewValueNode(kw_key_c));
        values.push_back(ParseExprNode(block, kw_value));
      }
    }
    auto keys_tuple = GenerateMakeTuple(block, keys);
    auto values_tuple = GenerateMakeTuple(block, values);
    auto make_dict_op = block->MakeResolveOperation(NAMED_PRIMITIVE_MAKEDICT);
    std::vector<AnfNodePtr> make_dict_nodes;
    make_dict_nodes.push_back(make_dict_op);
    make_dict_nodes.push_back(keys_tuple);
    make_dict_nodes.push_back(values_tuple);
    packed_arguments->push_back(block->func_graph()->NewCNodeInOrder(make_dict_nodes));
  }
  return need_unpack;
}

// Process call attributes of class type define, eg: x.y()
AnfNodePtr Parser::ParseAttribute(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Attribute";

  // Process class value,eg: self.xx
  if (ast()->target_type() == PARSE_TARGET_OBJECT_INSTANCE) {
    if (ast_->IsClassMember(node)) {
      std::string var_name = "self.";
      std::string attr_name = node.attr("attr").cast<std::string>();
      (void)var_name.append(attr_name);
      auto attr_obj = ast()->obj().attr(attr_name.c_str());
      if (py::hasattr(ast()->obj(), attr_name.c_str()) &&
          (py::hasattr(attr_obj, PYTHON_PRIMITIVE_FLAG) || py::isinstance<py::int_>(attr_obj) ||
           py::isinstance<py::float_>(attr_obj) || py::isinstance<py::bool_>(attr_obj) ||
           py::isinstance<py::str>(attr_obj) || data_converter::IsCellInstance(attr_obj))) {
        return block->MakeResolveSymbol(var_name);
      } else {
        return block->ReadVariable(var_name);
      }
    }
  }

  // Process the get attr
  // Use the Primitive replace the operation resolve node (getattr),
  // because the getattr will eventually be converted to Primitive node
  AnfNodePtr op_node = NewValueNode(prim::kPrimGetAttr);

  // Process the attr body
  py::object value_body = python_adapter::GetPyObjAttr(node, "value");
  AnfNodePtr value_node = ParseExprNode(block, value_body);
  if (value_node == nullptr) {
    MS_LOG(WARNING) << "Parse attribute failed";
    return nullptr;
  }

  // Process the node attr
  auto attr_str = python_adapter::GetPyObjAttr(node, "attr").cast<std::string>();
  MS_LOG(DEBUG) << "Attr = " << attr_str;
  AnfNodePtr attr_node = nullptr;
  {
    TraceGuard guard(GetLocation(python_adapter::GetPyObjAttr(node, "attr")));
    attr_node = NewValueNode(attr_str);
  }

  // Create the apply node
  return block->func_graph()->NewCNodeInOrder({op_node, value_node, attr_node});
}

// Process comparison expression : a == b. a > b  etc.
AnfNodePtr Parser::ParseCompare(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Compare";

  // For python comparison ,there may be if x>y>5 ,
  // Which there is two ops , but we only support one now
  py::list ops = python_adapter::GetPyObjAttr(node, "ops");
  if (ops.size() > MAX_COMPARISON_OPS_SUPPORTED) {
    MS_LOG(ERROR) << "MindSpore does not support comparison with operators more than one now, ops size =" << ops.size();
    return nullptr;
  }

  py::object left = python_adapter::GetPyObjAttr(node, "left");
  py::list comparators = python_adapter::GetPyObjAttr(node, "comparators");
  AnfNodePtr left_node = ParseExprNode(block, left);
  AnfNodePtr right_node = ParseExprNode(block, comparators[0]);

  MS_EXCEPTION_IF_NULL(block);
  AnfNodePtr op_node = block->MakeResolveAstOp(ops[0]);

  return block->func_graph()->NewCNodeInOrder({op_node, left_node, right_node});
}

AnfNodePtr Parser::ProcessBoolOpValueList(const FunctionBlockPtr &block, const py::list &value_list, AstSubType mode) {
  // If there is only one bool op now
  if (value_list.size() == 1) {
    AnfNodePtr first_node = ParseExprNode(block, value_list[0]);
    return first_node;
  } else {
    py::object first = value_list[0];
    py::list rest;
    for (size_t i = 1; i < value_list.size(); i++) {
      rest.append(value_list[i]);
    }
    MS_EXCEPTION_IF_NULL(block);
    FunctionBlockPtr true_block = nullptr;
    FunctionBlockPtr false_block = nullptr;
    {
      TraceGuard guard(std::make_shared<TraceIfExpTrueBranch>(block->func_graph()->debug_info()));
      true_block = MakeFunctionBlock(*this);
    }
    {
      TraceGuard guard(std::make_shared<TraceIfExpFalseBranch>(block->func_graph()->debug_info()));
      false_block = MakeFunctionBlock(*this);
    }
    MakeConditionBlocks(block, true_block, false_block);
    FunctionBlockPtr b1, b2;

    // If it is and, we need to process the rest nodes;
    // If it is or, we continue to next
    if (mode == AST_SUB_TYPE_AND) {
      b1 = true_block;
      b2 = false_block;
    } else if (mode == AST_SUB_TYPE_OR) {
      b2 = true_block;
      b1 = false_block;
    } else {
      MS_LOG(ERROR) << "Not supported mode: " << mode;
      return nullptr;
    }
    AnfNodePtr test_node = ParseExprNode(block, first);
    AnfNodePtr rest_node = ProcessBoolOpValueList(b1, rest, mode);
    b1->func_graph()->set_output(rest_node);
    b2->func_graph()->set_output(test_node);

    auto cond_node = block->ForceToBoolNode(test_node);
    auto switch_app = block->func_graph()->NewCNodeInOrder({NewValueNode(prim::kPrimSwitch), cond_node,
                                                            NewValueNode(true_block->func_graph()),
                                                            NewValueNode(false_block->func_graph())});

    std::vector<AnfNodePtr> call_graph_nodes{switch_app};
    auto switch_app_call = block->func_graph()->NewCNodeInOrder(call_graph_nodes);
    return switch_app_call;
  }
}

// Process comparison expression : a and b. a or b .
AnfNodePtr Parser::ParseBoolOp(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast BoolOp";
  py::object op_node = python_adapter::GetPyObjAttr(node, "op");
  AstSubType op_type = ast_->GetOpType(op_node);
  if (op_type == AST_SUB_TYPE_UNKNOWN) {
    MS_LOG(WARNING) << "ProcessBoolOp, got unknown op type";
    return nullptr;
  }
  py::list op_values = python_adapter::GetPyObjAttr(node, "values");
  return ProcessBoolOpValueList(block, op_values, op_type);
}

// Process a function def
FunctionBlockPtr Parser::ParseFunctionDef(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast FunctionDef";
  FunctionBlockPtr function_block = ParseFunction(node, block);
  MS_EXCEPTION_IF_NULL(function_block);

  // Get function name
  py::str name = python_adapter::GetPyObjAttr(node, "name");
  std::string function_name = name;
  ValueNodePtr valuenode_graph = NewValueNode(function_block->func_graph());
  block->WriteVariable(function_name, valuenode_graph);
  return block;
}

// Process a lambda expression . like lambda x,y: x + y
AnfNodePtr Parser::ParseLambda(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Lambda";
  FunctionBlockPtr func_block = MakeFunctionBlock(*this);
  func_block->AddPrevBlock(block);
  func_block->Mature();

  // Get lambda args
  py::list args = ast_->GetArgs(node);
  for (std::size_t i = 0; i < args.size(); i++) {
    std::string arg = py::cast<std::string>(args[i].attr("arg"));
    TraceGuard guard(GetLocation(args[i]));
    auto para_node = std::make_shared<Parameter>(func_block->func_graph());
    para_node->debug_info()->set_name(arg);
    func_block->func_graph()->add_parameter(para_node);
    func_block->WriteVariable(arg, para_node);
    MS_LOG(DEBUG) << "The arg[" << i << "] is " << arg;
  }

  py::object body_node = python_adapter::GetPyObjAttr(node, "body");
  AnfNodePtr lambda_body_node = ParseExprNode(func_block, body_node);
  func_block->func_graph()->set_output(lambda_body_node);
  ValueNodePtr const_graph = NewValueNode(func_block->func_graph());
  return const_graph;
}

// Process a tuple
AnfNodePtr Parser::ParseTuple(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Tuple";
  MS_EXCEPTION_IF_NULL(block);
  py::tuple elts = python_adapter::GetPyObjAttr(node, "elts");
  if (elts.size() == 0) {
    auto empty_tuple = std::vector<ValuePtr>();
    return NewValueNode(std::make_shared<ValueTuple>(empty_tuple));
  }

  std::vector<AnfNodePtr> tuple_vec;
  AnfNodePtr make_tuple_op = block->MakeResolveOperation(NAMED_PRIMITIVE_MAKETUPLE);
  tuple_vec.emplace_back(make_tuple_op);
  for (size_t i = 0; i < elts.size(); i++) {
    AnfNodePtr node_ptr = ParseExprNode(block, elts[i]);
    tuple_vec.emplace_back(node_ptr);
  }
  CNodePtr tuple_app = block->func_graph()->NewCNodeInOrder(tuple_vec);
  return tuple_app;
}

// Process a list
AnfNodePtr Parser::ParseList(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast List";
  MS_EXCEPTION_IF_NULL(block);
  py::list elts = python_adapter::GetPyObjAttr(node, "elts");
  if (elts.size() == 0) {
    auto empty_list = std::vector<ValuePtr>();
    return NewValueNode(std::make_shared<ValueList>(empty_list));
  }

  std::vector<AnfNodePtr> list_vec;
  AnfNodePtr make_list_op = block->MakeResolveOperation(NAMED_PRIMITIVE_MAKELIST);
  list_vec.emplace_back(make_list_op);
  for (size_t i = 0; i < elts.size(); i++) {
    AnfNodePtr node_ptr = ParseExprNode(block, elts[i]);
    list_vec.emplace_back(node_ptr);
  }
  CNodePtr list_app = block->func_graph()->NewCNodeInOrder(list_vec);
  return list_app;
}

// Process a subscript, such as x[y] , node expressed as value[slice]
AnfNodePtr Parser::ParseSubscript(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Subscript";
  MS_EXCEPTION_IF_NULL(block);
  AnfNodePtr op_getitem = block->MakeResolveOperation(NAMED_PRIMITIVE_GETITEM);
  py::object value_node = python_adapter::GetPyObjAttr(node, "value");
  py::object slice_node = python_adapter::GetPyObjAttr(node, "slice");
  AnfNodePtr value = ParseExprNode(block, value_node);
  AnfNodePtr slice = ParseExprNode(block, slice_node);

  return block->func_graph()->NewCNodeInOrder({op_getitem, value, slice});
}

// Process a slice, get the slice value
AnfNodePtr Parser::ParseSlice(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Slice";
  MS_EXCEPTION_IF_NULL(block);
  AnfNodePtr op_makeslice = block->MakeResolveOperation(NAMED_PRIMITIVE_MAKESLICE);
  py::object start = python_adapter::GetPyObjAttr(node, "lower");
  py::object stop = python_adapter::GetPyObjAttr(node, "upper");
  py::object step = python_adapter::GetPyObjAttr(node, "step");
  AnfNodePtr start_node = ParseExprNode(block, start);
  AnfNodePtr stop_node = ParseExprNode(block, stop);
  AnfNodePtr step_node = ParseExprNode(block, step);

  return block->func_graph()->NewCNodeInOrder({op_makeslice, start_node, stop_node, step_node});
}

// Process a extslice
AnfNodePtr Parser::ParseExtSlice(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast ExtSlice";
  MS_EXCEPTION_IF_NULL(block);
  AnfNodePtr make_tuple_op = block->MakeResolveOperation(NAMED_PRIMITIVE_MAKETUPLE);
  py::tuple slice_tuple = python_adapter::GetPyObjAttr(node, "dims");

  std::vector<AnfNodePtr> node_vec;
  node_vec.emplace_back(make_tuple_op);
  for (size_t i = 0; i < slice_tuple.size(); i++) {
    AnfNodePtr node_ptr = ParseExprNode(block, slice_tuple[i]);
    node_vec.emplace_back(node_ptr);
  }
  CNodePtr tuple_conde = block->func_graph()->NewCNodeInOrder(node_vec);
  return tuple_conde;
}

// Process a index, get the index number
AnfNodePtr Parser::ParseIndex(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Index";
  py::object value_node = python_adapter::GetPyObjAttr(node, "value");
  return ParseExprNode(block, value_node);
}

// Process a  UnaryOp, +a, -b
AnfNodePtr Parser::ParseUnaryOp(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast UnaryOp";
  py::object op = python_adapter::GetPyObjAttr(node, "op");

  MS_EXCEPTION_IF_NULL(block);
  // Resolve the op
  AnfNodePtr op_node = block->MakeResolveAstOp(op);

  py::object operand = python_adapter::GetPyObjAttr(node, "operand");
  AnfNodePtr operand_node = ParseExprNode(block, operand);
  return block->func_graph()->NewCNodeInOrder({op_node, operand_node});
}

// Process a dict ast node expression
AnfNodePtr Parser::ParseDict(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Dict";
  py::list keys = node.attr("keys");
  py::list values = node.attr("values");
  std::vector<AnfNodePtr> key_nodes;
  std::vector<AnfNodePtr> value_nodes;
  for (size_t i = 0; i < keys.size(); i++) {
    key_nodes.push_back(ParseExprNode(block, keys[i]));
    value_nodes.push_back(ParseExprNode(block, values[i]));
  }
  auto keys_tuple = GenerateMakeTuple(block, key_nodes);
  auto values_tuple = GenerateMakeTuple(block, value_nodes);
  MS_EXCEPTION_IF_NULL(block);
  auto make_dict_op = block->MakeResolveOperation(NAMED_PRIMITIVE_MAKEDICT);
  return block->func_graph()->NewCNodeInOrder({make_dict_op, keys_tuple, values_tuple});
}

// Process a  augment assign such as a += b or mat[stride_slice] += b.
FunctionBlockPtr Parser::ParseAugAssign(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast AugAssign";
  MS_EXCEPTION_IF_NULL(block);
  MS_EXCEPTION_IF_NULL(ast_);

  py::object target_obj = python_adapter::GetPyObjAttr(node, "target");
  py::object op_obj = python_adapter::GetPyObjAttr(node, "op");
  py::object value_obj = python_adapter::GetPyObjAttr(node, "value");
  AnfNodePtr target_node = nullptr;
  AnfNodePtr op_node = block->MakeResolveAstOp(op_obj);
  AnfNodePtr value_node = ParseExprNode(block, value_obj);
  auto ast_type = AstSubType(py::cast<int32_t>(ast_->CallParseModFunction(PYTHON_PARSE_GET_AST_TYPE, target_obj)));

  if (ast_type == AST_SUB_TYPE_NAME) {
    target_node = ParseName(block, target_obj);
  } else if (ast_type == AST_SUB_TYPE_SUBSCRIPT) {
    target_node = ParseSubscript(block, target_obj);
  } else if (ast_->IsClassMember(target_obj)) {
    target_node = ParseAttribute(block, target_obj);
  } else {
    MS_LOG(EXCEPTION) << "Not supported augassign";
  }
  if (target_node == nullptr) {
    MS_LOG(EXCEPTION) << "Can not get target node ";
  }
  CNodePtr augassign_app = block->func_graph()->NewCNodeInOrder({op_node, target_node, value_node});
  WriteAssignVars(block, target_obj, augassign_app);
  return block;
}
// Process global declaration such as 'global x';
FunctionBlockPtr Parser::ParseGlobal(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Global";
  MS_EXCEPTION_IF_NULL(block);
  py::list vars = python_adapter::GetPyObjAttr(node, "names");
  for (auto &item : vars) {
    block->AddGlobalVar(py::cast<std::string>(item));
  }
  return block;
}

// Process a if statement
FunctionBlockPtr Parser::ParseIf(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast If";
  py::object test_node = python_adapter::GetPyObjAttr(node, "test");
  AnfNodePtr condition_node = ParseExprNode(block, test_node);
  MS_EXCEPTION_IF_NULL(block);
  CNodePtr bool_node = block->ForceToBoolNode(condition_node);

  FunctionBlockPtr true_block = nullptr;
  FunctionBlockPtr false_block = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceIfStmtTrueBranch>(block->func_graph()->debug_info()));
    true_block = MakeFunctionBlock(*this);
  }
  {
    TraceGuard guard(std::make_shared<TraceIfStmtFalseBranch>(block->func_graph()->debug_info()));
    false_block = MakeFunctionBlock(*this);
  }

  MakeConditionBlocks(block, true_block, false_block);

  FunctionBlockPtr after_block = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceIfStmtAfterBranch>(block->func_graph()->debug_info()));
    after_block = MakeFunctionBlock(*this);
  }

  if (MsContext::GetInstance()->backend_policy() != "ge") {
    // For backends excludes 'ge', it can handle multi graph call, use this flag to
    // generate call not inline `after_block` graph to reduce if by if switch expansion.
    after_block->func_graph()->set_flag(FUNC_GRAPH_FLAG_AFTER_BLOCK, true);
  }

  // Process the if-true branch
  py::object bodyNode = python_adapter::GetPyObjAttr(node, "body");
  FunctionBlockPtr true_end = ParseStatements(true_block, bodyNode);

  // If the return_ is set ,it has its own continuation block
  if (true_end->func_graph()->get_return() == nullptr) {
    true_end->Jump(after_block, nullptr);
  }

  // Process the orelse branch
  py::object orelseNode = python_adapter::GetPyObjAttr(node, "orelse");
  FunctionBlockPtr false_end = ParseStatements(false_block, orelseNode);

  // If the return_ is set ,it has its own continuation block
  if (false_end->func_graph()->get_return() == nullptr) {
    false_end->Jump(after_block, nullptr);
  }

  block->ConditionalJump(bool_node, true_block, false_block);
  after_block->Mature();
  return after_block;
}

FunctionBlockPtr Parser::ParseWhile(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast While";
  MS_EXCEPTION_IF_NULL(block);
  MS_LOG(INFO) << "Parse while statement";
  FunctionBlockPtr header_block = nullptr;
  FunctionBlockPtr body_block = nullptr;
  FunctionBlockPtr after_block = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceWhileHeader>(block->func_graph()->debug_info()));
    header_block = MakeFunctionBlock(*this);
  }
  {
    TraceGuard guard(std::make_shared<TraceWhileBody>(block->func_graph()->debug_info()));
    body_block = MakeFunctionBlock(*this);
  }
  {
    TraceGuard guard(std::make_shared<TraceWhileAfter>(block->func_graph()->debug_info()));
    after_block = MakeFunctionBlock(*this);
  }

  body_block->AddPrevBlock(header_block);
  after_block->AddPrevBlock(header_block);
  block->Jump(header_block, nullptr);

  py::object test_node = python_adapter::GetPyObjAttr(node, "test");
  AnfNodePtr condition_node = ParseExprNode(header_block, test_node);
  condition_node = header_block->ForceToWhileCond(condition_node);
  body_block->Mature();
  header_block->ConditionalJump(condition_node, body_block, after_block);

  // Parse loop body statements with loop context.
  LoopContext loop_context{&loops_, header_block, nullptr};
  py::object body_node = python_adapter::GetPyObjAttr(node, "body");
  FunctionBlockPtr after_body = ParseStatements(body_block, body_node);
  if (after_body->func_graph()->get_return() == nullptr) {
    after_body->Jump(header_block, nullptr);
  }

  header_block->Mature();
  after_block->Mature();
  auto &end_block = loop_context.EndBlock();
  if (end_block) {
    // end_block exists if we encounter 'break' in loop body.
    after_block->Jump(end_block, nullptr);
    end_block->Mature();
    return end_block;
  }
  // No 'break', no end_block.
  return after_block;
}

CNodePtr Parser::GenerateIteratorInFor(const FunctionBlockPtr &block, const py::object &node,
                                       const AnfNodePtr &op_iter) {
  py::object iter_node = python_adapter::GetPyObjAttr(node, "iter");
  AnfNodePtr iter_anf_node = ParseExprNode(block, iter_node);
  return block->func_graph()->NewCNodeInOrder({op_iter, iter_anf_node});
}

CNodePtr Parser::GenerateCondInFor(const ParameterPtr &iter_param, const FunctionBlockPtr &header_block,
                                   const AnfNodePtr &op_hasnext) {
  MS_EXCEPTION_IF_NULL(header_block);
  return header_block->func_graph()->NewCNodeInOrder({op_hasnext, iter_param});
}

FunctionBlockPtr Parser::GenerateBlockInFor(const TraceInfoPtr &trace_info) {
  TraceGuard trace_guard(trace_info);
  FunctionBlockPtr body_block = MakeFunctionBlock(*this);
  return body_block;
}

int64_t GetForTransToWhileLoop() {
  static const auto loop_str = common::GetEnv("ENV_FOR_TO_WHILE_LOOP");
  // int64 support 63bits positive num mostly.
  if (loop_str.size() > 63 || loop_str.empty()) {
    return MAX_FOR_LOOP_COUNT;
  }
  if (std::any_of(loop_str.begin(), loop_str.end(), [](char c) { return c < '0' || c > '9'; })) {
    return MAX_FOR_LOOP_COUNT;
  }
  int64_t loop_count;
  std::stringstream ss;
  ss << loop_str;
  ss >> loop_count;
  return loop_count;
}
// A for loop will generate 3 functions :the test, the body, and the continuation
// for x in xs:
//    body
// It is compiled to be following statement
// if len(xs) < max_loop_cnt:
//    ParseForIter()  // use iter to implement for loop, which always unroll loop
// else:
//    ParseForLoop()  // use loop var to implement for loop, which always sink loop
FunctionBlockPtr Parser::ParseFor(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast For, create an if else statement";
  MS_EXCEPTION_IF_NULL(block);
  // Create statement 'len(xs) < MAX_FOR_LOOP_COUNT'
  AnfNodePtr op_len = block->MakeResolveSymbol(NAMED_PRIMITIVE_LEN);
  py::object iter_obj = python_adapter::GetPyObjAttr(node, NAMED_PRIMITIVE_ITER);
  AnfNodePtr iter_node = ParseExprNode(block, iter_obj);
  CNodePtr len_iter = block->func_graph()->NewCNodeInOrder({op_len, iter_node});
  CNodePtr bool_node = block->func_graph()->NewCNodeInOrder(
    {NewValueNode(prim::kPrimScalarLt), len_iter, NewValueNode(GetForTransToWhileLoop())});

  // Create statement 'if len(xs) < prim::MAX_FOR_LOOP_COUNT then ParseForIter else ParseForLoop'
  FunctionBlockPtr true_block = nullptr;
  FunctionBlockPtr false_block = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceIfStmtTrueBranch>(block->func_graph()->debug_info()));
    true_block = MakeFunctionBlock(*this);
  }
  {
    TraceGuard guard(std::make_shared<TraceIfStmtFalseBranch>(block->func_graph()->debug_info()));
    false_block = MakeFunctionBlock(*this);
  }

  MakeConditionBlocks(block, true_block, false_block);

  FunctionBlockPtr after_block = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceIfStmtAfterBranch>(block->func_graph()->debug_info()));
    after_block = MakeFunctionBlock(*this);
  }

  FunctionBlockPtr true_end = ParseForIter(true_block, node);
  true_end->Jump(after_block, nullptr);

  FunctionBlockPtr false_end = ParseForLoop(false_block, node);
  false_end->Jump(after_block, nullptr);

  block->ConditionalJump(bool_node, true_block, false_block);
  after_block->Mature();
  return after_block;
}

// A for loop will generate 3 functions :the test, the body, and the continuation
// for x in xs:
//    body
// It is compiled to be following statement
// it = iter(xs)
// while hastnext(it)
//    x, it = next(it)
//    body
FunctionBlockPtr Parser::ParseForIter(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast For";
  MS_EXCEPTION_IF_NULL(block);
  AnfNodePtr op_iter = block->MakeResolveOperation(NAMED_PRIMITIVE_ITER);
  AnfNodePtr op_next = block->MakeResolveOperation(NAMED_PRIMITIVE_NEXT);
  AnfNodePtr op_getitem = block->MakeResolveOperation(NAMED_PRIMITIVE_GETITEM);
  AnfNodePtr op_hasnext = block->MakeResolveOperation(NAMED_PRIMITIVE_HASNEXT);
  // Generate the iterator apply
  CNodePtr iter_apply = GenerateIteratorInFor(block, node, op_iter);
  MS_EXCEPTION_IF_NULL(iter_apply);
  FunctionBlockPtr header_block =
    GenerateBlockInFor(std::make_shared<TraceForHeader>(block->func_graph()->debug_info()));
  MS_EXCEPTION_IF_NULL(header_block);
  // Generate the hasnext apply which is a condition
  ParameterPtr iter_param = header_block->func_graph()->add_parameter();
  CNodePtr cond_apply = GenerateCondInFor(iter_param, header_block, op_hasnext);
  // Generate the body of the for statement
  FunctionBlockPtr body_block = GenerateBlockInFor(std::make_shared<TraceForBody>(block->func_graph()->debug_info()));
  MS_EXCEPTION_IF_NULL(body_block);
  body_block->AddPrevBlock(header_block);
  // Generate the iterator next apply
  // Process as following: `app = next(it); target = app[0]; it = app[1];`
  CNodePtr app = body_block->func_graph()->NewCNodeInOrder({op_next, iter_param});
  CNodePtr target_app =
    body_block->func_graph()->NewCNodeInOrder({op_getitem, app, NewValueNode(static_cast<int64_t>(0))});
  py::object target_node = python_adapter::GetPyObjAttr(node, "target");

  CNodePtr iter2_app =
    body_block->func_graph()->NewCNodeInOrder({op_getitem, app, NewValueNode(static_cast<int64_t>(1))});
  WriteAssignVars(body_block, target_node, target_app);

  // Link the variable name with the target
  auto it_info = std::make_shared<TraceIterator>(target_app->debug_info());
  iter_param->debug_info()->set_trace_info(it_info);
  iter2_app->debug_info()->set_trace_info(it_info);
  iter_apply->debug_info()->set_trace_info(it_info);

  FunctionBlockPtr after_block = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceForAfter>(block->func_graph()->debug_info()));
    after_block = MakeFunctionBlock(*this);
  }
  MS_EXCEPTION_IF_NULL(after_block);
  after_block->AddPrevBlock(header_block);

  block->Jump(header_block, iter_apply);
  body_block->Mature();
  header_block->ConditionalJump(cond_apply, body_block, after_block);

  // Parse loop body statements with loop context.
  LoopContext loop_context{&loops_, header_block, iter2_app};
  py::object body_node = python_adapter::GetPyObjAttr(node, "body");
  FunctionBlockPtr after_body_block = ParseStatements(body_block, body_node);
  if (after_body_block->func_graph()->get_return() == nullptr) {
    after_body_block->Jump(header_block, iter2_app);
  }

  header_block->Mature();
  after_block->Mature();
  auto &end_block = loop_context.EndBlock();
  if (end_block) {
    // end_block exists if we encounter 'break' in loop body.
    after_block->Jump(end_block, nullptr);
    end_block->Mature();
    return end_block;
  }
  // No 'break', no end_block.
  return after_block;
}

// A for loop will generate 3 functions :the test, the body, and the continuation
// for x in xs:
//    body
// It is compiled to be following statement
// i = 0
// while i < len(xs)
//    x = xs[i]
//    i = i + 1
//    body
FunctionBlockPtr Parser::ParseForLoop(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast For by loop variable";
  MS_EXCEPTION_IF_NULL(block);
  AnfNodePtr op_len = block->MakeResolveSymbol(NAMED_PRIMITIVE_LEN);
  AnfNodePtr op_getitem = block->MakeResolveOperation(NAMED_PRIMITIVE_GETITEM);

  // Get variable name of 'x' in statement 'for x in xs'
  py::object target_node = python_adapter::GetPyObjAttr(node, "target");

  // Create statement 'len(xs)'
  py::object iter_obj = python_adapter::GetPyObjAttr(node, "iter");
  AnfNodePtr iter_node = ParseExprNode(block, iter_obj);
  MS_EXCEPTION_IF_NULL(iter_node);
  // Generate node for loop count and convert it to tensor, to make the loop not unroll
  CNodePtr scalar_len = block->func_graph()->NewCNodeInOrder({op_len, iter_node});
  auto scalar_to_tensor = prim::GetPythonOps("ScalarToTensor", "mindspore.ops.operations");
  auto scalar_to_tensor_node = block->func_graph()->NewCNodeInOrder({NewValueNode(scalar_to_tensor)});

  CNodePtr len_iter = block->func_graph()->NewCNodeInOrder({scalar_to_tensor_node, scalar_len});

  FunctionBlockPtr header_block =
    GenerateBlockInFor(std::make_shared<TraceForHeader>(block->func_graph()->debug_info()));
  MS_EXCEPTION_IF_NULL(header_block);
  // Create loop variable 'i'
  ParameterPtr loop_var = header_block->func_graph()->add_parameter();
  // Create loop condition 'i < len(xs)'
  auto prim_less = prim::GetPythonOps("Less", "mindspore.ops.operations");
  auto less_node = header_block->func_graph()->NewCNodeInOrder({NewValueNode(prim_less)});
  CNodePtr cond_node = header_block->func_graph()->NewCNodeInOrder({less_node, loop_var, len_iter});

  // Generate the body of the for statement
  FunctionBlockPtr body_block = GenerateBlockInFor(std::make_shared<TraceForBody>(block->func_graph()->debug_info()));
  MS_EXCEPTION_IF_NULL(body_block);
  body_block->AddPrevBlock(header_block);
  // Create 'x = xs[i]'
  CNodePtr target_var = body_block->func_graph()->NewCNodeInOrder({op_getitem, iter_node, loop_var});
  WriteAssignVars(body_block, target_node, target_var);
  // Create 'i = i + 1'
  CNodePtr loop_var_inc = body_block->func_graph()->NewCNodeInOrder(
    {NewValueNode(prim::kPrimScalarAdd), loop_var, NewValueNode(static_cast<int64_t>(1))});
  body_block->WriteVariable(loop_var->name(), loop_var_inc);

  // Link the variable name with the target
  auto it_info = std::make_shared<TraceIterator>(loop_var_inc->debug_info());
  loop_var->debug_info()->set_trace_info(it_info);
  len_iter->debug_info()->set_trace_info(it_info);

  FunctionBlockPtr after_block = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceForAfter>(block->func_graph()->debug_info()));
    after_block = MakeFunctionBlock(*this);
  }
  MS_EXCEPTION_IF_NULL(after_block);
  after_block->AddPrevBlock(header_block);

  block->Jump(header_block, NewValueNode(static_cast<int64_t>(0)));
  body_block->Mature();

  header_block->ConditionalJump(cond_node, body_block, after_block, false);

  // Parse loop body statements with loop context.
  LoopContext loop_context{&loops_, header_block, loop_var_inc};
  py::object body_node = python_adapter::GetPyObjAttr(node, "body");
  FunctionBlockPtr after_body_block = ParseStatements(body_block, body_node);
  if (after_body_block->func_graph()->get_return() == nullptr) {
    after_body_block->Jump(header_block, loop_var_inc);
  }

  header_block->Mature();
  after_block->Mature();
  auto &end_block = loop_context.EndBlock();
  if (end_block) {
    // end_block exists if we encounter 'break' in loop body.
    after_block->Jump(end_block, nullptr);
    end_block->Mature();
    return end_block;
  }
  // No 'break', no end_block.
  return after_block;
}

AnfNodePtr Parser::ParseIfExp(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast IfExp";
  MS_EXCEPTION_IF_NULL(block);
  py::object test_node = python_adapter::GetPyObjAttr(node, "test");
  AnfNodePtr condition_node = ParseExprNode(block, test_node);
  CNodePtr bool_node = block->ForceToBoolNode(condition_node);

  FunctionBlockPtr true_block = nullptr;
  FunctionBlockPtr false_block = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceIfExpTrueBranch>(block->func_graph()->debug_info()));
    true_block = MakeFunctionBlock(*this);
  }
  {
    TraceGuard guard(std::make_shared<TraceIfExpFalseBranch>(block->func_graph()->debug_info()));
    false_block = MakeFunctionBlock(*this);
  }

  MakeConditionBlocks(block, true_block, false_block);

  // Process the if-true branch
  py::object bodyNode = python_adapter::GetPyObjAttr(node, "body");
  true_block->func_graph()->debug_info()->set_location(GetLocation(bodyNode));
  AnfNodePtr true_node = ParseExprNode(true_block, bodyNode);

  // Process the orelse branch
  py::object orelseNode = python_adapter::GetPyObjAttr(node, "orelse");
  false_block->func_graph()->debug_info()->set_location(GetLocation(orelseNode));
  AnfNodePtr false_node = ParseExprNode(false_block, orelseNode);

  true_block->func_graph()->set_output(true_node);
  false_block->func_graph()->set_output(false_node);

  // Use the Primitive replace the operation resolve node (switch),
  // because the switch will eventually be converted to Primitive node
  CNodePtr switch_app = block->func_graph()->NewCNodeInOrder({NewValueNode(prim::kPrimSwitch), bool_node,
                                                              NewValueNode(true_block->func_graph()),
                                                              NewValueNode(false_block->func_graph())});

  std::vector<AnfNodePtr> call_graph_nodes{switch_app};
  CNodePtr switch_app_call = block->func_graph()->NewCNodeInOrder(call_graph_nodes);
  return switch_app_call;
}

void Parser::HandleAssignName(const FunctionBlockPtr &block, const py::object &targ, const AnfNodePtr &assigned_node) {
  MS_EXCEPTION_IF_NULL(block);
  MS_EXCEPTION_IF_NULL(assigned_node);
  py::str name = python_adapter::GetPyObjAttr(targ, "id");
  std::string name_id = name;
  assigned_node->debug_info()->set_name(name_id);
  // Set the debug name of the constant graph
  if (IsValueNode<FuncGraph>(assigned_node)) {
    // The value should be graph
    auto fg = GetValueNode<FuncGraphPtr>(assigned_node);
    if (fg->debug_info()->name().empty()) {
      fg->debug_info()->set_name(name_id);
    }
  }
  block->WriteVariable(name_id, assigned_node);
}

void Parser::HandleAssignTuple(const FunctionBlockPtr &block, const py::object &targ, const AnfNodePtr &assigned_node) {
  MS_EXCEPTION_IF_NULL(block);
  AnfNodePtr op_getitem = block->MakeResolveOperation(NAMED_PRIMITIVE_GETITEM);
  py::list items = python_adapter::GetPyObjAttr(targ, "elts");
  for (size_t i = 0; i < items.size(); i++) {
    // Use the Primitive replace the operation resolve node (getitem),
    // because the getitem will eventually be converted to Primitive node
    CNodePtr item_apply =
      block->func_graph()->NewCNodeInOrder({op_getitem, assigned_node, NewValueNode(static_cast<int64_t>(i))});

    py::object elt = items[i];
    WriteAssignVars(block, elt, item_apply);
  }
}

void Parser::HandleAssignClassMember(const FunctionBlockPtr &block, const py::object &targ,
                                     const AnfNodePtr &assigned_node) {
  // Now only support the self.xx = xxxxx, can't support x.y = xxxx
  AnfNodePtr target_node = ParseExprNode(block, targ);
  MS_EXCEPTION_IF_NULL(target_node);

  auto attr_name = targ.attr("attr").cast<std::string>();
  std::string var_name = "self." + attr_name;

  // Now only support the self.xxx = yyy, where self.xxx must be a defined Parameter type
  if (!py::hasattr(ast()->obj(), common::SafeCStr(attr_name))) {
    MS_EXCEPTION(TypeError) << "'" << var_name << "' should be a Parameter, but not defined.";
  }
  auto obj = ast()->obj().attr(common::SafeCStr(attr_name));
  auto obj_type = obj.attr("__class__").attr("__name__");
  if (!py::hasattr(obj, "__parameter__")) {
    MS_EXCEPTION(TypeError) << "'" << var_name << "' should be a Parameter, but got '"
                            << py::str(obj).cast<std::string>() << "' with type '"
                            << py::str(obj_type).cast<std::string>() << ".";
  }

  MS_EXCEPTION_IF_NULL(block);
  MS_LOG(DEBUG) << "SetState write " << var_name << " : " << target_node->ToString();
  block->SetStateAssign(target_node, assigned_node);
}

void Parser::HandleAssignSubscript(const FunctionBlockPtr &block, const py::object &targ,
                                   const AnfNodePtr &assigned_node) {
  MS_EXCEPTION_IF_NULL(block);
  AnfNodePtr op_setitem = block->MakeResolveOperation(NAMED_PRIMITIVE_SETITEM);
  py::object value_obj = python_adapter::GetPyObjAttr(targ, "value");
  py::object slice_obj = python_adapter::GetPyObjAttr(targ, "slice");
  AnfNodePtr value_node = ParseExprNode(block, value_obj);
  AnfNodePtr slice_node = ParseExprNode(block, slice_obj);
  CNodePtr setitem_app = block->func_graph()->NewCNodeInOrder({op_setitem, value_node, slice_node, assigned_node});
  // Getitem apply should return the sequence data structure itself
  std::string var_name;
  if (ast_->IsClassMember(value_obj)) {
    auto attr_name = value_obj.attr("attr").cast<std::string>();
    var_name = "self." + attr_name;
    if (!py::hasattr(ast()->obj(), common::SafeCStr(attr_name))) {
      MS_EXCEPTION(TypeError) << "'" << var_name << "' was not defined in the class '__init__' function.";
    }
    auto obj = ast()->obj().attr(common::SafeCStr(attr_name));
    auto obj_type = obj.attr("__class__").attr("__name__");
    if (!py::hasattr(obj, "__parameter__")) {
      MS_EXCEPTION(TypeError) << "'" << var_name << "' should be a Parameter, but got '"
                              << py::str(obj).cast<std::string>() << "' with type '"
                              << py::str(obj_type).cast<std::string>() << "'.";
    }
    block->WriteVariable(var_name, setitem_app);
    return;
  }
  if (AstSubType(py::cast<int32_t>(ast_->CallParseModFunction(PYTHON_PARSE_GET_AST_TYPE, value_obj))) ==
      AST_SUB_TYPE_SUBSCRIPT) {
    HandleAssignSubscript(block, value_obj, setitem_app);
    return;
  }
  if (!py::hasattr(value_obj, "id")) {
    MS_EXCEPTION(TypeError) << "Attribute id not found in " << py::str(value_obj).cast<std::string>();
  }
  var_name = value_obj.attr("id").cast<std::string>();
  block->WriteVariable(var_name, setitem_app);
}

void Parser::WriteAssignVars(const FunctionBlockPtr &block, const py::object &targ, const AnfNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  MS_LOG(DEBUG) << "Process WriteAssignVars";
  auto ast_type = AstSubType(py::cast<int32_t>(ast_->CallParseModFunction(PYTHON_PARSE_GET_AST_TYPE, targ)));
  if (ast_type == AST_SUB_TYPE_NAME) {
    HandleAssignName(block, targ, value_node);
  } else if (ast_type == AST_SUB_TYPE_TUPLE) {
    HandleAssignTuple(block, targ, value_node);
  } else if (ast_type == AST_SUB_TYPE_SUBSCRIPT) {
    HandleAssignSubscript(block, targ, value_node);
  } else if (ast_->IsClassMember(targ)) {
    HandleAssignClassMember(block, targ, value_node);
  } else if (ast_type == AST_SUB_TYPE_ATTRIBUTE) {
    MS_LOG(EXCEPTION) << "The subnet attributes cannot be changed in the network"
                      << " NodeInfo: " << trace::GetDebugInfo(value_node->debug_info());
  } else {
    MS_LOG(EXCEPTION) << "Not supported assign type: " << ast_type
                      << " NodeInfo: " << trace::GetDebugInfo(value_node->debug_info());
  }
}

// Process a assign statement, such as a =b,  a,b = tup
FunctionBlockPtr Parser::ParseAssign(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast assign";
  py::object value_object = python_adapter::GetPyObjAttr(node, "value");
  AnfNodePtr value_node = ParseExprNode(block, value_object);
  py::object targets_object = python_adapter::GetPyObjAttr(node, "targets");
  py::int_ pcount = python_adapter::CallPyObjMethod(targets_object, "__len__");
  size_t count = LongToSize(pcount);
  MS_LOG(DEBUG) << "The nodes count is " << count;
  for (size_t i = 0; i < count; i++) {
    auto target_node = py::cast<py::list>(targets_object)[i];
    WriteAssignVars(block, target_node, value_node);
  }

  return block;
}

FunctionBlockPtr Parser::ParseBreak(const FunctionBlockPtr &block, const py::object &node) {
  if (loops_.empty()) {
    // Report error if loop context not set for the 'break' statement.
    MS_LOG(EXCEPTION) << "Unexpected 'break'.";
  }
  // Get current loop.
  Loop &loop = loops_.top();
  if (loop.end == nullptr) {
    // Create end_block if it is not existed.
    TraceGuard trace_guard(std::make_shared<TraceLoopEnd>(block->func_graph()->debug_info()));
    loop.end = MakeFunctionBlock(*this);
  }
  // Jump to the end_block.
  block->Jump(loop.end, nullptr);
  return block;
}

FunctionBlockPtr Parser::ParseContinue(const FunctionBlockPtr &block, const py::object &node) {
  if (loops_.empty()) {
    // Report error if loop context not set for the 'continue' statement.
    MS_LOG(EXCEPTION) << "Unexpected 'continue.";
  }
  // Jump to the header of the loop with iterator called.
  Loop &loop = loops_.top();
  block->Jump(loop.header, loop.iterator);
  return block;
}

FunctionBlockPtr Parser::ParsePass(const FunctionBlockPtr &block, const py::object &node) {
  // We just bypass 'pass' statement.
  return block;
}

AnfNodePtr FindPhis(const std::unordered_map<ParameterPtr, AnfNodePtr> &removable_phis, const AnfNodePtr &node) {
  const auto &inp = node->cast<ParameterPtr>();
  const auto &iter = removable_phis.find(inp);
  if (iter == removable_phis.end()) {
    return node;
  }
  return FindPhis(removable_phis, iter->second);
}

void Parser::RemoveUnnecessaryPhis() {
  // Merge all removable phis to one map;
  std::unordered_map<ParameterPtr, AnfNodePtr> removable_phis;
  std::vector<ParameterPtr> phis;
  for (FunctionBlockPtr &block : func_block_list_) {
    MS_EXCEPTION_IF_NULL(block);
    removable_phis.insert(block->removable_phis().begin(), block->removable_phis().end());
    std::transform(block->removable_phis().begin(), block->removable_phis().end(), std::back_inserter(phis),
                   [](const std::pair<ParameterPtr, AnfNodePtr> &pair) { return pair.first; });
  }
  if (removable_phis.empty()) {
    return;
  }
  auto fg_name = func_graph_->ToString();
  auto mng = Manage(func_graph_, false);
  // Replace the nodes
  // Remove from inside to outside
  for (int64_t idx = SizeToLong(phis.size() - 1); idx >= 0; idx--) {
    auto phi = phis[LongToSize(idx)];
    auto new_node = FindPhis(removable_phis, phi);
    mng->Replace(phi, new_node);
  }
  // Remove the parameter
  for (FunctionBlockPtr &block : func_block_list_) {
    MS_EXCEPTION_IF_NULL(block);
    auto &local_removable_phis = block->removable_phis();
    if (local_removable_phis.empty()) {
      continue;
    }
    auto func_graph = block->func_graph();
    auto &parameters = func_graph->parameters();
    std::vector<AnfNodePtr> new_parameters(parameters.size());
    auto it = std::copy_if(
      parameters.begin(), parameters.end(), new_parameters.begin(), [&local_removable_phis](const AnfNodePtr &param) {
        return local_removable_phis.find(param->cast<ParameterPtr>()) == local_removable_phis.end();
      });

    // Shrink container to new size
    new_parameters.resize(std::distance(new_parameters.begin(), it));
    func_graph->set_parameters(new_parameters);
  }
}

// ParseAst class code
bool ParseAst::InitParseAstInfo(const std::string &python_mod_get_parse_method) {
  // Init the type
  target_type_ = PARSE_TARGET_UNKNOW;

  // Call python parse, get the parser fn
  module_ = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::object parse_method = python_adapter::GetPyObjAttr(obj_, PYTHON_EXTERN_PARSE_METHOD);

  // Get the obj type
  auto type = data_converter::GetObjType(obj_);
  if (type == RESOLVE_TYPE_FUNCTION) {
    target_type_ = PARSE_TARGET_FUNCTION;
    function_ = obj_;
  } else if (type == RESOLVE_TYPE_METHOD) {
    // Process the method ,need get the method's self obj
    target_type_ = PARSE_TARGET_METHOD;
    py::object method_object = python_adapter::GetPyObjAttr(obj_, PYTHON_GET_METHOD_SELF_CLASS);
    if (py::isinstance<py::none>(method_object)) {
      MS_LOG(ERROR) << "Get method's self object instance failed.";
      return false;
    }
    target_type_ = PARSE_TARGET_OBJECT_INSTANCE;
    function_ = obj_;
    obj_ = method_object;
  } else if (type == RESOLVE_TYPE_CLASS_INSTANCE) {
    // obj is class instance, get the method to parse.
    function_ = python_adapter::CallPyModFn(module_, python_mod_get_parse_method, obj_, parse_method);
    if (py::isinstance<py::none>(function_)) {
      MS_LOG(ERROR) << "Get obj method function failed.";
      return false;
    }
    target_type_ = PARSE_TARGET_OBJECT_INSTANCE;
    // Check the fn is method
    auto obj_type = data_converter::GetObjType(function_);
    if (obj_type != RESOLVE_TYPE_METHOD) {
      MS_LOG(WARNING) << "Parse method function is invalid.";
      return false;
    }
  } else {
    MS_LOG(WARNING) << "Parse obj is invalid, only can parse function and obj, type = " << type;
    return false;
  }

  // Call python parse get ast tree
  parser_ = python_adapter::CallPyModFn(module_, PYTHON_MOD_PARSE_OBJECT_FUNCTION, function_, parse_method);
  ast_tree_ = python_adapter::CallPyObjMethod(parser_, "parse");

  // Get fn name and module
  function_module_ = py::cast<std::string>(python_adapter::GetPyObjAttr(parser_, "function_module"));
  function_name_ = py::cast<std::string>(python_adapter::GetPyObjAttr(parser_, "function_name"));
  function_filename_ = py::cast<std::string>(python_adapter::GetPyObjAttr(parser_, "filename"));
  function_line_offset_ = py::cast<int64_t>(python_adapter::GetPyObjAttr(parser_, "line_offset"));

  return true;
}

// Get ast tree node : is the tree bode list[0]
py::object ParseAst::GetAstNode() {
  py::list tree_body = python_adapter::GetPyObjAttr(ast_tree_, "body");
  py::object ast_node = tree_body[0];
  return ast_node;
}

py::list ParseAst::GetArgs(const py::object &func_node) {
  py::list ret = python_adapter::CallPyModFn(module_, PYTHON_PARSE_GET_ARGS, func_node);
  return ret;
}

py::list ParseAst::GetArgsDefaultValues(const py::object &func_node) {
  py::list ret = python_adapter::CallPyModFn(module_, PYTHON_PARSE_GET_ARGS_DEFAULT_VALUES, func_node);
  return ret;
}

AstNodeTypePtr ParseAst::GetNodeType(const py::object &node) {
  py::list list_value = python_adapter::CallPyModFn(module_, PYTHON_PARSE_GET_NODE_TYPE, node);
  if (list_value.size() < 2) {
    MS_LOG(ERROR) << "The node of python method must has 2 values.";
    return nullptr;
  }
  auto node_name = py::cast<std::string>(list_value[0]);
  auto type = AstMainType(py::cast<int32_t>(list_value[1]));
  return std::make_shared<AstNodeType>(node, node_name, type);
}

AstSubType ParseAst::GetOpType(const py::object &node) {
  auto op_type = AstSubType(python_adapter::CallPyModFn(module_, PYTHON_PARSE_GET_AST_TYPE, node).cast<int32_t>());
  return op_type;
}

bool ParseAst::IsClassMember(const py::object &node) {
  py::object ret = CallParseModFunction(PYTHON_MOD_PARSE_CHECK_IS_CLASS_MEMBER, node);
  if (!py::isinstance<py::bool_>(ret)) {
    MS_LOG(ERROR) << "The result of mod function parse, should be bool type.";
    return false;
  }
  return ret.cast<bool>();
}

bool UpdateFuncGraphFlags(const py::object &obj, const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "FuncGraph is null";
    return false;
  }

  if (!py::hasattr(obj, PYTHON_EXTERN_MINDSPORE_FLAG)) {
    MS_LOG(DEBUG) << "No flags";
    return true;
  }
  py::dict flags = python_adapter::GetPyObjAttr(obj, PYTHON_EXTERN_MINDSPORE_FLAG);
  for (auto &item : flags) {
    if (!py::isinstance<py::str>(item.first)) {
      MS_LOG(ERROR) << "Type error in flags dict convert";
      return false;
    }
    auto name = py::cast<std::string>(item.first);
    if (py::isinstance<py::bool_>(item.second)) {
      auto value = py::cast<bool>(item.second);
      MS_LOG(DEBUG) << "Flag name: " << name << ". Value: " << value;
      func_graph->set_flag(name, value);
    } else if (py::isinstance<py::str>(item.second)) {
      auto value = py::cast<std::string>(item.second);
      MS_LOG(DEBUG) << "Flag name: " << name << ". Value: " << value;
      func_graph->set_attr(name, MakeValue(value));
    } else {
      MS_LOG(ERROR) << "Type error in flags/attrs dict convert";
      return false;
    }
  }
  return true;
}

// Generate and copy a ValueNode, or a CNode with its child nodes
static AnfNodePtr CopyNodesFromParamDefaultValue(const FuncGraphPtr &func_graph, const AnfNodePtr &param_node) {
  MS_EXCEPTION_IF_NULL(param_node);
  if (param_node->isa<ValueNode>()) {
    return std::make_shared<ValueNode>(param_node->cast<ValueNodePtr>()->value());
  }

  // Parameter default value is CNode.
  std::size_t index = 0;
  std::vector<AnfNodePtr> old_cnodes;
  old_cnodes.emplace_back(param_node);
  auto res = func_graph->NewCNodeInOrder({});
  std::vector<CNodePtr> new_cnodes;
  new_cnodes.emplace_back(res);
  while (index < old_cnodes.size()) {
    auto current = old_cnodes[index];
    auto current_new_cnode = new_cnodes[index];
    index++;
    MS_EXCEPTION_IF_NULL(current);
    if (current->isa<CNode>()) {
      auto &inputs = current->cast<CNodePtr>()->inputs();
      for (auto it = inputs.begin(); it != inputs.end(); it++) {
        AnfNodePtr input = *it;
        if (input != nullptr && input->isa<CNode>()) {
          old_cnodes.emplace_back(input);
          auto new_cnode = func_graph->NewCNodeInOrder({});
          new_cnodes.emplace_back(new_cnode);
          current_new_cnode->add_input(new_cnode);
        } else if (input->isa<ValueNode>()) {
          current_new_cnode->add_input(std::make_shared<ValueNode>(input->cast<ValueNodePtr>()->value()));
        } else {
          MS_LOG(EXCEPTION) << "Wrong type item in default parameters: " << input->ToString();
        }
      }
    }
  }
  return res;
}

FuncGraphPtr MakeTopGraph(const py::object &cell, const ValuePtr &cell_ptr) {
  auto current_graph = dyn_cast<FuncGraph>(cell_ptr);
  if (current_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Current graph cast failed from " << cell_ptr->ToString();
  }

  auto func_graph = std::make_shared<FuncGraph>();
  func_graph->debug_info()->set_name(current_graph->debug_info()->name() + "_wrapper");

  // Copy all parameters information
  for (auto &para : current_graph->parameters()) {
    auto param = func_graph->add_parameter();
    auto orig_param = para->cast<ParameterPtr>();
    auto name = orig_param->name();
    param->set_name(name);
    param->debug_info()->set_name(name);
  }
  func_graph->set_has_vararg(current_graph->has_vararg());
  func_graph->set_has_kwarg(current_graph->has_kwarg());
  func_graph->set_kwonlyargs_count(current_graph->kwonlyargs_count());
  // Copy all default values
  for (auto &d : current_graph->parameter_default_value()) {
    func_graph->set_param_default_value(d.first, CopyNodesFromParamDefaultValue(func_graph, d.second));
  }

  // cell_obj
  MS_LOG(DEBUG) << "add Flag for " << std::string(py::str(cell));
  parse::UpdateFuncGraphFlags(cell, func_graph);
  // Top graph's construct flag
  if (py::hasattr(cell, "construct")) {
    parse::UpdateFuncGraphFlags(cell.attr("construct"), func_graph);
  }

  auto unpacking = func_graph->has_vararg() || func_graph->has_kwarg();
  if (!unpacking) {
    std::vector<AnfNodePtr> inputs;
    inputs.emplace_back(NewValueNode(cell_ptr));
    auto &params = func_graph->parameters();
    (void)std::transform(params.begin(), params.end(), std::back_inserter(inputs),
                         [](AnfNodePtr node) -> AnfNodePtr { return node; });
    func_graph->set_output(func_graph->NewCNodeInOrder(inputs));
  } else {
    // ret = cell_obj(*arg, *kwargs)
    auto call_fn = MakeUnpackCall(func_graph, NewValueNode(cell_ptr), func_graph->parameters());
    // Set output as ret
    func_graph->set_output(call_fn);
  }
  return func_graph;
}
}  // namespace parse
}  // namespace mindspore
