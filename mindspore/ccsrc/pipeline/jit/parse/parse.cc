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

#include "pipeline/jit/parse/parse.h"

#include <utility>
#include <string>
#include <memory>
#include <sstream>
#include <algorithm>
#include <stack>
#include "utils/hash_map.h"
#include "pipeline/jit/parse/resolve.h"
#include "pipeline/jit/parse/data_converter.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/composite.h"
#include "utils/ms_context.h"
#include "utils/log_adapter.h"
#include "utils/interpret_node_recorder.h"
#include "pipeline/jit/debug/trace.h"
#include "mindspore/core/ir/cell.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/python_adapter.h"

namespace mindspore {
namespace parse {
FuncGraphPtr ParsePythonCode(const py::object &obj, const std::string &python_mod_get_parse_method) {
  (void)python_adapter::set_python_scoped();
  py::gil_scoped_acquire gil;

  if (!obj || py::isinstance<py::none>(obj)) {
    MS_LOG(ERROR) << "Parse the python code failed, obj is nullptr or none";
    return nullptr;
  }

  auto ast = std::make_shared<ParseFunctionAst>(obj);
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

FuncGraphWeakPtr Parser::top_func_graph_ = FuncGraphWeakPtr();

Parser::Parser(const std::shared_ptr<ParseFunctionAst> &ast)
    : ast_(ast), errcode_(PARSE_SUCCESS), support_fallback_(common::GetEnv("MS_DEV_ENABLE_FALLBACK")) {
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
  stmt_method_map_["Raise"] = &Parser::ParseRaise;
  stmt_method_map_["Assert"] = &Parser::ParseAssert;
  stmt_method_map_["With"] = &Parser::ParseWith;
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
  expr_method_map_["ListComp"] = &Parser::ParseListComp;
  expr_method_map_["GeneratorExp"] = &Parser::ParseListComp;  // We treat 'GeneratorExp' the same as 'ListComp'.
  expr_method_map_["JoinedStr"] = &Parser::ParseJoinedStr;
  expr_method_map_["FormattedValue"] = &Parser::ParseFormattedValue;
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

void Parser::CheckFuncReturn(const FuncGraphPtr &fn) {
  // Check whether the functions referred by this function and itself are missing 'return' statement
  auto manager = Manage(fn, false);
  MS_EXCEPTION_IF_NULL(ast_);
  for (const auto &func_graph : manager->func_graphs()) {
    MS_EXCEPTION_IF_NULL(func_graph);
    if (func_graph->get_return() != nullptr) {
      continue;
    }
    py::object node = ast_->GetAstNode();
    const auto &location = GetLocation(node);
    py::str desc = python_adapter::CallPyModFn(ast_->module(), PYTHON_MOD_GET_OBJECT_DESCRIPTION, ast_->function(),
                                               location->file_name(), location->line());
    MS_LOG(WARNING) << "Function must has 'return' statement, but missing in " << desc.cast<std::string>()
                    << ". FuncGraph: " << func_graph->ToString()
                    << ". We will add a 'return None' statement automatically.";
    // If the def function has no return statement, mean that return none.
    auto none_node = NewValueNode(kNone);
    auto return_node = func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimReturn), none_node});
    func_graph->set_return(return_node);
  }
}

std::vector<std::pair<CNodePtr, size_t>> GetFreeVariable(const FuncGraphPtr &func_graph) {
  // Considering the performance, we didn't use Manager here.
  std::vector<std::pair<CNodePtr, size_t>> free_variables;
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> nodes =
    TopoSort(func_graph->get_return(), SuccIncoming, [&func_graph](const AnfNodePtr &node) -> IncludeType {
      MS_EXCEPTION_IF_NULL(node);
      // Not follow FV's inputs.
      if (node->func_graph() != nullptr && node->func_graph() != func_graph) {
        return NOFOLLOW;
      }
      return FOLLOW;
    });
  for (auto &node : nodes) {
    // Only check Non-FV CNode.
    auto cnode = dyn_cast<CNode>(node);
    if (cnode == nullptr || cnode->func_graph() != func_graph) {
      continue;
    }

    for (size_t i = 0; i < cnode->inputs().size(); ++i) {
      auto &input = cnode->input(i);
      if (input->func_graph() != nullptr && input->func_graph() != func_graph) {
        (void)free_variables.emplace_back(std::make_pair(cnode, i));
        constexpr auto recur_2 = 2;
        MS_LOG(DEBUG) << "Found FV: input[" << i << "] of " << cnode->DebugString(recur_2);
      }
    }
  }
  return free_variables;
}

void Parser::LiftRolledBodyGraphFV() {
  for (auto &rolled_call_pair : rolled_body_calls_) {
    auto rolled_call_cnode = rolled_call_pair.first;
    auto rolled_graph = rolled_call_pair.second->func_graph();
    MS_EXCEPTION_IF_NULL(rolled_graph);
    const auto &free_variables = GetFreeVariable(rolled_graph);
    for (auto &free_node_pair : free_variables) {
      auto &cnode = free_node_pair.first;
      auto index = free_node_pair.second;
      // Move the free variable to parent.
      auto &free_node = cnode->input(index);
      rolled_call_cnode->add_input(free_node);
      // Change the free variable to the parameter.
      auto parameter = rolled_graph->add_parameter();
      cnode->set_input(index, parameter);
      constexpr auto recur_2 = 2;
      MS_LOG(DEBUG) << "Change FV: " << cnode->DebugString(recur_2);
    }
  }
}

void Parser::LiftIfBranchGraphFV() {
  for (auto &branch_call_tuple : if_branch_calls_) {
    auto call_cnode = std::get<0>(branch_call_tuple);
    auto true_branch_graph = std::get<1>(branch_call_tuple)->func_graph();
    MS_EXCEPTION_IF_NULL(true_branch_graph);
    auto false_branch_graph = std::get<2>(branch_call_tuple)->func_graph();
    MS_EXCEPTION_IF_NULL(false_branch_graph);
    const auto &true_free_variables = GetFreeVariable(true_branch_graph);
    const auto &false_free_variables = GetFreeVariable(false_branch_graph);
    // Handle true branch.
    for (auto &free_node_pair : true_free_variables) {
      auto &cnode = free_node_pair.first;
      MS_EXCEPTION_IF_NULL(cnode);
      auto index = free_node_pair.second;
      // Move the free variable to parent.
      auto &free_node = cnode->input(index);
      call_cnode->add_input(free_node);
      // Change the free variable to the parameter.
      auto parameter = true_branch_graph->add_parameter();
      cnode->set_input(index, parameter);
      // Add a unused parameter in other branch.
      (void)false_branch_graph->add_parameter();
      constexpr auto recur_2 = 2;
      MS_LOG(DEBUG) << "True branch, change FV: " << cnode->DebugString(recur_2);
    }
    // Handle false branch.
    for (auto &free_node_pair : false_free_variables) {
      auto &cnode = free_node_pair.first;
      MS_EXCEPTION_IF_NULL(cnode);
      auto index = free_node_pair.second;
      // Move the free variable to parent.
      auto &free_node = cnode->input(index);
      call_cnode->add_input(free_node);
      // Change the free variable to the parameter.
      auto parameter = false_branch_graph->add_parameter();
      cnode->set_input(index, parameter);
      // Add a unused parameter in other branch.
      (void)true_branch_graph->add_parameter();
      constexpr auto recur_2 = 2;
      MS_LOG(DEBUG) << "False branch, change FV: " << cnode->DebugString(recur_2);
    }
  }
}

namespace {
bool IsDependOfIsolatedNodes(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimDepend)) {
    return false;
  }
  auto cnode = dyn_cast<CNode>(node);
  MS_EXCEPTION_IF_NULL(cnode);
  auto attr_sort_rhs_first = cnode->GetAttr(kAttrTopoSortRhsFirst);
  auto sort_rhs_first =
    attr_sort_rhs_first != nullptr && attr_sort_rhs_first->isa<BoolImm>() && GetValue<bool>(attr_sort_rhs_first);
  return sort_rhs_first;
}

std::pair<CNodePtr, AnfNodePtr> GetRealOutputNodes(const FuncGraphPtr &call_graph) {
  MS_EXCEPTION_IF_NULL(call_graph);
  auto graph_output = call_graph->output();
  if (graph_output == nullptr) {
    MS_LOG(EXCEPTION) << "graph_output is null, call_graph: " << call_graph->ToString();
  }
  auto graph_output_cnode = dyn_cast<CNode>(graph_output);
  MS_EXCEPTION_IF_NULL(graph_output_cnode);
  // If output cnode is not the tail call but a Depend CNode, keep the dependency node for later use.
  AnfNodePtr graph_dependency_node = nullptr;
  if (IsDependOfIsolatedNodes(graph_output_cnode)) {
    auto graph_real_output_cnode = dyn_cast<CNode>(graph_output_cnode->input(1));
    // Get the dependency node;
    constexpr auto dependency_node_index = 2;
    graph_dependency_node = graph_output_cnode->input(dependency_node_index);
    MS_EXCEPTION_IF_NULL(graph_real_output_cnode);
    graph_output_cnode = graph_real_output_cnode;
  }
  return {graph_output_cnode, graph_dependency_node};
}

void TransformParallelCallFormerToMiddle(const FuncGraphPtr &former_call_graph, const FuncGraphPtr &latter_call_graph,
                                         size_t middle_graph_output_cnode_size, bool use_arguments_pack) {
  // The 'former_graph_output' is middle graph call or depend.
  const auto &[former_graph_output_cnode, former_graph_dependency_node] = GetRealOutputNodes(former_call_graph);
  MS_EXCEPTION_IF_NULL(former_graph_output_cnode);
  MS_EXCEPTION_IF_NULL(former_call_graph);
  std::vector<AnfNodePtr> inputs({NewValueNode(latter_call_graph)});
  if (use_arguments_pack) {
    for (size_t i = 0; i < middle_graph_output_cnode_size - 1; ++i) {
      auto getitem_input = former_call_graph->NewCNodeInOrder(
        {NewValueNode(prim::kPrimTupleGetItem), former_graph_output_cnode, NewValueNode(SizeToLong(i))});
      (void)inputs.emplace_back(getitem_input);
    }
  } else {
    (void)inputs.emplace_back(former_graph_output_cnode);
  }
  auto new_output = former_call_graph->NewCNodeBefore(former_call_graph->return_node(), std::move(inputs));
  if (former_graph_dependency_node != nullptr) {
    // Adjust the former funcgraph output with Depend.
    new_output = former_call_graph->NewCNodeAfter(
      new_output, {NewValueNode(prim::kPrimDepend), new_output, former_graph_dependency_node});
    // Origin dependency_node has this attribute(refer to function IsDependOfIsolatedNodes), so we keep it.
    new_output->AddAttr(kAttrTopoSortRhsFirst, MakeValue(true));
  }
  former_call_graph->set_output(new_output);
}

bool TransformParallelCallMiddleToLatter(const FuncGraphPtr &middle_call_graph,
                                         const CNodePtr &middle_graph_output_cnode,
                                         const AnfNodePtr &middle_graph_dependency_node,
                                         size_t middle_graph_output_cnode_size) {
  MS_EXCEPTION_IF_NULL(middle_graph_output_cnode);
  MS_EXCEPTION_IF_NULL(middle_call_graph);
  bool use_arguments_pack = false;
  constexpr auto output_inputs_num = 2;
  AnfNodePtr new_middle_graph_output = nullptr;
  if (middle_graph_output_cnode_size == output_inputs_num) {  // Only one argument.
    new_middle_graph_output = middle_graph_output_cnode->input(1);
  } else {  // More than one argument, pack them with tuple.
    use_arguments_pack = true;
    middle_graph_output_cnode->set_input(0, NewValueNode(prim::kPrimMakeTuple));
    new_middle_graph_output = middle_graph_output_cnode;
  }
  // Adjust the middle funcgraph output with Depend.
  if (middle_graph_dependency_node != nullptr) {
    new_middle_graph_output = middle_graph_output_cnode->func_graph()->NewCNode(
      {NewValueNode(prim::kPrimDepend), new_middle_graph_output, middle_graph_dependency_node});
  }
  middle_call_graph->set_output(new_middle_graph_output);
  return use_arguments_pack;
}

bool IsValueContainScalar(const ValuePtr &value) {
  if (value->isa<Scalar>()) {
    return true;
  }
  return false;
}

bool IsOutputContainScalar(const CNodePtr &output_cnode) {
  return std::any_of(output_cnode->inputs().cbegin() + 1, output_cnode->inputs().end(), [](const AnfNodePtr &node) {
    if (node->isa<ValueNode>()) {
      auto value_node = node->cast<ValueNodePtr>();
      return IsValueContainScalar(value_node->value());
    }
    return false;
  });
}

bool CheckMiddleGraphOutputContainScalar(
  const std::vector<std::pair<FunctionBlockPtr, FunctionBlockPtr>> &parallel_call_vec) {
  std::vector<bool> contains_scalar;
  for (auto &call_graphs_pair : parallel_call_vec) {
    MS_EXCEPTION_IF_NULL(call_graphs_pair.second);
    auto middle_call_graph = call_graphs_pair.second->func_graph();
    constexpr auto recur_2 = 2;
    const auto &middle_graph_output_pair = GetRealOutputNodes(middle_call_graph);
    const auto middle_graph_output_cnode = middle_graph_output_pair.first;
    MS_EXCEPTION_IF_NULL(middle_graph_output_cnode);
    auto middle_graph_output_cnode_size = middle_graph_output_cnode->inputs().size();
    if (middle_graph_output_cnode_size <= 1) {
      MS_LOG(DEBUG) << "CNode's inputs size should exceed 1, " << middle_graph_output_cnode->DebugString(recur_2);
      return false;
    }

    static const auto transform_if_const_scalar = (common::GetEnv("MS_DEV_IF_PARALLEL_CALL") == "2");
    if (!transform_if_const_scalar && IsOutputContainScalar(middle_graph_output_cnode)) {
      MS_LOG(DEBUG) << "CNode's inputs contain const scalar, " << middle_graph_output_cnode->DebugString(recur_2);
      contains_scalar.push_back(true);
    } else {
      contains_scalar.push_back(false);
    }
  }

  return std::all_of(contains_scalar.cbegin(), contains_scalar.cend(), [](bool is_scalar) { return is_scalar; });
}

bool CheckMiddleGraphOutputPyInterpret(
  const std::vector<std::pair<FunctionBlockPtr, FunctionBlockPtr>> &parallel_call_vec) {
  bool contain_py_interpret = false;
  for (auto &call_graphs_pair : parallel_call_vec) {
    MS_EXCEPTION_IF_NULL(call_graphs_pair.second);
    auto middle_call_graph = call_graphs_pair.second->func_graph();
    constexpr auto recur_2 = 2;
    const auto &middle_graph_output_pair = GetRealOutputNodes(middle_call_graph);
    const auto middle_graph_output_cnode = middle_graph_output_pair.first;
    MS_EXCEPTION_IF_NULL(middle_graph_output_cnode);
    auto middle_graph_output_cnode_size = middle_graph_output_cnode->inputs().size();
    if (middle_graph_output_cnode_size <= 1) {
      MS_LOG(DEBUG) << "CNode's inputs size should exceed 1, " << middle_graph_output_cnode->DebugString(recur_2);
      return false;
    }
    bool exist_interpret =
      std::any_of(middle_graph_output_cnode->inputs().cbegin() + 1, middle_graph_output_cnode->inputs().cend(),
                  [](const AnfNodePtr &node) { return IsPrimitiveCNode(node, prim::kPrimPyInterpret); });
    contain_py_interpret |= exist_interpret;
    if (contain_py_interpret) {
      return true;
    }
  }

  return false;
}
}  // namespace

// Transform tail call to parallel call.
void Parser::TransformParallelCall() {
  mindspore::HashSet<FuncGraphPtr> latter_call_graphs_set;
  for (auto &parallel_call_vec : parallel_call_graphs_) {
    bool all_middle_graphs_output_scalar = CheckMiddleGraphOutputContainScalar(parallel_call_vec);
    if (all_middle_graphs_output_scalar) {
      MS_LOG(DEBUG) << "All middle func graph's output contain const scalar, cannot transform to Parallel_If.";
      continue;
    }
    // After Join, Value in Abstract of PyInterpret CNode will be kAnyValue, it cannot be PyInterpreted again, so
    // ignore the transformation.
    bool is_middle_graphs_output_py_interpret = CheckMiddleGraphOutputPyInterpret(parallel_call_vec);
    if (is_middle_graphs_output_py_interpret) {
      MS_LOG(DEBUG) << "Middle func graph's output contain PyInterpret CNode, cannot transform to Parallel_If.";
      continue;
    }
    for (auto &call_graphs_pair : parallel_call_vec) {
      MS_EXCEPTION_IF_NULL(call_graphs_pair.first);
      auto former_call_graph = call_graphs_pair.first->func_graph();
      MS_EXCEPTION_IF_NULL(call_graphs_pair.second);
      auto middle_call_graph = call_graphs_pair.second->func_graph();
      // Transform the call of {middle_graph -> latter_graph}.
      auto middle_graph_return = middle_call_graph->get_return();
      if (middle_graph_return == nullptr) {
        MS_LOG(INFO) << "middle_graph_return is null, middle_call_graph: " << middle_call_graph->ToString();
        continue;
      }
      constexpr auto recur_3 = 3;
      constexpr auto recur_2 = 2;
      MS_LOG(DEBUG) << "Tail call graphs return: {former: " << former_call_graph->get_return()->DebugString(recur_3)
                    << ", middle: " << middle_call_graph->get_return()->DebugString(recur_3) << "}";
      const auto &[middle_graph_output_cnode, middle_graph_dependency_node] = GetRealOutputNodes(middle_call_graph);
      auto middle_graph_output_cnode_size = middle_graph_output_cnode->inputs().size();
      if (middle_graph_output_cnode_size <= 1) {
        MS_LOG(DEBUG) << "CNode's inputs size should exceed 1, " << middle_graph_output_cnode->DebugString(recur_2);
        continue;
      }

      auto latter_graph_node = middle_graph_output_cnode->input(0);
      bool use_arguments_pack = TransformParallelCallMiddleToLatter(
        middle_call_graph, middle_graph_output_cnode, middle_graph_dependency_node, middle_graph_output_cnode_size);

      // Transform the call of {former_graph -> middle_graph}.
      auto latter_call_graph = GetValueNode<FuncGraphPtr>(latter_graph_node);
      if (latter_call_graph == nullptr) {
        MS_LOG(ERROR) << "The latter graph node is not FuncGraph, " << latter_graph_node->DebugString(recur_2);
        continue;
      }
      if (latter_call_graphs_set.find(latter_call_graph) != latter_call_graphs_set.end()) {
        MS_LOG(DEBUG) << "The latter graph is handled before, " << latter_call_graph->ToString();
        continue;
      }
      (void)latter_call_graphs_set.emplace(latter_call_graph);
      TransformParallelCallFormerToMiddle(former_call_graph, latter_call_graph, middle_graph_output_cnode_size,
                                          use_arguments_pack);

      MS_LOG(DEBUG) << "Parallel call graphs return: {former: " << former_call_graph->get_return()->DebugString(recur_3)
                    << ", middle: " << middle_call_graph->get_return()->DebugString(recur_3) << "}";
    }
  }

  // Lift inner, then lift outer.
  LiftIfBranchGraphFV();
  LiftRolledBodyGraphFV();
}

FuncGraphPtr Parser::ParseFuncGraph() {
  // Get ast FunctionDef node
  py::object node = ast_->GetAstNode();
  constexpr char function_def_name[] = "FunctionDef";
  constexpr char lambda_name[] = "Lambda";
  FunctionBlockPtr fn_block = nullptr;
  MS_EXCEPTION_IF_NULL(ast_->GetNodeType(node));
  if (ast_->GetNodeType(node)->node_name() == function_def_name) {
    fn_block = ParseDefFunction(node);
  } else {
    auto lambda_node = python_adapter::GetPyObjAttr(node, "value");
    if (py::isinstance<py::none>(lambda_node) || ast_->GetNodeType(lambda_node)->node_name() != lambda_name) {
      MS_EXCEPTION(TypeError) << "Parse Lambda Function Fail. Node type must be Lambda, but got "
                              << ast_->GetNodeType(lambda_node)->node_name() << ".";
    }
    fn_block = ParseLambdaFunction(lambda_node);
  }
  if (errcode() != PARSE_SUCCESS) {
    MS_LOG(ERROR) << "Parse function error, code is " << errcode();
    return nullptr;
  }
  RemoveUnnecessaryPhis();
  MS_EXCEPTION_IF_NULL(fn_block);
  CheckFuncReturn(fn_block->func_graph());
  TransformParallelCall();
  return fn_block->func_graph();
}

// If any mixed precision flag add a cast node after the parameter node.
AnfNodePtr GetMixedPrecisionCastHelp(const FuncGraphPtr &func_graph, const AnfNodePtr &param) {
  MS_EXCEPTION_IF_NULL(func_graph);
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

void Parser::GenerateArgsNodeForFunction(const FunctionBlockPtr &block, const py::object &fn_node) {
  py::object func_args = python_adapter::GetPyObjAttr(fn_node, "args");
  py::object var_arg_node = python_adapter::GetPyObjAttr(func_args, "vararg");
  MS_EXCEPTION_IF_NULL(block);
  auto block_fg = block->func_graph();
  block_fg->set_has_vararg(!py::isinstance<py::none>(var_arg_node));

  py::object kw_arg_node = python_adapter::GetPyObjAttr(func_args, "kwarg");
  block_fg->set_has_kwarg(!py::isinstance<py::none>(kw_arg_node));

  py::list kwonly_args = python_adapter::GetPyObjAttr(func_args, "kwonlyargs");
  block_fg->set_kwonlyargs_count(SizeToInt(kwonly_args.size()));

  MS_EXCEPTION_IF_NULL(ast_);
  py::list args = ast_->GetArgs(fn_node);
  for (std::size_t i = 0; i < args.size(); i++) {
    std::string arg_name = py::cast<std::string>(args[i].attr("arg"));
    if (ast_->target_type() == PARSE_TARGET_OBJECT_INSTANCE) {
      if (arg_name == "self") {
        continue;
      }
    }
    TraceGuard guard(GetLocation(args[i]));
    auto para_node = std::make_shared<Parameter>(block_fg);
    MS_EXCEPTION_IF_NULL(para_node);
    para_node->set_name(arg_name);
    MS_EXCEPTION_IF_NULL(para_node->debug_info());
    para_node->debug_info()->set_name(arg_name);
    block_fg->add_parameter(para_node);
    AnfNodePtr para_after_cast = GetMixedPrecisionCastHelp(block_fg, para_node);
    MS_LOG(DEBUG) << "The arg[" << i << "] is " << arg_name;
    block->WriteVariable(arg_name, para_after_cast);
  }
}

void Parser::GenerateArgsDefaultValueForFunction(const FunctionBlockPtr &block, const py::object &fn_node) {
  MS_EXCEPTION_IF_NULL(block);
  py::list defaults = ast_->GetArgsDefaultValues(fn_node);
  py::list args = ast_->GetArgs(fn_node);
  std::vector<std::string> namelist_for_default_value;
  std::vector<AnfNodePtr> default_values;
  for (std::size_t i = 0; i < args.size(); i++) {
    std::string arg_name = py::cast<std::string>(args[i].attr("arg"));
    if (ast_->target_type() == PARSE_TARGET_OBJECT_INSTANCE) {
      if (arg_name == "self") {
        continue;
      }
    }

    namelist_for_default_value.push_back(arg_name);
    if (i >= defaults.size()) {
      MS_LOG(EXCEPTION) << "Index: " << i << " out of range: " << defaults.size();
    }
    if (py::isinstance<py::none>(defaults[i])) {
      default_values.push_back(NewValueNode(kNull));
    } else {
      AnfNodePtr arg_node = ParseExprNode(block, defaults[i]);
      arg_node = HandleInterpret(block, arg_node, defaults[i]);
      default_values.push_back(arg_node);
    }
  }
  MS_EXCEPTION_IF_NULL(block->func_graph());
  block->func_graph()->SetDefaultValues(namelist_for_default_value, default_values);
}

ScopePtr Parser::GetScopeForParseFunction() {
  ScopePtr scope = ScopeManager::GetInstance().GetCurrentScope();
  if (ast_->target_type() == PARSE_TARGET_OBJECT_INSTANCE) {
    py::object scope_str = python_adapter::CallPyFn(PYTHON_MOD_PARSE_MODULE, PYTHON_PARSE_GET_SCOPE_NAME, ast_->obj());
    if (!py::isinstance<py::none>(scope_str)) {
      auto scope_name = py::cast<std::string>(scope_str);
      scope = std::make_shared<Scope>(scope_name);
    }
  }
  return scope;
}

FunctionBlockPtr Parser::ParseDefFunction(const py::object &node, const FunctionBlockPtr &block) {
  const std::string cell_construct = "construct";
  const std::string cell_scope_name_split_reserve = "-";
  const std::string cell_scope_name_split_no_reserve = "/";
  const std::string cell_name_split = ".";
  ScopePtr scope = GetScopeForParseFunction();
  // The node created in the parsefunction context, will inherit the scope created using scope_guard
  ScopeGuard scope_guard(scope);
  TraceGuard trace_guard(data_converter::GetObjKey(ast_->obj())[0], GetLocation(node));
  FunctionBlockPtr func_block = MakeFunctionBlock(*this);
  if (block != nullptr) {
    func_block->AddPrevBlock(block);
  } else {
    func_graph_ = func_block->func_graph();
  }
  func_block->Mature();
  auto current_fg = func_block->func_graph();
  auto function_name = py::cast<std::string>(python_adapter::GetPyObjAttr(node, "name"));
  MS_LOG(DEBUG) << "The function name is " << function_name;
  // Replace the construct function name with the cell name
  if (function_name == cell_construct) {
    auto find = scope->name().rfind(cell_scope_name_split_reserve);
    if (find != std::string::npos) {
      function_name = scope->name().substr(find + 1);
    } else {
      find = scope->name().rfind(cell_scope_name_split_no_reserve);
      if (find != std::string::npos) {
        function_name = scope->name().substr(find + 1);
      } else {
        function_name = scope->name();
      }
    }
    function_name = cell_construct + cell_name_split + function_name;
    MS_LOG(DEBUG) << scope->name() << " func name:" << function_name;
  }
  MS_EXCEPTION_IF_NULL(current_fg->debug_info());
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
  GenerateArgsNodeForFunction(func_block, node);

  // When parsing the top graph of construct, save the top graph
  if (GetTopFuncGraph() == nullptr) {
    UpdateTopFuncGraph(func_block->func_graph());
  }

  // Save the function node to block
  func_block->WriteVariable(function_name, NewValueNode(current_fg));

  py::object func_obj = python_adapter::GetPyObjAttr(node, "body");
  (void)ParseStatements(func_block, func_obj);
  if (current_fg->get_return() == nullptr) {
    // If the def function has no return statement, mean that return none.
    py::object node = ast_->GetAstNode();
    const auto &location = GetLocation(node);
    py::str desc = python_adapter::CallPyModFn(ast_->module(), PYTHON_MOD_GET_OBJECT_DESCRIPTION, ast_->function(),
                                               location->file_name(), location->line());
    MS_LOG(WARNING) << "Function must has 'return' statement, but missing in " << desc.cast<std::string>()
                    << ". FuncGraph: " << current_fg->ToString()
                    << ". We will add a 'return None' statement automatically.";
    auto none_node = NewValueNode(kNone);
    auto return_node = current_fg->NewCNodeInOrder({NewValueNode(prim::kPrimReturn), none_node});
    current_fg->set_return(return_node);
  }

  // Add unused variables as isolate nodes.
  for (auto &func_block_item : func_block_list_) {
    MS_EXCEPTION_IF_NULL(func_block_item);
    MS_EXCEPTION_IF_NULL(func_block_item->func_graph());
    if (func_block_item->func_graph()->get_return() != nullptr) {
      // Find unused variables.
      func_block_item->FindIsolatedNodes();
      // Attach all isolated nodes.
      func_block_item->AttachIsolatedNodesBeforeReturn();
    }
  }

  GenerateArgsDefaultValueForFunction(func_block, node);
  return func_block;
}

FunctionBlockPtr Parser::ParseLambdaFunction(const py::object &node, const FunctionBlockPtr &block) {
  MS_EXCEPTION_IF_NULL(ast_);
  ScopePtr scope = GetScopeForParseFunction();
  ScopeGuard scope_guard(scope);
  TraceGuard trace_guard(data_converter::GetObjKey(ast_->obj())[0], GetLocation(node));

  FunctionBlockPtr func_block = MakeFunctionBlock(*this);
  if (block != nullptr) {
    func_block->AddPrevBlock(block);
  } else {
    func_graph_ = func_block->func_graph();
  }
  func_block->Mature();
  auto current_fg = func_block->func_graph();

  MS_EXCEPTION_IF_NULL(current_fg);
  auto function_name = ast_->function_name();
  MS_LOG(DEBUG) << "The function name is " << function_name;
  MS_EXCEPTION_IF_NULL(current_fg->debug_info());
  current_fg->debug_info()->set_name(function_name);
  GenerateArgsNodeForFunction(func_block, node);

  py::object body_node = python_adapter::GetPyObjAttr(node, "body");
  AnfNodePtr lambda_body_node = ParseExprNode(func_block, body_node);
  lambda_body_node = HandleInterpret(block, lambda_body_node, body_node);
  current_fg->set_output(lambda_body_node);
  GenerateArgsDefaultValueForFunction(func_block, node);
  return func_block;
}

FunctionBlockPtr Parser::ParseStatements(const FunctionBlockPtr &block, const py::object &nodes) {
  auto node_list = py::cast<py::list>(nodes);
  size_t count = py::len(node_list);
  MS_LOG(DEBUG) << "The nodes count is " << count;
  auto sub_block = block;
  for (size_t i = 0; i < count; ++i) {
    MS_LOG(DEBUG) << "Start parse statement[" << i << "]: " << py::str(node_list[i])
                  << ", block: " << sub_block->ToString();
    auto node = node_list[i];
    // Flag of return statement is set on sub_block inside ParseStatement, so use next_block
    // to store the returned block temporarily.
    auto next_block = ParseStatement(sub_block, node);
    MS_EXCEPTION_IF_NULL(next_block);
    MS_EXCEPTION_IF_NULL(next_block->func_graph());
    // Propagate flag of return statement back;
    if (sub_block != block && sub_block->is_return_statement_inside()) {
      MS_LOG(DEBUG) << "Sub block: " << sub_block->ToString()
                    << " has return statement inside, propagate flag back to block: " << block->ToString();
      block->SetReturnStatementInside();
    }
    // Propagate flag of break or continue statement back;
    if (sub_block != block && sub_block->is_break_continue_statement_inside()) {
      MS_LOG(DEBUG) << "Sub block: " << sub_block->ToString()
                    << " has break or continue statement inside, propagate flag back to block: " << block->ToString();
      block->SetBreakContinueStatementInside();
    }
    sub_block = next_block;
    // Insert appropriate depended items for the function block if it has a return node
    if (sub_block->func_graph()->get_return() != nullptr || sub_block->is_dead_block()) {
      // If break is not the last expr.
      if (i != count - 1) {
        TraceGuard trace_guard(GetLocation(node_list[i + 1]));
        MS_LOG(EXCEPTION) << "Dead code exist, please remove it.";
      }
      // Skip statements after 'return' (or 'break', 'continue').
      break;
    }
  }
  return sub_block;
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
  if (stmt_method_map_.count(node_name) != 0) {
    auto stmt_block = (this->*stmt_method_map_[node_name])(block, node);
    TraceManager::ClearParseOrResolveDebugInfo();
    return stmt_block;
  } else {
    errcode_ = PARSE_NODE_METHOD_UNSUPPORTED;
    MS_LOG(EXCEPTION) << "Unsupported statement '" << node_name
                      << "'.\nMore details please refer to syntax support at https://www.mindspore.cn";
  }
}

AnfNodePtr Parser::ParseExprNode(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast expr.";
  TraceGuard trace_guard(GetLocation(node));
  auto node_type = ast_->GetNodeType(node);
  // Check the node type
  AstMainType node_main_type = node_type->main_type();
  if (node_main_type != AST_MAIN_TYPE_EXPR) {
    errcode_ = PARSE_NODE_TYPE_NO_MATCH;
    MS_LOG(EXCEPTION) << "Node type is error : " << node_main_type;
  }
  // Call the process function
  std::string node_name = node_type->node_name();
  MS_LOG(DEBUG) << "Ast node is " << node_name;
  if (expr_method_map_.count(node_name) != 0) {
    auto expr_node = (this->*expr_method_map_[node_name])(block, node);
    TraceManager::ClearParseOrResolveDebugInfo();
    return expr_node;
  } else {
    errcode_ = PARSE_NODE_METHOD_UNSUPPORTED;
    MS_LOG(EXCEPTION) << "Unsupported expression '" << node_name
                      << "'.\nMore details please refer to syntax support at https://www.mindspore.cn";
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
  if (expand_info.empty()) {
    MS_LOG(EXCEPTION) << "Empty expand_info.";
  }
  auto is_expand = py::cast<bool>(expand_info[0]);
  if (is_expand) {
    // Process the expr statement
    constexpr size_t expect_size = 2;
    if (expand_info.size() < expect_size) {
      MS_LOG(EXCEPTION) << "expand_info size:" << expand_info.size() << " less than " << expect_size << ".";
    }
    py::object value_object = expand_info[1];
    // Make a Expr CNode.
    AnfNodePtr call_node = ParseExprNode(block, value_object);
    if (py::len(expand_info) == expect_size) {
      // Expression that not assigned to any variable.
      // This is usually a call with side effects.
      // e.g.: print(x)
      // We save it as an isolated node.
      auto &no_return_node = call_node;
      MS_LOG(INFO) << "Isolated node found(NoReturn), no_return_node: " << no_return_node->DebugString(2)
                   << ", block: " << block << "/"
                   << (block->func_graph() ? block->func_graph()->ToString() : "FG(Null)")
                   << ", Line: " << trace::GetDebugInfo(no_return_node->debug_info(), "", kSourceLineTipDiscard);
      // Some builtin functions need to be implemented using fallback.
      auto isolated_node = HandleInterpret(block, no_return_node, value_object);
      block->AddIsolatedNode(isolated_node);
    } else {
      // Expand the assign statement,
      // e.g.: x.append(y)  -> x = x.append(y)
      py::object target_node = expand_info[2];
      // Check whether the target_node is class member recursively.
      // e.g.: self.a1.a1.update()
      if (ast_->target_type() == PARSE_TARGET_OBJECT_INSTANCE && ast_->IsClassMemberRecursive(target_node)) {
        // self.x = [xx, xx]
        // self.x.append()
        MS_LOG(DEBUG) << "The variables whose type is not parameter do not support assign operation.";
        block->AddIsolatedNode(call_node);
      } else {
        WriteAssignVars(block, target_node, call_node);
      }
    }
  }
  return block;
}

LocationPtr Parser::GetLocation(const py::object &node) const {
  MS_EXCEPTION_IF_NULL(ast_);
  py::list res = ast_->CallParserObjMethod(PYTHON_PARSE_GET_LOCATION, node);
  constexpr size_t list_size = 6;
  if (res.size() < list_size) {
    MS_LOG(EXCEPTION) << "List size should not be less than 5.";
  }
  constexpr size_t file_name_index = 0;
  constexpr size_t line_index = 1;
  constexpr size_t column_index = 2;
  constexpr size_t line_end_index = 3;
  constexpr size_t column_end_index = 4;
  constexpr size_t expr_src_index = 5;
  // Refer to Location::Location() for each member of res: line, column, line_end, column_end, expr_src.
  auto location =
    std::make_shared<Location>(res[file_name_index].cast<std::string>(), res[line_index].cast<int64_t>(),
                               res[column_index].cast<int64_t>(), res[line_end_index].cast<int64_t>(),
                               res[column_end_index].cast<int64_t>(), res[expr_src_index].cast<std::string>());
  return location;
}

void Parser::MakeConditionBlocks(const FunctionBlockPtr &pre_block, const FunctionBlockPtr &true_block,
                                 const FunctionBlockPtr &false_block) const {
  MS_EXCEPTION_IF_NULL(true_block);
  MS_EXCEPTION_IF_NULL(false_block);
  true_block->AddPrevBlock(pre_block);
  true_block->Mature();

  false_block->AddPrevBlock(pre_block);
  false_block->Mature();

  static const auto use_fallback = (support_fallback() != "0");
  if (use_fallback) {
    true_block->UpdateGlobalPyParam(pre_block->global_py_params());
    false_block->UpdateGlobalPyParam(pre_block->global_py_params());
  }
}

AnfNodePtr Parser::HandelReturnExprNode(const FunctionBlockPtr &block, const AnfNodePtr &return_expr_node,
                                        const py::object &value_object) {
  MS_EXCEPTION_IF_NULL(return_expr_node);
  // The fallback feature is enabled in default.
  static const auto use_fallback = (support_fallback() != "0");
  if (!use_fallback) {
    return return_expr_node;
  }

  // Handle the case of returning tuple.
  py::object obj = python_adapter::GetPyObjAttr(value_object, "elts");
  if (!py::isinstance<py::none>(obj)) {
    auto elts = py::cast<py::tuple>(obj);
    if (!elts.empty()) {
      auto cnode = return_expr_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      // The first input of cnode is MakeTuple.
      if (cnode->size() != elts.size() + 1) {
        MS_LOG(EXCEPTION) << "The size of make_tuple's inputs must be equal to " << (elts.size() + 1) << ".";
      }
      for (size_t i = 0; i < elts.size(); i++) {
        auto input = cnode->input(i + 1);
        if (input->interpret()) {
          auto interpreted_node = HandleInterpret(block, input, elts[i]);
          cnode->set_input(i + 1, interpreted_node);
        }
      }
      return cnode;
    }
  }

  // Handle the case of a single return value.
  return HandleInterpret(block, return_expr_node, value_object);
}

FunctionBlockPtr Parser::ParseReturn(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast return";
  MS_EXCEPTION_IF_NULL(block);
  // Parse the return Statements value.
  py::object value_object = python_adapter::GetPyObjAttr(node, "value");
  AnfNodePtr return_expr_node = ParseExprNode(block, value_object);
  return_expr_node = HandelReturnExprNode(block, return_expr_node, value_object);
  // Create the `return` CNode.
  auto func_graph = block->func_graph();
  CNodePtr return_cnode = func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimReturn), return_expr_node});
  func_graph->set_return(return_cnode);
  MS_LOG(DEBUG) << "Inside the block has return statement, block: " << block->ToString();
  block->SetReturnStatementInside();
  return block;
}

// Process binary operators,eg: `a + b`, `a | b`, etc.
AnfNodePtr Parser::ParseBinOp(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast BinOP";

  MS_EXCEPTION_IF_NULL(block);
  py::object left = python_adapter::GetPyObjAttr(node, "left");
  py::object right = python_adapter::GetPyObjAttr(node, "right");
  py::object op = python_adapter::GetPyObjAttr(node, "op");
  // Create left and right ANF node
  AnfNodePtr left_node = ParseExprNode(block, left);
  if (left_node == nullptr) {
    MS_LOG(EXCEPTION) << "DoBinOp process left node failed: " << errcode();
  }
  left_node = HandleInterpret(block, left_node, left);
  AnfNodePtr right_node = ParseExprNode(block, right);
  if (right_node == nullptr) {
    MS_LOG(EXCEPTION) << "DoBinOp process right node failed:" << errcode();
  }
  right_node = HandleInterpret(block, right_node, right);
  // Resolve the op
  AnfNodePtr op_node = block->MakeResolveAstOp(op);
  // Create apply node
  MS_EXCEPTION_IF_NULL(block->func_graph());
  AnfNodePtr new_node = block->func_graph()->NewCNodeInOrder({op_node, left_node, right_node});
  UpdateInterpretForUserNode(new_node, {left_node, right_node});
  new_node = HandleInterpret(block, new_node, node);
  // Handling % symbol in formatted string values by JIT Fallback.
  // The string AnfNode may be created by ParseJoinedStr or ParseStr.
  // For example, string % var, f"The string is: %s." % str  or "The number is: %d." % num
  static const auto use_fallback = (support_fallback() != "0");
  if (use_fallback) {
    auto op_cnode = op_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(op_cnode);
    const size_t symbol_index = 2;
    if (op_cnode->inputs().size() <= symbol_index) {
      MS_LOG(EXCEPTION) << "Unexpected symbol node:" << op_node->DebugString();
    }
    auto mod_node = op_cnode->input(symbol_index);
    auto symbol = GetValueNode<parse::SymbolPtr>(mod_node);
    // Only support the pattern (string % xxx) by fallback.
    if (symbol != nullptr && symbol->symbol() == "mod") {
      if (IsPrimitiveCNode(left_node, prim::kPrimJoinedStr)) {
        // left_node created by ParseJoinedStr
        auto inputs = left_node->cast<CNodePtr>()->inputs();
        if (inputs.size() <= 1) {
          MS_LOG(EXCEPTION) << "Unexpected maketuple node:" << left_node->DebugString();
        }
        auto str_node = inputs[1];
        if (IsValueNode<StringImm>(str_node)) {
          new_node->set_interpret(true);
          auto new_interpret_node = HandleInterpret(block, new_node, node);
          return new_interpret_node;
        }
      } else if (IsValueNode<StringImm>(left_node)) {
        // left_node created by ParseStr
        new_node->set_interpret(true);
        auto new_interpret_node = HandleInterpret(block, new_node, node);
        return new_interpret_node;
      }
    }
  }

  return new_node;
}

AnfNodePtr Parser::ParseName(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Name";
  auto name_id = py::cast<std::string>(python_adapter::GetPyObjAttr(node, "id"));
  MS_LOG(DEBUG) << "The Name id is " << name_id;
  MS_EXCEPTION_IF_NULL(block);
  static const auto use_fallback = (support_fallback() != "0");
  // The Tensor object will be parsed into an Interpret node. For example, Tensor(0).astype("int32")
  if (block->IsGlobalVar(name_id) || (use_fallback && name_id == "Tensor")) {
    MS_LOG(DEBUG) << "name_id: " << name_id;
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
    errcode_ = PARSE_NODE_TYPE_UNKNOWN;
    MS_EXCEPTION(TypeError) << "Only support 'Number' type of 'int` and 'float', but got type: " << obj.get_type()
                            << " Value:" << py::str(obj);
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
    return NewValueNode(py::cast<bool>(obj));
  } else if (py::isinstance<py::int_>(obj)) {
    MS_LOG(INFO) << "The Constant is int64_t:" << (std::string)py::str(obj);
    return NewValueNode(py::cast<int64_t>(obj));
  } else if (py::isinstance<py::float_>(obj)) {
    MS_LOG(INFO) << "The Constant is float:" << (std::string)py::str(obj);
    return NewValueNode(py::cast<float>(obj));
  } else if (py::isinstance<py::str>(obj)) {
    MS_LOG(INFO) << "The Constant is string:" << (std::string)py::str(obj);
    return NewValueNode(py::cast<std::string>(obj));
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
  }
  // no else actually
  errcode_ = PARSE_NODE_TYPE_UNKNOWN;
  MS_LOG(EXCEPTION) << "Unsupported NameConstant type: " << (std::string)py::str(obj);
}

AnfNodePtr Parser::GenerateMakeTuple(const FunctionBlockPtr &block, const std::vector<AnfNodePtr> &element_nodes) {
  MS_EXCEPTION_IF_NULL(block);
  AnfNodePtr make_tuple_op = block->MakeResolveOperation(NAMED_PRIMITIVE_MAKETUPLE);
  std::vector<AnfNodePtr> make_tuple_nodes;
  make_tuple_nodes.push_back(make_tuple_op);
  (void)std::transform(element_nodes.begin(), element_nodes.end(), std::back_inserter(make_tuple_nodes),
                       [](AnfNodePtr arg) -> AnfNodePtr { return arg; });
  MS_EXCEPTION_IF_NULL(block->func_graph());
  return block->func_graph()->NewCNodeInOrder(std::move(make_tuple_nodes));
}

AnfNodePtr Parser::ParseSuper(const FunctionBlockPtr &block, const py::list &args) {
  MS_EXCEPTION_IF_NULL(block);
  py::object father_class;
  const size_t expect_args_size = 2;
  if (args.empty()) {
    father_class = py::none();
  } else if (args.size() == expect_args_size) {
    father_class = args[0];
    auto arg_type = AstSubType(py::cast<int32_t>(ast_->CallParseModFunction(PYTHON_PARSE_GET_AST_TYPE, args[1])));
    if (arg_type != AST_SUB_TYPE_NAME || py::cast<std::string>(python_adapter::GetPyObjAttr(args[1], "id")) != "self") {
      MS_EXCEPTION(ArgumentError) << "Argument 2 of 'super()' must be 'self', but got '"
                                  << py::cast<std::string>(python_adapter::GetPyObjAttr(args[1], "id")) << "'.";
    }
  } else {
    MS_EXCEPTION(ArgumentError) << "Arguments number of 'super()' should be 0 or 2, but got " << args.size() << ".";
  }
  py::object target_class_instance = ast_->CallParserObjMethod(PYTHON_PARSE_ANALYZE_SUPER, father_class, ast_->obj());
  py::object namespace_var = ast_->CallParseModFunction(PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL, target_class_instance);
  NameSpacePtr name_space = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, namespace_var);
  SymbolPtr symbol = std::make_shared<Symbol>("namespace");
  MS_LOG(DEBUG) << "name_space: " << name_space->ToString() << ", symbol: " << symbol->ToString();
  return block->MakeResolve(name_space, symbol);
}

void Parser::ParseStrInError(const FunctionBlockPtr &block, const py::list &args, std::vector<AnfNodePtr> *str_nodes) {
  for (size_t i = 0; i < args.size(); ++i) {
    AnfNodePtr node = ParseExprNode(block, args[i]);
    node = HandleInterpret(block, node, args[i]);
    (void)str_nodes->emplace_back(node);
  }
}

std::vector<AnfNodePtr> Parser::ParseException(const FunctionBlockPtr &block, const py::list &args,
                                               const std::string &name) {
  auto exception_type_node = NewValueNode(name);
  std::vector<AnfNodePtr> node_inputs = {exception_type_node};
  ParseStrInError(block, args, &node_inputs);
  return node_inputs;
}

std::vector<AnfNodePtr> Parser::ParseRaiseCall(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Call, the current node is raise.";
  // Process function call
  py::object function_ast_node = python_adapter::GetPyObjAttr(node, "func");
  // Process raise ValueError
  if (py::isinstance<py::none>(function_ast_node)) {
    auto name_id = py::cast<std::string>(python_adapter::GetPyObjAttr(node, "id"));
    if (exception_types_map.find(name_id) != exception_types_map.end()) {
      return {NewValueNode(name_id)};
    } else {
      MS_LOG(EXCEPTION) << "Unsupported exception type: " << name_id
                        << ". Raise only support some Python standard exception types: "
                        << SupportedExceptionsToString();
    }
  }

  py::list args = python_adapter::GetPyObjAttr(node, "args");

  auto arg_type =
    AstSubType(py::cast<int32_t>(ast_->CallParseModFunction(PYTHON_PARSE_GET_AST_TYPE, function_ast_node)));
  if (arg_type == AST_SUB_TYPE_NAME) {
    auto name_id = py::cast<std::string>(python_adapter::GetPyObjAttr(function_ast_node, "id"));
    MS_LOG(DEBUG) << "The name of call node is: " << name_id;
    if (exception_types_map.find(name_id) != exception_types_map.end()) {
      return ParseException(block, args, name_id);
    } else {
      MS_LOG(EXCEPTION) << "Unsupported exception type: " << name_id
                        << ". Raise only support some Python standard exception types: "
                        << SupportedExceptionsToString();
    }
  }
  return {};
}

// Process function call, eg : f1(x, y) ...
AnfNodePtr Parser::ParseCall(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Call";
  // Process function call
  py::object function_ast_node = python_adapter::GetPyObjAttr(node, "func");
  py::list args = python_adapter::GetPyObjAttr(node, "args");

  std::string name_id = "";
  auto arg_type =
    AstSubType(py::cast<int32_t>(ast_->CallParseModFunction(PYTHON_PARSE_GET_AST_TYPE, function_ast_node)));
  if (arg_type == AST_SUB_TYPE_NAME) {
    name_id = py::cast<std::string>(python_adapter::GetPyObjAttr(function_ast_node, "id"));
    MS_LOG(DEBUG) << "The name of call node is: " << name_id;
    if (name_id == "super") {
      return ParseSuper(block, args);
    }
  }

  AnfNodePtr call_function_node = ParseExprNode(block, function_ast_node);
  // Function call arguments should be passed in as groups and unpacked later using unpack call
  ArgsContext args_context = ArgsContext();
  ParseArgsInCall(block, args, &args_context);
  ParseKeywordsInCall(block, node, &args_context);

  auto call_cnode = GenerateAnfNodeForCall(block, call_function_node, args_context);
  static const auto use_fallback = (support_fallback() != "0");
  if (!use_fallback) {
    return call_cnode;
  }
  UpdateInterpretForUserNode(call_cnode, call_function_node);
  MS_EXCEPTION_IF_NULL(call_cnode);
  MS_LOG(DEBUG) << "call_cnode: " << call_cnode->DebugString()
                << ", call_function_node: " << call_function_node->DebugString();

  // Process bulitin function, for example, sum(np.array(xx))
  py::tuple namespace_info = ast_->CallParserObjMethod(PYTHON_PARSE_GET_BUILTIN_NAMESPACE_SYMBOL, name_id);
  constexpr size_t namespace_info_size = 4;
  constexpr size_t flag_index = 3;
  if (namespace_info.size() == namespace_info_size) {
    auto syntax_support = namespace_info[flag_index].cast<int32_t>();
    if (syntax_support == SYNTAX_HYBRID_TYPE) {
      // For hybrid type function, such as print, the inputs to the function determine whether the call_cnode is
      // a graph node or the interpret node. If the inputs contain interpret node (not Tensor), the call_cnode will
      // be interpretive executived. Otherwise, call_cnode will be a graph node.
      if (args_context.has_interpret_without_internal) {
        call_cnode->set_interpret(true);
        call_cnode = HandleInterpret(block, call_cnode, node);
      }
      return call_cnode;
    } else if (syntax_support != SYNTAX_SUPPORTED) {
      call_cnode->set_interpret(true);
      call_cnode = HandleInterpret(block, call_cnode, node);
      // For the unsupported type function, if the input to the function contains tensor, the return value of
      // the function should be graph node too.
      if (args_context.has_interpret_internal) {
        call_cnode->set_interpret_internal_type(true);
      }
    }
  }
  return call_cnode;
}

CNodePtr MakeUnpackCall(const FuncGraphPtr &func_graph, const AnfNodePtr &call_function_node,
                        const std::vector<AnfNodePtr> &packed_arguments) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> unpack_call_nodes;
  auto unpack_call_op = NewValueNode(std::make_shared<prim::UnpackCall>(NAMED_METAGRAPH_UNPACKCALL));
  unpack_call_nodes.push_back(unpack_call_op);
  unpack_call_nodes.push_back(call_function_node);
  (void)std::transform(packed_arguments.begin(), packed_arguments.end(), std::back_inserter(unpack_call_nodes),
                       [](AnfNodePtr node) -> AnfNodePtr { return node; });
  CNodePtr unpack_call = func_graph->NewCNodeInOrder(std::move(unpack_call_nodes));
  return unpack_call;
}

AnfNodePtr Parser::GenerateAnfNodeForCall(const FunctionBlockPtr &block, const AnfNodePtr &call_function_node,
                                          const ArgsContext &args_context) const {
  // If there is keyword arguments or starred, using an unpack_call op to unpack the argument
  MS_EXCEPTION_IF_NULL(block);
  if (args_context.need_unpack) {
    return MakeUnpackCall(block->func_graph(), call_function_node, args_context.packed_arguments);
  }
  // else there is no keyword arguments and starred, parsed as normal arguments without unpack
  static const auto use_fallback = (support_fallback() != "0");
  const auto &group_arguments = args_context.group_arguments;
  if (use_fallback && group_arguments.size() == 0 && IsPrimitiveCNode(call_function_node, prim::kPrimPyInterpret)) {
    // call Interpret node is invalid. Do not new call Interpret node.
    // %1 = Interpret_node
    // %2 = %1()
    return call_function_node;
  }
  std::vector<AnfNodePtr> func_call_nodes;
  func_call_nodes.push_back(call_function_node);
  (void)std::transform(group_arguments.begin(), group_arguments.end(), std::back_inserter(func_call_nodes),
                       [](AnfNodePtr node) -> AnfNodePtr { return node; });
  MS_EXCEPTION_IF_NULL(block->func_graph());
  CNodePtr call_anf_node = block->func_graph()->NewCNodeInOrder(std::move(func_call_nodes));
  return call_anf_node;
}

void Parser::ParseArgsInCall(const FunctionBlockPtr &block, const py::list &args, ArgsContext *args_context) {
  MS_LOG(DEBUG) << "Process ast args in call";
  MS_EXCEPTION_IF_NULL(args_context);
  for (size_t i = 0; i < args.size(); i++) {
    auto arg_node = AstSubType(py::cast<int32_t>(ast_->CallParseModFunction(PYTHON_PARSE_GET_AST_TYPE, args[i])));
    if (arg_node == AST_SUB_TYPE_STARRED) {
      if (!args_context->group_arguments.empty()) {
        args_context->packed_arguments.push_back(GenerateMakeTuple(block, args_context->group_arguments));
      }
      args_context->packed_arguments.push_back(ParseExprNode(block, python_adapter::GetPyObjAttr(args[i], "value")));
      args_context->group_arguments.clear();
      args_context->need_unpack = true;
    } else {
      auto node = ParseExprNode(block, args[i]);
      node = HandleInterpret(block, node, args[i]);
      auto internal = node->interpret_internal_type();
      auto interpret_without_internal =
        ((node->interpret() || IsPrimitiveCNode(node, prim::kPrimPyInterpret)) && !internal);
      if (internal) {
        args_context->has_interpret_internal = true;
      } else if (interpret_without_internal) {
        args_context->has_interpret_without_internal = true;
      }
      args_context->group_arguments.push_back(node);
    }
  }
  if (!args_context->group_arguments.empty()) {
    args_context->packed_arguments.push_back(GenerateMakeTuple(block, args_context->group_arguments));
  }
}

void Parser::ParseKeywordsInCall(const FunctionBlockPtr &block, const py::object &node, ArgsContext *args_context) {
  MS_LOG(DEBUG) << "Process ast key words in call";
  py::list keywords = python_adapter::GetPyObjAttr(node, "keywords");
  if (!keywords.empty()) {
    MS_EXCEPTION_IF_NULL(block);
    args_context->need_unpack = true;
    std::vector<AnfNodePtr> keys;
    std::vector<AnfNodePtr> values;
    for (size_t index = 0; index < keywords.size(); index++) {
      auto kw_key = python_adapter::GetPyObjAttr(keywords[index], "arg");
      auto kw_value = python_adapter::GetPyObjAttr(keywords[index], "value");
      if (py::isinstance<py::none>(kw_key)) {
        args_context->packed_arguments.push_back(ParseExprNode(block, kw_value));
      } else {
        auto kw_key_c = kw_key.cast<std::string>();
        keys.push_back(NewValueNode(kw_key_c));
        auto ret_node = ParseExprNode(block, kw_value);
        ret_node = HandleInterpret(block, ret_node, kw_value);
        values.push_back(ret_node);
      }
    }
    if (!keys.empty()) {
      auto keys_tuple = GenerateMakeTuple(block, keys);
      auto values_tuple = GenerateMakeTuple(block, values);
      auto make_dict_op = block->MakeResolveOperation(NAMED_PRIMITIVE_MAKEDICT);
      std::vector<AnfNodePtr> make_dict_nodes = {make_dict_op, keys_tuple, values_tuple};
      MS_EXCEPTION_IF_NULL(block->func_graph());
      args_context->packed_arguments.push_back(block->func_graph()->NewCNodeInOrder(std::move(make_dict_nodes)));
    }
  }
}

AnfNodePtr Parser::ProcessAttributeWithClassMember(const FunctionBlockPtr &block, const py::object &node) const {
  std::string var_name = "self.";
  std::string attr_name = node.attr("attr").cast<std::string>();
  (void)var_name.append(attr_name);
  auto attr_obj = ast()->obj().attr(attr_name.c_str());
  MS_EXCEPTION_IF_NULL(block);
  MS_LOG(DEBUG) << "var_name: " << var_name;
  bool check_need_resolve = py::hasattr(ast()->obj(), attr_name.c_str()) &&
                            (py::hasattr(attr_obj, PYTHON_PRIMITIVE_FLAG) || py::isinstance<py::int_>(attr_obj) ||
                             py::isinstance<py::float_>(attr_obj) || py::isinstance<py::bool_>(attr_obj) ||
                             py::isinstance<py::str>(attr_obj) || data_converter::IsCellInstance(attr_obj));
  if (check_need_resolve) {
    return block->MakeResolveSymbol(var_name);
  }
  auto var_node = block->ReadVariable(var_name);
  // Process numpy array, eg: self.x = np.array([1, 2])
  // The fallback feature is enabled in default.
  static const auto use_fallback = (support_fallback() != "0");
  if (use_fallback && py::hasattr(ast()->obj(), attr_name.c_str()) && data_converter::IsNumpyArrayInstance(attr_obj)) {
    var_node->set_interpret(true);
  }
  return var_node;
}

AnfNodePtr Parser::ParseMsTensor(const FunctionBlockPtr &block, const py::object &node, const py::object &value_body,
                                 const AnfNodePtr &value_node) {
  if (py::hasattr(value_body, "id")) {
    std::string module_name = py::cast<std::string>(python_adapter::GetPyObjAttr(value_body, "id"));
    py::dict global_dict = const_cast<py::dict &>(block->global_py_params());
    if (global_dict.contains(module_name)) {
      py::object module_obj = global_dict[py::str(module_name)];
      std::string module_str = py::cast<std::string>(py::str(module_obj));
      bool the_module_is_mindspore = module_str.find("module 'mindspore'") != std::string::npos;
      if (the_module_is_mindspore) {
        std::string script_text = py::cast<std::string>(ast()->GetAstNodeText(node));
        AnfNodePtr interpret_node = MakeInterpretNode(block, value_node, script_text);
        interpret_node->set_interpret(true);
        interpret_node->set_interpret_internal_type(true);
        return interpret_node;
      }
    }
  }
  return nullptr;
}

AnfNodePtr Parser::ParseNull(const FunctionBlockPtr &block, const py::object &value_body) const {
  if (py::hasattr(value_body, "id")) {
    std::string module_name = py::cast<std::string>(python_adapter::GetPyObjAttr(value_body, "id"));
    py::dict global_dict = const_cast<py::dict &>(block->global_py_params());
    if (global_dict.contains(module_name)) {
      py::object module_obj = global_dict[py::str(module_name)];
      std::string module_str = py::cast<std::string>(py::str(module_obj));
      if (module_str.find("module 'mindspore.common.dtype'") != std::string::npos) {
        return NewValueNode(std::make_shared<TypeNull>());
      }
    }
  }
  return nullptr;
}

// Process call attributes of class type define, eg: x.y()
AnfNodePtr Parser::ParseAttribute(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Attribute";
  // Process class value, eg: self.xx
  if (ast_->target_type() == PARSE_TARGET_OBJECT_INSTANCE) {
    if (ast_->IsClassMember(node)) {
      return ProcessAttributeWithClassMember(block, node);
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
    MS_LOG(EXCEPTION) << "Parse attribute failed";
  }

  // Process the node attr
  auto attr_str = python_adapter::GetPyObjAttr(node, "attr").cast<std::string>();
  MS_LOG(DEBUG) << "node: " << node << ", attr: " << attr_str << ", value: " << value_body;
  // The fallback feature is enabled in default.
  static const auto use_fallback = (support_fallback() != "0");
  if (use_fallback && attr_str == "Tensor") {
    // Process xxx.Tensor() and xxx is mindspore.
    auto ret = ParseMsTensor(block, node, value_body, value_node);
    if (ret != nullptr) {
      return ret;
    }
  }
  if (attr_str == "_null") {
    // For stype._null, return TypeNull value node directly.
    auto ret = ParseNull(block, value_body);
    if (ret != nullptr) {
      return ret;
    }
  }

  MS_EXCEPTION_IF_NULL(block->func_graph());
  // Create the apply node
  AnfNodePtr attr_node = NewValueNode(attr_str);
  auto attr_cnode = block->func_graph()->NewCNodeInOrder({op_node, value_node, attr_node});
  if (use_fallback) {
    // Check whether it is constant, constant does not need interpret.
    auto value_str = py::cast<std::string>(ast()->GetAstNodeText(value_body));
    py::bool_ is_const_value =
      ast()->CallParserObjMethod(PYTHON_PARSE_CHECK_IS_CONSTANT_VALUE, value_str, common::SafeCStr(attr_str));
    auto is_constant = py::cast<bool>(is_const_value);
    if (!is_constant) {
      UpdateInterpretForUserNode(attr_cnode, value_node);
    }
  }
  if (attr_str == "pop") {
    list_pop_target_obj_ = value_body;
  }
  return attr_cnode;
}

// Process comparison expression : a == b. a > b  etc.
AnfNodePtr Parser::ParseCompare(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Compare";

  // For python comparison ,there may be if x>y>5 ,
  // Which there is two ops , but we only support one now
  py::list ops = python_adapter::GetPyObjAttr(node, "ops");
  if (ops.size() != MAX_COMPARISON_OPS_SUPPORTED) {
    MS_EXCEPTION(NotSupportError) << "Only support comparison with 1 operator, but got " << ops.size() << ", which is "
                                  << py::str(ops);
  }

  py::object left = python_adapter::GetPyObjAttr(node, "left");
  py::list comparators = python_adapter::GetPyObjAttr(node, "comparators");
  if (comparators.empty()) {
    MS_LOG(EXCEPTION) << "Comparators can't be empty.";
  }
  AnfNodePtr left_node = ParseExprNode(block, left);
  left_node = HandleInterpret(block, left_node, left);
  AnfNodePtr right_node = ParseExprNode(block, comparators[0]);
  right_node = HandleInterpret(block, right_node, comparators[0]);

  MS_EXCEPTION_IF_NULL(block);
  AnfNodePtr op_node = block->MakeResolveAstOp(ops[0]);
  MS_EXCEPTION_IF_NULL(block->func_graph());
  AnfNodePtr new_node = block->func_graph()->NewCNodeInOrder({op_node, left_node, right_node});
  UpdateInterpretForUserNode(new_node, {left_node, right_node});
  new_node = HandleInterpret(block, new_node, node);
  return new_node;
}

AnfNodePtr Parser::ProcessBoolOpValueList(const FunctionBlockPtr &block, const py::list &value_list, AstSubType mode) {
  // If there is only one bool op now
  MS_EXCEPTION_IF_NULL(block);
  if (value_list.empty()) {
    MS_LOG(EXCEPTION) << "value list is empty.";
  }
  if (value_list.size() == 1) {
    AnfNodePtr first_node = ParseExprNode(block, value_list[0]);
    return first_node;
  } else {
    py::object first = value_list[0];
    py::list rest;
    for (size_t i = 1; i < value_list.size(); i++) {
      rest.append(value_list[i]);
    }
    FunctionBlockPtr true_block = nullptr;
    FunctionBlockPtr false_block = nullptr;
    auto block_fg = block->func_graph();
    {
      TraceGuard guard(std::make_shared<TraceIfExpTrueBranch>(block_fg->debug_info()));
      true_block = MakeFunctionBlock(*this);
    }
    {
      TraceGuard guard(std::make_shared<TraceIfExpFalseBranch>(block_fg->debug_info()));
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
    MS_EXCEPTION_IF_NULL(b1->func_graph());
    MS_EXCEPTION_IF_NULL(b2->func_graph());
    b1->func_graph()->set_output(rest_node);
    b2->func_graph()->set_output(test_node);

    AnfNodePtr cond_node = block->ForceToBoolNode(test_node);
    UpdateInterpretForUserNode(cond_node, test_node);
    cond_node = HandleInterpret(block, cond_node, first);

    auto switch_app =
      block_fg->NewCNodeInOrder({NewValueNode(prim::kPrimSwitch), cond_node, NewValueNode(true_block->func_graph()),
                                 NewValueNode(false_block->func_graph())});

    std::vector<AnfNodePtr> call_graph_nodes{switch_app};
    auto switch_app_call = block_fg->NewCNodeInOrder(std::move(call_graph_nodes));
    UpdateInterpretForUserNode(switch_app_call, {test_node, rest_node});
    return switch_app_call;
  }
}

// Process comparison expression : a and b. a or b .
AnfNodePtr Parser::ParseBoolOp(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast BoolOp";
  py::object op_node = python_adapter::GetPyObjAttr(node, "op");
  AstSubType op_type = ast_->GetOpType(op_node);
  if (op_type == AST_SUB_TYPE_UNKNOWN) {
    MS_LOG(EXCEPTION) << "ProcessBoolOp, got unknown op type";
  }
  py::list op_values = python_adapter::GetPyObjAttr(node, "values");
  return ProcessBoolOpValueList(block, op_values, op_type);
}

// Process a function def
FunctionBlockPtr Parser::ParseFunctionDef(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast FunctionDef";
  FunctionBlockPtr function_block = ParseDefFunction(node, block);
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
  FunctionBlockPtr function_block = ParseLambdaFunction(node, block);
  MS_EXCEPTION_IF_NULL(function_block);

  auto block_fg = function_block->func_graph();
  ValueNodePtr const_graph = NewValueNode(block_fg);
  return const_graph;
}

// Process a tuple
AnfNodePtr Parser::ParseTuple(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Tuple";
  MS_EXCEPTION_IF_NULL(block);
  py::tuple elts = python_adapter::GetPyObjAttr(node, "elts");
  if (elts.empty()) {
    auto empty_tuple = std::vector<ValuePtr>();
    return NewValueNode(std::make_shared<ValueTuple>(empty_tuple));
  }

  std::vector<AnfNodePtr> tuple_vec;
  AnfNodePtr make_tuple_op = block->MakeResolveOperation(NAMED_PRIMITIVE_MAKETUPLE);
  tuple_vec.emplace_back(make_tuple_op);
  for (size_t i = 0; i < elts.size(); i++) {
    AnfNodePtr node_ptr = ParseExprNode(block, elts[i]);
    node_ptr = HandleInterpret(block, node_ptr, elts[i]);
    tuple_vec.emplace_back(node_ptr);
  }
  MS_EXCEPTION_IF_NULL(block->func_graph());
  CNodePtr tuple_app = block->func_graph()->NewCNodeInOrder(std::move(tuple_vec));
  return tuple_app;
}

// Process a list
AnfNodePtr Parser::ParseList(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast List";
  MS_EXCEPTION_IF_NULL(block);
  py::list elts = python_adapter::GetPyObjAttr(node, "elts");
  if (elts.empty()) {
    auto empty_list = std::vector<ValuePtr>();
    return NewValueNode(std::make_shared<ValueList>(empty_list));
  }

  std::vector<AnfNodePtr> list_vec;
  AnfNodePtr make_list_op = block->MakeResolveOperation(NAMED_PRIMITIVE_MAKELIST);
  list_vec.emplace_back(make_list_op);
  for (size_t i = 0; i < elts.size(); i++) {
    AnfNodePtr node_ptr = ParseExprNode(block, elts[i]);
    node_ptr = HandleInterpret(block, node_ptr, elts[i]);
    list_vec.emplace_back(node_ptr);
  }
  MS_EXCEPTION_IF_NULL(block->func_graph());
  CNodePtr list_app = block->func_graph()->NewCNodeInOrder(std::move(list_vec));
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
  value = HandleInterpret(block, value, value_node);
  AnfNodePtr slice = ParseExprNode(block, slice_node);
  slice = HandleInterpret(block, slice, slice_node);
  MS_EXCEPTION_IF_NULL(block->func_graph());
  AnfNodePtr new_node = block->func_graph()->NewCNodeInOrder({op_getitem, value, slice});
  UpdateInterpretForUserNode(new_node, {value, slice});
  new_node = HandleInterpret(block, new_node, node);
  return new_node;
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
  start_node = HandleInterpret(block, start_node, start);
  AnfNodePtr stop_node = ParseExprNode(block, stop);
  stop_node = HandleInterpret(block, stop_node, stop);
  AnfNodePtr step_node = ParseExprNode(block, step);
  step_node = HandleInterpret(block, step_node, step);
  MS_EXCEPTION_IF_NULL(block->func_graph());
  AnfNodePtr slice_node = block->func_graph()->NewCNodeInOrder({op_makeslice, start_node, stop_node, step_node});
  UpdateInterpretForUserNode(slice_node, {start_node, stop_node, step_node});
  slice_node = HandleInterpret(block, slice_node, node);
  return slice_node;
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
  MS_EXCEPTION_IF_NULL(block->func_graph());
  CNodePtr tuple_conde = block->func_graph()->NewCNodeInOrder(std::move(node_vec));
  return tuple_conde;
}

// Process a index, get the index number
AnfNodePtr Parser::ParseIndex(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Index";
  py::object value_node = python_adapter::GetPyObjAttr(node, "value");
  return ParseExprNode(block, value_node);
}

// Process a UnaryOp, +a, -b
AnfNodePtr Parser::ParseUnaryOp(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast UnaryOp";
  py::object op = python_adapter::GetPyObjAttr(node, "op");

  MS_EXCEPTION_IF_NULL(block);
  // Resolve the op
  AnfNodePtr op_node = block->MakeResolveAstOp(op);

  py::object operand = python_adapter::GetPyObjAttr(node, "operand");
  AnfNodePtr operand_node = ParseExprNode(block, operand);
  operand_node = HandleInterpret(block, operand_node, operand);
  MS_EXCEPTION_IF_NULL(block->func_graph());
  auto new_node = block->func_graph()->NewCNodeInOrder({op_node, operand_node});
  UpdateInterpretForUserNode(new_node, operand_node);
  return new_node;
}

// Process a dict ast node expression
AnfNodePtr Parser::ParseDictByKeysAndValues(const FunctionBlockPtr &block, const std::vector<AnfNodePtr> &key_nodes,
                                            const std::vector<AnfNodePtr> &value_nodes) {
  auto keys_tuple = GenerateMakeTuple(block, key_nodes);
  auto values_tuple = GenerateMakeTuple(block, value_nodes);
  MS_EXCEPTION_IF_NULL(block);
  auto make_dict_op = block->MakeResolveOperation(NAMED_PRIMITIVE_MAKEDICT);
  MS_EXCEPTION_IF_NULL(block->func_graph());
  return block->func_graph()->NewCNodeInOrder({make_dict_op, keys_tuple, values_tuple});
}

AnfNodePtr Parser::ParseDict(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Dict";
  py::list keys = node.attr("keys");
  py::list values = node.attr("values");
  std::vector<AnfNodePtr> key_nodes;
  std::vector<AnfNodePtr> value_nodes;
  for (size_t i = 0; i < keys.size(); i++) {
    AnfNodePtr key_node = ParseExprNode(block, keys[i]);
    key_node = HandleInterpret(block, key_node, keys[i]);
    key_nodes.push_back(key_node);
    AnfNodePtr value_node = ParseExprNode(block, values[i]);
    value_node = HandleInterpret(block, value_node, values[i]);
    value_nodes.push_back(value_node);
  }
  return ParseDictByKeysAndValues(block, key_nodes, value_nodes);
}

AnfNodePtr Parser::HandleInterpretForAugassign(const FunctionBlockPtr &block, const AnfNodePtr &augassign_node,
                                               const py::object &op_object, const py::object &target_object,
                                               const py::object &value_object) {
  MS_EXCEPTION_IF_NULL(augassign_node);
  // The fallback feature is enabled in default.
  static const auto use_fallback = (support_fallback() != "0");
  if (!use_fallback || !augassign_node->interpret()) {
    return augassign_node;
  }

  std::string op_text =
    py::cast<std::string>(ast()->CallParseModFunction(PYTHON_PARSE_GET_OPERATION_SYMBOL, op_object));
  // Check the symbol in the Augasssign expression.
  if (op_text.empty()) {
    MS_LOG(EXCEPTION) << "Invalid augasssign operator, only support `+=`, `-=`, `*=`, `/=`, `%=`, `**=`, `//=`, `<<=`, "
                      << "`>>=`, `&=`, `|=`, `^=`.";
  }

  const auto target_text = py::cast<std::string>(ast()->GetAstNodeText(target_object));
  const auto value_text = py::cast<std::string>(ast()->GetAstNodeText(value_object));
  std::string script_text = target_text + op_text + value_text;
  return MakeInterpretNode(block, augassign_node, script_text);
}

// Process a augment assign such as a += b or mat[stride_slice] += b.
FunctionBlockPtr Parser::ParseAugAssign(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast AugAssign";
  MS_EXCEPTION_IF_NULL(block);
  MS_EXCEPTION_IF_NULL(ast_);

  py::object target_object = python_adapter::GetPyObjAttr(node, "target");
  py::object op_object = python_adapter::GetPyObjAttr(node, "op");
  py::object value_object = python_adapter::GetPyObjAttr(node, "value");
  AnfNodePtr target_node = nullptr;
  AnfNodePtr op_node = block->MakeResolveAstOp(op_object);
  AnfNodePtr value_node = ParseExprNode(block, value_object);
  value_node = HandleInterpret(block, value_node, value_object);
  auto ast_type = AstSubType(py::cast<int32_t>(ast_->CallParseModFunction(PYTHON_PARSE_GET_AST_TYPE, target_object)));

  if (ast_type == AST_SUB_TYPE_NAME) {
    target_node = ParseName(block, target_object);
  } else if (ast_type == AST_SUB_TYPE_SUBSCRIPT) {
    target_node = ParseSubscript(block, target_object);
  } else if (ast_->IsClassMember(target_object)) {
    target_node = ParseAttribute(block, target_object);
  } else if (ast_type == AST_SUB_TYPE_ATTRIBUTE) {
    TraceGuard(GetLocation(target_object));
    MS_EXCEPTION(TypeError) << "Only support augassign to attribute of self, but got attribute of "
                            << py::str(target_object.attr("value").attr("id")) << ".\n"
                            << "More details please refer to syntax support at https://www.mindspore.cn";
  } else {
    TraceGuard(GetLocation(target_object));
    MS_EXCEPTION(TypeError) << "Only supported augassign to attribute of self, variable and index value, but got "
                            << target_object.get_type()
                            << ".\nMore details please refer to syntax support at https://www.mindspore.cn";
  }

  if (target_node == nullptr) {
    MS_LOG(EXCEPTION) << "Can not get target node ";
  }
  MS_EXCEPTION_IF_NULL(block->func_graph());
  AnfNodePtr augassign_app = block->func_graph()->NewCNodeInOrder({op_node, target_node, value_node});

  // Check whether the augassign expression needs to be interpreted.
  UpdateInterpretForUserNode(augassign_app, {target_node, value_node});
  augassign_app = HandleInterpretForAugassign(block, augassign_app, op_object, target_object, value_object);

  WriteAssignVars(block, target_object, augassign_app);
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

void Parser::CheckControlFlowAlterationInIf(std::pair<FunctionBlockPtr, FunctionBlockPtr> *branch_graphs_pair,
                                            const FunctionBlockPtr &branch_block, const FunctionBlockPtr &branch_end,
                                            const FunctionBlockPtr &after_block, const FunctionBlockPtr &block) const {
  if (branch_block->is_return_statement_inside()) {
    MS_LOG(DEBUG)
      << "Inside the branch block has return statement, ignore for transformation to parallel-if call, branch block:"
      << branch_block->ToString() << ", block: " << block->ToString();
    block->SetReturnStatementInside();
    return;
  }
  if (branch_block->is_break_continue_statement_inside()) {
    MS_LOG(DEBUG) << "Inside the branch block has break or continue statement, ignore for transformation to "
                     "parallel-if call, branch block: "
                  << branch_block->ToString() << ", branch end: " << branch_end->ToString()
                  << ", block: " << block->ToString();
    MS_LOG(DEBUG) << "Propagate flag of break or continue statement from branch block to block, branch block:"
                  << branch_block->ToString() << ", block: " << block->ToString();
    block->SetBreakContinueStatementInside();
  } else if (branch_end->func_graph()->get_return() != nullptr) {
    // Currently, this can only happen with raise statement inside. As try/expect is not supported now,
    // and contional for raise will be evaluated in Compile time. If raise condition is met, it will
    // cause compile fail, so no need to propagate the flag back.
    MS_LOG(DEBUG) << "Ignore the block as branch_end will not call after_block, branch_block: "
                  << branch_block->ToString() << ", branch_end: " << branch_end->ToString()
                  << ", after_block: " << after_block->ToString();
  } else {
    branch_graphs_pair->second = branch_end;
  }
}

// Process a if statement
FunctionBlockPtr Parser::ParseIf(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast If";
  MS_EXCEPTION_IF_NULL(block);
  py::object test_node = python_adapter::GetPyObjAttr(node, "test");
  AnfNodePtr condition_node = ParseExprNode(block, test_node);
  condition_node = HandleInterpret(block, condition_node, test_node);

  AnfNodePtr bool_node = block->ForceToBoolNode(condition_node);
  UpdateInterpretForUserNode(bool_node, condition_node);
  bool_node = HandleInterpret(block, bool_node, test_node);

  FunctionBlockPtr true_block = nullptr, false_block = nullptr, after_block = nullptr;
  auto block_fg = block->func_graph();
  MS_EXCEPTION_IF_NULL(block_fg);
  {
    TraceGuard guard(std::make_shared<TraceIfStmtTrueBranch>(block_fg->debug_info()));
    true_block = MakeFunctionBlock(*this);
  }
  {
    TraceGuard guard(std::make_shared<TraceIfStmtFalseBranch>(block_fg->debug_info()));
    false_block = MakeFunctionBlock(*this);
  }

  MakeConditionBlocks(block, true_block, false_block);

  {
    TraceGuard guard(std::make_shared<TraceIfStmtAfterBranch>(block_fg->debug_info()));
    after_block = MakeFunctionBlock(*this);
  }

  if (MsContext::GetInstance()->backend_policy() != "ge") {
    // For backends excludes 'ge', it can handle multi graph call, use this flag to
    // generate call not inline `after_block` graph to reduce if by if switch expansion.
    MS_EXCEPTION_IF_NULL(after_block->func_graph());
    after_block->func_graph()->set_flag(FUNC_GRAPH_FLAG_AFTER_BLOCK, true);
  }
  static const auto use_fallback = (support_fallback() != "0");
  // Process the if-true branch
  std::pair<FunctionBlockPtr, FunctionBlockPtr> true_branch_graphs;
  py::object bodyNode = python_adapter::GetPyObjAttr(node, "body");
  FunctionBlockPtr true_end = ParseStatements(true_block, bodyNode);
  MS_EXCEPTION_IF_NULL(true_end->func_graph());
  CheckControlFlowAlterationInIf(&true_branch_graphs, true_block, true_end, after_block, block);
  // If the return_ is set, it has its own continuation block
  if (true_end->func_graph()->get_return() == nullptr) {
    true_end->Jump(after_block, {});
    MS_LOG(DEBUG) << "The true_end block jump to after, true_block: " << true_block->ToString()
                  << ", true_end: " << true_end->ToString() << ", after: " << after_block->ToString();
    if (use_fallback) {
      after_block->UpdateGlobalPyParam(true_end->global_py_params());
    }
  }
  // Process the orelse branch
  std::pair<FunctionBlockPtr, FunctionBlockPtr> false_branch_graphs;
  py::object orelseNode = python_adapter::GetPyObjAttr(node, "orelse");
  FunctionBlockPtr false_end = ParseStatements(false_block, orelseNode);
  MS_EXCEPTION_IF_NULL(false_end->func_graph());
  CheckControlFlowAlterationInIf(&false_branch_graphs, false_block, false_end, after_block, block);
  // If the return_ is set, it has its own continuation block
  if (false_end->func_graph()->get_return() == nullptr) {
    false_end->Jump(after_block, {});
    MS_LOG(DEBUG) << "The false_end block jump to after, false_block: " << false_block->ToString()
                  << ", false_end: " << false_end->ToString() << ", after: " << after_block->ToString();
    if (use_fallback) {
      after_block->UpdateGlobalPyParam(false_end->global_py_params());
    }
  }

  auto switch_app = block->ConditionalJump(bool_node, true_block, false_block);

  // Record the former, middle, latter graphs info.
  static const auto transform_tail_call_to_parallel_call = (common::GetEnv("MS_DEV_IF_PARALLEL_CALL") != "0");
  if (transform_tail_call_to_parallel_call && true_branch_graphs.second != nullptr &&
      false_branch_graphs.second != nullptr) {
    true_branch_graphs.first = block;
    false_branch_graphs.first = block;
    MS_LOG(DEBUG) << "Record tail call {former: " << block->func_graph()->ToString()
                  << ", true middle: " << true_branch_graphs.second->func_graph()->ToString()
                  << ", false middle: " << false_branch_graphs.second->func_graph()->ToString() << "}";
    std::vector<std::pair<FunctionBlockPtr, FunctionBlockPtr>> branch_graphs_vec{true_branch_graphs,
                                                                                 false_branch_graphs};
    (void)parallel_call_graphs_.emplace_back(branch_graphs_vec);
  }

  static const auto transform_for_half_unroll_call = (common::GetEnv("MS_DEV_FOR_HALF_UNROLL") == "1");
  if (transform_for_half_unroll_call) {
    // Lift the if branches in for statement.
    (void)if_branch_calls_.emplace_back(std::make_tuple(switch_app, true_block, false_block));
  }

  if (after_block->prev_blocks().empty()) {
    after_block->SetAsDeadBlock();
  }
  after_block->Mature();
  return after_block;
}

void Parser::CheckReturnInLoop(const FunctionBlockPtr &block, const FunctionBlockPtr &body_block) const {
  // Propagate flag of return statement in body_block back.
  if (body_block->is_return_statement_inside()) {
    MS_LOG(DEBUG) << "Propagate flag of return statement in body_block back, body_block: " << body_block->ToString()
                  << ", block: " << block->ToString();
    block->SetReturnStatementInside();
  }
}

FunctionBlockPtr Parser::ParseWhile(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast While";
  MS_EXCEPTION_IF_NULL(block);
  FunctionBlockPtr header_block = nullptr;
  FunctionBlockPtr body_block = nullptr;
  FunctionBlockPtr after_block = nullptr;
  MS_EXCEPTION_IF_NULL(block->func_graph());
  {
    TraceGuard guard(std::make_shared<TraceWhileHeader>(block->func_graph()->debug_info()));
    header_block = MakeFunctionBlock(*this);
    auto func_graph = header_block->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    func_graph->set_flag(GRAPH_FLAG_IS_WHILE_HEADER, true);
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
  block->Jump(header_block, {});

  py::object test_node = python_adapter::GetPyObjAttr(node, "test");
  static const auto use_fallback = (support_fallback() != "0");
  if (use_fallback) {
    header_block->UpdateGlobalPyParam(block->global_py_params());
    body_block->UpdateGlobalPyParam(block->global_py_params());
    after_block->UpdateGlobalPyParam(block->global_py_params());
  }
  AnfNodePtr condition_node = ParseExprNode(header_block, test_node);
  condition_node = HandleInterpret(header_block, condition_node, test_node);
  AnfNodePtr while_condition_node = header_block->ForceToWhileCond(condition_node);
  UpdateInterpretForUserNode(while_condition_node, condition_node);
  while_condition_node = HandleInterpret(header_block, while_condition_node, test_node);
  (void)header_block->ConditionalJump(while_condition_node, body_block, after_block);

  body_block->Mature();
  // Parse loop body statements with loop context.
  LoopContext loop_context{&loops_, header_block, nullptr};
  py::object body_node = python_adapter::GetPyObjAttr(node, "body");
  FunctionBlockPtr after_body = ParseStatements(body_block, body_node);
  MS_EXCEPTION_IF_NULL(after_body->func_graph());
  if (after_body->func_graph()->get_return() == nullptr) {
    after_body->Jump(header_block, {});
  }
  header_block->Mature();
  after_block->Mature();
  py::object orelse_obj = python_adapter::GetPyObjAttr(node, "orelse");
  if (py::len_hint(orelse_obj) != 0) {
    TraceGuard(GetLocation(orelse_obj));
    MS_LOG(EXCEPTION) << "The 'while...else...' statement is not supported now.";
  }
  auto &end_block = loop_context.EndBlock();
  // end_block exists if we encounter 'break' in loop body.
  if (end_block) {
    after_block->Jump(end_block, {});
    end_block->Mature();
    CheckReturnInLoop(block, body_block);
    return end_block;
  }
  // No 'break', no end_block.
  CheckReturnInLoop(block, body_block);
  return after_block;
}

FunctionBlockPtr Parser::GenerateBlock(const TraceInfoPtr &trace_info) {
  TraceGuard trace_guard(trace_info);
  FunctionBlockPtr block = MakeFunctionBlock(*this);
  MS_EXCEPTION_IF_NULL(block);
  return block;
}

FunctionBlockPtr Parser::ParseFor(const FunctionBlockPtr &block, const py::object &node) {
  // Check for-else
  py::object orelse_obj = python_adapter::GetPyObjAttr(node, "orelse");
  if (py::len_hint(orelse_obj) != 0) {
    TraceGuard(GetLocation(orelse_obj));
    MS_LOG(EXCEPTION) << "The 'for...else...' statement is not supported now.";
  }
  static const auto transform_for_half_unroll_call = (common::GetEnv("MS_DEV_FOR_HALF_UNROLL") == "1");
  if (transform_for_half_unroll_call) {
    return ParseForRepeat(block, node);
  }
  return ParseForUnroll(block, node);
}

AnfNodePtr Parser::ConvertInterpretIterNodeToList(const FunctionBlockPtr &block, const AnfNodePtr &iter_node,
                                                  const py::object &iter_obj) {
  // For interpret iter_node, convert it to list. xs --> list(xs).
  py::object iter_id = python_adapter::GetPyObjAttr(iter_obj, "id");
  if (!py::isinstance<py::none>(iter_id)) {
    // If variable is assigned, for example:
    //     xs = np.array([1, 2, 3, 4])
    //     for x in xs
    const std::string &iter_id_str = iter_id.cast<py::str>();
    return MakeInterpretNode(block, iter_node, "list(" + iter_id_str + ")");
  }
  // If variable is not assigned, for example:
  //     for x in np.array([1, 2, 3, 4])
  const auto &interpret_iter_node =
    IsPrimitiveCNode(iter_node, prim::kPrimPyInterpret) ? iter_node : HandleInterpret(block, iter_node, iter_obj);
  constexpr size_t script_index = 1;
  auto iter_cnode = interpret_iter_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(iter_cnode);
  auto iter_cnode_inputs = iter_cnode->inputs();
  auto iter_script_input = iter_cnode_inputs[script_index];
  if (!IsValueNode<Script>(iter_script_input)) {
    MS_LOG(EXCEPTION) << "The second input to iter node: " << interpret_iter_node->DebugString()
                      << " should be a script value node but got: " << iter_script_input->DebugString() << ".";
  }
  auto script = iter_script_input->cast<ValueNodePtr>();
  auto script_val = script->value()->cast<ScriptPtr>();
  auto script_text = script_val->script();
  auto new_script_val = NewValueNode(std::make_shared<Script>("list(" + script_text + ")"));
  iter_cnode_inputs[script_index] = new_script_val;
  return block->func_graph()->NewCNodeInOrder(iter_cnode_inputs);
}

CNodePtr GenerateInterpretGetItem(const FuncGraphPtr &fg, const AnfNodePtr &iter_node, const AnfNodePtr &loop_var) {
  // Construct global dict node.
  auto global_dict_key = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple)});
  auto global_dict_value = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple)});
  auto global_dict_node = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeDict), global_dict_key, global_dict_value});

  // Construct local dict node.
  auto local_dict_key = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), NewValueNode("x"), NewValueNode("i")});
  auto local_dict_value = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), iter_node, loop_var});
  auto local_dict_node = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeDict), local_dict_key, local_dict_value});

  // Construct script text node.
  auto script = std::make_shared<Script>("x[i]");
  auto script_node = NewValueNode(script);

  return fg->NewCNodeInOrder({NewValueNode(prim::kPrimPyInterpret), script_node, global_dict_node, local_dict_node});
}

// Implement unroll for statement with tuple/getitem.
FunctionBlockPtr Parser::ParseForUnroll(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast For by loop variable";
  MS_EXCEPTION_IF_NULL(block);

  const py::function len_with_check = python_adapter::GetPyFn(kStandardMethodModelName, kMsLenWithCheck);
  auto len_with_check_fg = ParsePythonCode(len_with_check);
  auto op_len_with_check = NewValueNode(len_with_check_fg);
  AnfNodePtr op_getitem = block->MakeResolveOperation(NAMED_PRIMITIVE_GETITEM);
  AnfNodePtr op_iter = block->MakeResolveOperation(NAMED_PRIMITIVE_ITER);

  // Get variable name of 'x' in statement 'for x in xs'
  py::object target_node = python_adapter::GetPyObjAttr(node, "target");

  // Create statement 'len(xs)'
  py::object iter_obj = python_adapter::GetPyObjAttr(node, "iter");
  AnfNodePtr iter_node = ParseExprNode(block, iter_obj);
  MS_EXCEPTION_IF_NULL(iter_node);
  // Generate node for loop count and convert it to tensor, to make the loop not unroll
  if (iter_node->interpret() && !IsPrimitiveCNode(iter_node, prim::kPrimPyInterpret)) {
    iter_node = HandleInterpret(block, iter_node, iter_obj);
  }
  CNodePtr scalar_len = block->func_graph()->NewCNodeInOrder({op_len_with_check, iter_node});
  FunctionBlockPtr header_block = GenerateBlock(std::make_shared<TraceForHeader>(block->func_graph()->debug_info()));
  MS_EXCEPTION_IF_NULL(header_block);
  // Create loop variable 'i'
  ParameterPtr loop_var = header_block->func_graph()->add_parameter();

  std::string less_module_name = "mindspore.ops.composite.multitype_ops.less_impl";
  ValuePtr less_op = prim::GetPythonOps("less", less_module_name);
  CNodePtr cond_node = header_block->func_graph()->NewCNodeInOrder({NewValueNode(less_op), loop_var, scalar_len});

  // Generate the body of the for statement
  FunctionBlockPtr body_block = GenerateBlock(std::make_shared<TraceForBody>(block->func_graph()->debug_info()));
  MS_EXCEPTION_IF_NULL(body_block);
  body_block->AddPrevBlock(header_block);
  // Create 'x = xs[i]'
  auto body_func_graph = body_block->func_graph();
  bool interpret_without_internal =
    IsPrimitiveCNode(iter_node, prim::kPrimPyInterpret) && !iter_node->interpret_internal_type();
  CNodePtr target_var = nullptr;
  if (iter_node->interpret() || interpret_without_internal) {
    target_var = GenerateInterpretGetItem(body_func_graph, iter_node, loop_var);
  } else {
    CNodePtr iterated_node = body_func_graph->NewCNodeInOrder({op_iter, iter_node});
    target_var = body_func_graph->NewCNodeInOrder({op_getitem, iterated_node, loop_var});
  }
  static const auto use_fallback = (support_fallback() != "0");
  if (use_fallback) {
    header_block->UpdateGlobalPyParam(block->global_py_params());
    body_block->UpdateGlobalPyParam(block->global_py_params());
  }
  WriteAssignVars(body_block, target_node, target_var);

  // Create 'i = i + 1'
  std::string add_module_name = "mindspore.ops.composite.multitype_ops.add_impl";
  ValuePtr add_op = prim::GetPythonOps("add", add_module_name);
  CNodePtr loop_var_inc =
    body_func_graph->NewCNodeInOrder({NewValueNode(add_op), loop_var, NewValueNode(static_cast<int64_t>(1))});
  body_block->WriteVariable(loop_var->name(), loop_var_inc);

  // Link the variable name with the target
  auto it_info = std::make_shared<TraceIterator>(loop_var_inc->debug_info());
  loop_var->debug_info()->set_trace_info(it_info);

  FunctionBlockPtr after_block = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceForAfter>(block->func_graph()->debug_info()));
    after_block = MakeFunctionBlock(*this);
  }
  MS_EXCEPTION_IF_NULL(after_block);
  after_block->AddPrevBlock(header_block);
  block->Jump(header_block, {NewValueNode(static_cast<int64_t>(0))});
  body_block->Mature();
  if (use_fallback) {
    after_block->UpdateGlobalPyParam(block->global_py_params());
  }

  (void)header_block->ConditionalJump(cond_node, body_block, after_block);

  // Parse loop body statements with loop context.
  LoopContext loop_context{&loops_, header_block, loop_var_inc};
  py::object body_node = python_adapter::GetPyObjAttr(node, "body");
  FunctionBlockPtr after_body_block = ParseStatements(body_block, body_node);
  if (use_fallback) {
    after_body_block->UpdateGlobalPyParam(block->global_py_params());
  }
  if (after_body_block->func_graph()->get_return() == nullptr) {
    after_body_block->Jump(header_block, {loop_var_inc});
  }

  header_block->Mature();
  after_block->Mature();
  auto &end_block = loop_context.EndBlock();
  if (end_block) {
    // end_block exists if we encounter 'break' in loop body.
    after_block->Jump(end_block, {});
    end_block->Mature();
    CheckReturnInLoop(block, body_block);
    return end_block;
  }
  // No 'break', no end_block.
  CheckReturnInLoop(block, body_block);
  return after_block;
}

// Implement for statement with repeat calling sub graph.
FunctionBlockPtr Parser::ParseForRepeat(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast For by loop variable";
  MS_EXCEPTION_IF_NULL(block);
  FunctionBlockPtr header_block = GenerateBlock(std::make_shared<TraceForHeader>(block->func_graph()->debug_info()));
  MS_EXCEPTION_IF_NULL(header_block);

  // Create statement 'len(xs)'
  py::object iter_obj = python_adapter::GetPyObjAttr(node, "iter");
  AnfNodePtr iter_node = ParseExprNode(block, iter_obj);
  MS_EXCEPTION_IF_NULL(iter_node);
  iter_node = HandleInterpret(header_block, iter_node, iter_obj);
  // Generate node for loop count and convert it to tensor, to make the loop not unroll
  ParameterPtr header_iter_param = header_block->func_graph()->add_parameter();
  AnfNodePtr header_len = header_block->MakeResolveSymbol(NAMED_PRIMITIVE_LEN);
  CNodePtr scalar_len = header_block->func_graph()->NewCNodeInOrder({header_len, header_iter_param});

  // Create loop variable 'i'
  ParameterPtr loop_var = header_block->func_graph()->add_parameter();
  // Create loop condition 'i < len(xs)'
  std::string less_module_name = "mindspore.ops.composite.multitype_ops.less_impl";
  ValuePtr less_op = prim::GetPythonOps("less", less_module_name);
  CNodePtr cond_node = header_block->func_graph()->NewCNodeInOrder({NewValueNode(less_op), loop_var, scalar_len});

  // Generate the body of the for statement
  FunctionBlockPtr body_block = GenerateBlock(std::make_shared<TraceForBody>(block->func_graph()->debug_info()));
  MS_EXCEPTION_IF_NULL(body_block);
  body_block->AddPrevBlock(header_block);
  // Create 'x = xs[i]'
  auto body_func_graph = body_block->func_graph();
  AnfNodePtr body_getitem = body_block->MakeResolveOperation(NAMED_PRIMITIVE_GETITEM);
  CNodePtr target_var = body_func_graph->NewCNodeInOrder({body_getitem, header_iter_param, loop_var});

  static const auto use_fallback = (support_fallback() != "0");
  if (use_fallback) {
    header_block->UpdateGlobalPyParam(block->global_py_params());
    body_block->UpdateGlobalPyParam(block->global_py_params());
  }

  // Get variable name of 'x' in statement 'for x in xs'
  py::object target_node = python_adapter::GetPyObjAttr(node, "target");
  WriteAssignVars(body_block, target_node, target_var);

  // Create 'i = i + 1'
  std::string add_module_name = "mindspore.ops.composite.multitype_ops.add_impl";
  ValuePtr add_op = prim::GetPythonOps("add", add_module_name);
  CNodePtr loop_var_inc =
    body_func_graph->NewCNodeInOrder({NewValueNode(add_op), loop_var, NewValueNode(static_cast<int64_t>(1))});
  body_block->WriteVariable(loop_var->name(), loop_var_inc);

  // Link the variable name with the target
  auto it_info = std::make_shared<TraceIterator>(loop_var_inc->debug_info());
  loop_var->debug_info()->set_trace_info(it_info);

  FunctionBlockPtr after_block = nullptr;
  {
    TraceGuard guard(std::make_shared<TraceForAfter>(block->func_graph()->debug_info()));
    after_block = MakeFunctionBlock(*this);
  }
  MS_EXCEPTION_IF_NULL(after_block);
  after_block->AddPrevBlock(header_block);
  block->Jump(header_block, {iter_node, NewValueNode(static_cast<int64_t>(0))});
  body_block->Mature();
  if (use_fallback) {
    after_block->UpdateGlobalPyParam(block->global_py_params());
  }
  (void)header_block->ConditionalJump(cond_node, body_block, after_block);

  // Generate the body of the for statement
  FunctionBlockPtr rolled_body_block =
    GenerateBlock(std::make_shared<TraceForRolledBody>(body_block->func_graph()->debug_info()));
  MS_EXCEPTION_IF_NULL(rolled_body_block);

  rolled_body_block->Mature();
  body_block->Jump(rolled_body_block, {});
  auto rolled_body_call = dyn_cast<CNode>(body_block->func_graph()->output());
  if (use_fallback) {
    rolled_body_block->UpdateGlobalPyParam(block->global_py_params());
  }

  // Parse loop body statements with loop context.
  LoopContext loop_context{&loops_, header_block, loop_var_inc};
  py::object body_node = python_adapter::GetPyObjAttr(node, "body");
  FunctionBlockPtr after_body_block = ParseStatements(rolled_body_block, body_node);
  if (use_fallback) {
    after_body_block->UpdateGlobalPyParam(block->global_py_params());
  }
  MS_LOG(DEBUG) << "Finish rolled block, after_body_block: " << after_body_block->ToString()
                << ", rolled_body_block: " << rolled_body_block->ToString();
  if (after_body_block->func_graph()->get_return() == nullptr) {
    after_body_block->Jump(header_block, {header_iter_param, loop_var_inc});
  }

  // Record the former/middle/latter graphs for later transforming.
  static const auto transform_for_half_unroll_call = (common::GetEnv("MS_DEV_FOR_HALF_UNROLL") == "1");
  if (transform_for_half_unroll_call) {
    std::pair<FunctionBlockPtr, FunctionBlockPtr> loop_graphs;
    loop_graphs.first = body_block;
    loop_graphs.second = after_body_block;
    std::vector<std::pair<FunctionBlockPtr, FunctionBlockPtr>> loop_graphs_vec{loop_graphs};
    (void)parallel_call_graphs_.emplace_back(loop_graphs_vec);
    MS_LOG(DEBUG) << "Record tail call graphs, loop: {former: " << loop_graphs.first->func_graph()->ToString()
                  << ", middle: " << loop_graphs.second->func_graph()->ToString() << "}";
    // Record the rolled body function, for later lifting operation.
    if (rolled_body_call != nullptr) {
      (void)rolled_body_calls_.emplace_back(std::make_pair(rolled_body_call, rolled_body_block));
      constexpr int recursive_level = 2;
      MS_LOG(DEBUG) << "Record rolled body call: {CNode: " << rolled_body_call->DebugString(recursive_level)
                    << ", rolled_graph: " << rolled_body_block->ToString() << "}";
    }
    auto rolled_body_func_graph = rolled_body_block->func_graph();
    rolled_body_func_graph->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, true);
    rolled_body_func_graph->set_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE, true);
  }

  header_block->Mature();
  after_block->Mature();
  auto &end_block = loop_context.EndBlock();
  if (end_block) {
    // end_block exists if we encounter 'break' in loop body.
    after_block->Jump(end_block, {});
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

  AnfNodePtr bool_node = block->ForceToBoolNode(condition_node);
  UpdateInterpretForUserNode(bool_node, condition_node);
  bool_node = HandleInterpret(block, bool_node, test_node);

  FunctionBlockPtr true_block = nullptr;
  FunctionBlockPtr false_block = nullptr;
  MS_EXCEPTION_IF_NULL(block->func_graph());
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
  MS_EXCEPTION_IF_NULL(true_block->func_graph());
  MS_EXCEPTION_IF_NULL(true_block->func_graph()->debug_info());
  true_block->func_graph()->debug_info()->set_location(GetLocation(bodyNode));
  AnfNodePtr true_node = ParseExprNode(true_block, bodyNode);
  true_node = HandleInterpret(true_block, true_node, bodyNode);

  // Process the orelse branch
  py::object orelseNode = python_adapter::GetPyObjAttr(node, "orelse");
  MS_EXCEPTION_IF_NULL(false_block->func_graph());
  MS_EXCEPTION_IF_NULL(false_block->func_graph()->debug_info());
  false_block->func_graph()->debug_info()->set_location(GetLocation(orelseNode));
  AnfNodePtr false_node = ParseExprNode(false_block, orelseNode);
  false_node = HandleInterpret(false_block, false_node, orelseNode);

  true_block->func_graph()->set_output(true_node);
  false_block->func_graph()->set_output(false_node);

  // Use the Primitive replace the operation resolve node (switch),
  // because the switch will eventually be converted to Primitive node
  CNodePtr switch_app = block->func_graph()->NewCNodeInOrder({NewValueNode(prim::kPrimSwitch), bool_node,
                                                              NewValueNode(true_block->func_graph()),
                                                              NewValueNode(false_block->func_graph())});
  std::vector<AnfNodePtr> call_graph_nodes{switch_app};
  CNodePtr switch_app_call = block->func_graph()->NewCNodeInOrder(std::move(call_graph_nodes));
  return switch_app_call;
}

FunctionBlockPtr Parser::ParseListCompIter(const FunctionBlockPtr &block, const py::object &node,
                                           const py::object &generator_node) {
  // Create a header block.
  MS_EXCEPTION_IF_NULL(block->func_graph());
  FunctionBlockPtr top_block = GenerateBlock(std::make_shared<TraceListComp>(block->func_graph()->debug_info()));
  top_block->AddPrevBlock(block);
  // Handle iter attribute.
  py::object iter_node = python_adapter::GetPyObjAttr(generator_node, "iter");
  AnfNodePtr iter_anf_node = ParseExprNode(block, iter_node);
  MS_EXCEPTION_IF_NULL(iter_anf_node);
  bool interpret_without_internal =
    IsPrimitiveCNode(iter_anf_node, prim::kPrimPyInterpret) && !iter_anf_node->interpret_internal_type();
  if (iter_anf_node->interpret() || interpret_without_internal) {
    iter_anf_node = ConvertInterpretIterNodeToList(block, iter_anf_node, iter_node);
  }

  // Create header graph.
  FunctionBlockPtr list_header_block =
    GenerateBlock(std::make_shared<TraceForHeader>(block->func_graph()->debug_info()));

  // Create hasNext apply.
  AnfNodePtr op_hasnext = top_block->MakeResolveOperation(NAMED_PRIMITIVE_HASNEXT);
  MS_EXCEPTION_IF_NULL(list_header_block->func_graph());
  ParameterPtr iter_param = list_header_block->func_graph()->add_parameter();
  constexpr auto iter_param_name = "iter";
  iter_param->set_name(iter_param_name);
  MS_EXCEPTION_IF_NULL(iter_param->debug_info());
  iter_param->debug_info()->set_name(iter_param_name);
  CNodePtr cond_apply = list_header_block->func_graph()->NewCNodeInOrder({op_hasnext, iter_param});

  // Call the header graph with iter.
  ParameterPtr list_param = list_header_block->func_graph()->add_parameter();
  constexpr auto list_param_name = "list";
  list_param->set_name(list_param_name);
  MS_EXCEPTION_IF_NULL(list_param->debug_info());
  list_param->debug_info()->set_name(list_param_name);
  auto empty_list = std::vector<ValuePtr>();
  AnfNodePtr empty_list_node = NewValueNode(std::make_shared<ValueList>(empty_list));
  top_block->Jump(list_header_block, {iter_anf_node, empty_list_node});

  // Create body graph.
  FunctionBlockPtr list_body_block = GenerateBlock(std::make_shared<TraceForBody>(block->func_graph()->debug_info()));
  list_body_block->AddPrevBlock(list_header_block);
  AnfNodePtr op_next = top_block->MakeResolveOperation(NAMED_PRIMITIVE_NEXT);
  MS_EXCEPTION_IF_NULL(list_body_block->func_graph());
  CNodePtr next_apply = list_body_block->func_graph()->NewCNodeInOrder({op_next, iter_param});
  AnfNodePtr op_getitem = top_block->MakeResolveOperation(NAMED_PRIMITIVE_GETITEM);
  CNodePtr item_apply =
    list_body_block->func_graph()->NewCNodeInOrder({op_getitem, next_apply, NewValueNode(static_cast<int64_t>(0))});
  CNodePtr new_iter =
    list_body_block->func_graph()->NewCNodeInOrder({op_getitem, next_apply, NewValueNode(static_cast<int64_t>(1))});

  // Save the `target` in a variable.
  py::object gen_target_node = python_adapter::GetPyObjAttr(generator_node, "target");
  WriteAssignVars(list_body_block, gen_target_node, item_apply);

  auto ifs_new_list = ParseListCompIfs(list_body_block, list_param, node, generator_node);
  list_body_block->Jump(list_header_block, {new_iter, ifs_new_list});

  // Create after graph.
  FunctionBlockPtr list_after_block = GenerateBlock(std::make_shared<TraceForAfter>(block->func_graph()->debug_info()));
  list_after_block->AddPrevBlock(list_header_block);
  // Return the list in after graph.
  MS_EXCEPTION_IF_NULL(list_after_block->func_graph());
  list_after_block->func_graph()->set_output(list_param);

  // Run the branches.
  (void)list_header_block->ConditionalJump(cond_apply, list_body_block, list_after_block);

  top_block->Mature();
  list_header_block->Mature();
  list_body_block->Mature();
  list_after_block->Mature();
  return top_block;
}

AnfNodePtr Parser::ParseListCompIfs(const FunctionBlockPtr &list_body_block, const ParameterPtr &list_param,
                                    const py::object &node, const py::object &generator_node) {
  // Handle ifs attribute.
  py::list ifs_node = python_adapter::GetPyObjAttr(generator_node, "ifs");
  AnfNodePtr ifs_bool_node;
  if (ifs_node.empty()) {
    ifs_bool_node = NewValueNode(true);
  } else {
    ifs_bool_node = ProcessBoolOpValueList(list_body_block, ifs_node, AST_SUB_TYPE_AND);
  }

  // Create if-true graph.
  FunctionBlockPtr if_true_block =
    GenerateBlock(std::make_shared<TraceIfStmtTrueBranch>(list_body_block->func_graph()->debug_info()));
  if_true_block->AddPrevBlock(list_body_block);
  // Handle elt attribute in body block.
  py::object elt_obj = python_adapter::GetPyObjAttr(node, "elt");
  AnfNodePtr elt_node = ParseExprNode(list_body_block, elt_obj);
  // Append the element.
  MS_EXCEPTION_IF_NULL(list_body_block->func_graph());
  auto new_list = list_body_block->func_graph()->NewCNodeInOrder(
    {NewValueNode(std::make_shared<prim::ListAppend>("ListAppend")), list_param, elt_node});
  // Return new list in true branch graph.
  if_true_block->func_graph()->set_output(new_list);

  // Create if-false graph.
  FunctionBlockPtr if_false_block =
    GenerateBlock(std::make_shared<TraceIfStmtFalseBranch>(list_body_block->func_graph()->debug_info()));
  if_false_block->AddPrevBlock(list_body_block);
  // Return original list in false branch graph.
  MS_EXCEPTION_IF_NULL(if_false_block->func_graph());
  if_false_block->func_graph()->set_output(list_param);

  // We don't want to create a header graph, where to get and wrap the result of Switch().
  // So just call ConditionalJump() to set Switch() as output, and reset it later, as tricky.
  (void)list_body_block->ConditionalJump(ifs_bool_node, if_true_block, if_false_block);
  // Output is Switch() result, i.e. updated list.
  auto switch_apply_node = list_body_block->func_graph()->output();
  auto ifs_new_list = switch_apply_node;
  // Since we call ConditionalJump() above, to reset the Return as null before call Jump().
  list_body_block->func_graph()->set_return(nullptr);
  if_true_block->Mature();
  if_false_block->Mature();
  return ifs_new_list;
}

// A ListComp contains: `elt` and `generators`.
// `generators` contains: `target`, `iter` and `ifs`.
// For example:
// [x * x for x in range(0, 10) if x % 2 == 0]
// It is compiled to be following statement:
// list = []
// for x in range(0, 10):
//    if x % 2 == 0:
//        list.append(x * x)
// return list
AnfNodePtr Parser::ParseListComp(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast ListComp";
  MS_EXCEPTION_IF_NULL(block);

  // Handle generators attribute.
  py::list generators_node = python_adapter::GetPyObjAttr(node, "generators");
  if (generators_node.size() != 1) {
    MS_EXCEPTION(TypeError) << "The 'generators' supports 1 'comprehension' in ListComp/GeneratorExp, but got "
                            << generators_node.size() << " comprehensions.";
  }
  py::object generator_node = generators_node[0];
  auto generator_node_type = ast_->GetNodeType(generator_node);
  auto generator_node_name = generator_node_type->node_name();
  constexpr auto comprehension_name = "comprehension";
  if (generator_node_name != comprehension_name) {
    MS_LOG(EXCEPTION) << "Generator node name should be " << comprehension_name << ", but got " << generator_node_name;
  }

  // Parse ListComp's `iter` and add `elt` in it.
  auto top_block = ParseListCompIter(block, node, generator_node);

  // Call the top graph and return the list.
  auto call_function_node = NewValueNode(top_block->func_graph());
  std::vector<AnfNodePtr> func_call_nodes;
  func_call_nodes.push_back(call_function_node);
  MS_EXCEPTION_IF_NULL(block->func_graph());
  AnfNodePtr output = block->func_graph()->NewCNodeInOrder(std::move(func_call_nodes));
  return output;
}

AnfNodePtr Parser::ParseJoinedStr(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast JoinedStr.";
  MS_EXCEPTION_IF_NULL(block);
  const auto script_text = py::cast<std::string>(ast()->GetAstNodeText(node));
  py::list py_values = python_adapter::GetPyObjAttr(node, "values");
  std::vector<AnfNodePtr> value_nodes{NewValueNode(prim::kPrimJoinedStr)};
  for (size_t i = 0; i < py_values.size(); ++i) {
    AnfNodePtr str_value = ParseExprNode(block, py_values[i]);
    str_value = HandleInterpret(block, str_value, py_values[i]);
    // If exist interpret node in JoinedStr, all object in py_values will convert to interpret node.
    if (IsPrimitiveCNode(str_value, prim::kPrimPyInterpret)) {
      return MakeInterpretNode(block, str_value, script_text);
    }
    (void)value_nodes.emplace_back(str_value);
  }
  auto func_graph = block->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtr output = func_graph->NewCNodeInOrder(std::move(value_nodes));
  return output;
}

AnfNodePtr Parser::ParseFormattedValue(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast FormattedValue.";
  MS_EXCEPTION_IF_NULL(block);
  py::object value_object = python_adapter::GetPyObjAttr(node, "value");
  AnfNodePtr value_node = ParseExprNode(block, value_object);
  return value_node;
}

void Parser::HandleAssignName(const FunctionBlockPtr &block, const py::object &targ,
                              const AnfNodePtr &assigned_node) const {
  MS_EXCEPTION_IF_NULL(block);
  MS_EXCEPTION_IF_NULL(assigned_node);
  py::str name = python_adapter::GetPyObjAttr(targ, "id");
  std::string name_id = name;
  MS_EXCEPTION_IF_NULL(assigned_node->debug_info());
  assigned_node->debug_info()->set_name(name_id);
  // Set the debug name of the constant graph
  if (IsValueNode<FuncGraph>(assigned_node)) {
    // The value should be graph
    auto fg = GetValueNode<FuncGraphPtr>(assigned_node);
    MS_EXCEPTION_IF_NULL(fg->debug_info());
    if (fg->debug_info()->name().empty()) {
      fg->debug_info()->set_name(name_id);
    }
  }
  MS_LOG(DEBUG) << "Assign name: `" << name_id << "` to node: " << assigned_node->DebugString();
  block->WriteVariable(name_id, assigned_node);
}

void Parser::HandleAssignTuple(const FunctionBlockPtr &block, const py::object &targ, const AnfNodePtr &assigned_node) {
  MS_EXCEPTION_IF_NULL(block);
  AnfNodePtr op_getitem = block->MakeResolveOperation(NAMED_PRIMITIVE_GETITEM);
  py::list items = python_adapter::GetPyObjAttr(targ, "elts");
  for (size_t i = 0; i < items.size(); i++) {
    // Use the Primitive replace the operation resolve node (getitem),
    // because the getitem will eventually be converted to Primitive node
    MS_EXCEPTION_IF_NULL(block->func_graph());
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
  target_node = HandleInterpret(block, target_node, targ);
  MS_EXCEPTION_IF_NULL(target_node);

  auto attr_name = targ.attr("attr").cast<std::string>();
  std::string var_name = "self." + attr_name;

  // Now only support the self.xxx = yyy, where self.xxx must be a defined Parameter type
  if (!py::hasattr(ast()->obj(), common::SafeCStr(attr_name))) {
    MS_EXCEPTION(TypeError)
      << "'" << var_name << "' should be initialized as a 'Parameter' in the '__init__' function before assigning.\n\n"
      << trace::GetDebugInfo(target_node->debug_info());
  }
  auto obj = ast()->obj().attr(common::SafeCStr(attr_name));
  auto obj_type = obj.attr("__class__").attr("__name__");
  if (!py::hasattr(obj, "__parameter__")) {
    MS_EXCEPTION(TypeError) << "'" << var_name
                            << "' should be initialized as a 'Parameter' type in the '__init__' function, but got '"
                            << py::str(obj).cast<std::string>() << "' with type '"
                            << py::str(obj_type).cast<std::string>() << ".\n\n"
                            << trace::GetDebugInfo(target_node->debug_info());
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
  value_node = HandleInterpret(block, value_node, value_obj);
  AnfNodePtr slice_node = ParseExprNode(block, slice_obj);
  slice_node = HandleInterpret(block, slice_node, slice_obj);
  MS_EXCEPTION_IF_NULL(block->func_graph());
  CNodePtr setitem_app = block->func_graph()->NewCNodeInOrder({op_setitem, value_node, slice_node, assigned_node});
  // Getitem apply should return the sequence data structure itself
  std::string var_name;
  if (ast_->IsClassMember(value_obj)) {
    auto attr_name = value_obj.attr("attr").cast<std::string>();
    var_name = "self." + attr_name;
    if (!py::hasattr(ast()->obj(), common::SafeCStr(attr_name))) {
      MS_EXCEPTION(TypeError)
        << "'" << var_name
        << "' should be initialized as a 'Parameter' in the '__init__' function before assigning.\n\n"
        << trace::GetDebugInfo(value_node->debug_info());
    }
    auto obj = ast()->obj().attr(common::SafeCStr(attr_name));
    auto obj_type = obj.attr("__class__").attr("__name__");
    if (!py::hasattr(obj, "__parameter__")) {
      MS_EXCEPTION(TypeError) << "'" << var_name
                              << "' should be initialized as a 'Parameter' in the '__init__' function, but got '"
                              << py::str(obj).cast<std::string>() << "' with type '"
                              << py::str(obj_type).cast<std::string>() << "'.\n\n"
                              << trace::GetDebugInfo(value_node->debug_info());
    }
    block->WriteVariable(var_name, setitem_app);
    return;
  }
  if (AstSubType(py::cast<int32_t>(ast_->CallParseModFunction(PYTHON_PARSE_GET_AST_TYPE, value_obj))) ==
      AST_SUB_TYPE_SUBSCRIPT) {
    if (IsSubscriptReferenceType(value_obj)) {
      HandleAssignSubscript(block, value_obj, setitem_app);
      return;
    }
  }
  if (py::hasattr(value_obj, "id")) {
    var_name = value_obj.attr("id").cast<std::string>();
  }
  block->WriteVariable(var_name, setitem_app);
}

void Parser::WriteAssignVars(const FunctionBlockPtr &block, const py::object &target_object,
                             const AnfNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  MS_LOG(DEBUG) << "Process WriteAssignVars";
  auto ast_type = AstSubType(py::cast<int32_t>(ast_->CallParseModFunction(PYTHON_PARSE_GET_AST_TYPE, target_object)));
  if (ast_type == AST_SUB_TYPE_NAME) {
    HandleAssignName(block, target_object, value_node);
  } else if (ast_type == AST_SUB_TYPE_TUPLE) {
    HandleAssignTuple(block, target_object, value_node);
  } else if (ast_type == AST_SUB_TYPE_SUBSCRIPT) {
    HandleAssignSubscript(block, target_object, value_node);
  } else if (ast_->IsClassMember(target_object)) {
    HandleAssignClassMember(block, target_object, value_node);
  } else if (ast_type == AST_SUB_TYPE_ATTRIBUTE) {
    TraceGuard(GetLocation(target_object));
    MS_EXCEPTION(TypeError) << "Only support assign to attribute of self, but got attribute of "
                            << py::str(target_object.attr("value").attr("id")) << ".\n"
                            << "More details please refer to syntax support at https://www.mindspore.cn";
  } else {
    TraceGuard(GetLocation(target_object));
    MS_EXCEPTION(TypeError) << "Only supported augassign to attribute of self, variable and index value, but got "
                            << target_object.get_type()
                            << ".\nMore details please refer to syntax support at https://www.mindspore.cn";
  }
}

void Parser::UpdateInterpretForUserNode(const AnfNodePtr &user_node, const AnfNodePtr &node) const {
  // The fallback feature is enabled in default.
  static const auto use_fallback = (support_fallback() != "0");
  if (!use_fallback) {
    return;
  }

  MS_EXCEPTION_IF_NULL(user_node);
  MS_EXCEPTION_IF_NULL(node);
  // Do not handle user node with internal type such as Tensor.abs().
  bool interpret_without_internal = IsPrimitiveCNode(node, prim::kPrimPyInterpret) && !node->interpret_internal_type();
  if (node->interpret() || interpret_without_internal) {
    user_node->set_interpret(true);
    if (node->interpret_internal_type()) {
      user_node->set_interpret_internal_type(true);
    }
  }
}

void Parser::UpdateInterpretForUserNode(const AnfNodePtr &user_node, const std::vector<AnfNodePtr> &nodes) const {
  for (auto &node : nodes) {
    UpdateInterpretForUserNode(user_node, node);
  }
}

bool Parser::IsScriptInParams(const std::string &script_text, const py::dict &global_dict,
                              const std::map<std::string, AnfNodePtr> &local_keys,
                              const FuncGraphPtr &func_graph) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  // Check global parameters.
  if (global_dict.contains(script_text)) {
    MS_LOG(DEBUG) << "[" << func_graph->ToString() << "] Found `" << script_text << "` in global params.";
    return true;
  }

  // Check local parameters.
  if (local_keys.find(script_text) != local_keys.end()) {
    MS_LOG(DEBUG) << "[" << func_graph->ToString() << "] Found `" << script_text << "` in local params.";
    return true;
  }
  return false;
}

AnfNodePtr Parser::HandleInterpret(const FunctionBlockPtr &block, const AnfNodePtr &value_node,
                                   const py::object &value_object) {
  // The fallback feature is enabled in default.
  // Not support change the flag during the process is alive.
  static const auto use_fallback = (support_fallback() != "0");
  MS_EXCEPTION_IF_NULL(value_node);
  if (!use_fallback || !value_node->interpret()) {
    return value_node;
  }
  const auto script_text = py::cast<std::string>(ast()->GetAstNodeText(value_object));
  return MakeInterpretNode(block, value_node, script_text);
}

bool Parser::CheckNeedConvertInterpret(const FunctionBlockPtr &block, const AnfNodePtr &node,
                                       const string &script_text) const {
  MS_EXCEPTION_IF_NULL(block);
  MS_EXCEPTION_IF_NULL(node);
  // Check if script_text is in global/local params.
  py::dict global_dict = block->global_py_params();
  auto keys = std::get<0>(block->local_py_params());
  if (IsScriptInParams(script_text, global_dict, keys, block->func_graph())) {
    return false;
  }
  return true;
}

AnfNodePtr Parser::MakeInterpretNode(const FunctionBlockPtr &block, const AnfNodePtr &value_node,
                                     const string &script_text) {
  MS_EXCEPTION_IF_NULL(block);
  MS_EXCEPTION_IF_NULL(value_node);
  if (!CheckNeedConvertInterpret(block, value_node, script_text)) {
    return value_node;
  }

  // Prepare global parameters.
  ValuePtr globals_converted_value = nullptr;
  py::dict global_dict = block->global_py_params();
  if (!ConvertData(global_dict, &globals_converted_value)) {
    MS_LOG(EXCEPTION) << "Convert data failed";
  }
  auto global_dict_node = NewValueNode(globals_converted_value);
  // Prepare local parameters. Select the needed local parameters for script.
  auto [keys, values] = block->local_py_params();
  std::vector<AnfNodePtr> filter_keys;
  std::vector<AnfNodePtr> filter_values;
  const py::set &ids = data_converter::GetPythonScriptIds(py::str(script_text));
  for (const auto &id : ids) {
    const auto &id_str = py::str(id);
    const auto &iter = values.find(id_str);
    if (iter != values.end()) {
      (void)filter_keys.emplace_back(keys[iter->first]);
      (void)filter_values.emplace_back(iter->second);
    }
  }

  auto local_dict_node = ParseDictByKeysAndValues(block, filter_keys, filter_values);
  // Update the valued node if it need interpreting.
  constexpr int recursive_level = 2;
  MS_EXCEPTION_IF_NULL(block->func_graph());
  MS_LOG(INFO) << "[" << block->func_graph()->ToString() << "] script_text: `" << script_text
               << "`,\nvalue_node: " << value_node->DebugString(recursive_level)
               << ",\nglobal_dict_node: " << global_dict_node->ToString()
               << ",\nlocal_dict_node: " << local_dict_node->DebugString(recursive_level);
  AnfNodePtr interpreted_node = block->MakeInterpret(script_text, global_dict_node, local_dict_node, value_node);

  // Print a hint for user.
  auto line_info = trace::GetDebugInfo(value_node->debug_info());
  MS_LOG(INFO) << "Found unsupported syntax in graph mode, those codes would be fallen back to Python interpreter:"
               << "\n\n"
               << line_info;
  InterpretNodeRecorder::GetInstance().PushLineInfo(line_info);
  return interpreted_node;
}

bool Parser::IsPopOperation(const AnfNodePtr &node) const {
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return false;
  }
  auto attr_node = cnode->input(0);
  if (IsPrimitiveCNode(attr_node, prim::kPrimGetAttr)) {
    auto attr_cnode = attr_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(attr_cnode);
    constexpr size_t attr_cnode_size = 3;
    constexpr size_t member_index = 2;
    if (attr_cnode->inputs().size() < attr_cnode_size) {
      MS_LOG(EXCEPTION) << "The attr operate has wrong input.";
    }
    auto member_node = attr_cnode->input(member_index);
    if (IsValueNode<StringImm>(member_node)) {
      auto attr_name = GetValue<std::string>(GetValueNode(member_node));
      if (attr_name == "pop") {
        return true;
      }
    }
  }
  return false;
}

// Process a assign statement, such as a = b,  a, b = tuple(xx, xx)
FunctionBlockPtr Parser::ParseAssign(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast assign";
  py::object value_object = python_adapter::GetPyObjAttr(node, "value");
  AnfNodePtr value_node = ParseExprNode(block, value_object);
  value_node = HandleInterpret(block, value_node, value_object);

  py::object targets_object = python_adapter::GetPyObjAttr(node, "targets");
  py::int_ pcount = python_adapter::CallPyObjMethod(targets_object, "__len__");
  size_t count = LongToSize(pcount);
  MS_LOG(DEBUG) << "The nodes count is " << count;

  // b = list_x.pop(a)
  // -->  list_x, b = list_x.pop(a) need renew the list_x.
  if (IsPopOperation(value_node)) {
    if (ast_->target_type() == PARSE_TARGET_OBJECT_INSTANCE && ast_->IsClassMember(list_pop_target_obj_)) {
      // self.list_x = [xx, xx]
      // y = self.list_x.pop()
      MS_LOG(DEBUG) << "The variables whose type is not parameter do not support pop operation.";
    } else {
      auto func_graph = block->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      auto new_list =
        func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), value_node, NewValueNode(SizeToLong(0))});
      auto pop_node =
        func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), value_node, NewValueNode(SizeToLong(1))});
      WriteAssignVars(block, list_pop_target_obj_, new_list);
      if (count != 1) {
        MS_LOG(EXCEPTION) << "The pop operate has wrong input.";
      }
      auto pop_obj = py::cast<py::list>(targets_object)[0];
      WriteAssignVars(block, pop_obj, pop_node);
      return block;
    }
  }
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
    MS_EXCEPTION_IF_NULL(block->func_graph());
    TraceGuard trace_guard(std::make_shared<TraceLoopEnd>(block->func_graph()->debug_info()));
    loop.end = MakeFunctionBlock(*this);
  }
  block->SetBreakContinueStatementInside();
  MS_LOG(DEBUG) << "Inside the block has break statement, block: " << block->ToString();

  // Jump to the end_block.
  block->Jump(loop.end, {});
  return block;
}

FunctionBlockPtr Parser::ParseContinue(const FunctionBlockPtr &block, const py::object &node) {
  if (loops_.empty()) {
    // Report error if loop context not set for the 'continue' statement.
    MS_LOG(EXCEPTION) << "Unexpected 'continue'.";
  }
  // Jump to the header of the loop with iterator called.
  Loop &loop = loops_.top();
  std::vector<AnfNodePtr> args;
  if (loop.iterator != nullptr) {
    args.emplace_back(loop.iterator);
  }
  block->SetBreakContinueStatementInside();
  MS_LOG(DEBUG) << "Inside the block has continue statement, block: " << block->ToString();

  block->Jump(loop.header, args);
  return block;
}

FunctionBlockPtr Parser::ParsePass(const FunctionBlockPtr &block, const py::object &node) {
  // We just bypass 'pass' statement.
  return block;
}

FunctionBlockPtr Parser::ParseRaise(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process raise statement";
  MS_EXCEPTION_IF_NULL(block);
  auto func_graph = block->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  py::object exc_ast_node = python_adapter::GetPyObjAttr(node, "exc");
  // raise
  if (py::isinstance<py::none>(exc_ast_node)) {
    CNodePtr raise_node = func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimRaise)});
    func_graph->set_return(raise_node);
    return block;
  }
  auto exc_node_inputs = ParseRaiseCall(block, exc_ast_node);
  // raise ExceptionType or raise ExceptionType(ExceptionString)
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimRaise)};
  (void)inputs.insert(inputs.end(), exc_node_inputs.begin(), exc_node_inputs.end());
  CNodePtr raise_node = func_graph->NewCNodeInOrder(inputs);
  CNodePtr return_node = func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimReturn), raise_node});
  func_graph->set_return(return_node);
  return block;
}

FunctionBlockPtr Parser::MakeAssertErrorBlock(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process make AssertError block";
  MS_EXCEPTION_IF_NULL(block);
  const std::string kAssertionError = "AssertionError";
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimRaise), NewValueNode(kAssertionError)};

  py::object msg_node = python_adapter::GetPyObjAttr(node, "msg");
  if (!py::isinstance<py::none>(msg_node)) {
    AnfNodePtr msg = ParseExprNode(block, msg_node);
    msg = HandleInterpret(block, msg, msg_node);
    (void)inputs.emplace_back(msg);
  }

  auto func_graph = block->func_graph();
  CNodePtr raise_node = func_graph->NewCNodeInOrder(inputs);
  CNodePtr return_node = func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimReturn), raise_node});
  func_graph->set_return(return_node);
  return block;
}

// assert expression [, arguments]
// =>
// if not expression:
//     raise AssertionError(arguments)
FunctionBlockPtr Parser::ParseAssert(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast Assert";
  MS_EXCEPTION_IF_NULL(block);
  py::object test_node = python_adapter::GetPyObjAttr(node, "test");
  AnfNodePtr condition_node = ParseExprNode(block, test_node);

  AnfNodePtr bool_node = block->ForceToBoolNode(condition_node);
  UpdateInterpretForUserNode(bool_node, condition_node);
  bool_node = HandleInterpret(block, bool_node, test_node);

  FunctionBlockPtr true_block = MakeFunctionBlock(*this);
  FunctionBlockPtr false_block = MakeFunctionBlock(*this);
  FunctionBlockPtr after_block = MakeFunctionBlock(*this);
  MakeConditionBlocks(block, true_block, false_block);

  true_block->Jump(after_block, {});
  false_block = MakeAssertErrorBlock(false_block, node);
  (void)block->ConditionalJump(bool_node, true_block, false_block);

  after_block->Mature();
  return after_block;
}

AnfNodePtr Parser::ParseWithitem(const FunctionBlockPtr &block, const py::object &node,
                                 const AnfNodePtr &context_expr_node) {
  MS_LOG(DEBUG) << "Process ast Withitem";
  // Handle __enter__(self)
  std::vector<AnfNodePtr> enter_inputs{NewValueNode(prim::kPrimWithEnter), context_expr_node};
  auto func_graph = block->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtr enter_node = func_graph->NewCNodeInOrder(enter_inputs);
  py::object optional_vars_obj = python_adapter::GetPyObjAttr(node, "optional_vars");
  if (!py::isinstance<py::none>(optional_vars_obj)) {
    // with Sample() as sample: mean that sample = Sample()
    WriteAssignVars(block, optional_vars_obj, enter_node);
  }
  return enter_node;
}

// with expression [as variable]:
//      with-block
FunctionBlockPtr Parser::ParseWith(const FunctionBlockPtr &block, const py::object &node) {
  MS_LOG(DEBUG) << "Process ast With";
  py::list items_objs = python_adapter::GetPyObjAttr(node, "items");
  if (items_objs.empty()) {
    MS_LOG(EXCEPTION) << "Unexpected 'with'.";
  }
  std::stack<AnfNodePtr> context_expr_nodes;
  std::stack<AnfNodePtr> entered_nodes;
  for (size_t i = 0; i < items_objs.size(); ++i) {
    auto items_obj = items_objs[i];
    // with Sample() as sample:
    // mean context_expr is Sample(), sample is optional_vars
    py::object context_expr_obj = python_adapter::GetPyObjAttr(items_obj, "context_expr");
    AnfNodePtr context_expr_node = ParseExprNode(block, context_expr_obj);
    context_expr_nodes.push(context_expr_node);
    auto enter_node = ParseWithitem(block, items_obj, context_expr_node);
    entered_nodes.push(enter_node);
  }
  auto func_graph = block->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  py::object body_node = python_adapter::GetPyObjAttr(node, "body");
  FunctionBlockPtr body_block = ParseStatements(block, body_node);
  auto body_func = body_block->func_graph();
  MS_EXCEPTION_IF_NULL(body_func);

  while (!context_expr_nodes.empty()) {
    auto context_expr_node = context_expr_nodes.top();
    auto entered_node = entered_nodes.top();
    context_expr_nodes.pop();
    entered_nodes.pop();
    // Use the depend node to ensure the execution order of enter and exit node.
    std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), context_expr_node, entered_node};
    context_expr_node = func_graph->NewCNodeInOrder(depend_inputs);
    // Handle __exit__(self, type, value, trace)
    std::vector<AnfNodePtr> exit_inputs{NewValueNode(prim::kPrimWithExit), context_expr_node};
    AnfNodePtr exit_node = func_graph->NewCNodeInOrder(exit_inputs);
    block->AddIsolatedNode(exit_node);
  }
  FunctionBlockPtr after_block = MakeFunctionBlock(*this);
  if (body_func->get_return() == nullptr) {
    body_block->Jump(after_block, {});
  }
  after_block->Mature();
  return after_block;
}

void Parser::PrintPhiArgMaps(const std::map<ParameterPtr, std::set<AnfNodePtr>> &phi_to_args,
                             const std::map<AnfNodePtr, std::set<ParameterPtr>> &arg_to_phis) {
  if (!IS_OUTPUT_ON(mindspore::kDebug)) {
    return;
  }
  std::ostringstream oss;
  oss << "==============================Start=============================="
      << "\n";
  size_t m = 0;
  for (const auto &[phi, args] : phi_to_args) {
    MS_EXCEPTION_IF_NULL(phi);
    oss << "phi[" << m++ << "]: " << phi->DebugString() << "\n";
    size_t n = 0;
    for (auto &arg : args) {
      MS_EXCEPTION_IF_NULL(arg);
      oss << "    args[" << n++ << "]: " << arg->DebugString() << "\n";
    }
  }

  m = 0;
  for (const auto &[arg, phis] : arg_to_phis) {
    MS_EXCEPTION_IF_NULL(arg);
    oss << "arg[" << m++ << "]: " << arg->DebugString() << "\n";
    size_t n = 0;
    for (auto &phi : phis) {
      MS_EXCEPTION_IF_NULL(phi);
      oss << "    phis[" << n++ << "]: " << phi->DebugString() << "\n";
    }
  }
  oss << "===============================End==============================="
      << "\n";
  MS_LOG(DEBUG) << "\n" << oss.str();
}

bool UpdatePhiArgMaps(std::map<ParameterPtr, std::set<AnfNodePtr>> *phi_to_args,
                      std::map<AnfNodePtr, std::set<ParameterPtr>> *arg_to_phis) {
  bool phi_arg_updated = false;
  auto copy_phi_to_args = *phi_to_args;
  for (const auto &[phi, args] : copy_phi_to_args) {
    // The phi node has only one arg can be replaced as arg.
    if (args.size() != 1) {
      continue;
    }
    auto phi_arg = *args.begin();
    MS_EXCEPTION_IF_NULL(phi_arg);
    MS_LOG(DEBUG) << "phi: " << phi->DebugString() << ", get one arg: " << phi_arg->DebugString();
    // If this phi is a arg of other phi.
    auto arg_to_phi_it = arg_to_phis->find(phi);
    if (arg_to_phi_it == arg_to_phis->end()) {
      continue;
    }
    // Use the new phi arg as the arg of other phi's arg. Usually other phi is a deeper subgraph's phi node.
    auto other_phis = arg_to_phi_it->second;
    MS_LOG(DEBUG) << "Find phi as arg of other phi, other phis num: " << other_phis.size();
    // Update all other phis' arg from phi to phi_arg.
    for (auto &other_phi : other_phis) {
      MS_EXCEPTION_IF_NULL(other_phi);
      MS_LOG(DEBUG) << "other phi: " << other_phi->DebugString();
      phi_arg_updated = true;
      // The phi will not be arg of any other phis.Erase map1.
      (void)(*phi_to_args)[other_phi].erase(phi);
      // If arg is same to the parameter phi, ignore the arg, keep maps don't have self arg.
      if (phi_arg == other_phi) {
        MS_LOG(DEBUG) << "Get phi arg of phi self.";
        continue;
      }
      MS_LOG(DEBUG) << "phi arg: " << phi_arg->DebugString()
                    << " as new arg of other phi: " << other_phi->DebugString();
      // Replace other phi's arg as this phi's arg, instead of phi. (other_phi , phi) -> (other_phi, phi_arg)
      (void)(*phi_to_args)[other_phi].insert(phi_arg);
      // Add other phi to the phi_arg's phis set. (phi_arg, {phi_x, }) -> (phi_arg, {phi_x, other_phi})
      (void)(*arg_to_phis)[phi_arg].insert(other_phi);
    }
    MS_LOG(DEBUG) << "Remove phi type arg: " << phi;
    // The phi will not be arg of any other phis.Erase map2.
    (void)(*arg_to_phis).erase(phi);
  }
  return phi_arg_updated;
}

void Parser::UpdatePhiArgMapsRepeatedly(std::map<ParameterPtr, std::set<AnfNodePtr>> *phi_to_args,
                                        std::map<AnfNodePtr, std::set<ParameterPtr>> *arg_to_phis) {
  bool phi_arg_updated = true;
  size_t loop_count = 0;
  while (phi_arg_updated) {
    MS_LOG(DEBUG) << "update loop count: " << loop_count++;
    PrintPhiArgMaps(*phi_to_args, *arg_to_phis);
    phi_arg_updated = UpdatePhiArgMaps(phi_to_args, arg_to_phis);
  }
}

void Parser::CreatePhiArgMaps(std::map<ParameterPtr, std::set<AnfNodePtr>> *phi_to_args,
                              std::map<AnfNodePtr, std::set<ParameterPtr>> *arg_to_phis) {
  for (FunctionBlockPtr &block : func_block_list_) {
    MS_EXCEPTION_IF_NULL(block);
    for (const auto &[phi, args] : block->phi_args()) {
      // Filtered args exclude the arg pointer equals to phi pointer.
      for (const auto &arg : args) {
        if (phi == arg) {
          continue;
        }
        (void)(*phi_to_args)[phi].insert(arg);
        (void)(*arg_to_phis)[arg].insert(phi);
      }
    }
  }
}

std::shared_ptr<std::map<ParameterPtr, AnfNodePtr>> Parser::CollectRemovablePhiArgs(
  const std::map<ParameterPtr, std::set<AnfNodePtr>> &phi_to_args) {
  auto need_remove_phi_args = std::make_shared<std::map<ParameterPtr, AnfNodePtr>>();
  for (const auto &[phi, args] : phi_to_args) {
    if (args.empty()) {
      // phi's arg is phi self.
      (*need_remove_phi_args)[phi] = nullptr;
      continue;
    }
    if (args.size() == 1) {
      (*need_remove_phi_args)[phi] = *(args.begin());
    }
  }
  if (IS_OUTPUT_ON(mindspore::kDebug)) {
    size_t m = 0;
    std::ostringstream oss;
    oss << "=====================Need removed phis and args====================="
        << "\n";
    for (const auto &[phi, arg] : *need_remove_phi_args) {
      MS_EXCEPTION_IF_NULL(phi);
      oss << "phi[" << m << "]: " << phi->DebugString() << "\n";
      oss << "arg[" << m++ << "]: " << arg->DebugString() << "\n";
    }
    MS_LOG(DEBUG) << "\n" << oss.str();
  }
  return need_remove_phi_args;
}

std::shared_ptr<std::map<ParameterPtr, AnfNodePtr>> Parser::CalRemovablePhis() {
  std::map<ParameterPtr, std::set<AnfNodePtr>> phi_to_args;
  std::map<AnfNodePtr, std::set<ParameterPtr>> arg_to_phis;
  CreatePhiArgMaps(&phi_to_args, &arg_to_phis);
  // Update phi arg maps by phi arg map relations, some phi can be replaced as arg.
  UpdatePhiArgMapsRepeatedly(&phi_to_args, &arg_to_phis);
  // Collect all one arg phis.
  return CollectRemovablePhiArgs(phi_to_args);
}

void ReplacePhiAsArg(const std::map<ParameterPtr, AnfNodePtr> &removable_phis, const FuncGraphManagerPtr &manager) {
  MS_LOG(DEBUG) << "Removable phi size: " << removable_phis.size();
  for (const auto &[phi, arg] : removable_phis) {
    MS_LOG(DEBUG) << "Removable phi: " << phi->DebugString()
                  << ", arg: " << (arg == nullptr ? "null" : arg->DebugString());
    if (arg != nullptr) {
      (void)manager->Replace(phi, arg);
    }
  }
}

// Remove the removable phi parameter and get the corresponding index.
HashSet<size_t> RemovePhiParametersAndGetRemoveIndex(const FunctionBlockPtr &block,
                                                     const std::map<ParameterPtr, AnfNodePtr> &removable_phis) {
  MS_EXCEPTION_IF_NULL(block);
  auto func_graph = block->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "Check removable parameters of block: " << block->ToString();
  const auto &parameters = func_graph->parameters();
  std::vector<AnfNodePtr> new_parameters;
  // Remove the unnecessary phi parameters.
  HashSet<size_t> need_removed_indexes;
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto parameter_i = parameters[i];
    MS_EXCEPTION_IF_NULL(parameter_i);
    if (removable_phis.find(parameter_i->cast<ParameterPtr>()) == removable_phis.end()) {
      new_parameters.push_back(parameter_i);
      continue;
    }
    // Record all removed indexes.
    (void)need_removed_indexes.insert(i);
  }
  MS_LOG(DEBUG) << "parameters.size(): " << parameters.size()
                << ", need_removed_indexes.size(): " << need_removed_indexes.size();
  // Only if need_removed_indexes not empty, parameters need be updated.
  if (!need_removed_indexes.empty()) {
    func_graph->set_parameters(new_parameters);
  }
  return need_removed_indexes;
}

// If phi parameter is removable, then the corresponding arg should be removed.
void RemoveJumpNodeArgs(const FunctionBlockPtr &block, const HashSet<size_t> &need_removed_indexes,
                        const FuncGraphManagerPtr &manager) {
  if (need_removed_indexes.empty()) {
    return;
  }
  for (const auto &prev_block : block->prev_blocks()) {
    MS_EXCEPTION_IF_NULL(prev_block);
    const auto &jump_node = prev_block->GetJumpNode(block.get());
    // Switch call has no jump node.
    if (jump_node == nullptr) {
      continue;
    }
    std::vector<AnfNodePtr> new_inputs = {jump_node->input(0)};
    for (size_t arg_index = 0; arg_index < jump_node->inputs().size() - 1; ++arg_index) {
      if (need_removed_indexes.find(arg_index) == need_removed_indexes.end()) {
        new_inputs.push_back(jump_node->input(arg_index + 1));
      }
    }
    MS_EXCEPTION_IF_NULL(prev_block->func_graph());
    const auto &new_jump_node = prev_block->func_graph()->NewCNodeInOrder(new_inputs);
    MS_LOG(DEBUG) << "Replace old jump node: " << jump_node->DebugString()
                  << " as new jump node: " << new_jump_node->DebugString()
                  << ", jump node block: " << prev_block->ToString();
    (void)manager->Replace(jump_node, new_jump_node);
  }
}

void Parser::RemoveUnnecessaryPhis() {
  // Merge all removable phis to one map;
  const auto &removable_phis = CalRemovablePhis();
  if (removable_phis->empty()) {
    return;
  }
  const auto &manager = Manage(func_graph_, false);
  MS_EXCEPTION_IF_NULL(manager);
  // Replace all phi node as arg.
  ReplacePhiAsArg(*removable_phis, manager);
  // Remove the unnecessary phi parameters.
  for (const auto &block : func_block_list_) {
    MS_EXCEPTION_IF_NULL(block);
    MS_LOG(DEBUG) << "Start remove phi of block: " << block->ToString();
    // Remove the unnecessary phi parameters.
    const auto &need_removed_indexes = RemovePhiParametersAndGetRemoveIndex(block, *removable_phis);
    // Remove all block->prev_blocks()'s jump node corresponding args.
    RemoveJumpNodeArgs(block, need_removed_indexes, manager);
  }
}

// ParseFunctionAst class code
bool ParseFunctionAst::InitParseAstInfo(const std::string &python_mod_get_parse_method) {
  // Init the type
  target_type_ = PARSE_TARGET_UNKNOW;

  // Call python parse, get the parser fn
  module_ = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::object parse_method = python_adapter::GetPyObjAttr(obj_, PYTHON_PARSE_METHOD);

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
    // 'obj' is class instance, get the method to parse.
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
    MS_LOG(WARNING) << "Parse obj is invalid, only can parse function and obj, type: " << type;
    return false;
  }

  // Call python parse get ast tree
  parser_ = python_adapter::CallPyModFn(module_, PYTHON_MOD_PARSE_OBJECT_FUNCTION, function_, parse_method);
  py::tuple ast_info = python_adapter::CallPyObjMethod(parser_, "parse");
  const size_t ast_info_size = 2;
  if (ast_info.size() != ast_info_size) {
    MS_EXCEPTION(NameError) << "ast info size is not equal to 2.";
  }
  ast_tokens_ = ast_info[0];
  ast_tree_ = ast_info[1];

  // Get fn name and module
  function_module_ = py::cast<std::string>(python_adapter::GetPyObjAttr(parser_, "function_module"));
  function_name_ = py::cast<std::string>(python_adapter::GetPyObjAttr(parser_, "function_name"));
  function_filename_ = py::cast<std::string>(python_adapter::GetPyObjAttr(parser_, "filename"));
  function_line_offset_ = py::cast<int64_t>(python_adapter::GetPyObjAttr(parser_, "line_offset"));

  return true;
}

// Get ast tree node : is the tree bode list[0]
py::object ParseFunctionAst::GetAstNode() {
  py::list tree_body = python_adapter::GetPyObjAttr(ast_tree_, "body");
  py::object ast_node = tree_body[0];
  return ast_node;
}

// Get ast tokens node text.
py::str ParseFunctionAst::GetAstNodeText(const py::object &node_obj) {
  return python_adapter::CallPyObjMethod(ast_tokens_, "get_text", node_obj);
}

py::list ParseFunctionAst::GetArgs(const py::object &func_node) {
  py::list ret = python_adapter::CallPyModFn(module_, PYTHON_PARSE_GET_ARGS, func_node);
  return ret;
}

py::list ParseFunctionAst::GetArgsDefaultValues(const py::object &func_node) {
  py::list ret = python_adapter::CallPyModFn(module_, PYTHON_PARSE_GET_ARGS_DEFAULT_VALUES, func_node);
  return ret;
}

AstNodeTypePtr ParseFunctionAst::GetNodeType(const py::object &node) {
  py::list list_value = python_adapter::CallPyModFn(module_, PYTHON_PARSE_GET_NODE_TYPE, node);
  const size_t list_value_size = 2;
  if (list_value.size() < list_value_size) {
    MS_LOG(EXCEPTION) << "The node of python method must has 2 values.";
  }
  auto node_name = py::cast<std::string>(list_value[0]);
  auto type = AstMainType(py::cast<int32_t>(list_value[1]));
  return std::make_shared<AstNodeType>(node, node_name, type);
}

AstSubType ParseFunctionAst::GetOpType(const py::object &node) {
  auto op_type = AstSubType(python_adapter::CallPyModFn(module_, PYTHON_PARSE_GET_AST_TYPE, node).cast<int32_t>());
  return op_type;
}

bool ParseFunctionAst::IsClassMember(const py::object &node) {
  py::object ret = CallParseModFunction(PYTHON_MOD_PARSE_CHECK_IS_CLASS_MEMBER, node);
  if (!py::isinstance<py::bool_>(ret)) {
    MS_LOG(ERROR) << "The result of mod function parse, should be bool type.";
    return false;
  }
  return ret.cast<bool>();
}

bool ParseFunctionAst::IsClassMemberRecursive(const py::object &node) {
  py::object ret = CallParseModFunction(PYTHON_MOD_PARSE_CHECK_IS_CLASS_MEMBER_RECURSIVE, node);
  if (!py::isinstance<py::bool_>(ret)) {
    MS_LOG(ERROR) << "The result of mod function parse, should be bool type.";
    return false;
  }
  return ret.cast<bool>();
}

void SetMixedPrecisionFlag(const py::object &obj, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!py::isinstance<Cell>(obj)) {
    return;
  }
  auto cell = py::cast<CellPtr>(obj);
  MS_EXCEPTION_IF_NULL(cell);
  auto mixed_type = cell->GetMixedPrecisionType();
  if (mixed_type != MixedPrecisionType::kNotSet) {
    func_graph->set_flag(GRAPH_FLAG_MIX_PRECISION_FP16, mixed_type == MixedPrecisionType::kFP16);
    func_graph->set_flag(GRAPH_FLAG_MIX_PRECISION_FP32, mixed_type == MixedPrecisionType::kFP32);
  }
}

bool UpdateFuncGraphFlags(const py::object &obj, const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "FuncGraph is null";
    return false;
  }

  SetMixedPrecisionFlag(obj, func_graph);

  if (!py::hasattr(obj, PYTHON_FUNC_GRAPH_FLAGS)) {
    MS_LOG(DEBUG) << "No flags";
    return true;
  }
  py::dict flags = python_adapter::GetPyObjAttr(obj, PYTHON_FUNC_GRAPH_FLAGS);
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
  MS_EXCEPTION_IF_NULL(func_graph);
  auto res = func_graph->NewCNodeInOrder({});
  std::vector<CNodePtr> new_cnodes;
  new_cnodes.emplace_back(res);
  while (index < old_cnodes.size()) {
    auto current = old_cnodes[index];
    auto current_new_cnode = new_cnodes[index];
    index++;
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
  MS_EXCEPTION_IF_NULL(func_graph->debug_info());
  func_graph->debug_info()->set_name(current_graph->debug_info()->name() + "_wrapper");
  func_graph->debug_info()->set_location(current_graph->debug_info()->location());

  // Copy all parameters information
  for (auto &para : current_graph->parameters()) {
    auto param = func_graph->add_parameter();
    auto orig_param = para->cast<ParameterPtr>();
    auto name = orig_param->name();
    param->set_name(name);
    MS_EXCEPTION_IF_NULL(param->debug_info());
    param->debug_info()->set_name(name);
    param->debug_info()->set_location(orig_param->debug_info()->location());
    param->set_is_top_graph_param(true);
  }
  func_graph->set_has_vararg(current_graph->has_vararg());
  func_graph->set_has_kwarg(current_graph->has_kwarg());
  func_graph->set_kwonlyargs_count(current_graph->kwonlyargs_count());
  // Copy all default values
  for (const auto &d : current_graph->parameter_default_value()) {
    func_graph->set_param_default_value(d.first, CopyNodesFromParamDefaultValue(func_graph, d.second));
  }

  // cell_obj
  MS_LOG(DEBUG) << "Add flag for " << std::string(py::str(cell));
  parse::UpdateFuncGraphFlags(cell, func_graph);
  // Top graph's construct flag
  if (py::hasattr(cell, "construct")) {
    parse::UpdateFuncGraphFlags(cell.attr("construct"), func_graph);
  }
  if (current_graph->has_flag(FUNC_GRAPH_FLAG_DUMP) && func_graph->has_flag(FUNC_GRAPH_FLAG_DUMP)) {
    current_graph->erase_flag(FUNC_GRAPH_FLAG_DUMP);
  }

  auto unpacking = func_graph->has_vararg() || func_graph->has_kwarg();
  MS_EXCEPTION_IF_NULL(current_graph->get_return());
  MS_EXCEPTION_IF_NULL(current_graph->get_return()->debug_info());
  if (!unpacking) {
    std::vector<AnfNodePtr> inputs;
    inputs.emplace_back(NewValueNode(cell_ptr));
    auto &params = func_graph->parameters();
    (void)std::transform(params.begin(), params.end(), std::back_inserter(inputs),
                         [](AnfNodePtr node) -> AnfNodePtr { return node; });
    auto call_node = func_graph->NewCNodeInOrder(std::move(inputs));
    // Set output as ret
    TraceGuard guard(current_graph->get_return()->debug_info()->location());
    func_graph->set_output(call_node);
  } else {
    // ret = cell_obj(*arg, *kwargs)
    auto call_fn = MakeUnpackCall(func_graph, NewValueNode(cell_ptr), func_graph->parameters());
    // Set output as ret
    TraceGuard guard(current_graph->get_return()->debug_info()->location());
    func_graph->set_output(call_fn);
  }
  return func_graph;
}

bool Parser::IsSubscriptReferenceType(const py::object &obj) {
  py::object slice_node = python_adapter::GetPyObjAttr(obj, "slice");
  auto node_type = ast_->GetNodeType(slice_node);
  auto node_name = node_type->node_name();
  return node_name != "Slice";
}
}  // namespace parse
}  // namespace mindspore
