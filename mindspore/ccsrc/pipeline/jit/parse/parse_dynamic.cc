/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include <set>
#include <vector>
#include <string>
#include <memory>
#include "utils/hash_set.h"
#include "pipeline/jit/parse/parse_dynamic.h"
#include "mindspore/core/ir/cell.h"

namespace mindspore::parse {
static mindspore::HashSet<std::string> cell_input_args_ = {};
static mindspore::HashSet<std::string> cell_assigned_attributes_ = {};
static mindspore::HashSet<std::string> cell_condition_expr_attributes_ = {};
static const std::set<std::string> ignore_judge_dynamic_cell = {
  "Cell mindspore.nn.layer.basic.Dense",
  "Cell mindspore.nn.probability.distribution.normal.Normal",
  "Cell src.transformer.create_attn_mask.CreateAttentionMaskFromInputMask",
  "Cell mindspore.nn.layer.math.MatMul",
  "Cell mindspore.nn.layer.rnns.LSTM",
  "Cell mindspore.nn.layer.rnns._DynamicLSTMAscend"};
static const std::set<std::string> unchanged_named_primitive = {
  parse::NAMED_PRIMITIVE_ATTRIBUTE, parse::NAMED_PRIMITIVE_NAMECONSTANT, parse::NAMED_PRIMITIVE_CONSTANT,
  parse::NAMED_PRIMITIVE_NUM, parse::NAMED_PRIMITIVE_STR};

std::string DynamicParser::ParseNodeName(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node,
                                         parse::AstMainType type) {
  MS_EXCEPTION_IF_NULL(ast);
  if (py::isinstance<py::none>(node)) {
    MS_LOG(DEBUG) << "Get none type node!";
    return "";
  }
  auto node_type = ast->GetNodeType(node);
  MS_EXCEPTION_IF_NULL(node_type);
  // Check node type
  parse::AstMainType node_main_type = node_type->main_type();
  if (node_main_type != type) {
    MS_LOG(ERROR) << "Node type is wrong: " << node_main_type << ", it should be " << type;
    return "";
  }
  std::string node_name = node_type->node_name();
  MS_LOG(DEBUG) << "Ast node is " << node_name;
  return node_name;
}

bool DynamicParser::CheckAttributeInExpr(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node,
                                         const ExprType &expr_type) {
  MS_EXCEPTION_IF_NULL(ast);
  auto variable_type = ParseNodeName(ast, node, parse::AST_MAIN_TYPE_EXPR);
  if (variable_type == parse::NAMED_PRIMITIVE_ATTRIBUTE) {
    auto target_value = python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_VALUE);
    std::string target_variabe;
    if (py::hasattr(target_value, "id") && py::hasattr(node, "attr")) {
      target_variabe = py::cast<std::string>(target_value.attr("id")) + py::cast<std::string>(node.attr("attr"));
    }
    if (expr_type == ExprType::kAssignedExpr) {
      if (cell_condition_expr_attributes_.find(target_variabe) != cell_condition_expr_attributes_.end()) {
        return true;
      }
      cell_assigned_attributes_.insert(target_variabe);
    } else if (expr_type == ExprType::kConditionExpr) {
      MS_LOG(DEBUG) << "target variable " << target_variabe << " kConditionExpr";
      if (cell_assigned_attributes_.find(target_variabe) != cell_assigned_attributes_.end()) {
        return true;
      }
      cell_condition_expr_attributes_.insert(target_variabe);
    }
  }
  return false;
}

void DynamicParser::ParseInputArgs(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &fn_node) {
  MS_EXCEPTION_IF_NULL(ast);
  py::list args = ast->GetArgs(fn_node);
  for (size_t i = 1; i < args.size(); i++) {
    std::string arg_name = py::cast<std::string>(args[i].attr("arg"));
    MS_LOG(DEBUG) << "Input arg name: " << arg_name;
    (void)cell_input_args_.emplace(arg_name);
  }
}

bool DynamicParser::ParseIfWhileExprNode(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node) {
  MS_LOG(DEBUG) << "Parse if/while expr";
  py::object test_node = python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_TEST);
  if (ParseConditionExper(ast, test_node)) {
    return true;
  }
  return ParseBodyOrElseContext(ast, node, parse::NAMED_PRIMITIVE_BODY) ||
         ParseBodyOrElseContext(ast, node, parse::NAMED_PRIMITIVE_ORELSE);
}

bool DynamicParser::ParseConditionExper(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node) {
  MS_LOG(DEBUG) << "Parse condition expr";
  const auto &node_name = ParseNodeName(ast, node, parse::AST_MAIN_TYPE_EXPR);
  if (node_name == parse::NAMED_PRIMITIVE_BOOLOP) {
    py::object value_object = python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_VALUES);
    py::list value_nodes = py::cast<py::list>(value_object);
    if (value_nodes.empty()) {
      MS_LOG(DEBUG) << "parse value_nodes none";
      return false;
    }
    for (size_t i = 0; i < value_nodes.size(); ++i) {
      py::object value_node = value_nodes[i];
      if (ParseConditionExper(ast, value_node)) {
        return true;
      }
    }
  } else if (node_name == parse::NAMED_PRIMITIVE_COMPARE && ParseCompareExpr(ast, node)) {
    return true;
  } else if (node_name == parse::NAMED_PRIMITIVE_SUBSCRIPT && ParseSubsciptExpr(ast, node)) {
    return true;
  } else if (node_name == parse::NAMED_PRIMITIVE_ATTRIBUTE &&
             CheckAttributeInExpr(ast, node, ExprType::kConditionExpr)) {
    return true;
  } else if (node_name == parse::NAMED_PRIMITIVE_NAME) {  // if flag:
    std::string id = py::cast<std::string>(node.attr("id"));
    if (cell_input_args_.find(id) != cell_input_args_.end()) {
      return true;
    }
  }
  return false;
}

bool DynamicParser::ParseCompareExpr(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &cmp_node) {
  MS_LOG(DEBUG) << "Parse compare expr";
  py::object left_node = python_adapter::GetPyObjAttr(cmp_node, parse::NAMED_PRIMITIVE_LEFT);
  py::list comparators_node = python_adapter::GetPyObjAttr(cmp_node, parse::NAMED_PRIMITIVE_COMPARATORS);
  if (comparators_node.empty() || py::isinstance<py::none>(left_node)) {
    MS_LOG(DEBUG) << "parse cmp node failed!";
    return false;
  }

  auto left = ParseNodeName(ast, left_node, parse::AST_MAIN_TYPE_EXPR);
  auto right = ParseNodeName(ast, comparators_node[0], parse::AST_MAIN_TYPE_EXPR);
  if (left == parse::NAMED_PRIMITIVE_SUBSCRIPT && ParseSubsciptExpr(ast, left_node)) {
    return true;
  } else if (left == parse::NAMED_PRIMITIVE_ATTRIBUTE &&
             CheckAttributeInExpr(ast, left_node, ExprType::kConditionExpr)) {
    return true;
  }
  if (right == parse::NAMED_PRIMITIVE_SUBSCRIPT && ParseSubsciptExpr(ast, comparators_node[0])) {
    return true;
  } else if (right == parse::NAMED_PRIMITIVE_ATTRIBUTE &&
             CheckAttributeInExpr(ast, comparators_node[0], ExprType::kConditionExpr)) {
    return true;
  }

  MS_LOG(DEBUG) << "Left is " << left << " Right is " << right;
  if (unchanged_named_primitive.find(left) == unchanged_named_primitive.end() ||
      unchanged_named_primitive.find(right) == unchanged_named_primitive.end()) {
    return true;
  }
  return false;
}

bool DynamicParser::ParseSubsciptExpr(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node) {
  MS_LOG(DEBUG) << "Parse subscipt expr";
  py::object value_in_subscript = python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_VALUE);
  if (py::isinstance<py::none>(value_in_subscript)) {
    MS_LOG(DEBUG) << "parse subscipt node fail";
    return false;
  }
  auto value_node_name = ParseNodeName(ast, value_in_subscript, parse::AST_MAIN_TYPE_EXPR);
  if (value_node_name == parse::NAMED_PRIMITIVE_ATTRIBUTE &&
      CheckAttributeInExpr(ast, value_in_subscript, ExprType::kConditionExpr)) {
    return true;
  }
  return false;
}

bool DynamicParser::ParseAssignExprNode(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node) {
  MS_LOG(DEBUG) << "Parse assign expr";
  py::object value_node = python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_VALUE);
  const auto &node_name = ParseNodeName(ast, value_node, parse::AST_MAIN_TYPE_EXPR);
  if (node_name == parse::NAMED_PRIMITIVE_CALL) {
    py::object func_node = python_adapter::GetPyObjAttr(value_node, parse::NAMED_PRIMITIVE_FUNC);
    const auto &func_name = ParseNodeName(ast, func_node, parse::AST_MAIN_TYPE_EXPR);
    if (func_name == parse::NAMED_PRIMITIVE_SUBSCRIPT) {
      py::object slice_node = python_adapter::GetPyObjAttr(func_node, parse::NAMED_PRIMITIVE_SLICE);
      py::object value_in_slice_node = python_adapter::GetPyObjAttr(slice_node, parse::NAMED_PRIMITIVE_VALUE);
      if (py::isinstance<py::none>(value_in_slice_node)) {
        MS_LOG(DEBUG) << "Parse value node is none!";
        return false;
      }
      const auto &node_name_in_slice_node = ParseNodeName(ast, value_in_slice_node, parse::AST_MAIN_TYPE_EXPR);
      std::string id;
      if (py::hasattr(value_in_slice_node, "id")) {
        id = py::cast<std::string>(value_in_slice_node.attr("id"));
      }
      if (cell_input_args_.find(node_name_in_slice_node) != cell_input_args_.end() ||
          (!id.empty() && cell_input_args_.find(id) != cell_input_args_.end())) {
        return true;
      }
    }
  }

  py::object target = python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_TARGETS);
  py::list target_nodes = py::cast<py::list>(target);
  if (target_nodes.empty()) {
    MS_LOG(DEBUG) << "Parse target nodes is none!";
    return false;
  }
  for (size_t i = 0; i < target_nodes.size(); ++i) {
    py::object target_node = target_nodes[i];
    if (CheckAttributeInExpr(ast, target_node, ExprType::kAssignedExpr)) {
      return true;
    }
  }
  return false;
}

bool DynamicParser::ParseAugAssignExprNode(const std::shared_ptr<parse::ParseFunctionAst> &ast,
                                           const py::object &node) {
  MS_LOG(DEBUG) << "Parse augassign expr";
  py::object target_node = python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_TARGET);
  if (py::isinstance<py::none>(target_node)) {
    MS_LOG(DEBUG) << "Parse target node is none!";
    return false;
  }
  if (CheckAttributeInExpr(ast, target_node, ExprType::kAssignedExpr)) {
    return true;
  }
  return false;
}

bool DynamicParser::ParseForExprNode(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node) {
  MS_LOG(DEBUG) << "Parse for expr";
  py::object iter_node = python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_ITER);
  if (py::isinstance<py::none>(iter_node)) {
    MS_LOG(DEBUG) << "Parse iter node is none!";
    return false;
  }
  const auto &iter_node_name = ParseNodeName(ast, iter_node, parse::AST_MAIN_TYPE_EXPR);
  // for i in range(self.a) and for i in self.a[0] and for i in self.a
  if (iter_node_name == parse::NAMED_PRIMITIVE_CALL) {
    py::object func_obj = python_adapter::GetPyObjAttr(iter_node, parse::NAMED_PRIMITIVE_ARGS);
    py::list arg_nodes = py::cast<py::list>(func_obj);
    if (arg_nodes.empty()) {
      MS_LOG(DEBUG) << "Parse arg_nodes is none!";
      return false;
    }
    for (size_t i = 0; i < arg_nodes.size(); ++i) {
      py::object arg_node = arg_nodes[i];
      const auto arg_name = ParseNodeName(ast, arg_node, parse::AST_MAIN_TYPE_EXPR);
      if (arg_name == parse::NAMED_PRIMITIVE_SUBSCRIPT && ParseSubsciptExpr(ast, arg_node)) {
        return true;
      } else if (arg_name == parse::NAMED_PRIMITIVE_ATTRIBUTE &&
                 CheckAttributeInExpr(ast, arg_node, ExprType::kConditionExpr)) {
        return true;
      }
    }
  } else if (iter_node_name == parse::NAMED_PRIMITIVE_SUBSCRIPT && ParseSubsciptExpr(ast, iter_node)) {
    return true;
  } else if (iter_node_name == parse::NAMED_PRIMITIVE_ATTRIBUTE &&
             CheckAttributeInExpr(ast, iter_node, ExprType::kConditionExpr)) {
    return true;
  }
  return ParseBodyOrElseContext(ast, node, parse::NAMED_PRIMITIVE_BODY);
}

bool DynamicParser::ParseBodyOrElseContext(const std::shared_ptr<parse::ParseFunctionAst> &ast,
                                           const py::object &fn_node, const std::string &body_type) {
  MS_EXCEPTION_IF_NULL(ast);
  py::object func_obj = python_adapter::GetPyObjAttr(fn_node, body_type);
  if (py::isinstance<py::none>(func_obj)) {
    MS_LOG(DEBUG) << "Parse body of cell is none!";
    return false;
  }
  py::int_ pcount = python_adapter::CallPyObjMethod(func_obj, parse::PYTHON_GET_METHOD_LEN);
  size_t count = IntToSize(pcount);
  MS_LOG(DEBUG) << "The nodes count in body is " << count;
  bool ret = false;
  for (size_t i = 0; i < count; ++i) {
    auto node = py::cast<py::list>(func_obj)[i];
    const auto &node_name = ParseNodeName(ast, node, parse::AST_MAIN_TYPE_STMT);
    if (node_name == parse::NAMED_PRIMITIVE_ASSIGN) {
      ret = ParseAssignExprNode(ast, node);
    } else if (node_name == parse::NAMED_PRIMITIVE_AUGASSIGN) {
      ret = ParseAugAssignExprNode(ast, node);
    } else if (node_name == parse::NAMED_PRIMITIVE_FOR) {
      ret = ParseForExprNode(ast, node);
    } else if (node_name == parse::NAMED_PRIMITIVE_IF || node_name == parse::NAMED_PRIMITIVE_WHILE) {
      ret = ParseIfWhileExprNode(ast, node);
    }
    if (ret) {
      MS_LOG(INFO) << "Current cell is dynamic!";
      break;
    }
  }
  return ret;
}

std::string DynamicParser::GetCellInfo(const py::object &cell) {
  if (py::isinstance<Cell>(cell)) {
    auto c_cell = py::cast<CellPtr>(cell);
    MS_EXCEPTION_IF_NULL(c_cell);
    auto cell_info = c_cell->ToString();
    return cell_info;
  }
  return "";
}

bool DynamicParser::IsDynamicCell(const py::object &cell) {
  std::string cell_info = GetCellInfo(cell);
  if (ignore_judge_dynamic_cell.find(cell_info) != ignore_judge_dynamic_cell.end()) {
    return false;
  }
  // Using ast parse to check whether the construct of cell will be changed
  auto ast = std::make_shared<parse::ParseFunctionAst>(cell);
  bool success = ast->InitParseAstInfo(parse::PYTHON_MOD_GET_PARSE_METHOD);
  if (!success) {
    MS_LOG(DEBUG) << "Parse code to ast tree failed";
    return false;
  }
  py::object fn_node = ast->GetAstNode();
  // get the name of input args as the initialize of dynamic_variables
  ParseInputArgs(ast, fn_node);
  // parse body context
  bool ret = ParseBodyOrElseContext(ast, fn_node, parse::NAMED_PRIMITIVE_BODY);
  if (ret) {
    MS_LOG(INFO) << "Cell is dynamic, filename:" << ast->function_filename() << " line:" << ast->function_line_offset()
                 << " module:" << ast->function_module() << " function name:" << ast->function_name();
  }
  cell_input_args_.clear();
  cell_assigned_attributes_.clear();
  cell_condition_expr_attributes_.clear();
  return ret;
}
}  // namespace mindspore::parse
