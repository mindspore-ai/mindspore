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
#include "pipeline/jit/parse/parse_dynamic.h"
#include <set>
#include <vector>
#include <string>
#include <memory>
#include "utils/hash_set.h"
#include "ir/cell.h"

namespace mindspore::parse {
static mindspore::HashSet<std::string> cell_input_args_ = {};
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

void DynamicParser::ParseInputArgs(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &fn_node) {
  MS_EXCEPTION_IF_NULL(ast);
  py::list args = ast->GetArgs(fn_node);
  for (size_t i = 1; i < args.size(); i++) {
    std::string arg_name = py::cast<std::string>(args[i].attr("arg"));
    MS_LOG(DEBUG) << "Input arg name: " << arg_name;
    (void)cell_input_args_.emplace(arg_name);
  }
}

bool DynamicParser::ParseIfCompare(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &test_node) {
  py::object left_node = python_adapter::GetPyObjAttr(test_node, parse::NAMED_PRIMITIVE_LEFT);
  py::list comparators_node = python_adapter::GetPyObjAttr(test_node, parse::NAMED_PRIMITIVE_COMPARATORS);
  if (comparators_node.empty()) {
    MS_LOG(DEBUG) << "Get comparators node failed!";
    return false;
  }
  auto left = ParseNodeName(ast, left_node, parse::AST_MAIN_TYPE_EXPR);
  auto right = ParseNodeName(ast, comparators_node[0], parse::AST_MAIN_TYPE_EXPR);
  // while self.a > self.b and changed self.a or self.b
  if (left == parse::NAMED_PRIMITIVE_ATTRIBUTE && right == parse::NAMED_PRIMITIVE_ATTRIBUTE) {
    auto left_value = python_adapter::GetPyObjAttr(left_node, parse::NAMED_PRIMITIVE_VALUE);
    std::string left_variable;
    if (py::hasattr(left_node, "attr") && py::hasattr(left_value, "id")) {
      left_variable = py::cast<std::string>(left_value.attr("id")) + py::cast<std::string>(left_node.attr("attr"));
    }
    auto right_value = python_adapter::GetPyObjAttr(comparators_node[0], parse::NAMED_PRIMITIVE_VALUE);
    std::string right_variable;
    if (py::hasattr(comparators_node[0], "attr") && py::hasattr(right_value, "id")) {
      right_variable =
        py::cast<std::string>(right_value.attr("id")) + py::cast<std::string>(comparators_node[0].attr("attr"));
    }
    py::object func_obj = python_adapter::GetPyObjAttr(test_node, parse::NAMED_PRIMITIVE_BODY);
    if (py::isinstance<py::none>(func_obj)) {
      MS_LOG(DEBUG) << "Parse body of cell is none!";
      return false;
    }
    return ParseContext(ast, func_obj, {left_variable, right_variable});
  }
  // if a[0]
  if (left == parse::NAMED_PRIMITIVE_SUBSCRIPT) {
    py::object value_in_subscript = python_adapter::GetPyObjAttr(left_node, parse::NAMED_PRIMITIVE_VALUE);
    left = ParseNodeName(ast, value_in_subscript, parse::AST_MAIN_TYPE_EXPR);
    if (left == parse::NAMED_PRIMITIVE_NAME) {
      std::string id = py::cast<std::string>(value_in_subscript.attr("id"));
      if (cell_input_args_.find(id) != cell_input_args_.end()) {
        return true;
      }
    }
  }
  MS_LOG(DEBUG) << "Left is " << left << " Right is " << right;
  if (unchanged_named_primitive.find(left) == unchanged_named_primitive.end() ||
      unchanged_named_primitive.find(right) == unchanged_named_primitive.end()) {
    return true;
  }
  return false;
}

bool DynamicParser::ParseIfWhileExprNode(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node) {
  MS_LOG(DEBUG) << "Parse if/while expr";
  py::object test_node = python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_TEST);
  const auto &node_name = ParseNodeName(ast, test_node, parse::AST_MAIN_TYPE_EXPR);
  if (node_name == parse::NAMED_PRIMITIVE_COMPARE && ParseIfCompare(ast, test_node)) {
    return true;
  }
  // if flag:
  if (node_name == parse::NAMED_PRIMITIVE_NAME) {
    std::string id = py::cast<std::string>(test_node.attr("id"));
    if (cell_input_args_.find(id) != cell_input_args_.end()) {
      return true;
    }
  }
  // parse body context
  py::object body_obj = python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_BODY);
  if (!py::isinstance<py::none>(body_obj) && ParseContext(ast, body_obj)) {
    return true;
  }
  // parse orelse context
  py::object orelse_obj = python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_ORELSE);
  if (!py::isinstance<py::none>(orelse_obj) && ParseContext(ast, orelse_obj)) {
    return true;
  }
  return false;
}

void DynamicParser::ParseInputArgsUsedInFuncCall(const std::shared_ptr<parse::ParseFunctionAst> &ast,
                                                 const py::object &value_node, const py::object &node) {
  py::list args = python_adapter::GetPyObjAttr(value_node, parse::NAMED_PRIMITIVE_ARGS);
  // Func whether use input, if func use input, save targes id. And checkout in If/For/While
  bool is_used_input = false;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &arg_node_name = ParseNodeName(ast, args[i], parse::AST_MAIN_TYPE_EXPR);
    if (arg_node_name == parse::NAMED_PRIMITIVE_NAME) {
      std::string id = py::cast<std::string>(args[i].attr("id"));
      if (cell_input_args_.find(id) != cell_input_args_.end()) {
        is_used_input = true;
        break;
      }
    }
  }
  if (is_used_input) {
    py::list targets_args = python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_TARGETS);
    for (size_t j = 0; j < targets_args.size(); ++j) {
      const auto &arg_node_name = ParseNodeName(ast, targets_args[j], parse::AST_MAIN_TYPE_EXPR);
      if (arg_node_name == parse::NAMED_PRIMITIVE_NAME) {
        const auto &target_id = py::cast<std::string>(targets_args[j].attr("id"));
        MS_LOG(DEBUG) << "Add target id " << target_id;
        (void)cell_input_args_.emplace(target_id);
      }
    }
  }
}

bool DynamicParser::ParseAssignExprNode(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node) {
  MS_LOG(DEBUG) << "Parse assign expr";
  py::object value_node = python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_VALUE);
  const auto &node_name = ParseNodeName(ast, value_node, parse::AST_MAIN_TYPE_EXPR);
  if (node_name == parse::NAMED_PRIMITIVE_CALL) {
    py::object func_node = python_adapter::GetPyObjAttr(value_node, parse::NAMED_PRIMITIVE_FUNC);
    ParseInputArgsUsedInFuncCall(ast, value_node, node);
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
  return false;
}

bool DynamicParser::ParseAugAssignExprNode(const std::shared_ptr<parse::ParseFunctionAst> &, const py::object &node,
                                           const std::vector<std::string> &compare_prim) {
  MS_LOG(DEBUG) << "Parse augassign expr";
  bool ret = false;
  if (compare_prim.empty()) {
    return ret;
  }
  py::object target_node = python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_TARGET);
  if (py::isinstance<py::none>(target_node)) {
    MS_LOG(DEBUG) << "Parse target node is none!";
    return ret;
  }
  py::object value_node = python_adapter::GetPyObjAttr(target_node, parse::NAMED_PRIMITIVE_VALUE);
  if (py::isinstance<py::none>(value_node)) {
    MS_LOG(DEBUG) << "Parse value node is none!";
    return ret;
  }
  std::string assign_prim;
  if (py::hasattr(target_node, "attr") && py::hasattr(value_node, "id")) {
    assign_prim = py::cast<std::string>(value_node.attr("id")) + py::cast<std::string>(target_node.attr("attr"));
  }
  auto iter = std::find(compare_prim.begin(), compare_prim.end(), assign_prim);
  if (iter != compare_prim.end()) {
    ret = true;
  }
  return ret;
}

bool DynamicParser::ParseForExprNode(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node) {
  MS_LOG(DEBUG) << "Parse for expr";
  py::list body_node = python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_BODY);
  if (py::isinstance<py::none>(body_node)) {
    MS_LOG(DEBUG) << "Parse body of for expression is none!";
    return false;
  }
  for (size_t i = 0; i < body_node.size(); ++i) {
    const auto &node_name = ParseNodeName(ast, body_node[i], parse::AST_MAIN_TYPE_STMT);
    if (node_name == parse::NAMED_PRIMITIVE_ASSIGN && ParseAssignExprNode(ast, body_node[i])) {
      return true;
    }
    if (ParseContext(ast, body_node[i])) {
      return true;
    }
  }
  return false;
}

bool DynamicParser::ParseContext(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &fn_node,
                                 const std::vector<std::string> &compare_prim) {
  MS_EXCEPTION_IF_NULL(ast);
  auto fn = [&ast, &compare_prim](const py::object &node) {
    bool ret = false;
    const auto &node_name = ParseNodeName(ast, node, parse::AST_MAIN_TYPE_STMT);
    if (node_name == parse::NAMED_PRIMITIVE_ASSIGN) {
      ret = ParseAssignExprNode(ast, node);
    } else if (node_name == parse::NAMED_PRIMITIVE_AUGASSIGN) {
      ret = ParseAugAssignExprNode(ast, node, compare_prim);
    } else if (node_name == parse::NAMED_PRIMITIVE_FOR) {
      ret = ParseForExprNode(ast, node);
    } else if (node_name == parse::NAMED_PRIMITIVE_IF || node_name == parse::NAMED_PRIMITIVE_WHILE) {
      ret = ParseIfWhileExprNode(ast, node);
    }
    return ret;
  };
  if (!py::isinstance<py::list>(fn_node) && !py::isinstance<py::tuple>(fn_node)) {
    return fn(fn_node);
  }
  auto list_node = py::cast<py::list>(fn_node);
  for (size_t i = 0; i < list_node.size(); ++i) {
    if (fn(list_node[i])) {
      return true;
    }
  }
  return false;
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
  py::object func_obj = python_adapter::GetPyObjAttr(fn_node, parse::NAMED_PRIMITIVE_BODY);
  if (py::isinstance<py::none>(func_obj)) {
    MS_LOG(DEBUG) << "Parse body of cell is none!";
    return false;
  }
  bool ret = ParseContext(ast, func_obj);
  if (ret) {
    MS_LOG(INFO) << "Cell is dynamic, filename:" << ast->function_filename() << " line:" << ast->function_line_offset()
                 << " module:" << ast->function_module() << " function name:" << ast->function_name();
  }
  cell_input_args_.clear();
  return ret;
}
}  // namespace mindspore::parse
