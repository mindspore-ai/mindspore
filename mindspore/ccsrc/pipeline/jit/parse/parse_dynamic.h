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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_PARSE_DYNAMIC_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_PARSE_DYNAMIC_H_

#include <vector>
#include <memory>
#include <string>
#include "pipeline/jit/parse/parse.h"
#include "pybind11/pytypes.h"

namespace mindspore::parse {

enum class ExprType { kAssignedExpr, kConditionExpr };
class DynamicParser {
 public:
  DynamicParser() = default;
  ~DynamicParser() = default;

  // Check cell struct
  static bool IsDynamicCell(const py::object &cell);

 private:
  static std::string GetCellInfo(const py::object &cell);
  static void ParseInputArgs(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &fn_node);
  static bool ParseBodyOrElseContext(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &fn_node,
                                     const std::string &body_type);
  static bool ParseConditionExper(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node);
  static bool ParseIfWhileExprNode(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node);
  static bool ParseCompareExpr(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &cmp_node);
  static bool ParseSubsciptExpr(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node);
  static bool ParseAssignExprNode(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node);
  static bool ParseAugAssignExprNode(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node);
  static bool ParseForExprNode(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node);
  static std::string ParseNodeName(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node,
                                   parse::AstMainType type);
  static bool CheckAttributeInExpr(const std::shared_ptr<parse::ParseFunctionAst> &ast, const py::object &node,
                                   const ExprType &type);
};
}  // namespace mindspore::parse

#endif
