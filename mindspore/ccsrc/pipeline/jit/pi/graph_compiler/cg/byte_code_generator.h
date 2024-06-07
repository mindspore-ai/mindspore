/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PI_JIT_BYTE_CODE_GENERATOR_H_
#define MINDSPORE_PI_JIT_BYTE_CODE_GENERATOR_H_

#include <Python.h>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "pipeline/jit/pi/graph_compiler/pi_ir/ir_visitor.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace pijit {
namespace py = pybind11;

// ByteCodeGenerator to parse python byte code
class ByteCodeGenerator : public ir::IRVisitor {
 public:
  virtual ~ByteCodeGenerator() = default;
  static py::object GenFunction(const ir::FunctionNodePtr &func);
  py::object Generate(const ir::FunctionNodePtr &func);

  // overloadable Mutate function.
  void Visit_(const ir::ParameterPtr &node) override;
  void Visit_(const ir::ValuePtr &node) override;
  void Visit_(const ir::UnaryOperationPtr &node) override;
  void Visit_(const ir::BinaryOperationPtr &node) override;
  void Visit_(const ir::NaryOperationPtr &node) override;
  void Visit_(const ir::NegativeNodePtr &node) override;
  void Visit_(const ir::NotNodePtr &node) override;
  void Visit_(const ir::InvertNodePtr &node) override;
  void Visit_(const ir::ReturnNodePtr &node) override;
  void Visit_(const ir::CastNodePtr &node) override;
  void Visit_(const ir::DeleteNodePtr &node) override;
  void Visit_(const ir::GetNodePtr &node) override;
  void Visit_(const ir::FormatNodePtr &node) override;
  void Visit_(const ir::AddNodePtr &node) override;
  void Visit_(const ir::SubNodePtr &node) override;
  void Visit_(const ir::MulNodePtr &node) override;
  void Visit_(const ir::DivNodePtr &node) override;
  void Visit_(const ir::BitwiseNodePtr &node) override;
  void Visit_(const ir::IsNodePtr &node) override;
  void Visit_(const ir::ContainsNodePtr &node) override;
  void Visit_(const ir::StoreNodePtr &node) override;
  void Visit_(const ir::JumpNodePtr &node) override;
  void Visit_(const ir::CompareNodePtr &node) override;
  void Visit_(const ir::UpdateNodePtr &node) override;
  void Visit_(const ir::LoadValueNodePtr &node) override;
  void Visit_(const ir::LoadFieldNodePtr &node) override;
  void Visit_(const ir::BuildNodePtr &node) override;
  void Visit_(const ir::CallNodePtr &node) override;
  void Visit_(const ir::NaryWithFlagNodePtr &node) override;

 private:
  class CellVarCounter : public ir::IRVisitor {
   public:
    static int GetCount(const ir::NodePtr node) {
      auto counter = std::make_shared<CellVarCounter>();
      counter->Visit(node);
      return counter->GetCount();
    }

   private:
    int GetCount() const { return cell_var_cnt_; }

    void Visit_(const ir::ValuePtr &node) override {
      if (node->GetScope() == ir::kScopeCellVar) {
        cell_var_cnt_++;
      }
    }

    int cell_var_cnt_{0};
  };

  int GetValueIndex(const ir::ValuePtr &node);
  void CheckInstrOffset(const ir::NodePtr &node);
  void GenerateInstr(ir::OpCode op, int arg = 0);
  void SetStartsLine(const ir::NodePtr &node);

  PyFunctionObject *func_graph_{nullptr};
  std::vector<_Py_CODEUNIT> co_code_;
  // A string encoding the mapping from bytecode offsets to line numbers.
  std::vector<char> co_lnotab_;
  int first_line_no_{0};
  ir::NodePtr last_starts_instr_{nullptr};
  py::dict globals_;
  py::dict builtins_;
  py::list co_consts_;
  std::unordered_map<std::string, int> co_consts_map_;
  py::list co_var_names_;
  std::unordered_map<std::string, int> co_var_names_map_;
  py::list co_names_;
  std::unordered_map<std::string, int> co_names_map_;
  py::list co_free_vars_;
  std::unordered_map<std::string, int> co_free_vars_map_;
  py::list co_cell_vars_;
  std::unordered_map<std::string, int> co_cell_vars_map_;
  py::list clousre_;
  std::unordered_map<std::string, int> clousre_map_;
  py::list defaults_;
  py::dict kwdefaults_;
  const std::map<ir::Scope, std::unordered_map<std::string, int> *> scope_inquire_map_ = {
    {ir::kScopeConst, &co_consts_map_},      {ir::kScopeLocal, &co_var_names_map_},
    {ir::kScopeName, &co_names_map_},        {ir::kScopeFreeVar, &co_free_vars_map_},
    {ir::kScopeCellVar, &co_cell_vars_map_}, {ir::kScopeClousre, &co_free_vars_map_},
    {ir::kScopeBuiltIn, &co_names_map_},     {ir::kScopeGlobal, &co_names_map_}};
  const std::map<ir::Scope, std::pair<py::list, py::object>> scope_value_list_ = {
    {ir::kScopeConst, {co_consts_, co_consts_}},
    {ir::kScopeLocal, {co_var_names_, co_var_names_}},
    {ir::kScopeName, {co_names_, co_names_}},
    {ir::kScopeFreeVar, {co_free_vars_, co_free_vars_}},
    {ir::kScopeCellVar, {co_cell_vars_, co_cell_vars_}},
    {ir::kScopeClousre, {co_free_vars_, clousre_}},
    {ir::kScopeBuiltIn, {co_names_, builtins_}},
    {ir::kScopeGlobal, {co_names_, globals_}}};
  int cell_var_cnt_{0};
};

using ByteCodeGeneratorPtr = std::shared_ptr<ByteCodeGenerator>;
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_BYTE_CODE_GENERATOR_H_
