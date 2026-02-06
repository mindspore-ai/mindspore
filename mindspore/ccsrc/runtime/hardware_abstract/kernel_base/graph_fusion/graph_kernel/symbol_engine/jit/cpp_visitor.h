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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_JIT_CPP_VISITOR_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_JIT_CPP_VISITOR_H_

#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/symbol_engine/jit/syntax.h"

namespace mindspore::graphkernel::symshape {
class CppVisitor : public ast::Visitor {
 public:
  using DynFuncType = void (*)(const int64_t **, int64_t **);
  CppVisitor();
  explicit CppVisitor(const std::string &name) : name_(name) {}
  ~CppVisitor();

  /// \brief Generate c++ function corresponding to the ast
  /// \note func_name should be valid c++ function name
  /// \return name of the function
  std::string CodeGen(const std::vector<ast::ShapePtr> &shapes, const ast::SymbolTable &symbol_table,
                      const std::string &func_name = "");
  void Compile();
  DynFuncType LoadFunc(const std::string &func_name);

  //------ override  ast::Visitor -----------------

  void Visit(const ast::IntImm &intimm) override;
  void Visit(const ast::Var &intimm) override;
  void Visit(const ast::BinOp &op) override;
  void Visit(const ast::Shape &shape) override;
  void Visit(const ast::Input &input_smbl) override;
  // ------------------------------------------------

  std::string UniqueName() {
    static size_t idx = 1;
    return "s_" + std::to_string(idx++);
  }

 protected:
  // Do the actual compile work
  void CompileImpl();

 public:
  // for codegen
  const ast::SymbolTable *symbols_table_ = nullptr;  // a map: id -> symbol
  std::vector<std::string> cpp_sentences_;
  std::vector<int32_t> var_tag_;  // indicate if Var already generated code
  std::string name_;
  std::vector<std::string> func_blocks_;  // store generated functions
  std::thread compile_thread_;
  bool null_ = true;  // indicate whether no code is generated

  // for dynamic library
  void *dynlib_ = nullptr;
};
using CppVisitorPtr = std::shared_ptr<CppVisitor>;
}  // namespace mindspore::graphkernel::symshape
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_JIT_CPP_VISITOR_H_
