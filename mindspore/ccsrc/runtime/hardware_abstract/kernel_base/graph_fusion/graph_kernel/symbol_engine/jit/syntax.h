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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_JIT_SYNTAX_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_JIT_SYNTAX_H_
#include <string>
#include <memory>
#include <vector>

namespace mindspore::graphkernel::symshape::ast {
/*
 Term ::= SingleTerm | Shape
 Shape ::= List[SingleTerm]
 SingleTerm ::= IntImm int | Symbol
 Symbol ::= BinOp BinOpType Term Term   // binary shape function, BinOpType is tag indicating Operation type
         |  Input int int      // from shape of input: input_i.shape[j]
         | Var id               // a symbol represent
*/

enum class BinOpType { ScalarAdd, ScalarSub, ScalarMul, ScalarDiv, ScalarMin, ScalarMax };

class Visitor;

struct Term {
  virtual void Accept(Visitor *visitor) {}
  virtual std::string ToString() const { return "Term"; }
};
using TermPtr = std::shared_ptr<Term>;

struct SingleTerm : public Term {
  std::string ToString() const override { return "SingleTerm"; }
};
using SingleTermPtr = std::shared_ptr<SingleTerm>;

struct IntImm : public SingleTerm {
  int64_t shape_int;
  explicit IntImm(int64_t i) : shape_int(i) {}

  void Accept(Visitor *visitor);
  std::string ToString() const override;
};

struct Symbol : public SingleTerm {
  virtual void Accept(Visitor *visitor) = 0;
  std::string ToString() const override { return "Var"; }
};

struct Var : public Symbol {
  size_t id_;
  std::string name_;

  Var(size_t id, const std::string &name) : id_(id), name_(name) {}
  void Accept(Visitor *visitor) override;
  std::string ToString() const override;
};
using VarPtr = std::shared_ptr<Var>;

struct BinOp : public Symbol {
  BinOpType optype_;
  TermPtr a_;
  TermPtr b_;

  void Accept(Visitor *visitor) override;
  std::string ToString() const override;
};

struct Input : public Symbol {
  int64_t i_, j_;

  Input(int64_t i, int64_t j) : i_(i), j_(j) {}
  void Accept(Visitor *visitor) override;
  std::string ToString() const override;
};

struct Shape : public Term {
  std::vector<SingleTermPtr> smbls_;
  void Accept(Visitor *visitor) override;
  std::string ToString() const override;
};
using ShapePtr = std::shared_ptr<Shape>;

class Visitor {
 public:
  virtual void Visit(const IntImm &intimm) = 0;
  virtual void Visit(const BinOp &op) = 0;
  virtual void Visit(const Input &input) = 0;
  virtual void Visit(const Var &val) = 0;
  virtual void Visit(const Shape &shape) {}
};

using SymbolTable = std::vector<TermPtr>;
}  // namespace mindspore::graphkernel::symshape::ast

#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_JIT_SYNTAX_H_
