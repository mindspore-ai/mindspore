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
#ifndef MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_SYMBOL_ENGINE_H_
#define MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_SYMBOL_ENGINE_H_
#include <memory>
#include <string>
#include <unordered_map>
#include "base/base.h"
#include "ir/anf.h"
#include "abstract/abstract_value.h"

namespace mindspore {
class FuncGraph;
using FuncGraphPtr = std::shared_ptr<FuncGraph>;
using FuncGraphWeakPtr = std::weak_ptr<FuncGraph>;
namespace symshape {
class MS_CORE_API SymbolEngine : public Base {
 public:
  explicit SymbolEngine(const FuncGraphPtr &fg) : func_graph_(fg) {}
  ~SymbolEngine() = default;
  MS_DECLARE_PARENT(SymbolEngine, Base)

  virtual bool Infer(const AbstractBasePtrList &inputs) = 0;
  bool SupportInfer() { return support_infer_; }
  FuncGraphPtr func_graph() const { return func_graph_.lock(); }

 protected:
  FuncGraphWeakPtr func_graph_;
  bool support_infer_{true};
};
}  // namespace symshape
using SymbolEnginePtr = std::shared_ptr<symshape::SymbolEngine>;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_SYMBOL_ENGINE_H_
