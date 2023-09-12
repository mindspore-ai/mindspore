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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_SYMBOL_ENGINE_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_SYMBOL_ENGINE_H_
#include <vector>
#include <utility>
#include <unordered_map>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "mindapi/base/shape_vector.h"
#include "include/backend/visible.h"

namespace mindspore {
constexpr auto kAttrSymbolEngine = "symbol_engine";
constexpr auto kAttrFuncGraph = "func_graph";
class BACKEND_EXPORT SymbolEngine : public Value {
 public:
  SymbolEngine() = default;
  ~SymbolEngine() override = default;
  MS_DECLARE_PARENT(SymbolEngine, Value)
  bool operator==(const Value &rhs) const override { return &rhs == this; }

  virtual bool ShapeEqual(const std::pair<AnfNodePtr, size_t> &a, const std::pair<AnfNodePtr, size_t> &b) = 0;
  virtual bool Infer(const AbstractBasePtrList &inputs) = 0;
  virtual ShapeArray QueryShape(const AnfNodePtr &node) = 0;
  virtual ShapeArray QueryValue(const AnfNodePtr &node) = 0;
  virtual std::vector<std::string> QuerySymbolicShape(const AnfNodePtr &node) = 0;
  virtual void QuerySymbolExpr(const AnfNodePtr &node,
                               std::unordered_map<std::string, std::string> *symbol_expr_map) = 0;
};
using SymbolEnginePtr = std::shared_ptr<SymbolEngine>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_SYMBOL_ENGINE_H_
