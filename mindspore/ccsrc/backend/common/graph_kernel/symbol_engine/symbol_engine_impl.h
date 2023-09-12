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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_SYMBOL_ENGINE_IMPL_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_SYMBOL_ENGINE_IMPL_H_
#include <vector>
#include <utility>
#include <unordered_map>
#include <string>
#include <memory>

#include "ir/anf.h"
#include "utils/hash_map.h"
#include "backend/common/graph_kernel/symbol_engine/symbol_engine.h"
#include "backend/common/graph_kernel/symbol_engine/symbol.h"
#include "backend/common/graph_kernel/symbol_engine/operation_builder.h"
#include "backend/common/graph_kernel/symbol_engine/operations/operation.h"

namespace mindspore::graphkernel::symbol {
class SymbolEngineImpl : public SymbolEngine {
 public:
  explicit SymbolEngineImpl(const std::string &name) : name_("SymbolEngine-" + name) {}
  ~SymbolEngineImpl() = default;
  MS_DECLARE_PARENT(SymbolEngineImpl, SymbolEngine)

  void Build(const FuncGraphPtr &func_graph);
  bool Infer(const AbstractBasePtrList &inputs) override;
  ShapeArray QueryShape(const AnfNodePtr &node) override;
  ShapeArray QueryValue(const AnfNodePtr &node) override;
  bool ShapeEqual(const std::pair<AnfNodePtr, size_t> &, const std::pair<AnfNodePtr, size_t> &) override {
    return false;
  }
  std::vector<std::string> QuerySymbolicShape(const AnfNodePtr &node) override;
  void QuerySymbolExpr(const AnfNodePtr &node, std::unordered_map<std::string, std::string> *symbol_expr_map) override;
  std::string ToString() const override { return name_; }
  void Dump();

 protected:
  void BuildCNodeSmbl(const CNodePtr &cnode, bool infer_value = false);
  void BuildFuncSmbl(const CNodePtr &cnode, bool infer_value = false);

  std::string name_;
  AnfNodePtrList cnodes_;
  OpPtrList ops_;
  SymbolCache cache_;
  std::unique_ptr<OperationEmitter> emitter_;
};
}  // namespace mindspore::graphkernel::symbol
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_SYMBOL_ENGINE_IMPL_H_
