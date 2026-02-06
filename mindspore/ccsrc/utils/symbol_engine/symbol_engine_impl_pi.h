/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_SYMBOL_ENGINE_SYMBOL_ENGINE_IMPL_PI_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_SYMBOL_ENGINE_SYMBOL_ENGINE_IMPL_PI_H_
#include <vector>
#include <utility>
#include <unordered_map>
#include <map>
#include <string>
#include <memory>
#include <set>
#include <mutex>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "abstract/symbolic_shape/operation_builder.h"
#include "abstract/symbolic_shape/operation.h"
#include "mindspore/ccsrc/utils/symbol_engine/utils.h"
#include "mindspore/ccsrc/include/utils/symbol_engine/symbol_engine_impl.h"

namespace mindspore {
namespace symshape {
class SymbolEngine;
class Symbol;
using SymbolPtr = std::shared_ptr<Symbol>;

class SymbolEnginePIJIT : public SymbolEngine {
 public:
  explicit SymbolEnginePIJIT(const FuncGraphPtr &fg) : SymbolEngine(fg) {}
  ~SymbolEnginePIJIT() = default;
  MS_DECLARE_PARENT(SymbolEnginePIJIT, SymbolEngine)

  /// \brief Build SymbolEngine, and set to the FuncGraph.
  static std::shared_ptr<symshape::SymbolEnginePIJIT> Build(const FuncGraphPtr &func_graph);
  void AddInputAbs(const AbstractBasePtr &abs, const AbstractBasePtr &hint_abs = nullptr);
  void BuildCNodeSymbol(const CNodePtr &cnode);
  AbstractBasePtr EvalOnePrimSymbol(const PrimitivePtr &prim, const AbstractBasePtrList &inputs_abs,
                                    const AbstractBasePtr &output_abs);
  bool CheckCondition(const AbstractBasePtrList &inputs, const BoolSymbolPtr condition);
  SymbolPtr GetHint(const SymbolPtr &s) {
    if (s == nullptr) {
      return nullptr;
    }
    if (hint_map_.find(s) == hint_map_.end()) {
      return nullptr;
    }
    return hint_map_[s];
  }
  void DumpHintMap() {
    for (auto [k, v] : hint_map_) {
      MS_LOG(DEBUG) << k << " : " << k->ToString() << " : " << v->ToString();
    }
  }
  std::string DumpText() const override;

 private:
  void BuildImpl();
  DependStatus GetDependStatus(const AbstractBasePtrList &inputs, const PrimitivePtr &prim);
  AbstractBasePtrList ExtractInputsAbstractHint(const AbstractBasePtrList &inputs);
  bool Infer(const AbstractBasePtrList &inputs) override;
  SymbolPtr BuildCNodeSymbolicValue(OperationBuilder *builder, const PrimitivePtr &prim,
                                    const AbstractBasePtrList &inputs, const AbstractBasePtr &abstract);
  SymbolPtr BuildCNodeSymbolicShape(OperationBuilder *builder, const PrimitivePtr &prim,
                                    const AbstractBasePtrList &inputs, const AbstractBasePtr &abstract);
  void SetHintMap(const SymbolPtr s, const SymbolPtr hint) {
    if (s != nullptr) {
      hint_map_[s] = hint;
    }
  }

 private:
  std::map<SymbolPtr, SymbolPtr> hint_map_;
  AbstractBasePtrList inputs_abs_;
  OpPtrList ops_;
  std::unique_ptr<OperationEmitter> emitter_;
};
using SymbolEnginePIJITPtr = std::shared_ptr<symshape::SymbolEnginePIJIT>;
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_SYMBOL_ENGINE_SYMBOL_ENGINE_IMPL_PI_H_
