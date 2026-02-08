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
#ifndef MINDSPORE_CCSRC_BACKEND_MS_BACKEND_GRAPH_FUSION_MULTI_SYMBOL_ENGINE_H_
#define MINDSPORE_CCSRC_BACKEND_MS_BACKEND_GRAPH_FUSION_MULTI_SYMBOL_ENGINE_H_

#include <string>
#include <memory>
#include <vector>
#include <map>
#include "mindspore/ccsrc/include/utils/symbol_engine/symbol_engine_impl.h"

namespace mindspore {
namespace graphkernel {
namespace symshape {
using mindspore::symshape::SymbolEngineImpl;

/// \brief SymbolEngine for a graph. new symbol engines will be created for subgraphs.
class MultiSymbolEngine : public SymbolEngineImpl {
 public:
  using SymbolEngineImpl::SymbolEngineImpl;
  ~MultiSymbolEngine() = default;
  MS_DECLARE_PARENT(MultiSymbolEngine, SymbolEngineImpl)

  static void Build(const FuncGraphPtr &func_graph);
  static void BuildSubEngine(const AnfNodePtr &node);

  std::string ToString() const override { return "MultiSymbolEngine_" + name_; }
  void BuildSubgraphImpl(const CNodePtr &cnode, const FuncGraphPtr &sub_fg, size_t begin_input_index) override;
  void PreBuildQuerySubgraphDependStatus(const CNodePtr &cnode, const FuncGraphPtr &sub_fg,
                                         size_t begin_input_index) override;

 protected:
  void SaveInputParaMap(std::map<SymbolPtr, SymbolPtr> *input_para_map, const SymbolPtr &inp, const SymbolPtr &para);
  ListSymbolPtr BuildShapeWithInputHint(const AbstractBasePtr &para_abs, const std::vector<ListSymbolPtr> &inputs,
                                        std::map<SymbolPtr, SymbolPtr> *input_para_map);
  void GenInputSymbols(const CNodePtr &cnode, const FuncGraphPtr &sub_fg, size_t begin_input_index);
};
}  // namespace symshape
using MultiSymbolEnginePtr = std::shared_ptr<symshape::MultiSymbolEngine>;
}  // namespace graphkernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_MS_BACKEND_GRAPH_FUSION_MULTI_SYMBOL_ENGINE_H_
