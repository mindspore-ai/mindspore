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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_ADAPTER_SYMBOL_ENGINE_BUILDER_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_ADAPTER_SYMBOL_ENGINE_BUILDER_H_
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "include/backend/optimizer/pass.h"
#include "backend/common/graph_kernel/symbol_engine/symbol_engine_impl.h"

namespace mindspore::graphkernel {
class SymbolEngineBuilder : public opt::Pass {
 public:
  explicit SymbolEngineBuilder(bool multi_engine = false)
      : Pass("symbol_engine_builder"), multi_engine_(multi_engine) {}
  ~SymbolEngineBuilder() = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  bool multi_engine_;
};

/// \brief Build SymbolEngine for FuncGraph, and set it as attr of FuncGraph.
/// \param fg the funcgraph
/// \param multi_engine if true, new SymbolEngine will be created for each subgraphs. otherwise, the unique SymbolEngine
/// will be reused. Default false.
/// \return shared_ptr of SymbolEngine
SymbolEnginePtr BuildSymbolEngine(const FuncGraphPtr &fg, bool multi_engine = false);

// Build SubSymbolEngine for fused kernels.
SymbolEnginePtr BuildSubSymbolEngine(const FuncGraphPtr &sub_fg, const AnfNodePtr &node);
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_ADAPTER_SYMBOL_ENGINE_BUILDER_H_
