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
#include "backend/common/graph_kernel/adapter/symbol_engine_builder.h"
#include <memory>
#include "include/common/utils/anfalgo.h"

namespace mindspore::graphkernel {
bool SymbolEngineBuilder::Run(const FuncGraphPtr &func_graph) {
  auto todos = TopoSort(func_graph->output());
  bool changed = false;
  for (auto &node : todos) {
    auto fg = GetCNodeFuncGraph(node);
    if (fg != nullptr && common::AnfAlgo::IsDynamicShape(node)) {
      fg->set_attr(kAttrSymbolEngine, BuildSymbolEngine(fg));
      changed = true;
    }
  }
  return changed;
}

SymbolEnginePtr BuildSymbolEngine(const FuncGraphPtr &fg) {
  auto engine = std::make_shared<symbol::SymbolEngineImpl>(fg->ToString());
  engine->Build(fg);
  return engine;
}
}  // namespace mindspore::graphkernel
