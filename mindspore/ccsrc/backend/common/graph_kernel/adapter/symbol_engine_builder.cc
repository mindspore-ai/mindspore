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
  BuildSymbolEngine(func_graph, multi_engine_);
  return true;
}

SymbolEnginePtr BuildSymbolEngine(const FuncGraphPtr &fg, bool multi_engine) {
  auto engine = std::make_shared<symbol::SymbolEngineImpl>(fg, multi_engine);
  fg->set_attr(kAttrSymbolEngine, engine);
  engine->PreBuild();
  engine->Build();
  return engine;
}

SymbolEnginePtr BuildSubSymbolEngine(const FuncGraphPtr &sub_fg, const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto engine = std::make_shared<symbol::SymbolEngineImpl>(sub_fg, true);
  sub_fg->set_attr(kAttrSymbolEngine, engine);
  engine->PreBuild();
  if (node->func_graph()->has_attr(kAttrSymbolEngine)) {
    auto main_engine = node->func_graph()->get_attr(kAttrSymbolEngine)->cast_ptr<symbol::SymbolEngineImpl>();
    MS_EXCEPTION_IF_NULL(main_engine);
    main_engine->BuildSubgraph(cnode);
  } else {
    engine->Build();
  }
  return engine;
}
}  // namespace mindspore::graphkernel
