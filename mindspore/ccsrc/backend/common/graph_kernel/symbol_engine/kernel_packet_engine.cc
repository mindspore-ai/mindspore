/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/symbol_engine/kernel_packet_engine.h"

namespace mindspore {
namespace graphkernel {
namespace symshape {
void CloneAllAbstracts(const FuncGraphPtr &func_graph) {
  auto nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple, AlwaysInclude);
  for (auto &node : nodes) {
    auto old_abs = node->abstract();
    if (old_abs == nullptr) {
      continue;
    }
    auto new_abs = old_abs->Clone();
    new_abs->SetSymbolicShape(nullptr);
    new_abs->SetSymbolicValue(nullptr);
    node->set_abstract(new_abs);
  }
}

KernelPacketEnginePtr KernelPacketEngine::Build(const FuncGraphPtr &func_graph) {
  CloneAllAbstracts(func_graph);
  auto engine = std::make_shared<KernelPacketEngine>(func_graph);
  func_graph->set_symbol_engine(engine);
  engine->PreBuild();
  engine->BuildImpl();
  return engine;
}
}  // namespace symshape
}  // namespace graphkernel
}  // namespace mindspore
