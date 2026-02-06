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
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/kernel_packet/kernel_packet_engine.h"
#include "include/utils/utils.h"
#include "ir/graph_utils.h"

namespace mindspore {
namespace graphkernel {
namespace packet {
void CloneAllAbstracts(const FuncGraphPtr &func_graph) {
  auto nodes = TopoSort(func_graph->get_return(), SuccDeeperWithAttrGraph, AlwaysInclude);
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

void KernelPacketEngine::SetBaseNodeDepend(const CNodePtr &basenode) {
  depend_status_map_[basenode].shape = true;
  for (size_t i = 1; i < basenode->size(); i++) {
    if (basenode->input(i)->isa<CNode>()) {
      depend_status_map_[basenode->input(i)].value = true;
    }
  }
}

void KernelPacketEngine::PreBuildQuerySubgraphDependStatus(const CNodePtr &cnode, const FuncGraphPtr &sub_fg,
                                                           size_t begin_input_index) {
  SymbolEngineImpl::PreBuildQuerySubgraphDependStatus(cnode, sub_fg, begin_input_index);
  for (auto &param : sub_fg->parameters()) {
    if (IsDependValue(param)) {
      support_infer_ = false;
    }
  }
}

KernelPacketEnginePtr KernelPacketEngine::Build(const FuncGraphPtr &func_graph) {
  CloneAllAbstracts(func_graph);
  auto engine = std::make_shared<KernelPacketEngine>(func_graph);
  func_graph->set_symbol_engine(engine);
  auto basenode = func_graph->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(basenode);
  engine->SetBaseNodeDepend(basenode);
  engine->PreBuild();
  engine->BuildImpl();
  if (std::any_of(func_graph->parameters().begin(), func_graph->parameters().end(), [](const AnfNodePtr &p) {
        auto abs = p->abstract();
        MS_EXCEPTION_IF_NULL(abs);
        return abs->GetSymbolicValue() != nullptr && abs->GetSymbolicValue()->CanUpdate();
      })) {
    MS_LOG(INFO) << "KernelPacket should begin with shape-depend ops, drop the packet from "
                 << basenode->fullname_with_scope();
    return nullptr;
  }
  return engine;
}
}  // namespace packet
}  // namespace graphkernel
}  // namespace mindspore
