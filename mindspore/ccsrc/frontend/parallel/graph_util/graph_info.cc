/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/graph_util/graph_info.h"
#include "debug/anf_ir_dump.h"
#include "debug/anf_ir_utils.h"
#include "debug/draw.h"
#include "utils/ms_context.h"
#include "ir/graph_utils.h"

namespace mindspore {
namespace parallel {
std::vector<PrimitivePtr> FindPrimtive(const FuncGraphPtr &graph, const std::string &name) {
  AnfNodePtr ret = graph->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  std::vector<PrimitivePtr> prim_list;
  for (auto &node : all_nodes) {
    if (!IsValueNode<Primitive>(node)) {
      continue;
    }
    ValueNodePtr prim_node_anf = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_node_anf);
    PrimitivePtr node_prim = prim_node_anf->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(node_prim);
    if (node_prim->name() == name) {
      prim_list.emplace_back(node_prim);
    }
  }
  return prim_list;
}

void DumpGraph(const FuncGraphPtr &root, const std::string &name) {
  if (MsContext::GetInstance()->save_graphs_flag()) {
    draw::Draw(name + ".dot", root);
    DumpIR(name + ".ir", root);
    ExportIR(name + ".dat", "0", root);
  }
}
}  // namespace parallel
}  // namespace mindspore
