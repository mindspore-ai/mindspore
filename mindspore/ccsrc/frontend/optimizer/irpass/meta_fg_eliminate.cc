/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/meta_fg_eliminate.h"
#include "frontend/optimizer/irpass/gradient_eliminate.h"

namespace mindspore {
namespace opt {
namespace irpass {
bool ExpandMetaFg::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  AnfNodePtr return_node = func_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(return_node);
  // The expanding of meta fg may change the number of outer layer meta fgs.
  // So, find all kinds of candidate meta fgs together and then expands them.
  for (auto expand_meta_fg_element : expand_meta_fg_list_) {
    expand_meta_fg_element->GetMetaFgPrim(all_nodes);
  }
  bool ret = false;
  for (auto expand_meta_fg_element : expand_meta_fg_list_) {
    auto prim_nodes = expand_meta_fg_element->prim_nodes();
    if (prim_nodes.size() != 0) {
      ret = ret || (*expand_meta_fg_element)(func_graph, optimizer);
    }
  }
  return ret;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
