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
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &all_nodes = manager->all_nodes();
  // The expanding of meta fg may change the number of outer layer meta fgs.
  // So, find all kinds of candidate meta fgs together and then expands them.
  for (auto &expand_meta_fg_element : expand_meta_fg_list_) {
    expand_meta_fg_element.second->GetMetaFgPrim(all_nodes);
  }
  bool res = false;
  for (auto &expand_meta_fg_element : expand_meta_fg_list_) {
    auto prim_nodes = expand_meta_fg_element.second->prim_nodes();
    if (prim_nodes.size() != 0) {
      MS_LOG(INFO) << "Start expanding meta: " << expand_meta_fg_element.first;
      auto change = (*expand_meta_fg_element.second)(func_graph, optimizer);
      res = res || change;
      MS_LOG(INFO) << "End expanding meta: " << expand_meta_fg_element.first << ", change: " << change;
    }
  }
  return res;
}

bool ExpandMetaShardFg::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &all_nodes = manager->all_nodes();
  // The expanding of meta fg may change the number of outer layer meta fgs.
  // So, find all shard meta fg and then expands them.
  for (auto &expand_meta_fg_element : expand_meta_shard_fg_list_) {
    expand_meta_fg_element.second->GetMetaFgPrim(all_nodes);
  }
  bool res = false;
  for (auto &expand_meta_fg_element_ : expand_meta_shard_fg_list_) {
    auto prim_nodes = expand_meta_fg_element_.second->prim_nodes();
    if (prim_nodes.size() != 0) {
      MS_LOG(INFO) << "Start expanding meta: " << expand_meta_fg_element_.first;
      auto change = (*expand_meta_fg_element_.second)(func_graph, optimizer);
      res = res || change;
      MS_LOG(INFO) << "End expanding meta: " << expand_meta_fg_element_.first << ", change: " << change;
    }
  }
  return res;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
