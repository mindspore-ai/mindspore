/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/load_eliminate.h"

#include <algorithm>
#include <memory>
#include <set>
#include <vector>

#include "frontend/operator/ops.h"

namespace mindspore::opt::irpass {
// Covert:
// load1 = load(para1, u1)
// u2 = UpdateState(u1, load1)
// ...
// load2 = load(load1, u3)
// u4 = UpdateState(u3, load2)
// To:
// load1 = load(para1, u1)
// u2 = UpdateState(u1, load1)
// ...
// load2 = load(para1, u3)        # load1 replaced by para1
// u4 = UpdateState(u3, load2)
AnfNodePtr LoadEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  auto load_node = dyn_cast<CNode>(node);
  if (load_node == nullptr || load_node->inputs().empty()) {
    MS_LOG(WARNING) << "LoadEliminater encounter invalid node: " << node->DebugString();
    return nullptr;
  }
  auto load_cnode = load_node->cast<CNodePtr>();
  constexpr size_t kFirstInputIndex = 1;
  constexpr size_t kSecondInputIndex = 2;
  auto &input_load = load_cnode->input(kFirstInputIndex);
  if (IsPrimitiveCNode(input_load, prim::kPrimLoad)) {
    auto load_prim = NewValueNode(prim::kPrimLoad);
    auto input_load_cnode = input_load->cast<CNodePtr>();
    auto replace_input = input_load_cnode->input(kFirstInputIndex);
    auto monad = load_cnode->input(kSecondInputIndex);
    std::vector<AnfNodePtr> new_load_inputs = {load_prim, replace_input, monad};
    auto fg = load_cnode->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    auto new_load = fg->NewCNode(new_load_inputs);
    new_load->set_abstract(load_cnode->abstract());
    new_load->set_scope(load_cnode->scope());
    return new_load;
  }
  return nullptr;
}
}  // namespace mindspore::opt::irpass
