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

#define USE_DEPRECATED_API
#include <memory>
#include <vector>
#include "src/common/log_adapter.h"
#include "test/ut/utils/build_func_graph.h"
#include "ops/make_tuple.h"
#include "ops/return.h"
#include "ir/func_graph.h"
#include "mindspore/core/base/base.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace lite {
CNodePtr AddReturn(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &return_inputs) {
  if (return_inputs.empty()) {
    return nullptr;
  }
  AnfNodePtr return_input;
  if (return_inputs.size() == 1) {
    return_input = return_inputs.front();
  } else {
    auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new MakeTuple failed";
      return nullptr;
    }
    auto prim_c = make_tuple_prim_ptr->GetPrim();
    MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr");
    auto return_input_cnode = graph->NewCNode(prim_c, return_inputs);
    if (return_input_cnode == nullptr) {
      MS_LOG(ERROR) << "new make tuple cnode failed";
      return nullptr;
    }
    return_input_cnode->set_fullname_with_scope("return tuple");
    return_input = return_input_cnode;
  }

  auto return_prim = std::make_shared<ops::Return>();
  MS_CHECK_TRUE_MSG(return_prim != nullptr, nullptr, "create return primitivec failed");
  auto return_prim_c = return_prim->GetPrim();
  MS_CHECK_TRUE_MSG(return_prim_c != nullptr, nullptr, "prim_c is nullptr");
  auto return_cnode = graph->NewCNode(return_prim_c, {return_input});
  MS_CHECK_TRUE_MSG(return_cnode != nullptr, nullptr, "create Return failed");
  return_cnode->set_fullname_with_scope("Return");
  graph->set_return(return_cnode);
  return return_cnode;
}
}  // namespace lite
}  // namespace mindspore
