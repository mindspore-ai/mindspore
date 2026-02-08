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

#include "backend/common/pass/add_attr_to_node/add_attr_to_node_register.h"
#include "include/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const AnfNodePtr DynamicRNNFusionProcess(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  constexpr auto kInputSize = "input_size";
  constexpr auto kHiddenSize = "hidden_size";
  constexpr auto kPlaceHolderIndex = "placeholder_index";
  constexpr int64_t kDynRnnNum4 = 4;

  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  std::vector<ValuePtr> placeholder_index = {MakeValue((int64_t)3)};
  common::AnfAlgo::SetNodeAttr(kPlaceHolderIndex, MakeValue(placeholder_index), cnode);

  auto x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, kIndex0);
  auto w_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, kIndex1);
  if (IsDynamic(x_shape) || IsDynamic(w_shape)) {
    MS_LOG(ERROR) << "For '" << cnode->fullname_with_scope()
                  << "', can't add attr input_size or hidden_size while shape of inputs is dynamic.";
    return cnode;
  }

  int64_t input_size = x_shape[kIndex2];
  int64_t hidden_size = w_shape[w_shape.size() - 1] / kDynRnnNum4;

  common::AnfAlgo::SetNodeAttr(kInputSize, MakeValue(input_size), cnode);
  common::AnfAlgo::SetNodeAttr(kHiddenSize, MakeValue(hidden_size), cnode);

  return cnode;
}
}  // namespace opt
}  // namespace mindspore
