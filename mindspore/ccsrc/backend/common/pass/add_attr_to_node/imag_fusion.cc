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

#include "backend/common/pass/add_attr_to_node/add_attr_to_node_register.h"
#include "include/utils/anfalgo.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
const AnfNodePtr ImagFusionProcess(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  constexpr auto kTout = "Tout";

  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);

  if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice) {
    return node;
  }

  auto output_dtype = node->Type();
  MS_EXCEPTION_IF_NULL(output_dtype);
  auto output_tensortype = output_dtype->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(output_tensortype);
  auto type_id = output_tensortype->element()->type_id();
  common::AnfAlgo::SetNodeAttr(kTout, TypeIdToType(type_id), node);

  return node;
}
}  // namespace opt
}  // namespace mindspore
