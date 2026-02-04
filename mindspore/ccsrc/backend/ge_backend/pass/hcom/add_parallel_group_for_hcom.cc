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

#include "backend/ge_backend/pass/hcom/add_parallel_group_for_hcom.h"

#include "include/utils/anfalgo.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"

namespace mindspore {
namespace opt {
namespace {
const char kParallelGroup[] = "_parallel_group";
}

const AnfNodePtr AddParallelGroupForHcom::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!common::AnfAlgo::IsCommunicationOp(node)) {
    return node;
  }
  auto hcom_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(hcom_node);
  auto group = hccl::HcclAdapter::GetInstance().GetHcomGroup(hcom_node);
  common::AnfAlgo::SetNodeAttr(kParallelGroup, MakeValue(group), node);
  return node;
}
}  // namespace opt
}  // namespace mindspore
