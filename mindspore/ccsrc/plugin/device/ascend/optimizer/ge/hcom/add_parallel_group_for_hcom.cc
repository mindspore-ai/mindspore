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

#include "plugin/device/ascend/optimizer/ge/hcom/add_parallel_group_for_hcom.h"

#include <vector>
#include <algorithm>
#include <memory>
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"

namespace mindspore {
namespace opt {
namespace {
const char kParallelGroup[] = "_parallel_group";
}
std::string AddParallelGroupForHcom::GetHcomGroup(const CNodePtr &cnode) const {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!common::AnfAlgo::HasNodeAttr(kAttrGroup, cnode)) {
    MS_LOG(EXCEPTION) << "Hcom node " << cnode->fullname_with_scope() << " has no group attribute.";
  }

  auto group_name = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrGroup);
  auto rank_ids = common::AnfAlgo::HasNodeAttr(kAttrGroupRankIds, cnode)
                    ? common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cnode, kAttrGroupRankIds)
                    : std::vector<uint32_t>();
  auto new_group = hccl::HcclAdapter::GetInstance().GetHcomGroup(group_name, rank_ids);
  MS_LOG(INFO) << "hcom node: " << cnode->fullname_with_scope() << ", old group: " << group_name
               << ", new group: " << new_group;
  return new_group;
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
  auto group = GetHcomGroup(hcom_node);
  common::AnfAlgo::SetNodeAttr(kParallelGroup, MakeValue(group), node);
  return node;
}
}  // namespace opt
}  // namespace mindspore
