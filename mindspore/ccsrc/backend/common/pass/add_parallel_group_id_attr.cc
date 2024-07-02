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
#include "backend/common/pass/add_parallel_group_id_attr.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <list>
#include "include/backend/kernel_info.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/anf_utils.h"
#include "ops/array_op_name.h"
#include "ops/framework_ops.h"
#include "ops/sequence_ops.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kMirrorSubStr = "mirror";
constexpr auto kParallelGroupId = "_parallel_group_id";

enum class GroupId { FORWARD = 5, BACKWARD = 6, UNKNOWN = 7 };

enum class Index { DATA_PARALLEL = 0, MODEL_PARALLEL = 1, PIPELINE_PARALLEL = 2, COMPUTE = 3, UNKNOWN = 4 };
}  // namespace

bool AddParallelGroupIdAttr::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::list<CNodePtr> orders = func_graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.cbegin(), orders.cend());
  for (const auto &cnode : origin_nodes_topological) {
    if (!AnfUtils::IsRealKernel(cnode)) {
      continue;
    }
    auto prim = GetCNodePrimitive(cnode);

    GroupId group_id = GroupId::UNKNOWN;
    Index index = Index::UNKNOWN;

    if (cnode->HasPrimalAttr(kPrimalAttrUniqueId)) {
      group_id = GroupId::FORWARD;
    } else if (cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
      group_id = GroupId::BACKWARD;
    } else {
      MS_LOG(DEBUG) << "While adding group id, detects cnode is neither forward nor backward!";
    }

    if (common::AnfAlgo::IsCommunicationOp(cnode)) {
      if (IsPrimitiveCNode(cnode, prim::kPrimSend) || IsPrimitiveCNode(cnode, prim::kPrimReceive)) {
        // cnode is for pipeline parallel
        index = Index::PIPELINE_PARALLEL;
      } else if (cnode->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
        // cnode is for model parallel
        index = Index::MODEL_PARALLEL;
      } else if (prim && prim->instance_name().find(kMirrorSubStr) != std::string::npos) {
        // cnode is for data parallel
        index = Index::DATA_PARALLEL;
      } else {
        MS_LOG(DEBUG)
          << "while adding index, detects cnode is communition operator but is not for data, model, pipeline parallel";
      }
    } else {
      index = Index::COMPUTE;
    }

    uint32_t parallel_group_id = static_cast<uint32_t>(group_id) << 16 | static_cast<uint32_t>(index);
    cnode->AddAttr(kParallelGroupId, MakeValue(parallel_group_id));
    MS_LOG(DEBUG) << "Successfully add _parallel_group_id: " << parallel_group_id
                  << " to node: " << cnode->fullname_with_scope();
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
