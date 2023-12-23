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
#include "ops/other_ops.h"
#include "ops/array_ops.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kMirrorSubStr = "mirror";
constexpr auto kParallelGroupId = "_parallel_group_id";

enum class GroupId {
  FORWARD_COMPUTE = 5,
  FORWARD_ALLGATHER = 6,
  FORWARD_REDUCE_SCATTER = 7,
  FORWARD_ALLREDUCE = 8,
  FORWARD_SEND = 9,
  FORWARD_RECEIVE = 10,
  FORWARD_OTHER_COMM_OP = 11,
  FORWARD_UNKOWN = 12,
  BACKWARD_COMPUTE = 13,
  BACKWARD_ALLGATHER = 14,
  BACKWARD_REDUCE_SCATTER = 15,
  BACKWARD_ALLREDUCE = 16,
  BACKWARD_SEND = 17,
  BACKWARD_RECEIVE = 18,
  BACKWARD_OTHER_COMM_OP = 19,
  BACKWARD_UNKOWN = 20,
  UNKNOWN = 21
};

enum class Index { DEFAULT = 0 };
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
    Index index = Index::DEFAULT;

    if (cnode->HasPrimalAttr(kPrimalAttrUniqueId)) {
      // is forward op
      if (common::AnfAlgo::IsCommunicationOp(cnode)) {
        // is comm op
        if (IsPrimitiveCNode(cnode, prim::kPrimAllGather)) {
          group_id = GroupId::FORWARD_ALLGATHER;
        } else if (IsPrimitiveCNode(cnode, prim::kPrimReduceScatter)) {
          group_id = GroupId::FORWARD_REDUCE_SCATTER;
        } else if (IsPrimitiveCNode(cnode, prim::kPrimAllReduce)) {
          group_id = GroupId::FORWARD_ALLREDUCE;
        } else if (IsPrimitiveCNode(cnode, prim::kPrimSend)) {
          group_id = GroupId::FORWARD_SEND;
        } else if (IsPrimitiveCNode(cnode, prim::kPrimReceive)) {
          group_id = GroupId::FORWARD_RECEIVE;
        } else {
          group_id = GroupId::FORWARD_OTHER_COMM_OP;
        }
      } else {
        // is compute op
        group_id = GroupId::FORWARD_COMPUTE;
      }
    } else if (cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
      // BACKWARD
      if (common::AnfAlgo::IsCommunicationOp(cnode)) {
        // is comm op
        if (IsPrimitiveCNode(cnode, prim::kPrimAllGather)) {
          group_id = GroupId::BACKWARD_ALLGATHER;
        } else if (IsPrimitiveCNode(cnode, prim::kPrimReduceScatter)) {
          group_id = GroupId::BACKWARD_REDUCE_SCATTER;
        } else if (IsPrimitiveCNode(cnode, prim::kPrimAllReduce)) {
          group_id = GroupId::BACKWARD_ALLREDUCE;
        } else if (IsPrimitiveCNode(cnode, prim::kPrimSend)) {
          group_id = GroupId::BACKWARD_SEND;
        } else if (IsPrimitiveCNode(cnode, prim::kPrimReceive)) {
          group_id = GroupId::BACKWARD_RECEIVE;
        } else {
          group_id = GroupId::BACKWARD_OTHER_COMM_OP;
        }
      } else {
        // is compute op
        group_id = GroupId::BACKWARD_COMPUTE;
      }
    } else {
      MS_LOG(WARNING) << "while adding group id, detects cnode is neither forward nor backward!";
    }

    uint32_t parallel_group_id = static_cast<uint32_t>(group_id) << 16 | static_cast<uint32_t>(index);
    cnode->AddAttr(kParallelGroupId, MakeValue(parallel_group_id));
    MS_LOG(INFO) << "Successfully add _parallel_group_id: " << parallel_group_id
                 << " to node: " << cnode->fullname_with_scope();
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
