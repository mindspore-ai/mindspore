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

#include "plugin/device/ascend/optimizer/ge/hcom/insert_load_for_allgather.h"
#include <vector>
#include "ops/other_op_name.h"
#include "ops/framework_ops.h"
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/parallel_context.h"
#include "ops/other_ops.h"

namespace mindspore {
namespace opt {

const BaseRef InsertLoadForAllGather::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto allgather_prim = std::make_shared<Primitive>(kAllGatherOpName);
  return VectorRef({allgather_prim, Xs});
}

const AnfNodePtr InsertLoadForAllGather::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!IsPrimitiveCNode(node, prim::kPrimAllGather)) {
    MS_LOG(ERROR) << "Not target node AllGather, but is: " << node->fullname_with_scope();
    return nullptr;
  }
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }
  auto node_users = mng->node_users()[node];
  if (node_users.size() <= 1) {
    MS_LOG(DEBUG) << "Node users size not greater than 1, node: " << node->fullname_with_scope();
    return nullptr;
  }
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim::kPrimLoad), node};
  auto load = this->NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(load);
  load->set_abstract(node->abstract());
  load->set_scope(node->scope());
  MS_LOG(DEBUG) << "Insert Load for AllGather, Load node: " << load->fullname_with_scope()
                << ", AllGather node: " << node->fullname_with_scope();
  return load;
}

}  // namespace opt
}  // namespace mindspore
