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

#include "backend/ge_backend/pass/hcom/insert_load_for_allgather.h"
#include <vector>
#include "primitive/other_op_name.h"
#include "primitive/framework_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/parallel_context.h"
#include "primitive/other_ops.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

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
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    MS_LOG(DEBUG) << "AllGather parallel optimization is not required in pipeline parallel mode.";
    return nullptr;
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto cell_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  if (cell_reuse) {
    return nullptr;
  }
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
  auto &node_users = mng->node_users()[node];
  size_t node_user_num = 0;
  for (const auto &node_user : node_users) {
    if (!IsPrimitiveCNode(node_user.first, prim::kPrimDepend)) {
      ++node_user_num;
      continue;
    }
    auto depend_cnode = node_user.first->cast<CNodePtr>();
    if (depend_cnode != nullptr && depend_cnode->input(1) == node) {
      ++node_user_num;
    }
  }

  if (node_user_num <= 1) {
    MS_LOG(DEBUG) << "Node users size not greater than 1, node: " << node->fullname_with_scope();
    return nullptr;
  }
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim::kPrimTensorMove), node};
  auto load = this->NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(load);
  load->set_abstract(node->abstract());
  load->set_scope(node->scope());
  MS_LOG(DEBUG) << "Insert TensorMove for AllGather, Load node: " << load->fullname_with_scope()
                << ", AllGather node: " << node->fullname_with_scope();
  return load;
}

}  // namespace opt
}  // namespace mindspore
