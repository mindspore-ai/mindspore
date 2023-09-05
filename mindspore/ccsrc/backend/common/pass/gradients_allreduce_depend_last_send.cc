/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "backend/common/pass/gradients_allreduce_depend_last_send.h"
#include <algorithm>
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace opt {
bool GradientsAllReduceDependLastSend::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != parallel::kSemiAutoParallel && parallel_mode != parallel::kAutoParallel) {
    return false;
  }
  int32_t split_stage_num = parallel::ParallelContext::GetInstance()->pipeline_stage_split_num();
  if (split_stage_num <= 1) {
    return false;
  }
  auto enable_fold_pipeline = parallel::ParallelContext::GetInstance()->enable_fold_pipeline();
  if (enable_fold_pipeline) {
    return false;
  }
  if (common::GetEnv("MS_ENABLE_FRONTEND_SCHEDULING_OPTIMIZATION") == "1") {
    return false;
  }
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  std::vector<CNodePtr> addn_list;
  CNodePtr last_send;
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->cast<CNodePtr>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsPrimitiveCNode(cnode, prim::kPrimAllReduce) && common::AnfAlgo::IsFusion(cnode)) {
      auto last_input = cnode->inputs().back();
      if (IsPrimitiveCNode(last_input, prim::kPrimTensorMove)) {
        auto last_input_cnode = last_input->cast<CNodePtr>();
        auto real_input_node = last_input_cnode->input(1);
        if (IsPrimitiveCNode(real_input_node, prim::kPrimDepend)) {
          auto addn_node = real_input_node->cast<CNodePtr>()->input(2);
          if (IsPrimitiveCNode(addn_node, prim::kPrimAddN)) {
            MS_LOG(INFO) << "Find the pipeline addn " << addn_node->fullname_with_scope();
            addn_list.push_back(addn_node->cast<CNodePtr>());
          }
        }
      }
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimSend)) {
      last_send = cnode;
    }
  }
  return InsertDependBetweenAllReduceAndSend(graph, addn_list, last_send);
}

bool GradientsAllReduceDependLastSend::InsertDependBetweenAllReduceAndSend(const FuncGraphPtr &graph,
                                                                           const std::vector<CNodePtr> &addn_list,
                                                                           const CNodePtr &last_send) const {
  bool changed = false;
  if (last_send == nullptr) {
    MS_LOG(DEBUG) << "last depend is null " << graph->ToString();
    return false;
  }
  FuncGraphManagerPtr manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &addn : addn_list) {
    std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), addn,
                                      last_send};
    auto new_depend = graph->NewCNode(inputs);
    new_depend->set_abstract(addn->abstract());
    (void)manager->Replace(addn, new_depend);
    changed = true;
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
