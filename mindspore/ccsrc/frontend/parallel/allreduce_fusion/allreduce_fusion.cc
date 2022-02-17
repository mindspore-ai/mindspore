/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/allreduce_fusion/allreduce_fusion.h"
#include <memory>
#include <queue>
#include <string>
#include <functional>
#include <utility>
#include "utils/hash_set.h"
#include "ir/func_graph.h"
#include "frontend/parallel/costmodel_context.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/step_parallel.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
void SetMirrorFusion(const CNodePtr &mirror_cnode, int64_t fusion, const std::string &parameter_name) {
  MS_EXCEPTION_IF_NULL(mirror_cnode);
  MS_LOG(DEBUG) << "Set Mirror " << mirror_cnode->DebugString() << " fusion " << fusion;
  auto node_prim = GetValueNode<PrimitivePtr>(mirror_cnode->input(0));
  (void)node_prim->AddAttr(FUSION, MakeValue(std::make_shared<Int64Imm>(fusion)));
  (void)node_prim->AddAttr(PARAMETER, MakeValue(std::make_shared<StringImm>(parameter_name)));
}

Status AllreduceFusion::SetFusionBySize(const CNodePtr &ret, int64_t threshold) {
  auto filter = [](const AnfNodePtr &node) { return !IsPrimitiveCNode(node, prim::kPrimMirror); };
  auto todo = DeepScopedGraphSearchWithFilter(ret, AlwaysInclude, filter);
  auto temp = threshold;
  int64_t fusion = 1;
  bool init = true;
  for (auto &node : todo) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode->input(1)->Shape() == nullptr) continue;
    auto input_shapes = GetNodeShape(cnode->input(1));
    int64_t input_size = std::accumulate(input_shapes[0].begin(), input_shapes[0].end(), 1, std::multiplies<int64_t>());
    FuncGraphPtr func_graph = cnode->func_graph();
    std::pair<AnfNodePtr, bool> param_node_pair = FindParameter(cnode->input(1), func_graph);
    if (!param_node_pair.first) {
      continue;
    }
    auto parameter_name = ParameterName(param_node_pair.first);
    if (input_size < temp) {
      temp -= input_size;
    } else {
      temp = threshold;
      fusion++;
    }
    if (init) {
      SetMirrorFusion(cnode, 1, parameter_name);
    } else {
      SetMirrorFusion(cnode, fusion, parameter_name);
    }
    init = false;
  }
  MS_LOG(INFO) << "Allreduce fusion by size succeed.";
  return SUCCESS;
}

Status AllreduceFusion::ProcessAllreduceFusion(const CNodePtr &ret) {
  if (ret == nullptr) {
    MS_LOG(ERROR) << "ret is nullptr.";
    return FAILED;
  }
  ret_ = ret;
  root_graph_ = ret_->func_graph();
  MS_EXCEPTION_IF_NULL(root_graph_);
  auto graph_set = ForwardGraph(root_graph_);
  if (graph_set.size() > 1) {
    MS_LOG(WARNING) << "AllReduce fusion don't support multiple subgraphs now.";
    return SUCCESS;
  }
  auto forward_graph = *(graph_set.begin());
  MS_EXCEPTION_IF_NULL(forward_graph);
  forward_ret_ = forward_graph->get_return();
  MS_EXCEPTION_IF_NULL(forward_ret_);
  if (allreduce_graph_.set_head_cnode(forward_ret_) != SUCCESS) {
    MS_LOG(ERROR) << "AllreduceGraph set_head_cnode failed.";
    return FAILED;
  }
  int64_t threshold = ParallelContext::GetInstance()->fusion_threshold_mb() * 1024 * 1024 / 4;
  if (threshold <= 0) {
    MS_LOG(ERROR) << "The threshold of SetFusionBySize must be larger than 0, but got " << threshold << ".";
    return FAILED;
  }

  (void)SetFusionBySize(ret, threshold);
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
