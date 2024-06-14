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

#include "plugin/device/ascend/optimizer/ge/fa_alltoallv_parallel.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <list>
#include <tuple>
#include <algorithm>
#include "utils/anf_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/comm_manager.h"
#include "include/backend/optimizer/helper.h"
#include "ops/array_ops.h"
#include "ops/other_ops.h"
#include "ops/framework_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "mindspore/core/ops/ops_func_impl/flash_attention_score.h"

namespace mindspore {
namespace opt {
namespace {

bool FindTargetNodes(const std::vector<CNodePtr> &origin_nodes_topological,
                     std::map<std::string, CNodePtr> *alltoallv_map, std::map<std::string, CNodePtr> *fa_map,
                     std::map<std::string, CNodePtr> *update_node_map) {
  bool found = false;
  for (size_t i = 0; i < origin_nodes_topological.size(); i++) {
    auto cnode = origin_nodes_topological[i];
    if (IsPrimitiveCNode(cnode, prim::kPrimAllToAllv)) {
      auto AllToAllv_prim = GetCNodePrimitive(cnode);
      if (AllToAllv_prim->HasAttr("FLASH_INDEX")) {
        auto flash_index = GetValue<std::string>(AllToAllv_prim->GetAttr("FLASH_INDEX"));
        alltoallv_map->insert({flash_index, cnode});
      }
    }

    if (IsPrimitiveCNode(cnode, prim::kPrimFlashAttentionScore)) {
      auto FlashAttentionScore_prim = GetCNodePrimitive(cnode);
      if (FlashAttentionScore_prim->HasAttr("FLASH_INDEX")) {
        auto flash_index = GetValue<std::string>(FlashAttentionScore_prim->GetAttr("FLASH_INDEX"));
        fa_map->insert({flash_index, cnode});
      }
    }

    if (common::AnfAlgo::HasNodeAttr("AccumulatedAttention", cnode)) {
      auto node_prim = GetCNodePrimitive(cnode);
      if (node_prim->HasAttr("FLASH_INDEX")) {
        auto update_node_index = GetValue<std::string>(node_prim->GetAttr("FLASH_INDEX"));
        update_node_map->insert({update_node_index, cnode});
        found = true;
      }
    }
  }
  return found;
}

CNodePtr NewFlashAttentionScoreNode(const std::vector<AnfNodePtr> &input_nodes, const std::vector<ShapeVector> &shapes,
                                    const std::vector<TypeId> &dtypes) {
  std::vector<AnfNodePtr> fa_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimFlashAttentionScore->name()))};
  for (size_t i = 0; i < input_nodes.size(); i++) {
    fa_inputs.push_back(input_nodes[i]);
  }
  auto fa_score = input_nodes[0]->func_graph()->NewCNode(fa_inputs);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, fa_score.get());
  fa_score->set_scope(input_nodes[0]->scope());
  return fa_score;
}

CNodePtr CloneSplitVDNode(const AnfNodePtr &cloned_node, const AnfNodePtr &new_input_node) {
  MS_EXCEPTION_IF_NULL(cloned_node);
  MS_EXCEPTION_IF_NULL(new_input_node);
  std::vector<AnfNodePtr> new_inputs;
  auto cloned_cnode = cloned_node->cast<CNodePtr>();
  if (!IsPrimitiveCNode(cloned_cnode, prim::kPrimSplitVD)) {
    MS_LOG(INTERNAL_EXCEPTION) << "cloned_node should be SplitVD cnode.";
  }
  // Get all of the inputs of old SplitVD cnode
  auto inputs = cloned_cnode->inputs();
  for (size_t i = 0; i < inputs.size(); i++) {
    auto input = inputs[i];
    if (input->isa<CNode>()) {
      new_inputs.push_back(new_input_node->cast<CNodePtr>());
    } else if (input->isa<ValueNode>()) {
      ValueNodePtr new_value_node = NewValueNode(GetValueNode(input));
      new_inputs.push_back(new_value_node);
    } else if (input->isa<Parameter>()) {
      new_inputs.push_back(input);
    }
  }

  // create new splitvd CNode
  auto splitvd = cloned_node->func_graph()->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(splitvd);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(cloned_node, 0)};
  auto shape = common::AnfAlgo::GetOutputInferShape(cloned_node, 0);
  std::vector<ShapeVector> shapes(1, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, splitvd.get());

  splitvd->set_scope(cloned_node->scope());
  return splitvd;
}

CNodePtr CreateDepend(const AnfNodePtr &latter_node, const AnfNodePtr &former_node, const FuncGraphPtr &graph) {
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                        latter_node, former_node};
  auto depend = graph->NewCNode(depend_inputs);
  MS_EXCEPTION_IF_NULL(depend);
  depend->set_abstract(latter_node->abstract()->Clone());
  return depend;
}

bool AddDependForFA(const FuncGraphPtr &graph, std::map<std::string, CNodePtr> *alltoallv_map,
                    std::map<std::string, CNodePtr> *fa_map, std::map<std::string, CNodePtr> *update_node_map) {
  auto manager = graph->manager();
  for (auto it = alltoallv_map->begin(); it != alltoallv_map->end(); ++it) {
    auto alltoallv = alltoallv_map->at(it->first);
    auto fa_score_node = fa_map->at(it->first);

    std::vector<AnfNodePtr> fa_inputs;
    for (size_t idx = 0; idx < ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputsNum; idx++) {
      fa_inputs.push_back(fa_score_node->input(idx + 1));
    }
    auto mask = fa_score_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex + 1);

    std::vector<ShapeVector> fa_output_shapes;
    std::vector<TypeId> fa_output_dtypes;
    for (size_t idx = 0; idx < ops::FlashAttentionScoreOutputIndex::kFlashAttentionScoreOutputsNum; idx++) {
      fa_output_shapes.push_back(common::AnfAlgo::GetOutputInferShape(fa_score_node, idx));
      fa_output_dtypes.push_back(common::AnfAlgo::GetOutputInferDataType(fa_score_node, idx));
    }

    auto concatd = alltoallv->input(1);
    // The key is cnode, the corresponding value is the list of nodes that use the key as its input
    auto alltoallv_users = manager->node_users()[alltoallv];

    if (alltoallv_users.size() != 1) {
      MS_LOG(INTERNAL_EXCEPTION) << "alltoallv node is only 1 user, which is splitvd, but we got : "
                                 << alltoallv_users.size();
    }
    auto splitvd_node = alltoallv_users.front().first;

    auto cal_comm_depend_mask = CreateDepend(mask, concatd, graph);
    fa_inputs[ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputAttnMaskIndex] = cal_comm_depend_mask;

    auto depend_fa_node = NewFlashAttentionScoreNode(fa_inputs, fa_output_shapes, fa_output_dtypes);
    common::AnfAlgo::CopyNodeAttrs(fa_score_node, depend_fa_node);
    // here we add the depend node and new splitvd and insert the depend node
    manager->Replace(fa_score_node, depend_fa_node);
    CNodePtr splitvd_depend_alltoallv;

    if (update_node_map->find(it->first) != update_node_map->end()) {
      splitvd_depend_alltoallv = CreateDepend(alltoallv, update_node_map->at(it->first), graph);
    } else {
      splitvd_depend_alltoallv = CreateDepend(alltoallv, depend_fa_node, graph);
    }

    auto depend_splitvd_node = CloneSplitVDNode(splitvd_node, splitvd_depend_alltoallv);
    manager->Replace(splitvd_node, depend_splitvd_node);
  }
  return true;
}
}  // namespace

bool FaAlltoAllvParallel::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::list<CNodePtr> orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.cbegin(), orders.cend());
  std::map<std::string, CNodePtr> alltoallv_map, fa_map, update_node_map;
  if (!FindTargetNodes(origin_nodes_topological, &alltoallv_map, &fa_map, &update_node_map)) {
    return false;
  }
  return AddDependForFA(graph, &alltoallv_map, &fa_map, &update_node_map);
}
}  // namespace opt
}  // namespace mindspore
