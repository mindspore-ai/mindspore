/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_RECOMPUTE_HELPER_H
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_RECOMPUTE_HELPER_H

#include <vector>

#include "ir/anf.h"
#include "utils/hash_map.h"
#include "ir/func_graph.h"
#include "include/utils/visible.h"

namespace mindspore {
bool CanNotRecomputed(const CNodePtr &node);

bool IsBpropNode(const AnfNodePtr &node);

ValuePtr GetRecomputeCNodeAttr(const AnfNodePtr &node);

bool IsSetNoRecomputeCNodeAttr(const AnfNodePtr &node);

bool IsSetRecomputeCNodeAttr(const AnfNodePtr &node);

bool IsCandidateRecomputedNode(const CNodePtr &node);

bool HasGradInputs(const AnfNodePtr &node, mindspore::HashMap<AnfNodePtr, bool> *has_grad_inputs_map);

bool HasForwardOutput(const FuncGraphManagerPtr &mng, const AnfNodePtr &node);

void GetTupleGetItemOutputNodes(const FuncGraphManagerPtr &mng, const AnfNodePtr &node,
                                std::vector<AnfNodePtr> *tuple_getitem_output_nodes);

bool SetRecomputedScope(const CNodePtr &node);

CNodePtr CreateNewRecomputedNode(const FuncGraphPtr &graph, const CNodePtr &origin_node,
                                 const std::vector<AnfNodePtr> &new_inputs);

CNodePtr NewRecomputedNode(const FuncGraphPtr &graph, const CNodePtr &origin_node,
                           const std::vector<AnfNodePtr> &first_target_inputs,
                           const mindspore::HashSet<CNodePtr> &recomputed_origin_nodes,
                           mindspore::HashMap<CNodePtr, CNodePtr> *origin_to_recomputed_nodes);

COMMON_EXPORT void SetRecomputedAttr(const FuncGraphPtr &graph, const std::vector<CNodePtr> &origin_nodes_topological);

COMMON_EXPORT bool WithRecomputedScope(const AnfNodePtr &node);

COMMON_EXPORT std::vector<CNodePtr> FindCandidateRecomputedNodes(const FuncGraphManagerPtr &mng,
                                                                 const std::vector<CNodePtr> &cnodes);

COMMON_EXPORT void GetMaxSubGraph(const FuncGraphManagerPtr &mng, mindspore::HashSet<CNodePtr> *recomputed_nodes,
                                  bool get_inputs, bool get_outputs);

COMMON_EXPORT void GetOriginRecomputeAndTargetNodes(const FuncGraphManagerPtr &mng,
                                                    const mindspore::HashSet<CNodePtr> &max_recomputed_sub_graph,
                                                    mindspore::HashSet<CNodePtr> *recompute_nodes,
                                                    mindspore::HashSet<CNodePtr> *target_nodes);

COMMON_EXPORT std::vector<AnfNodePtr> GetFirstTargetInputs(const std::vector<CNodePtr> &origin_nodes_topological,
                                                           const mindspore::HashSet<CNodePtr> &max_recomputed_sub_graph,
                                                           const mindspore::HashSet<CNodePtr> &recomputed_origin_nodes,
                                                           const mindspore::HashSet<CNodePtr> &target_nodes);

COMMON_EXPORT void DuplicateRecomputedNodes(const FuncGraphPtr &graph, const mindspore::HashSet<CNodePtr> &target_nodes,
                                            const mindspore::HashSet<CNodePtr> &origin_recomputed_nodes,
                                            const std::vector<AnfNodePtr> &first_target_inputs,
                                            mindspore::HashMap<CNodePtr, CNodePtr> *origin_to_new_target_nodes,
                                            mindspore::HashMap<CNodePtr, CNodePtr> *origin_to_recomputed_nodes);
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_RECOMPUTE_HELPER_H
