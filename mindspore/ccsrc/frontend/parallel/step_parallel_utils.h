/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_PARALLEL_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_PARALLEL_UTILS_H_

#include <vector>
#include <string>
#include <utility>
#include <set>
#include <memory>
#include "base/base.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
const int64_t TWO_INPUT_SIZE = 2;

// common method
bool IsSomePrimitive(const CNodePtr &cnode, const std::string &name);
bool IsSomePrimitiveList(const CNodePtr &cnode, const std::set<string> &check_list);
bool IsParallelCareNode(const CNodePtr &cnode);
Shapes GetNodeShape(const AnfNodePtr &node);
std::string GetPrimName(const CNodePtr &node);
std::shared_ptr<Value> GetAttrsFromAnfNode(const std::shared_ptr<AnfNode> &node, const string &key);
std::vector<AnfNodePtr> ReplaceOpInput(const Operator &replace_op, const std::string &instance_name,
                                       const CNodePtr &node);
std::string CreateInstanceName(const CNodePtr &node, size_t index);
TensorInfo GetInputsTensorInfo(const std::pair<AnfNodePtr, int64_t> &param_info);
AnfNodePtr CheckMakeTupleSplit(const AnfNodePtr &node, const FuncGraphManagerPtr &manager);
AnfNodePtr MatchPattern(const AnfNodePtr &node, const NodeUsersMap &user_map,
                        const std::vector<std::pair<const std::string, int64_t>> &match_pattern);

// for specific scenarios
RankList FindCommonMirrorGroup(const FuncGraphPtr &root);
void SetCommunicationOpGroupLabel(std::vector<AnfNodePtr> new_node_input);
void SetStridedSliceSplitStrategy(const std::vector<AnfNodePtr> &all_nodes);
AnfNodePtr CreateFP16Cast(const CNodePtr &node, const AnfNodePtr &pre_node, const TypePtr &compute_node_type);
TypePtr FindChildCastWithFP32ToFP16(const CNodePtr &cnode_ptr, const NodeUsersMap &node_users_map);
void LabelGenMaskMicro(const FuncGraphPtr &root);
void SetCastForParamNotRecompute(const std::vector<AnfNodePtr> &all_nodes);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_PARALLEL_UTILS_H_
