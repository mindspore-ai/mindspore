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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARAMETER_MANAGER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARAMETER_MANAGER_H_

#include <vector>
#include <string>
#include <utility>
#include "base/base.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/step_parallel_utils.h"

namespace mindspore {
namespace parallel {
constexpr char EXP_AVG[] = "exp_avg_";
constexpr char EXP_AVG_SQ_ROW[] = "exp_avg_sq_row_";
constexpr char EXP_AVG_SQ_COL[] = "exp_avg_sq_col_";
constexpr char EXP_AVG_SQ[] = "exp_avg_sq_";
using RefKeyPair = std::pair<AnfNodePtr, std::vector<AnfNodePtr>>;
using ParameterUsersInfo = std::pair<std::string, std::pair<AnfNodePtr, AnfNodeIndexSet>>;
struct ParameterSliceInfo {
  Shape slice_shape;
  RankList group_ranks;
};

ParameterUsersInfo FindParameterUsers(const AnfNodePtr &node, bool (*IsCareNode)(const CNodePtr &));
void CheckParameterSplit(const std::vector<AnfNodePtr> &all_nodes);
void HandleSymbolicKeyInstance(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes);
void HandleNoUsedParameter(const FuncGraphPtr &root);
void HandleFullySplitParameters(const FuncGraphPtr &root);
void SetClonedTensorShapeForOptimizer(const FuncGraphPtr &root);
void HandleAdaFactorOpt(const FuncGraphPtr &root);
bool ParameterIsCloned(const AnfNodePtr &parameter_node);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARAMETER_MANAGER_H_
