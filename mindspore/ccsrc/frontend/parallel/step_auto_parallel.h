/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_AUTO_PARALLEL_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_AUTO_PARALLEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "frontend/optimizer/opt.h"
#include "frontend/parallel/status.h"
#include "ir/anf.h"
#include "pipeline/jit/pipeline.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_parse_graph.h"

namespace mindspore {
namespace parallel {
// main step of Auto-parallel
bool StepAutoParallel(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer);

bool IsSplittableOperator(const std::string &);

bool IsAutoParallelCareNode(const CNodePtr &);

void InitCostGraph();

Status ConstructCostGraphNodesByUniqueId(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root);

Status ConstructCostGraphNodesByUniqueIdTC(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root);

void ConstructCostGraphEdges(const std::vector<AnfNodePtr> &all_nodes);

void AugmentCostGraph(const std::vector<AnfNodePtr> &all_nodes);

Status IgnoreOperatorsInCostGraph();

Status ParallelStrategySearch(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root);

Status ParallelStrategyRecSearch(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root);

std::vector<std::vector<std::string>> RecInputTensorNames(const std::map<std::string, std::string>::iterator &it,
                                                          std::vector<std::vector<std::string>> input_tensor_names);

CNodePtr GetInternalOperatorInfo(const CNodePtr &cnode, const ValueNodePtr &prim_anf_node);

void ModifyInputsTensorNameListIfOperatorInfoCreated(const std::string &name, const std::string &uniqueid);

size_t FindOperatorIndexById(const std::string &unique_id,
                             const std::vector<std::vector<std::string>> &input_tensor_names);

void AddSharedTensorWhenMultiUsers(
  const std::pair<std::string, std::pair<AnfNodePtr, AnfNodeIndexSet>> &parameter_users_info);

std::vector<std::vector<size_t>> GetSharedTensorsOps(
  const std::shared_ptr<Graph> &graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
  const std::vector<std::vector<std::string>> &shared_tensors_ops_names,
  const std::vector<std::vector<std::string>> &input_tensor_names);
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_AUTO_PARALLEL_H_
