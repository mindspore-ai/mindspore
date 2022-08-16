/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_PARALLEL_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_PARALLEL_H_

#include <vector>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "utils/hash_map.h"
#include "frontend/optimizer/opt.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/pipeline.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"

using OperatorInfoPtr = std::shared_ptr<mindspore::parallel::OperatorInfo>;

namespace mindspore {
namespace parallel {
const uint64_t kUSecondInSecond = 1000000;
const int32_t RECURSION_LIMIT = 3;

struct LossNodeInfo {
  bool has_tuple_getitem = false;
  int64_t dout_index = 0;  // now don't support the sens is a tuple
  CNodePtr loss_node = nullptr;
};

std::vector<AnfNodePtr> CreateInput(const Operator &op, const AnfNodePtr &node, const std::string &instance_name);
void ForwardCommunication(OperatorVector forward_op, const CNodePtr &node);

void InsertRedistribution(const RedistributionOpListPtr &redistribution_oplist_ptr, const CNodePtr &node,
                          const FuncGraphPtr &func_graph, int64_t pos, const CNodePtr &pre_node);

TensorLayout GetTensorInLayout(const AnfNodePtr &pre_node, int get_item_index);

OperatorInfoPtr GetDistributeOperator(const CNodePtr &node);

void Redistribution(const std::pair<AnfNodePtr, int64_t> &node_pair, const AnfNodePtr &pre_node,
                    TensorRedistribution tensor_redistribution, int get_item_index);

bool StrategyFound(const mindspore::HashMap<std::string, ValuePtr> &attrs);

bool AttrFound(const mindspore::HashMap<std::string, ValuePtr> &attrs, const std::string &target);

AnfNodePtr GetAccuGrad(const std::vector<AnfNodePtr> &parameters, const std::string &weight_name);

void MarkForwardCNode(const FuncGraphPtr &root);

void ExceptionIfHasCommunicationOp(const std::vector<AnfNodePtr> &all_nodes);

void StepRedistribution(const CNodePtr &cnode, const TensorRedistribution &tensor_redistribution,
                        const NodeUsersMap &node_users_map);

void StepReplaceOp(OperatorVector replace_op, const CNodePtr &node);

void InsertVirtualDivOp(const VirtualDivOp &virtual_div_op, const CNodePtr &node);

std::pair<bool, CNodePtr> FindCNode(const AnfNodePtr &anode, const std::string &name, const FuncGraphPtr &func_graph,
                                    size_t max_depth);

// Extract strategy from attr
StrategyPtr ExtractStrategy(const ValuePtr &stra);

// Find finally sub graph
std::pair<AnfNodePtr, int64_t> FindSubGraph(const FuncGraphPtr &graph, const AnfNodePtr &parameter);

// Set distribute shape for parameters abstract
std::string SetParallelShape(const AnfNodePtr &parameter, const std::pair<AnfNodePtr, int64_t> &res);

// change parameters'shape in resource
void CoverSliceShape(const FuncGraphPtr &root);

void LableBatchSizeSplit(const CNodePtr &node);

void SetVirtualDatasetStrategy(const CNodePtr &node);
bool IsInsertVirtualOutput(const FuncGraphPtr &root);

void SetStridedSliceSplitStrategy(const std::vector<AnfNodePtr> &all_nodes);

// Create parallel operator for primitive node(has strategy)
void ExtractInformation(const std::vector<AnfNodePtr> &all_nodes);

TensorLayout GetInputLayoutFromCNode(const std::pair<AnfNodePtr, int64_t> &node_pair);

std::shared_ptr<TensorLayout> FindNextLayout(const CNodePtr &node);

std::shared_ptr<TensorLayout> GetOutputLayoutFromCNode(const CNodePtr &cnode, size_t output_index);

std::shared_ptr<TensorLayout> FindPrevParallelCareNodeLayout(const AnfNodePtr &node, size_t output_index);

std::shared_ptr<TensorLayout> FindPrevLayout(const AnfNodePtr &node);

void ReshapeInit(const std::vector<AnfNodePtr> &all_nodes);

StrategyPtr GenerateBatchParallelStrategy(const OperatorInfoPtr operator_, const PrimitivePtr prim);

// Add node for whole graph
void ParallelCommunication(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                           const FuncGraphManagerPtr &manager);

ParameterMap NodeParameterName(const CNodePtr &node, int64_t index, size_t curr_depth);

void CheckpointStrategy(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root);

// main step of Parallel
bool StepParallel(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer);

Status ParallelInit();

std::set<FuncGraphPtr> ForwardGraph(const FuncGraphPtr &root);

std::vector<std::string> ExtractInputsTensorName(const CNodePtr &node);

std::shared_ptr<TensorLayout> FindParameterNextLayout(const AnfNodePtr &node);

bool IsUsedParameter(const FuncGraphPtr &graph, const AnfNodePtr &parameter);

void ApplyParallelOptOnParam(TensorLayout *tensor_layout, const OperatorInfoPtr &distribute_operator,
                             const CNodePtr &cnode, const AnfNodePtr &parameter, size_t index);

void SetLastNodeStrategy(const StrategyPtr strategyPtr);

bool CreateGroupsByCkptFile(const std::string &file);

void FindLastNodesUniqueId(const FuncGraphPtr &root, std::vector<std::string> *unique_ids,
                           std::vector<size_t> *indexes);

void InsertVirtualOutput(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes);

std::string MirrorOpName();

std::string GetPrimName(const CNodePtr &node);

void ReorderForPipelineSplit(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager, int64_t pipeline_stages);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_PARALLEL_H_
