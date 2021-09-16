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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_PARALLEL_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_PARALLEL_H_

#include <vector>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

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

struct CommInfo {
  int64_t device_num = 1;
  int64_t global_rank = 0;
  std::string world_group;
  std::string communication_backend;
};

std::vector<AnfNodePtr> CreateInput(const Operator &op, const AnfNodePtr &node, const std::string &instance_name);
void ForwardCommunication(OperatorVector forward_op, const CNodePtr &node);

void InsertRedistribution(const RedistributionOpListPtr &redistribution_oplist_ptr, const CNodePtr &node,
                          const FuncGraphPtr &func_graph, int64_t pos, const CNodePtr &pre_node);

TensorLayout GetTensorInLayout(const CNodePtr &pre_node, const PrimitivePtr &pre_prim,
                               const OperatorInfoPtr &distribute_operator_pre);

OperatorInfoPtr GetDistributeOperator(const CNodePtr &node);

void Redistribution(const std::pair<AnfNodePtr, int64_t> &node_pair, const OperatorInfoPtr &distribute_operator,
                    const CNodePtr &middle_node, int64_t index, TensorRedistribution tensor_redistribution,
                    const CNodePtr &pre_node);

bool StrategyFound(std::unordered_map<std::string, ValuePtr> attrs);

void MarkForwardCNode(const FuncGraphPtr &root);

bool FindCommunicationOp(const std::vector<AnfNodePtr> &all_nodes);

void StepRedistribution(const CNodePtr &node, const OperatorInfoPtr &distribute_operator, const CNodePtr &insert_node,
                        const TensorRedistribution &tensor_redistribution, const CNodePtr &pre_node);

void StepReplaceOp(OperatorVector replace_op, const CNodePtr &node);

void InsertVirtualDivOp(const VirtualDivOp &virtual_div_op, const CNodePtr &node);

std::pair<AnfNodePtr, bool> FindParameter(const AnfNodePtr &node, const FuncGraphPtr &func_graph);

std::pair<bool, CNodePtr> FindCNode(const AnfNodePtr &anode, const std::string &name, const FuncGraphPtr &func_graph,
                                    size_t max_depth);

// Generate and init parallel operator
OperatorInfoPtr OperatorInstance(const PrimitivePtr &prim, const PrimitiveAttrs &attrs,
                                 const std::vector<Shapes> &shape_list);

// Generate without initing parallel operator
OperatorInfoPtr NewOperatorInstance(const PrimitivePtr &prim, const PrimitiveAttrs &attrs,
                                    std::vector<Shapes> shape_list);

// Extract strategy from attr
StrategyPtr ExtractStrategy(const ValuePtr &strategy);

// Extract shape from anfnode
std::vector<Shapes> ExtractShape(const CNodePtr &node);

// Find finally sub graph
std::pair<AnfNodePtr, int64_t> FindSubGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &parameter);

// Set distribute shape for parameters abstract
std::string SetParallelShape(const AnfNodePtr &parameter, const std::pair<AnfNodePtr, int64_t> &res);

// change parameters'shape in resource
void CoverSliceShape(const FuncGraphPtr &root);

void SetVirtualDatasetStrategy(const CNodePtr &node);
bool IsInsertVirtualOutput(const FuncGraphPtr &root);
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
bool StepParallel(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer);

int64_t GetTupleGetItemIndex(const CNodePtr &cnode);

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

CommInfo GetCommInfo();

std::string GetPrimName(const CNodePtr &node);

void ReorderForPipelineSplit(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager, int64_t pipeline_stages);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_PARALLEL_H_
