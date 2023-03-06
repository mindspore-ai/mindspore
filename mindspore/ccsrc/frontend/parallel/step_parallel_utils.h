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
#include <map>
#include <memory>
#include "base/base.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"

namespace mindspore {
namespace parallel {
const int64_t TWO_INPUT_SIZE = 2;
extern size_t TOTAL_OPS;
extern std::map<AnfNodePtr, std::pair<AnfNodePtr, int64_t>> g_RefMap;
struct CommInfo {
  int64_t device_num = 1;
  int64_t global_rank = 0;
  std::string world_group;
  std::string communication_backend;
};
const std::set<std::string> COMMUNICATION_OPS = {ALL_REDUCE, ALL_GATHER,       ALL_TO_ALL,         REDUCE_SCATTER,
                                                 BROADCAST,  NEIGHBOREXCHANGE, NEIGHBOREXCHANGEV2, SYNC_BATCH_NORM};
// common method
CommInfo GetCommInfo();
ShapeVector ToFullShape(const ShapeVector &input_shape, size_t index);
void ExtendInputArgsAbstractShape(const AbstractBasePtr &args_abstract_item, size_t index);
bool IsSomePrimitive(const CNodePtr &cnode, const std::string &name);
bool IsSomePrimitiveList(const CNodePtr &cnode, const std::set<string> &check_list);
bool IsParallelCareNode(const CNodePtr &cnode);
bool IsAutoParallelCareNode(const CNodePtr &cnode);
Shapes GetNodeShape(const AnfNodePtr &node);
// Extract shape from anfnode
std::vector<Shapes> ExtractShape(const CNodePtr &node);
// Generate and init parallel operator
OperatorInfoPtr OperatorInstance(const PrimitivePtr &prim, const PrimitiveAttrs &attrs,
                                 const std::vector<Shapes> &shape_list);
OperatorInfoPtr CreateOperatorInfo(const CNodePtr &cnode);
std::string GetPrimName(const CNodePtr &node);
std::shared_ptr<Value> GetAttrsFromAnfNode(const std::shared_ptr<AnfNode> &node, const string &key);
std::vector<AnfNodePtr> ReplaceOpInput(const Operator &replace_op, const std::string &instance_name,
                                       const CNodePtr &node);
std::string CreateInstanceName(const CNodePtr &node, size_t index);
TensorInfo GetInputsTensorInfo(const std::pair<AnfNodePtr, int64_t> &param_info);
AnfNodePtr CheckMakeTupleSplit(const AnfNodePtr &node, const FuncGraphManagerPtr &manager);
bool IsControlFlowNode(const AnfNodePtr &node);
int64_t GetTupleGetItemIndex(const CNodePtr &cnode);
std::pair<AnfNodePtr, int64_t> GetRealKernelNode(const AnfNodePtr &node, int64_t get_item_index,
                                                 CNodePtr *call_node = nullptr, bool ignore_get_item = true);

std::vector<std::pair<AnfNodePtr, int>> GetOutputNodesWithFilter(const AnfNodePtr &node,
                                                                 std::function<bool(const AnfNodePtr &)> filter);

void RedistributionPreNode(const CNodePtr &cnode, const FuncGraphManagerPtr &manager,
                           std::vector<AnfNodePtr> *pre_nodes);
void RedistributionNextNode(const AnfNodePtr &node, const FuncGraphManagerPtr &manager,
                            const NodeUsersMap &node_users_map, int64_t get_item_index, int64_t make_tuple_index,
                            std::vector<std::pair<std::pair<AnfNodePtr, int>, int>> *next_nodes);

// for specific scenarios
RankList FindCommonMirrorGroup(const FuncGraphPtr &root);
bool IsTraining(const FuncGraphManagerPtr &manager);
void SetCommunicationOpGroupLabel(std::vector<AnfNodePtr> new_node_input);
void SetStridedSliceSplitStrategy(const std::vector<AnfNodePtr> &all_nodes);
AnfNodePtr CreateFP16Cast(const CNodePtr &node, const AnfNodePtr &pre_node, const TypePtr &compute_node_type);
TypePtr FindChildCastWithFP32ToFP16(const CNodePtr &cnode_ptr, const NodeUsersMap &node_users_map);
void LabelGenMaskMicro(const FuncGraphPtr &root);
void AddNodeFusionInfo(const CNodePtr &node, const CNodePtr &comm_node, const std::string &backward_comm_name,
                       int32_t fusion_id);
void SetCastForParamNotRecompute(const std::vector<AnfNodePtr> &all_nodes);
bool IsPynativeParallel();
bool IsAutoParallelCareGraph(const FuncGraphPtr &func_graph);
bool HasNestedMetaFg(const FuncGraphPtr &func_graph);
bool IsEmbedShardNode(const FuncGraphPtr &func_graph);
bool IsSplittableOperator(const std::string &op_name);
std::vector<std::string> ExtractInputsTensorName(const CNodePtr &node);
OperatorInfoPtr GetDistributeOperator(const CNodePtr &node);
bool StrategyFound(const mindspore::HashMap<std::string, ValuePtr> &attrs);
bool AttrFound(const mindspore::HashMap<std::string, ValuePtr> &attrs, const std::string &target);
void ExceptionIfHasCommunicationOp(const std::vector<AnfNodePtr> &all_nodes);
std::string MirrorOpName();
// Extract strategy from attr
StrategyPtr ExtractStrategy(const ValuePtr &stra);
ParameterMap NodeParameterName(const CNodePtr &node, int64_t index, size_t curr_depth);
Status ParallelInit();
std::pair<bool, CNodePtr> FindCNode(const AnfNodePtr &anode, const std::string &name, const FuncGraphPtr &func_graph,
                                    size_t max_depth);
void SetSharedParameterFlag(const FuncGraphPtr &root, const AnfNodePtr &parameter);
StrategyPtr GenerateStandAloneStrategy(const Shapes &inputs_shape);
StrategyPtr GenerateBatchParallelStrategy(const OperatorInfoPtr operator_, const PrimitivePtr prim);
bool IsInsertVirtualOutput(const FuncGraphPtr &root);
TensorLayout GetInputLayoutFromCNode(const std::pair<AnfNodePtr, int64_t> &node_pair);
Shape mirror_group_list(const TensorLayoutPtr &layout);
// Transfer number to serial number string
std::string GetSerialNumberString(size_t number);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_PARALLEL_UTILS_H_
