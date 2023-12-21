/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

// maybe the input value is dynamic for these ops
static const std::set<std::string> CANDIDATE_DYNAMIC_VALUE_OPS = {RESHAPE, STRIDED_SLICE, PAD_V3};
bool IsDynamicShapeInput(const CNodePtr &node, const AnfNodePtr &input);
// split tensor only for first input
static const std::set<std::string> SPLIT_TENSOR_ONLY_FOR_FIRST_INPUT_OPS = {PAD_V3};
// the input is tuple or list
static const std::set<std::string> INPUT_IS_TUPLE_OR_LIST_OPS = {CONCAT, STACK, ADDN, INCRE_FLASH_ATTENTION};

const int64_t TWO_INPUT_SIZE = 2;

constexpr char KAttrAsLossDivisor[] = "as_loss_divisor";
constexpr char KAttrDevMatrixShape[] = "dev_matrix_shape";
constexpr char KAttrInputsTensorMap[] = "inputs_tensor_map";
constexpr char KAttrOutputsTensorMap[] = "outputs_tensor_map";

extern size_t TOTAL_OPS;
extern std::map<AnfNodePtr, std::pair<AnfNodePtr, int64_t>> g_RefMap;
struct CommInfo {
  int64_t device_num = 1;
  int64_t global_rank = 0;
  std::string world_group;
  std::string communication_backend;
};
const std::set<std::string> COMMUNICATION_OPS = {
  ALL_REDUCE,       ALL_GATHER,         ALL_TO_ALL,      REDUCE_SCATTER,     BROADCAST,
  NEIGHBOREXCHANGE, NEIGHBOREXCHANGEV2, SYNC_BATCH_NORM, COLLECTIVE_SCATTER, COLLECTIVE_GATHER};
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
AnfNodePtr GetInputNodeWithFilter(const AnfNodePtr &node,
                                  std::function<std::pair<bool, size_t>(const CNodePtr &)> filter);
void RedistributionPreNode(const CNodePtr &cnode, const FuncGraphManagerPtr &manager,
                           std::vector<AnfNodePtr> *pre_nodes);
void RedistributionNextNode(const AnfNodePtr &node, const FuncGraphManagerPtr &manager,
                            const NodeUsersMap &node_users_map, int64_t get_item_index, int64_t make_tuple_index,
                            std::vector<std::pair<std::pair<AnfNodePtr, int>, int>> *next_nodes);
AnfNodePtr NewMicroMirrorPrimByMicroMirror(const FuncGraphPtr &func_graph, const CNodePtr &micro_mirror,
                                           const AnfNodePtr &micro_mirror_new_input);
// for specific scenarios
RankList FindCommonMirrorGroup(const FuncGraphPtr &root);
bool IsTraining(const FuncGraphManagerPtr &manager);
bool HasBackward(const FuncGraphPtr &root);
void SetCommunicationOpGroupLabel(std::vector<AnfNodePtr> new_node_input);
void SetStridedSliceSplitStrategy(const std::vector<AnfNodePtr> &all_nodes);
AnfNodePtr CreateFP16Cast(const CNodePtr &node, const AnfNodePtr &pre_node, const TypePtr &compute_node_type);
TypePtr FindChildCastWithFP32ToFP16(const std::pair<AnfNodePtr, int> &res, const NodeUsersMap &node_users_map);
void LabelGenMaskMicro(const FuncGraphPtr &root);
void AddNodeFusionInfo(const CNodePtr &node, const CNodePtr &comm_node, const std::string &backward_comm_name,
                       int32_t fusion_id);
void SetCastForParamNotRecompute(const std::vector<AnfNodePtr> &all_nodes);
bool IsPynativeParallel();
bool IsAutoParallelCareGraph(const FuncGraphPtr &func_graph);
bool HasNestedMetaFg(const FuncGraphPtr &func_graph);
bool IsEmbedShardNode(const FuncGraphPtr &func_graph);
bool IsSplittableOperator(const std::string &op_name);
AnfNodePtr FindRealInputByFormalParameter(const CNodePtr &node, const AnfNodePtr &input,
                                          const std::vector<AnfNodePtr> &all_nodes);
std::vector<std::string> ExtractInputsTensorName(const CNodePtr &node, const std::vector<AnfNodePtr> &all_nodes);
OperatorInfoPtr GetDistributeOperator(const CNodePtr &node);
bool StrategyFound(const mindspore::HashMap<std::string, ValuePtr> &attrs);
bool AttrFound(const mindspore::HashMap<std::string, ValuePtr> &attrs, const std::string &target);
void ExceptionIfHasCommunicationOp(const std::vector<AnfNodePtr> &all_nodes);
std::string MirrorOpName();
// Extract strategy from attr
StrategyPtr ExtractStrategy(const ValuePtr &stra);
ParameterMap NodeParameterName(const CNodePtr &node, int64_t index, size_t curr_depth);
std::vector<std::pair<AnfNodePtr, int>> FuncGraphNodeUsers(const std::pair<AnfNodePtr, int> &node_pair);
Status ParallelInit(size_t rank_id = 0, const size_t devices = 0);
std::pair<bool, CNodePtr> FindCNode(const AnfNodePtr &anode, const std::string &name, const FuncGraphPtr &func_graph,
                                    size_t max_depth);
std::pair<std::shared_ptr<AnfNode>, int> BFSParallelCareNode(const AnfNodePtr &node_ptr,
                                                             const NodeUsersMap &node_users_map, const int index,
                                                             const std::vector<AnfNodePtr> &all_nodes);
void FindPreNodeCrossFuncGraph(CNodePtr *cnode, int64_t out_index);
bool CrossInterNode(CNodePtr *prev_cnode, ValueNodePtr *prev_prim_anf_node, PrimitivePtr *prev_prim);
bool IsCarePrevCNode(const CNodePtr &prev_cnode, const PrimitivePtr &prev_prim);
void SetSharedParameterFlag(const FuncGraphPtr &root, const AnfNodePtr &parameter);
StrategyPtr GenerateStandAloneStrategy(const Shapes &inputs_shape);
StrategyPtr GenerateBatchParallelStrategy(const OperatorInfoPtr operator_, const PrimitivePtr prim);
bool IsInsertVirtualOutput(const FuncGraphPtr &root);
TensorLayout GetInputLayoutFromCNode(const std::pair<AnfNodePtr, int64_t> &node_pair);
Shape mirror_group_list(const TensorLayoutPtr &layout);
// Transfer number to serial number string
std::string GetSerialNumberString(size_t number);
bool IsIgnoreSplitTensor(const CNodePtr &node, int64_t index);
bool MergeConcatSlice(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphManagerPtr &manager);
void UpdateMicroBatchInterleavedStatus(const std::vector<AnfNodePtr> &all_nodes);
Status ExtractUserConfigLayout(const mindspore::HashMap<std::string, ValuePtr> &prim_attrs, const Shapes &inputs_shape,
                               const Shapes &outputs_shape,
                               std::vector<std::shared_ptr<TensorLayout>> *in_tensor_layouts,
                               std::vector<std::shared_ptr<TensorLayout>> *out_tensor_layouts);
inline bool IsMakeSequence(const AnfNodePtr &node) {
  return AnfNodeIsPrimitive(node, MAKE_TUPLE) || AnfNodeIsPrimitive(node, MAKE_LIST);
}
inline bool IsValueSequence(const AnfNodePtr &node) {
  return IsValueNode<ValueList>(node) || IsValueNode<ValueTuple>(node);
}
bool IsCellReuseForwardGraph(const FuncGraphPtr &graph);
FuncGraphPtr GetCellReuseBackwardGraph(const FuncGraphPtr &forward_graph);
bool IsCommunicationOp(const PrimitivePtr &prim);

inline void SetReserved(const FuncGraphPtr &root) {
  // Keep all func graph for parallel before save result.
  root->set_reserved(true);
  for (auto &fg : root->func_graphs_used_total()) {
    MS_EXCEPTION_IF_NULL(fg);
    fg->set_reserved(true);
  }
}
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_PARALLEL_UTILS_H_
