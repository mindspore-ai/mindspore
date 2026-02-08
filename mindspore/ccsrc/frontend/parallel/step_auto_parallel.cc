/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/step_auto_parallel.h"

#include <algorithm>
#include <cinttypes>
#include <ctime>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "include/frontend/optimizer/optimizer.h"
#include "frontend/parallel/auto_parallel/dp_algo_costmodel.h"
#include "frontend/parallel/auto_parallel/edge_costmodel.h"
#include "include/frontend/parallel/graph_costmodel.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_generate_strategy.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_parse_graph.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_partition.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/grad_accumulation_utils.h"
#include "frontend/parallel/ops_info/reshape_info.h"
#include "frontend/parallel/ops_info/tmp_identity_info.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/strategy_loader.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"
#include "include/utils/parallel_context.h"
#include "ir/anf.h"
#include "ir/param_info.h"
#include "ir/tensor.h"
#include "ir/graph_utils.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "primitive/sequence_ops.h"
#include "frontend/jit/ps/pipeline_split.h"
#include "utils/hash_map.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace parallel {
void SearchParallelStrategy(const std::string &strategy_search_mode, const FuncGraphPtr &root,
                            const std::vector<AnfNodePtr> &all_nodes) {
  if (StrategyCheckpoint::GetInstance().LoadAutoOpStrategyOn()) {
    if (StrategyLoader::LoadStrategyFromFile(all_nodes) == SUCCESS) {
      MS_LOG(INFO) << "Load strategies success, jump search strategy.";
      return;
    }
    MS_LOG(EXCEPTION) << "Load strategies failed, please check whether your config is changed.";
  }
  if ((strategy_search_mode == kDynamicProgramming) || (strategy_search_mode == kShardingPropagation)) {
    PROF_START(parallel_strategy_search);
    if (ParallelStrategySearch(all_nodes, root) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Auto-parallel strategy search failed when using " << strategy_search_mode
                        << " searching mode";
    }
    PROF_END(parallel_strategy_search);
  } else if (strategy_search_mode == kRecursiveProgramming) {
    if (ParallelStrategyRecSearch(all_nodes, root) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Auto-parallel strategy search failed when using RP searching mode";
    }
  } else {
    MS_LOG(EXCEPTION) << "Auto-parallel strategy searching mode unexpected: " << strategy_search_mode;
  }
  if (StrategyCheckpoint::GetInstance().SaveAutoOpStrategyOn()) {
    StrategyLoader::SaveStrategyToFile(all_nodes);
  }
}

bool HasCellShard(const FuncGraphPtr &func_graph) {
  AnfNodePtr ret = func_graph->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = TopoSort(ret, SuccDeeperSimple);
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimShard) || IsPrimitiveCNode(node, prim::kPrimReshard)) {
      return true;
    }
  }
  return false;
}

bool IsSkipAutoParallel(const FuncGraphPtr &root, const std::string &strategy_search_mode, const bool is_pre_action) {
  root->set_flag(kHasShard, HasCellShard(root));
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (root->has_flag(kSkipAutoParallelCompile) || parallel_mode != kAutoParallel ||
      root->has_flag(AUTO_PARALLEL_RUN_ONCE_ONLY)) {
    return true;
  }

  // For parallel with shard, skip PreAutoParallel
  // Shard Prim will be deleted once shard is set, see pass.cc.
  if (root->has_flag(kHasShard)) {
    return true;
  }
  if (root->has_flag(kSharded)) {
    return false;
  }

  if ((is_pre_action && strategy_search_mode == kDynamicProgramming) ||
      (!is_pre_action && strategy_search_mode != kDynamicProgramming)) {
    return true;
  }
  return false;
}

bool PreprocessRootGraph(const FuncGraphPtr &root) {
  bool changes = false;
  bool is_pre_action = !root->has_flag(AUTO_PARALLEL_FINISH_PRE_ACTION);
  if (is_pre_action) {
    root->set_flag(AUTO_PARALLEL_FINISH_PRE_ACTION, true);
    auto manager = root->manager();
    const auto &graphs = manager->func_graphs();
    bool is_training = std::any_of(graphs.cbegin(), graphs.cend(),
                                   [](auto cur_graph) -> bool { return cur_graph->has_flag(kTraining); });
    if (is_training) {
      root->set_flag(kTraining, true);
    }
    changes = true;
  }
  return changes;
}

bool SkipAutoParallel(const FuncGraphPtr &root, const std::string &strategy_search_mode, bool is_pre_action,
                      MSLogTime msTime) {
  // control whether use model_parallel mode
  bool is_skip = IsSkipAutoParallel(root, strategy_search_mode, is_pre_action);
  if (is_skip && !ParallelContext::GetInstance()->direct_split()) {
    msTime.End();
    uint64_t time = msTime.GetRunTimeUS();
    MS_LOG(INFO) << "Now leaving step auto parallel, used time: " << time << " us";
    return true;
  }
  return false;
}

std::vector<AnfNodePtr> GetAllNodesForParallel(const FuncGraphPtr &root, const AnfNodePtr &ret) {
  std::vector<AnfNodePtr> all_nodes;
  if (CheckShardingPropagation()) {
    all_nodes = TopoSort(ret, SuccDeeperSimple);
  }

  // merge concat slice for recursive propagation
  if (!CheckShardingPropagation()) {
    MS_EXCEPTION_IF_NULL(root);
    auto manager = root->manager();
    MS_EXCEPTION_IF_NULL(manager);
    MS_EXCEPTION_IF_NULL(ret);
    all_nodes = DeepScopedGraphSearch(ret);
    bool merged = MergeConcatSlice(all_nodes, manager);
    if (merged) {
      all_nodes = TopoSort(ret, SuccDeeperSimple);
    }
  }
  return all_nodes;
}

void InferParallelStrategy(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                           const std::string &strategy_search_mode) {
  if (strategy_search_mode == kRecursiveProgramming &&
      ((g_device_manager->DeviceNum() & (g_device_manager->DeviceNum() - 1)) != 0)) {
    MS_LOG(EXCEPTION)
      << "The recursive auto parallel strategy searching mode requires the device num be the power of 2.";
  }

  if (strategy_search_mode == kDynamicProgramming) {
    MS_LOG(WARNING) << "The dynamic programming auto parallel strategy searching mode will soon not be used.";
  }

  // set grad accumulation step
  SetGradAccumulationStep(all_nodes);

  // mark the forward cnodes, parallel only care these nodes
  MarkForwardCNode(root);

  ExceptionIfHasCommunicationOp(all_nodes);

  std::vector<AnfNodePtr> final_all_nodes = all_nodes;
  if (IsInsertVirtualOutput(root)) {
    InsertVirtualOutput(root, all_nodes);
    const AnfNodePtr &ret_after = root->get_return();
    MS_EXCEPTION_IF_NULL(ret_after);
    final_all_nodes = DeepScopedGraphSearch(ret_after);
  }

  // search parallelization strategy
  SearchParallelStrategy(strategy_search_mode, root, final_all_nodes);
  root->set_flag(AUTO_PARALLEL_RUN_ONCE_ONLY, true);
}

bool StepAutoParallel(const FuncGraphPtr &root, const opt::OptimizerPtr &) {
  MSLogTime msTime;
  msTime.Start();

  // Mode 'dynamic programming' will run after pipeline_split, others don't.
  MS_EXCEPTION_IF_NULL(root);
  bool is_pre_action = !root->has_flag(AUTO_PARALLEL_FINISH_PRE_ACTION);
  bool changes = PreprocessRootGraph(root);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  std::string strategy_search_mode = ParallelContext::GetInstance()->strategy_search_mode();
  if (SkipAutoParallel(root, strategy_search_mode, is_pre_action, msTime)) {
    return changes;
  }
  MS_LOG(INFO) << "search_mode: " << strategy_search_mode;
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpGraph(root, std::string(STEP_AUTO_PARALLEL_BEGIN));
  }
#endif
  MS_LOG(INFO) << "Now entering step auto parallel";
  TOTAL_OPS = 0;
  AnfNodePtr ret = root->get_return();
  std::vector<AnfNodePtr> all_nodes = GetAllNodesForParallel(root, ret);

  // insert Virtual Dataset if not exist
  if (ParallelInit() != SUCCESS) {
    MS_LOG(EXCEPTION) << "Parallel init failed";
  }
  if (!mindspore::pipeline::HasVirtualDataset(all_nodes)) {
    mindspore::pipeline::InsertVirtualDataset(root, all_nodes);
  }

  // redo deepscoped search again to connected the Virtual Dataset into the graph
  all_nodes = CheckShardingPropagation() ? TopoSort(ret, SuccDeeperSimple) : DeepScopedGraphSearch(ret);

  InferParallelStrategy(root, all_nodes, strategy_search_mode);
  msTime.End();
  uint64_t time = msTime.GetRunTimeUS();
  MS_LOG(INFO) << "Now leaving step auto parallel, used time: " << time << " us";
  return changes;
}

bool IsElementWiseOperator(const std::string &op_name) {
  // clang-format off
  static const std::set<std::string> elementwise_op = {ACTIVATION, GELU,         TANH,
                                                       SOFTMAX,    LOG_SOFTMAX,  RELU,
                                                       SQRT,       CAST,         POW,
                                                       EXP,        LOG,          COS,
                                                       ACOS,       LOGICALNOT,   NEG,
                                                       SQUARE,     SIGMOID,      ABS,
                                                       ACOSH,      ASIN,         ASINH,
                                                       ATAN,       ATANH,        CEIL,
                                                       COSH,       EXPM1,        LOG1P,
                                                       SIN,        SINH,         TAN,
                                                       RSQRT,      RECIPROCAL,   INV,
                                                       ROUND,      FLOOR,        SIGN,
                                                       ERF,        ERFC,         ZEROSLIKE,
                                                       ONESLIKE,   BESSELI0E,    BESSELI0,
                                                       BESSELI1,   BESSELJ0,     BESSELJ0,
                                                       ASSIGN,     ASSIGN_ADD,   ATAN2,
                                                       DIVNONAN,   LOGICALAND,   ELU,
                                                       LOGICALOR,  RELU6,        SOFTPLUS,
                                                       SOFTSIGN,   LESS,         LESSEQUAL,
                                                       BESSELI1E,  GREATEREQUAL, APPROXIMATEEQUAL,
                                                       MOD,        REVERSEV2,    REPEAT_ELEMENTS,
                                                       TRUNC,      LGAMMA,       CHOLESKY,
                                                       SWIGLU, FMODSCALAR};
  // clang-format on
  auto iter = elementwise_op.find(op_name);
  return (iter != elementwise_op.cend());
}

// Recording the operators appearing in a for-loop.
// Currently, we assume that the operators in different for-loops are identical, and their traversal
// orderings are also identical.
// Therefore, we create OperatorInfo objects for the operators in a loop (say, loop-3), and reuse them in
// the rest of loops (loop-2, loop-1 and loop-0)
std::set<std::string> ops_in_a_loop_;
// Whether two operators are in different loops; if it is true, then return true.
// If at least one of the two operators is not in the loop, then return false.
// If two operators are in the same loop, the return false.
bool IsOperatorsInTwoSeparateLoops(const CNodePtr &a_cnode, const CNodePtr &b_cnode) {
  auto a_op_info = a_cnode->user_data<OperatorInfo>();
  MS_EXCEPTION_IF_NULL(a_op_info);
  auto b_op_info = b_cnode->user_data<OperatorInfo>();
  MS_EXCEPTION_IF_NULL(b_op_info);
  if ((ops_in_a_loop_.find(a_op_info->name()) == ops_in_a_loop_.end()) ||
      (ops_in_a_loop_.find(b_op_info->name()) == ops_in_a_loop_.end())) {
    return false;
  }
  size_t a_loop_index = 0;
  size_t b_loop_index = 0;
  const auto &a_fullname = a_cnode->fullname_with_scope();
  if (!GetLoopIndexFromCNode(a_cnode, &a_loop_index)) {
    MS_LOG_WITH_NODE(EXCEPTION, a_cnode) << "The operator with fullname_with_scope: " << a_fullname
                                         << " was not included in the set.";
  }
  const auto &b_fullname = b_cnode->fullname_with_scope();
  if (!GetLoopIndexFromCNode(b_cnode, &b_loop_index)) {
    MS_LOG_WITH_NODE(EXCEPTION, b_cnode) << "The operator with fullname_with_scope: " << b_fullname
                                         << " was not included in the set.";
  }
  if (a_loop_index == b_loop_index) {
    return false;
  }
  return true;
}

// 'configured_stra_ops_' includes all operators that are configured sharding strategies.
std::map<OperatorInfoPtr, StrategyPtr, OpsPtrCompare> configured_stra_ops_;
std::set<OperatorInfoPtr> ignore_candidate_;
void InitCostGraph() {
  if (entire_costgraph == nullptr) {
    entire_costgraph = std::make_shared<CostGraph>();
  }
  MS_EXCEPTION_IF_NULL(CostModelContext::GetInstance());
  CostModelContext::GetInstance()->PrintCostModel();
  entire_costgraph->Init();
  configured_stra_ops_.clear();
  ignore_candidate_.clear();
}

void CheckStrategyUsedDevices(const OperatorInfoPtr &operator_info) {
  const auto fully_use_devices = CostModelContext::GetInstance()->fully_use_device();
  if (fully_use_devices) {
    // If configured to fully use devices, then checking for the user-specified strategy
    int64_t used_devices = operator_info->used_devices();
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(0).size();

    // 'used_devices == -1' means that 'used_devices_' is not set
    // 'used_devices == 1' means that ALL-1 strategy, which is valid in auto-parallel
    if (used_devices == -1 || (used_devices != 1 && LongToSize(used_devices) != total_device_num)) {
      MS_LOG_WITH_NODE(EXCEPTION, operator_info->cnode())
        << "In current configuration 'fully_use_devices' = True, "
        << "but the specified strategy uses device: " << used_devices << ", total devices: " << total_device_num
        << ", try to set 'set_algo_parameters(fully_use_devices=False)' "
           "in package 'mindspore.parallel'.";
    }
  }
}

void SetLayoutToOperatorForNewShape(const OperatorInfoPtr &operator_info,
                                    const mindspore::HashMap<std::string, ValuePtr> &attrs) {
  auto cnode = operator_info->cnode();
  std::vector<TensorLayoutBasePtr> in_tensor_layouts_new;
  std::vector<TensorLayoutBasePtr> out_tensor_layouts_new;
  if (ExtractUserConfigLayoutForNewShape(attrs, operator_info->inputs_shape_new(), operator_info->outputs_shape_new(),
                                         &in_tensor_layouts_new, &out_tensor_layouts_new) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Failure:operator " << operator_info->name()
                                       << " extract configured layout failed";
  }

  // Only init operator with tensor_layouts_new, new shape operator currently doesn't support propagate strategy,
  // no need and doesn't support setting cost
  if (operator_info->Init(nullptr, nullptr, in_tensor_layouts_new, out_tensor_layouts_new) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Failure: operator " << operator_info->name() << " SetCostUnderLayout failed";
  }

  operator_info->set_is_new_shape_node(true);
  operator_info->set_config_by_layout(true);
}

void SetLayoutToOperator(const OperatorInfoPtr &operator_info, const mindspore::HashMap<std::string, ValuePtr> &attrs) {
  auto cnode = operator_info->cnode();
  auto is_new_shape_base_node = IsSupportNewShapeBaseNode(cnode);
  if (is_new_shape_base_node) {
    SetLayoutToOperatorForNewShape(operator_info, attrs);
    return;
  }
  std::vector<std::shared_ptr<TensorLayout>> in_tensor_layouts;
  std::vector<std::shared_ptr<TensorLayout>> out_tensor_layouts;
  StrategyPtr in_strategy_ptr = nullptr;
  StrategyPtr out_strategy_ptr = nullptr;
  if (ExtractUserConfigLayout(attrs, operator_info->inputs_shape(), operator_info->outputs_shape(), &in_tensor_layouts,
                              &out_tensor_layouts) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Failure:operator " << operator_info->name()
                                       << " extract configured layout failed";
  }
  Strategies in_strategy;
  (void)std::transform(in_tensor_layouts.begin(), in_tensor_layouts.end(), std::back_inserter(in_strategy),
                       [](const auto &layout) { return layout->get_in_layout_strategy(); });
  MS_LOG(INFO) << "Converted strategies from in_tensor_layouts: " << in_strategy;
  in_strategy_ptr = NewStrategy(0, in_strategy);

  if (OutLayoutFound(attrs)) {
    out_strategy_ptr = operator_info->out_strategy();
  }
  // Set cost for this configured strategy
  if (operator_info->SetCostUnderLayout(in_strategy_ptr, out_strategy_ptr, in_tensor_layouts, out_tensor_layouts) !=
      SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Failure: operator " << operator_info->name() << " SetCostUnderLayout failed";
  }
  (void)configured_stra_ops_.emplace(operator_info, in_strategy_ptr);
  operator_info->set_config_by_layout(true);
}

void SetOutLayoutToOperater(const OperatorInfoPtr &operator_info,
                            const mindspore::HashMap<std::string, ValuePtr> attrs) {
  auto cnode = operator_info->cnode();
  auto is_new_shape_base_node = IsSupportNewShapeBaseNode(cnode);
  if (is_new_shape_base_node) {
    return;
  }
  std::vector<std::shared_ptr<TensorLayout>> in_tensor_layouts;
  std::vector<std::shared_ptr<TensorLayout>> out_tensor_layouts;
  if (ExtractUserConfigLayout(attrs, operator_info->inputs_shape(), operator_info->outputs_shape(), &in_tensor_layouts,
                              &out_tensor_layouts) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Failure:operator " << operator_info->name()
                                       << " extract configured layout failed";
  }
  Strategies out_strategy;
  (void)std::transform(out_tensor_layouts.begin(), out_tensor_layouts.end(), std::back_inserter(out_strategy),
                       [](const auto &layout) { return layout->get_out_layout_strategy(); });
  MS_LOG(INFO) << "Converted strategies from out_tensor_layouts: " << out_strategy;
  StrategyPtr out_strategy_ptr = NewStrategy(0, out_strategy);
  operator_info->set_out_strategy(out_strategy_ptr);
}

void SetOutStrategyToOperator(const OperatorInfoPtr &operator_info, mindspore::HashMap<std::string, ValuePtr> attrs) {
  // In this case, when attrs has out_strategy, the out_strategy will be set to operator
  StrategyPtr strategyPtr;
  if (!OutStrategyFound(attrs)) {
    return;
  }
  strategyPtr = parallel::ExtractStrategy(attrs[OUT_STRATEGY]);
  if (strategyPtr == nullptr) {
    return;
  }
  operator_info->set_out_strategy(strategyPtr);
}

void SetStrategyToOperator(const OperatorInfoPtr &operator_info, const PrimitivePtr &prim,
                           mindspore::HashMap<std::string, ValuePtr> attrs, bool, StrategyMap *stra_map,
                           const std::string &strategy_key_name) {
  // In this case, the configured strategy should be extracted to help setting cost
  StrategyPtr strategyPtr;
  if (StrategyFound(attrs)) {
    strategyPtr = parallel::ExtractStrategy(attrs[IN_STRATEGY]);
  } else {
    strategyPtr = (*stra_map)[strategy_key_name];
  }

  if (strategyPtr == nullptr) {
    return;
  }

  if (prim->name() == RESHAPE) {
    MS_LOG(EXCEPTION) << "Setting strategy for Reshape goes for nothing!";
  }

  // Set cost for this configured strategy
  if (operator_info->SetCostUnderStrategy(strategyPtr) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Failure: operator " << prim->name() << " SetCostUnderStrategy failed";
  }
  CheckStrategyUsedDevices(operator_info);
  operator_info->SetDefaultLayoutInfo();
  (void)configured_stra_ops_.emplace(operator_info, strategyPtr);
}

OperatorInfoPtr SetStrategiesToOperator(const PrimitivePtr &prim, const OperatorInfoPtr &operator_info,
                                        const CNodePtr &cnode, bool is_last_nodes, StrategyMap *stra_map) {
  // key of strategy map
  std::string strategy_key_name;
  auto param_names = NodeParameterName(cnode, -1, 0);
  if (!param_names.empty()) {
    strategy_key_name = prim->name() + "_" + param_names[0].first;
  }
  bool load_strategy_from_ckpt =
    StrategyCheckpoint::GetInstance().LoadCheckPointOn() && stra_map->find(strategy_key_name) != stra_map->end();
  auto attrs = prim->attrs();
  // If the user input layout (mindspore.Layout instance), convert layout to strategy, then set to the operator_info.
  if (OutLayoutFound(attrs)) {
    SetOutLayoutToOperater(operator_info, attrs);
  }
  if (LayoutFound(attrs)) {
    SetLayoutToOperator(operator_info, attrs);
    return operator_info;
  }
  // If the user input strategy (tuple-like), set the strategy to the operator_info.
  if (OutStrategyFound(attrs)) {
    SetOutStrategyToOperator(operator_info, attrs);
  }
  if ((StrategyFound(attrs) && prim->name() != CAST) || load_strategy_from_ckpt) {
    SetStrategyToOperator(operator_info, prim, attrs, is_last_nodes, stra_map, strategy_key_name);
    return operator_info;
  }
  return nullptr;
}

void ApplyApproximationForNode(const OperatorInfoPtr &operator_info) {
  auto approximation = CostModelContext::GetInstance()->dp_algo_enable_approxi();
  if (approximation) {
    operator_info->ApproximateStrategies();
    MS_LOG(INFO) << "Approximated StrategyCost for: " << operator_info->name();
  }
}

void AddOperatorToIgnoreCandidates(const PrimitivePtr &prim, const OperatorInfoPtr &operator_info) {
  if (prim->name() == CAST) {
    // add CAST into ignore_candidate
    (void)ignore_candidate_.insert(operator_info);
  }
}

bool GenerateStrategiesByOperatorInfoPtr(const OperatorInfoPtr &operator_info) {
  Status retGenStra;
  MS_EXCEPTION_IF_NULL(operator_info);
  auto attrs = operator_info->attrs();
  if (AttrFound(attrs, STRATEGY_GEN_MODE) && GetValue<std::string>(attrs[STRATEGY_GEN_MODE]) == kDataParallel) {
    MS_LOG(INFO) << "generating batch parallel strategy...";
    auto prim = std::make_shared<Primitive>(operator_info->name());
    StrategyPtr strategyPtr = parallel::GenerateBatchParallelStrategy(operator_info, prim);
    retGenStra = operator_info->SetCostUnderStrategy(strategyPtr);
    attrs = prim->attrs();
    operator_info->addAttr(IN_STRATEGY, attrs[GEN_STRATEGY]);  // for d-rec
  } else {
    MS_LOG(INFO) << "auto-searching strategy...";
    operator_info->LayoutPropagationBegin();
    retGenStra = operator_info->GenerateStrategies(0);
    operator_info->LayoutPropagationEnd();
  }
  if (retGenStra != SUCCESS) {
    MS_LOG(ERROR) << "Strategy search for Operator " << operator_info->name() << " failed.";
    return false;
  }
  return true;
}

OperatorInfoPtr CreateTheOperatorInfo(const PrimitivePtr &prim, const CNodePtr &cnode, bool is_last_nodes,
                                      StrategyMap *stra_map) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(cnode);
  // Create an OperatorInfo instance
  OperatorInfoPtr operator_info = CreateOperatorInfo(cnode);
  MS_EXCEPTION_IF_NULL(operator_info);
  // Set the parameter information for this OperatorInfo (whether the inputs are parameters or not)
  std::vector<bool> parameter_info = ExtractInputParameterByNode(cnode);
  if (operator_info->set_is_parameter(parameter_info) != SUCCESS) {
    MS_LOG(ERROR) << "Initializing parameter information failed for operator: " << operator_info->name();
    return nullptr;
  }
  // Set the data type for inputs and outputs of this OperatorInfo
  auto inputs_type_length = ExtractInputTypeLengthByNode(cnode);
  auto outputs_type = ExtractOutputTypeByNode(cnode);
  if (ParallelContext::GetInstance()->strategy_search_mode() == kRecursiveProgramming) {
    std::string param_name = ExtractInputParameterNameByNode(cnode);
    if (!param_name.empty()) {
      operator_info->set_involved_param_name(param_name);
    }
  }
  std::vector<size_t> outputs_type_length;
  outputs_type_length.reserve(outputs_type.size());
  (void)std::transform(outputs_type.begin(), outputs_type.end(), std::back_inserter(outputs_type_length),
                       GetLengthOfDataType);
  if (operator_info->SetInputAndOutputTypeLength(inputs_type_length, outputs_type_length) != SUCCESS) {
    MS_LOG(ERROR) << "Setting the lengths of inputs and outputs failed for operator: " << operator_info->name();
    return nullptr;
  }
  if (operator_info->set_outputs_type(outputs_type) != SUCCESS) {
    MS_LOG(ERROR) << "Setting the types of outputs failed for operator: " << operator_info->name();
    return nullptr;
  }

  operator_info->set_auto_parallel(true);

  AddOperatorToIgnoreCandidates(prim, operator_info);

  // If no strategy has been configured for this operator, then candidate strategies are generated for
  // auto-strategy searching; if this primitive is CAST, we ignore the user-specified strategy.
  // if strategy is set to load from checkpoint, it is prefer to load strategy from checkpoint .
  if (ParallelContext::GetInstance()->strategy_search_mode() != kRecursiveProgramming) {
    auto operator_info_with_strategies = SetStrategiesToOperator(prim, operator_info, cnode, is_last_nodes, stra_map);
    if (operator_info_with_strategies != nullptr) {
      return operator_info_with_strategies;
    }
  }
  // Compute split_flag_list_, indicating which input has batch dimension. This is ONLY used for preparation for
  // BatchParallelInfo operator
  operator_info->ComputeBatchSplitFlagList();
  if ((ParallelContext::GetInstance()->strategy_search_mode() != kRecursiveProgramming)) {
    (void)GenerateStrategiesByOperatorInfoPtr(operator_info);
  }

  bool use_sp_and_dataset = ((ParallelContext::GetInstance()->strategy_search_mode() == kShardingPropagation) ||
                             (ParallelContext::GetInstance()->sharding_propagation())) &&
                            (operator_info->name().find(VIRTUAL_DATA_SET_INFO) != std::string::npos);
  if (use_sp_and_dataset) {
    const auto &swc_vec = operator_info->GetStrategyCost();
    if (swc_vec.empty()) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "No available strategy for: " << operator_info->name();
    }
    MS_EXCEPTION_IF_NULL(swc_vec[0]->strategy_ptr);
    (void)configured_stra_ops_.emplace(operator_info, swc_vec[0]->strategy_ptr);
  }
  // If 'approximation' is enabled, the 'strategy_cost' of each operator is approximated
  ApplyApproximationForNode(operator_info);
  return operator_info;
}

bool IsFindWrong(const OperatorInfoPtr &current_op_ptr, const std::string &prim_name) {
  if (current_op_ptr->name().find(STAND_ALONE_INFO) != std::string::npos) {
    // StandAlone can be different with prim
    return false;
  }
  bool is_find_wrong = (current_op_ptr->name().find(VIRTUAL_DATA_SET_INFO) == std::string::npos) &&
                       (current_op_ptr->name().find(BATCH_PARALLEL) == std::string::npos) &&
                       (current_op_ptr->name().find(prim_name + "Info") == std::string::npos);
  if (prim_name == GATHERV2) {
    is_find_wrong = is_find_wrong && (current_op_ptr->name().find(prim_name + "PInfo") == std::string::npos);
  }
  return is_find_wrong;
}

void AddUsersUniqueIdWhenSharingParameter(
  const std::pair<std::string, std::pair<AnfNodePtr, AnfNodeIndexSet>> &parameter_users_info) {
  auto users_set = parameter_users_info.second.second;
  if (users_set.size() > 1) {
    MS_LOG(INFO) << "Parameter " << parameter_users_info.first << " has " << users_set.size() << " users.";
    std::vector<std::string> param_users_uniqueid;
    for (const auto &user : users_set) {
      MS_LOG(INFO) << "with ID: " << user.first->UniqueId() << " and name: " << user.first->UniqueName();
      param_users_uniqueid.push_back(user.first->UniqueId());
    }
    entire_costgraph->add_param_users_uniqueid(param_users_uniqueid);
  }
}

void AddParamUsersForRec(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    if (node->isa<Parameter>()) {
      ParameterUsersInfo parameter_users_info = FindParameterUsers(node, IsParallelCareNode, all_nodes);
      AddUsersUniqueIdWhenSharingParameter(parameter_users_info);
    }
  }
}

// Using CNode's UniqueIds to construct nodes
Status ConstructCostGraphNodesByUniqueId(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &) {
  MS_LOG(INFO) << "Constructing nodes for cost graph begins.";
  // The map from CNode's UniqueId to its operatorInfo
  std::map<std::string, OperatorInfoPtr> from_cnode_to_info;
  // The operator_infos in a loop
  std::vector<OperatorInfoPtr> operators_in_forloop;
  // Key: i-th loop; Value: index of 'operators_in_forloop'
  std::map<size_t, size_t> loop_to_ops;
  // extract strategy from checkpoint for multi-train
  StrategyMap stra_map;
  if (StrategyCheckpoint::GetInstance().LoadCheckPointOn()) {
    if (StrategyCheckpoint::GetInstance().Load(&stra_map) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Load strategy checkpoint failed. Please check if the file exists and the file permission.";
    }
  }

  for (auto &node : all_nodes) {
    // NOTE: we only care about splittable Primitive operators
    auto cnode = node->cast<CNodePtr>();
    bool bool_result = (cnode == nullptr) || (!IsValueNode<Primitive>(cnode->input(0)));
    if (bool_result) {
      continue;
    }
    auto prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    if (!IsAutoParallelCareNode(cnode)) {
      // Needed by rec_parser
      if (ParallelContext::GetInstance()->strategy_search_mode() == kRecursiveProgramming) {
        auto prev_cnode = GetInternalOperatorInfo(cnode, prim_anf_node);
        if (prev_cnode != nullptr) {
          entire_costgraph->add_tuple_getitem(std::make_pair(cnode->UniqueId(), prev_cnode->UniqueId()));
        }
      }
      continue;
    }
    auto prim = GetValueNode<PrimitivePtr>(prim_anf_node);
    MS_EXCEPTION_IF_NULL(prim);

    auto search_cnode = from_cnode_to_info.find(cnode->UniqueId() + prim->name());
    if (search_cnode == from_cnode_to_info.cend()) {
      size_t loop_index = 0;
      bool is_in_loop = GetLoopIndexFromCNode(cnode, &loop_index);
      const auto single_loop = CostModelContext::GetInstance()->dp_algo_single_loop();
      if (single_loop && is_in_loop && (loop_to_ops[loop_index] < operators_in_forloop.size())) {
        const auto &current_op_ptr = operators_in_forloop[loop_to_ops[loop_index]];
        if (IsFindWrong(current_op_ptr, prim->name())) {
          MS_LOG_WITH_NODE(EXCEPTION, cnode)
            << "The OperatorInfo: " << current_op_ptr->name() << " does not match the Prim: " << prim->name()
            << ". The fullname_with_scope: " << cnode->fullname_with_scope();
        }
        loop_to_ops[loop_index]++;
        cnode->set_user_data<OperatorInfo>(current_op_ptr);
        MS_LOG(INFO) << "The CNode with UniqueId: " << cnode->UniqueId()
                     << " and UniqueIdThroughCopy: " << cnode->UniqueIdThroughCopy()
                     << ", CNode fullname_with_scope: " << cnode->fullname_with_scope()
                     << " is set OperatorInfo: " << current_op_ptr->name() << ", Primitive: " << prim->name();
        (void)from_cnode_to_info.emplace(std::make_pair(cnode->UniqueId() + prim->name(), current_op_ptr));
        continue;
      }
      bool is_last_nodes = IsPrimitiveCNode(cnode, prim::kPrimVirtualOutput);
      auto operator_info = CreateTheOperatorInfo(prim, cnode, is_last_nodes, &stra_map);
      if (operator_info == nullptr) {
        return FAILED;
      }
      if (ParallelContext::GetInstance()->strategy_search_mode() == kRecursiveProgramming) {
        operator_info->set_type(prim->name());
        operator_info->set_last_node_flag(is_last_nodes);
        std::vector<std::string> inputs_tensor_name = ExtractInputsTensorName(cnode, all_nodes);
        entire_costgraph->add_inputs_tensor_name(inputs_tensor_name);
      }

      entire_costgraph->AddOperator(operator_info);
      cnode->set_user_data<OperatorInfo>(operator_info);
      MS_LOG(INFO) << "The CNode with UniqueId: " << cnode->UniqueId()
                   << " and UniqueIdThroughCopy: " << cnode->UniqueIdThroughCopy()
                   << ", CNode fullname_with_scope: " << cnode->fullname_with_scope()
                   << " is set OperatorInfo: " << operator_info->name() << ", Primitive: " << prim->name();
      (void)from_cnode_to_info.emplace(std::make_pair(cnode->UniqueId() + prim->name(), operator_info));
      if (single_loop && is_in_loop) {
        operators_in_forloop.push_back(operator_info);
        (void)ops_in_a_loop_.insert(operator_info->name());
        loop_to_ops[loop_index]++;
      }
    } else {
      // Two CNODEs' UniqueIds should not be equal
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "The CNode with UniqueId: " << cnode->UniqueId()
                                         << " and UniqueIdThroughCopy: " << cnode->UniqueIdThroughCopy()
                                         << " is set OperatorInfo: " << search_cnode->second->name()
                                         << ", Primitive: " << prim->name();
    }
  }

  MS_LOG(INFO) << "Constructing nodes for cost graph ends.";
  // Needed by rec_parser 2
  AddParamUsersForRec(all_nodes);

  return SUCCESS;
}

void SetOperatorToCNode(const OperatorInfoPtr &current_op_ptr, const PrimitivePtr &prim, const CNodePtr &cnode) {
  if (current_op_ptr == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Find " << prim->name() << " from CostGraph failed.";
  } else {
    if (IsFindWrong(current_op_ptr, prim->name())) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "The OperatorInfo: " << current_op_ptr->name()
                                         << " does not match the Prim: " << prim->name();
    }

    // Needed by rec_parser
    ModifyInputsTensorNameListIfOperatorInfoCreated(current_op_ptr->name(), cnode->UniqueId());

    cnode->set_user_data<OperatorInfo>(current_op_ptr);
    current_op_ptr->set_cnode(cnode);
    MS_LOG(INFO) << "The CNode with UniqueId: " << cnode->UniqueId()
                 << " and UniqueIdThroughCopy: " << cnode->UniqueIdThroughCopy()
                 << ", CNode fullname_with_scope: " << cnode->fullname_with_scope()
                 << " is set OperatorInfo: " << current_op_ptr->name() << ", Primitive: " << prim->name();
  }
}

bool Need_Create_New_Op(const std::map<std::string, OperatorInfoPtr>::iterator &search_cnode, const CNodePtr &cnode,
                        const PrimitivePtr &prim, bool op_in_map) {
  bool use_sp = ParallelContext::GetInstance()->strategy_search_mode() == kShardingPropagation;
  if (!use_sp) {
    return !op_in_map;
  }
  bool is_same_graph = false;
  if (op_in_map) {
    auto &op_created = search_cnode->second;
    if (op_created->cnode()->func_graph() != nullptr && cnode->func_graph() != nullptr) {
      is_same_graph = op_created->cnode()->func_graph() == cnode->func_graph();
    }
  }
  return !op_in_map || is_same_graph;
}

// Using CNode's UniqueIdThroughCopys to construct nodes
Status ConstructCostGraphNodesByUniqueIdTC(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &) {
  MS_LOG(INFO) << "Constructing nodes for cost graph begins.";
  // The map from CNode's UniqueIdThroughCopy to its operatorInfo
  std::map<std::string, OperatorInfoPtr> from_cnode_to_info;
  // The operator_infos in a loop
  std::vector<OperatorInfoPtr> operators_in_forloop;
  // Key: i-th loop; Value: index of 'operators_in_forloop'
  std::map<size_t, size_t> loop_to_ops;
  // extract strategy from checkpoint for multi-train
  StrategyMap stra_map;
  if (StrategyCheckpoint::GetInstance().LoadCheckPointOn() &&
      StrategyCheckpoint::GetInstance().Load(&stra_map) != SUCCESS) {
    MS_LOG(WARNING) << "Load strategy checkpoint failed. Please check if the file exists and the file permission.";
    return FAILED;
  }
  for (auto &node : all_nodes) {
    // NOTE: we only care about splittable Primitive operators
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || (!IsValueNode<Primitive>(cnode->input(0)))) {
      continue;
    }
    auto prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    if (!IsAutoParallelCareNode(cnode)) {
      // Needed by rec_parser
      if (ParallelContext::GetInstance()->strategy_search_mode() == kRecursiveProgramming) {
        auto prev_cnode = GetInternalOperatorInfo(cnode, prim_anf_node);
        if (prev_cnode != nullptr) {
          entire_costgraph->add_tuple_getitem(std::make_pair(cnode->UniqueId(), prev_cnode->UniqueId()));
        }
      }
      continue;
    }
    auto prim = GetValueNode<PrimitivePtr>(prim_anf_node);
    MS_EXCEPTION_IF_NULL(prim);
    auto search_cnode = from_cnode_to_info.find(cnode->UniqueIdThroughCopy() + prim->name());
    bool op_in_map = search_cnode != from_cnode_to_info.cend();

    if (Need_Create_New_Op(search_cnode, cnode, prim, op_in_map)) {
      size_t loop_index = 0;
      bool is_in_loop = GetLoopIndexFromCNode(cnode, &loop_index);
      const auto single_loop = CostModelContext::GetInstance()->dp_algo_single_loop();
      bool is_op_created = single_loop && is_in_loop && (loop_to_ops[loop_index] < operators_in_forloop.size());
      if (is_op_created) {
        const auto &current_op_ptr = operators_in_forloop[loop_to_ops[loop_index]];
        if (IsFindWrong(current_op_ptr, prim->name())) {
          MS_LOG_WITH_NODE(EXCEPTION, cnode)
            << "The OperatorInfo: " << current_op_ptr->name() << " does not match the Prim: " << prim->name()
            << ". The fullname_with_scope: " << cnode->fullname_with_scope();
        }
        loop_to_ops[loop_index]++;
        cnode->set_user_data<OperatorInfo>(current_op_ptr);
        MS_LOG(INFO) << "The CNode with UniqueId: " << cnode->UniqueId()
                     << " and UniqueIdThroughCopy: " << cnode->UniqueIdThroughCopy()
                     << ", CNode fullname_with_scope: " << cnode->fullname_with_scope()
                     << " is set OperatorInfo: " << current_op_ptr->name() << ", Primitive: " << prim->name();
        (void)from_cnode_to_info.emplace(std::make_pair(cnode->UniqueIdThroughCopy() + prim->name(), current_op_ptr));
        continue;
      }
      // In this case, the corresponding OperatorInfo is not created, create the new one.
      bool is_last_nodes = IsPrimitiveCNode(cnode, prim::kPrimVirtualOutput);
      auto operator_info = CreateTheOperatorInfo(prim, cnode, is_last_nodes, &stra_map);
      MS_EXCEPTION_IF_NULL(operator_info);

      if (ParallelContext::GetInstance()->strategy_search_mode() == kRecursiveProgramming) {
        operator_info->set_type(prim->name());
        operator_info->set_last_node_flag(is_last_nodes);
        std::vector<std::string> inputs_tensor_name = ExtractInputsTensorName(cnode, all_nodes);
        entire_costgraph->add_inputs_tensor_name(inputs_tensor_name);
      }

      entire_costgraph->AddOperator(operator_info);
      cnode->set_user_data<OperatorInfo>(operator_info);
      MS_LOG(INFO) << "The CNode with UniqueId: " << cnode->UniqueId()
                   << " and UniqueIdThroughCopy: " << cnode->UniqueIdThroughCopy()
                   << ", CNode fullname_with_scope: " << cnode->fullname_with_scope()
                   << " is set OperatorInfo: " << operator_info->name() << ", Primitive: " << prim->name();
      (void)from_cnode_to_info.emplace(std::make_pair(cnode->UniqueIdThroughCopy() + prim->name(), operator_info));
      if (single_loop && is_in_loop) {
        operators_in_forloop.push_back(operator_info);
        (void)ops_in_a_loop_.insert(operator_info->name());
        loop_to_ops[loop_index]++;
      }
    } else {
      SetOperatorToCNode(search_cnode->second, prim, cnode);
    }
  }

  MS_LOG(INFO) << "Constructing nodes for cost graph ends.";
  // Needed by rec_parser 2
  AddParamUsersForRec(all_nodes);

  return SUCCESS;
}

void PreProcessPreCastForSP(const OperatorInfoPtr &prev_op_info, const OperatorInfoPtr &node_op_info,
                            const CNodePtr &cnode, const EdgePtr edge_ptr, size_t input_index) {
  if (IsPrimitiveCNode(cnode, prim::kPrimMatMul) && input_index == INDEX_TWO) {
    prev_op_info->set_repeated_num_in_dev_matrix_right(false);
    prev_op_info->ClearStrategyCost();
    (void)prev_op_info->GenerateStrategies(0);
  }
  if ((configured_stra_ops_.find(node_op_info) != configured_stra_ops_.end())) {
    const auto next_op_stra = configured_stra_ops_[node_op_info];
    if (edge_ptr->InitEdgeCost() != SUCCESS) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Edge cost initialization failed";
    }
    const auto &candidate_swc = edge_ptr->GetPrevOpSwcByNextOpStrategyWithMiniComm(next_op_stra);
    if (candidate_swc == nullptr) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "No available candidate swc for: " << prev_op_info->name();
    }
    const auto cast_stra = candidate_swc->strategy_ptr;
    if (cast_stra == nullptr) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "No available strategy for: " << prev_op_info->name();
    }
    prev_op_info->ClearStrategyCost();
    if (prev_op_info->SetCostUnderStrategy(cast_stra) != SUCCESS) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Failure: operator " << prev_op_info->name()
                                         << " SetCostUnderStrategy failed";
    }
    if (edge_ptr->InitEdgeCost() != SUCCESS) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Edge cost re-initialization failed.";
    }
    MS_LOG(INFO) << "Set strategy for: " << prev_op_info->name() << " under the strategy of: " << node_op_info->name();
    (void)configured_stra_ops_.emplace(prev_op_info, cast_stra);
  }
}

void CreateEdgeBetweenTwoOps(const OperatorInfoPtr &prev_op_info, const OperatorInfoPtr &node_op_info,
                             const CNodePtr &cnode, const CNodePtr &prev_cnode, const PrimitivePtr &prim,
                             const PrimitivePtr &prev_prim, size_t output_index, size_t input_index,
                             size_t *edge_count) {
  MS_EXCEPTION_IF_NULL(prev_op_info);
  MS_EXCEPTION_IF_NULL(node_op_info);
  MS_EXCEPTION_IF_NULL(prim);

  std::string edge_name = prev_op_info->name() + OPERATOR_TO_OPERATOR_CONNECTOR + node_op_info->name();
  // If the edge between these two operators already has been added, then the edge will not be added again.
  if (entire_costgraph->IsEdgeInCostGraph(edge_name, output_index, input_index - 1)) {
    return;
  }
  EdgePtr edge_ptr;
  MS_LOG(INFO) << "Creating edge: " << edge_name;
  if (IsOperatorsInTwoSeparateLoops(prev_cnode, cnode)) {
    MS_LOG(INFO) << "prev_cnode_fullname: " << prev_cnode->fullname_with_scope()
                 << ", cnode_fullname: " << cnode->fullname_with_scope();
    MS_LOG(INFO) << "The two operators in two separate for-loops, thus skip the edge.";
    return;
  }
  const auto stra_follow = CostModelContext::GetInstance()->elementwise_stra_follow();
  MS_EXCEPTION_IF_NULL(prev_prim);
  bool follow_strategy = (prim->name() == RESHAPE) || (prev_prim->name() == RESHAPE) ||
                         (stra_follow && IsElementWiseOperator(prev_prim->name()));
  if (follow_strategy) {
    // Redistribution in not allowed on the edge.
    // Elementwise operators have the same strategy as their previous operators.
    edge_ptr =
      std::make_shared<Edge>(edge_name, prev_op_info, node_op_info, output_index, input_index - 1, false, true);
  } else {
    edge_ptr = std::make_shared<Edge>(edge_name, prev_op_info, node_op_info, output_index, input_index - 1, false);
  }
  bool use_sp = (ParallelContext::GetInstance()->strategy_search_mode() == kShardingPropagation) ||
                (ParallelContext::GetInstance()->sharding_propagation());
  // Init costs for this edge
  if (ParallelContext::GetInstance()->strategy_search_mode() != kRecursiveProgramming) {
    if (!use_sp && edge_ptr->InitEdgeCost() != SUCCESS) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Edge cost initialization failed";
    }
  }
  node_op_info->AddPrevEdge(edge_ptr);
  prev_op_info->AddSuccEdge(edge_ptr);
  entire_costgraph->AddEdge(prev_op_info, node_op_info, edge_ptr);
  if (use_sp && prev_prim->name() == CAST) {
    PreProcessPreCastForSP(prev_op_info, node_op_info, cnode, edge_ptr, input_index);
  }
  MS_LOG(INFO) << "Successfully adding the edge between " << prev_op_info->name() << " and " << node_op_info->name();
  (*edge_count)++;
}

void ApplyApproximationForGraphs() {
  // If 'approximation' is enabled, the edges need to be checked have effective costs.
  auto approximation = CostModelContext::GetInstance()->dp_algo_enable_approxi();
  if (approximation) {
    entire_costgraph->CheckApproximateCostGraphEdges();
  }
}

static void CreateEdgeAccrossMakeList(const CNodePtr &cnode, const PrimitivePtr &prim,
                                      const OperatorInfoPtr &node_op_info, CNodePtr *prev_cnode,
                                      ValueNodePtr *prev_prim_anf_node, PrimitivePtr *prev_prim, size_t *edge_count) {
  MS_LOG(INFO) << "Creating edges across the 'make_list' operator.";
  MS_EXCEPTION_IF_NULL(prev_cnode);
  const auto &sub_inputs = (*prev_cnode)->inputs();
  for (size_t j = 1; j < sub_inputs.size(); ++j) {
    *prev_cnode = sub_inputs[j]->cast<CNodePtr>();
    bool bool_result_list = (*prev_cnode == nullptr) || !IsValueNode<Primitive>((*prev_cnode)->input(0)) ||
                            !IsAutoParallelCareNode(*prev_cnode);
    if (bool_result_list) {
      continue;
    }
    *prev_prim_anf_node = (*prev_cnode)->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(*prev_prim_anf_node);
    *prev_prim = (*prev_prim_anf_node)->value()->cast<PrimitivePtr>();
    auto prev_op_info = (*prev_cnode)->user_data<OperatorInfo>();
    CreateEdgeBetweenTwoOps(prev_op_info, node_op_info, cnode, *prev_cnode, prim, *prev_prim, 0, j, edge_count);
  }
}

namespace {
bool HasOperatorInfo(const CNodePtr &cnode) {
  return IsAutoParallelCareNode(cnode) || IsValueNode<FuncGraph>(cnode->input(0)) || IsSomePrimitive(cnode, GENERATOR);
}
}  // namespace

static void ConstructCNodeCostGraphEdges(const mindspore::CNodePtr &cnode, const std::vector<AnfNodePtr> &all_nodes) {
  auto &inputs = cnode->inputs();
  ValueNodePtr prim_anf_node = inputs[0]->cast<ValueNodePtr>();
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  MS_EXCEPTION_IF_NULL(prim);
  size_t edge_count = 0;
  auto node_op_info = cnode->user_data<OperatorInfo>();
  MS_EXCEPTION_IF_NULL(node_op_info);

  for (size_t i = 1; i < inputs.size(); ++i) {
    AnfNodePtr prev_node = inputs[i];
    if (inputs[i]->isa<Parameter>()) {
      prev_node = FindRealInputByFormalParameter(cnode, inputs[i], all_nodes);
      if (prev_node->UniqueId() == inputs[i]->UniqueId()) {
        continue;
      }
    }
    auto prev_cnode = prev_node->cast<CNodePtr>();
    PrimitivePtr prev_prim;
    ValueNodePtr prev_prim_anf_node;
    bool is_cross = CrossInterNode(&prev_cnode, &prev_prim_anf_node, &prev_prim);
    if (is_cross) {
      continue;
    }
    size_t output_index = 0;
    bool is_before_tuple_get_item = false;

    while (IsCarePrevCNode(prev_cnode, prev_prim)) {
      if (IsValueNode<FuncGraph>(prev_cnode->input(0))) {
        auto graph = GetValueNode<FuncGraphPtr>(prev_cnode->input(0));
        MS_EXCEPTION_IF_NULL(graph);
        auto output = graph->output();
        MS_EXCEPTION_IF_NULL(output);
        prev_cnode = output->cast<CNodePtr>();
        (void)CrossInterNode(&prev_cnode, &prev_prim_anf_node, &prev_prim);
        continue;
      } else if (IsAutoParallelCareNode(prev_cnode)) {
        auto prev_op_info = prev_cnode->user_data<OperatorInfo>();
        CreateEdgeBetweenTwoOps(prev_op_info, node_op_info, cnode, prev_cnode, prim, prev_prim, output_index, i,
                                &edge_count);
        break;
      }

      MS_EXCEPTION_IF_NULL(prev_prim);
      if (prev_prim->name() == prim::kPrimTupleGetItem->name()) {
        // In this case, 'prev_anf_node' is 'tuple_getitem', the actual precursor node is node before
        // this 'tuple_getitem'
        output_index = LongToSize(GetValue<int64_t>(GetValueNode(prev_cnode->input(2))));
        prev_cnode = prev_cnode->input(1)->cast<CNodePtr>();
        is_cross = CrossInterNode(&prev_cnode, &prev_prim_anf_node, &prev_prim);
        if (is_cross) {
          break;
        }
        bool has_op_info = HasOperatorInfo(prev_cnode);
        if (!has_op_info) {
          MS_LOG_WITH_NODE(EXCEPTION, prev_node) << "Did not create OperatorInfo for : " << prev_prim->name();
        }
        is_before_tuple_get_item = true;
      } else if (prev_prim->name() == kMakeTupleOpName) {
        if (!is_before_tuple_get_item) {
          CreateEdgeAccrossMakeList(cnode, prim, node_op_info, &prev_cnode, &prev_prim_anf_node, &prev_prim,
                                    &edge_count);
          break;
        }
        prev_cnode = prev_cnode->input(output_index + 1)->cast<CNodePtr>();
        output_index = 0;
        is_cross = CrossInterNode(&prev_cnode, &prev_prim_anf_node, &prev_prim);
        if (is_cross) {
          break;
        }
        is_before_tuple_get_item = false;
      } else if (prev_prim->name() == kMakeListOpName) {
        CreateEdgeAccrossMakeList(cnode, prim, node_op_info, &prev_cnode, &prev_prim_anf_node, &prev_prim, &edge_count);
        break;
      } else if (prev_prim->name() == kDependOpName || prev_prim->name() == kLoadOpName) {
        // In this case, 'prev_anf_node' is 'depend', the actual precursor node is node before
        // this 'depend'
        prev_cnode = prev_cnode->input(1)->cast<CNodePtr>();
        is_cross = CrossInterNode(&prev_cnode, &prev_prim_anf_node, &prev_prim);
        if (is_cross) {
          break;
        }
        is_before_tuple_get_item = true;
      }
    }
  }
  MS_LOG(INFO) << "Successfully created " << edge_count << " edges for: " << node_op_info->name();
}

void ConstructCostGraphEdges(const std::vector<AnfNodePtr> &all_nodes) {
  // Step 2
  MS_LOG(INFO) << "Constructing edges for cost graph begins.";
  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    if (!IsAutoParallelCareNode(cnode)) {
      continue;
    }
    ConstructCNodeCostGraphEdges(cnode, all_nodes);
  }
  ApplyApproximationForGraphs();

  MS_LOG(INFO) << "Constructing edges for cost graph ends.";
}

void ApplyApproximationForParaNode(const OperatorInfoPtr &target_op_info) {
  // If 'approximation' is enabled, the edges need to be checked have effective costs.
  auto approximation = CostModelContext::GetInstance()->dp_algo_enable_approxi();
  if (approximation) {
    target_op_info->ExactStrategiesAndRelatedEdges();
  }
}

std::pair<OperatorInfoPtr, bool> CreateIdentityOp(const std::string &parameter_name,
                                                  const AnfNodePtr &target_parameter) {
  // Here, it is sure that this Parameter (RefKey) is being used by multiple Operators.
  OperatorInfoPtr tmp_identity_ptr;
  bool new_identity = false;
  auto returned_identity = entire_costgraph->FindTmpIdentityByParameterName(parameter_name);
  if (returned_identity != nullptr) {
    // In this case, the TmpIdentityInfo instance has already been created
    new_identity = false;
    tmp_identity_ptr = returned_identity;
  } else {
    // In the case, the TmpIdentityInfo instance has NOT been created. Thus, a new one is created.
    new_identity = true;
    // 1) extract input shape from this Parameter
    MS_EXCEPTION_IF_NULL(target_parameter);
    AbstractBasePtr abstract = target_parameter->abstract();
    if (abstract == nullptr) {
      MS_LOG_WITH_NODE(EXCEPTION, target_parameter) << "Failure: abstract is nullptr";
    }
    auto input_shape = dyn_cast<abstract::Shape>(abstract->GetShapeTrack());
    if (input_shape == nullptr) {
      MS_LOG_WITH_NODE(EXCEPTION, target_parameter) << "Failure: input_shape is nullptr";
    }
    Shape shape = input_shape->shape();
    Shapes inputs_shape = {shape};
    Shapes outputs_shape = {shape};
    // 2) init the attr
    mindspore::HashMap<std::string, ValuePtr> attr = {};

    // Create the TmpIdentity instance
    tmp_identity_ptr = std::make_shared<TmpIdentityInfo>(inputs_shape, outputs_shape, attr);
    tmp_identity_ptr->set_name(tmp_identity_ptr->name() + std::to_string(TOTAL_OPS));
    TOTAL_OPS++;
    tmp_identity_ptr->set_refkey_parameter_name(parameter_name);
    // Set the parameter and type lengths for inputs and outputs
    std::vector<bool> is_parameter;
    auto casted_target_parameter = target_parameter->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(casted_target_parameter);
    is_parameter.push_back(ParameterRequireGrad(casted_target_parameter));
    if (tmp_identity_ptr->set_is_parameter(is_parameter) != SUCCESS) {
      MS_LOG_WITH_NODE(EXCEPTION, target_parameter) << "Setting parameter for TmpIdentityInfo failed";
    }
    auto node_type = target_parameter->Type();
    if (node_type->isa<mindspore::TensorType>()) {
      auto input_element_type = node_type->cast<mindspore::TensorTypePtr>()->element();
      std::vector<size_t> type_length = {GetLengthOfDataType(input_element_type)};
      if (tmp_identity_ptr->SetInputAndOutputTypeLength(type_length, type_length) != SUCCESS) {
        MS_LOG_WITH_NODE(EXCEPTION, target_parameter)
          << "Setting input and output type length for TmpIdentityInfo failed";
      }
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, target_parameter) << "Unknown type: " << node_type->type_name();
    }

    // Generate strategies for this TmpIdentityInfo instance;
    if (tmp_identity_ptr->GenerateStrategies(0) != SUCCESS) {
      MS_LOG_WITH_NODE(EXCEPTION, target_parameter)
        << "Strategy search for Operator failed : " << tmp_identity_ptr->name();
    }
  }
  return std::make_pair(tmp_identity_ptr, new_identity);
}

void AugmentCostGraph(const std::vector<AnfNodePtr> &all_nodes) {
  // Step 3
  for (auto &node : all_nodes) {
    ParameterUsersInfo parameter_users_info = FindParameterUsers(node, IsAutoParallelCareNode, all_nodes);
    auto parameter_name = parameter_users_info.first;
    auto target_parameter = parameter_users_info.second.first;
    auto target_set = parameter_users_info.second.second;
    if (target_set.size() <= 1) {
      continue;
    }

    // Rule out the case when a Parameter being used by a Operator, but the Operator appears in multiple CNODEs
    std::set<std::string> target_without_duplicate;
    for (auto &target : target_set) {
      auto target_cnode = target.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(target_cnode);
      // Eliminate the ops without cost.
      if (IsSomePrimitive(target_cnode, SEND)) {
        continue;
      }
      auto input_index = target.second;
      auto target_cnode_info = target_cnode->user_data<OperatorInfo>();
      MS_EXCEPTION_IF_NULL(target_cnode_info);
      (void)target_without_duplicate.insert(std::to_string(input_index) + target_cnode_info->name());
    }
    if (target_without_duplicate.size() <= 1 || parameter_name.empty()) {
      continue;
    }

    auto pair = CreateIdentityOp(parameter_name, target_parameter);
    OperatorInfoPtr tmp_identity_ptr = pair.first;
    bool new_identity = pair.second;
    // A flag recording whether new edges have been created or not
    bool add_identity_edge = false;

    // Create edges between this TmpIdentityInfo instance and subsequent Operator instances
    for (auto &target : target_set) {
      auto target_cnode = target.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(target_cnode);
      auto input_index = target.second;
      auto target_op_info = target_cnode->user_data<OperatorInfo>();
      MS_EXCEPTION_IF_NULL(target_op_info);
      if (!target_op_info->repeated_num_in_dev_matrix_right() && tmp_identity_ptr->repeated_num_in_dev_matrix_right()) {
        tmp_identity_ptr->set_repeated_num_in_dev_matrix_right(false);
        tmp_identity_ptr->ClearStrategyCost();
        (void)tmp_identity_ptr->GenerateStrategies(0);
      }

      std::string edge_name = std::string(IDENTITY_INFO) + OPERATOR_TO_OPERATOR_CONNECTOR + target_op_info->name();
      // If the edge between these two operators already has been added, then the edge will not be added again.
      if (entire_costgraph->IsEdgeInCostGraph(edge_name, 0, LongToSize(input_index - 1))) {
        continue;
      }
      std::shared_ptr<Edge> edge_ptr =
        std::make_shared<Edge>(edge_name, tmp_identity_ptr, target_op_info, 0, input_index - 1, false, true);
      ApplyApproximationForParaNode(target_op_info);

      bool use_sp = (ParallelContext::GetInstance()->strategy_search_mode() == kShardingPropagation) ||
                    (ParallelContext::GetInstance()->sharding_propagation());
      if (!use_sp && edge_ptr->InitEdgeCost() != SUCCESS) {
        MS_LOG_WITH_NODE(EXCEPTION, node) << "Edge cost initialization failed";
      }
      target_op_info->AddPrevEdge(edge_ptr);
      tmp_identity_ptr->AddSuccEdge(edge_ptr);
      entire_costgraph->AddEdge(tmp_identity_ptr, target_op_info, edge_ptr);
      MS_LOG(INFO) << "Successfully adding the edge between " << tmp_identity_ptr->name() << " and "
                   << target_op_info->name();
      add_identity_edge = true;
    }
    if (new_identity && add_identity_edge) {
      // Add the TmpIdentityInfo to CostGraph if BOTH two conditions are satisfied
      entire_costgraph->AddOperator(tmp_identity_ptr);
    }
  }
}

void ReshapeCostCompute(const std::vector<AnfNodePtr> &all_nodes) {
  mindspore::HashSet<std::string> op_cache;
  for (const auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (!FindReshape(cnode, &op_cache)) {
      continue;
    }
    MS_ASSERT(cnode->size() == 3);
    // get previous node's strategy_cost_
    auto pre_node = cnode->input(1);
    if (IsPrimitiveCNode(pre_node, prim::kPrimLoad)) {
      pre_node = pre_node->cast<CNodePtr>()->input(1);
    }
    int64_t out_index = 0;
    OperatorInfoPtr pre_operator_info;
    std::vector<std::shared_ptr<StrategyWithCost>> pre_stra_costs;
    auto operator_info = cnode->user_data<OperatorInfo>();
    bool is_prev_param = false;
    if (!FindReshapePreNodeStraCosts(pre_node, &pre_operator_info, &is_prev_param, &out_index, 0)) {
      MS_LOG(WARNING) << "FindReshapePreNodeStraCosts for reshape failed";
      continue;
    }
    // 如果是双递归的话枚举reshape和前向算子的策略
    if (ParallelContext::GetInstance()->strategy_search_mode() == kRecursiveProgramming) {
      ConstructCNodeCostGraphEdges(cnode, all_nodes);
    }
    if (is_prev_param) {
      auto reshape_info1 = std::dynamic_pointer_cast<ReshapeInfo>(operator_info);
      reshape_info1->SetCostForReshapeWithParameter();
      pre_operator_info = reshape_info1;
      pre_stra_costs = reshape_info1->GetStrategyCost();
    } else {
      pre_stra_costs = pre_operator_info->GetStrategyCost();
    }
    // get next node's strategy_cost_
    std::vector<std::pair<OperatorInfoPtr, int64_t>> next_ops_index;
    bool is_next_reshape = false;
    std::vector<std::pair<std::vector<std::shared_ptr<StrategyWithCost>>, int64_t>> next_costs_index;
    (void)FindReshapeNextNodeStraCosts(cnode, &next_ops_index, &is_next_reshape, 0);
    if (next_ops_index.empty()) {
      MS_LOG(INFO) << "FindReshapeNextNodeStraCosts for reshape failed";
    }
    // set input_layout and output_layout for reshape.
    // init reshape and set cost for each input_layout and output_layout.
    auto reshape_info = std::dynamic_pointer_cast<ReshapeInfo>(operator_info);
    reshape_info->set_pre_operator_name(pre_operator_info->name());
    reshape_info->set_pre_operator_index(out_index);
    if (!next_ops_index.empty()) {
      for (auto &op_index : next_ops_index) {
        // 如果是双递归的话枚举reshape的后向算子的策略
        if (ParallelContext::GetInstance()->strategy_search_mode() == kRecursiveProgramming) {
          ConstructCNodeCostGraphEdges(op_index.first->cnode(), all_nodes);
        }
        auto op_cost = op_index.first->GetStrategyCost();
        (void)next_costs_index.emplace_back(std::make_pair(op_cost, op_index.second));
      }
      reshape_info->set_next_operator_name(next_ops_index[0].first->name());
      reshape_info->set_next_operator_index(next_ops_index[0].second);
    }
    if (ParallelContext::GetInstance()->strategy_search_mode() != kRecursiveProgramming) {
      if (reshape_info->GenerateStrategyCosts(pre_stra_costs, next_costs_index, out_index, is_prev_param,
                                              is_next_reshape) != SUCCESS) {
        MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Reshape generate strategy costs failed";
      }
    }
  }
}

Status IgnoreOperatorsInCostGraph() {
  for (auto op = ignore_candidate_.cbegin(); op != ignore_candidate_.cend(); ++op) {
    auto cnodes = (*op)->cnodes();
    for (auto &cnode : cnodes) {
      MS_EXCEPTION_IF_NULL(cnode);
      cnode->set_user_data<OperatorInfo>(nullptr);
    }
  }
  return SUCCESS;
}

Status ParallelStrategySearch(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root) {
  // There are 4 meta-steps to determine the parallelization strategy for the ANF graph.
  // Step 1: Traverse the ANF graph, and create NODEs for costgraph:
  //      create the OperatorInfo object for each primitive, and enumerate the parallelization strategies
  //      for each OperatorInfo;
  // Step 1.1: Deal with 'Reshape':
  //      For 'Reshape', it takes its previous operator's layout as its input layout, and takes its next operator's
  //      layout as its output layout.
  // Step 2: Traverse the ANF graph, and create EDGES for costgraph:
  //      create the Edge object for each pair of OperatorInfo, and enumerate the parallelization strategies
  //      for each edge, based on the strategies of two OperatorInfos;
  // Step 3: Augment the costgraph:
  //      taking care for the case of a single Parameter being used by multiple operators. Create a TmpIdentity
  //      operator for this Parameter, and add an edge for the use of this Parameter by each
  //      subsequent operator;
  // Step 3.1: Calculate memory usage:
  //      note the memory usage calculation is different in training phase and inference phase.
  // Step 4: Run the strategy searching algorithm:
  //      If 'sharding_propagation' is configured to be true, then the configured-sharding-strategies will propagate
  //      to the non-configured operators, with the goal of minimizing redistribution cost.
  //      Otherwise, DP algorithm is used to search strategy of the costgraph. Note that there may be several connected
  //      components in the costgraph, and the DP algorithm runs on each of them.
  //
  // OUTPUT: the determined strategy for each operator.

  InitCostGraph();
  bool use_sp = (ParallelContext::GetInstance()->strategy_search_mode() == kShardingPropagation) ||
                (ParallelContext::GetInstance()->sharding_propagation());
  // Step 1
  if (CostModelContext::GetInstance()->is_multi_subgraphs() || use_sp) {
    if (ConstructCostGraphNodesByUniqueIdTC(all_nodes, root) == SUCCESS) {
      MS_LOG(INFO) << "Constructing nodes for cost graph succeeded. There are "
                   << entire_costgraph->GetOperators().size() << " operators.";
    } else {
      MS_LOG(EXCEPTION) << "Constructing nodes for cost graph failed.";
    }
  } else {
    if (ConstructCostGraphNodesByUniqueId(all_nodes, root) == SUCCESS) {
      MS_LOG(INFO) << "Constructing nodes for cost graph succeeded. There are "
                   << entire_costgraph->GetOperators().size() << " operators.";
    } else {
      MS_LOG(EXCEPTION) << "Constructing nodes for cost graph failed.";
    }
  }
  // Step 1.1
  ReshapeCostCompute(all_nodes);
  // Step 2
  ConstructCostGraphEdges(all_nodes);
  MS_LOG(INFO) << "Constructing edges for cost graph succeeded. There are " << entire_costgraph->GetOperators().size()
               << " operators, and " << entire_costgraph->GetNumEdges() << " edges.";

  // Step 3: Augment the costgraph.
  AugmentCostGraph(all_nodes);
  auto num_ops = entire_costgraph->GetOperators().size();
  SetOpsNumToExecutor(num_ops);
  auto num_edges = entire_costgraph->GetNumEdges();
  MS_LOG(INFO) << "After the augmenting procedure, there are " << num_ops << " operators, and " << num_edges
               << " edges.";

  // Step 3.1: Calculate the memory usage
  if (!use_sp && entire_costgraph->CalculateMemoryCost() != SUCCESS) {
    MS_LOG(EXCEPTION) << "Calculating memory cost failed.";
  }

  // Step 4: run the strategy searching algorithm
  if (use_sp) {
    entire_costgraph->StrategyPropagate(configured_stra_ops_);
  } else if (GetStrategy(entire_costgraph) != SUCCESS) {
    MS_LOG(ERROR) << "Strategy search for cost-graph fails";
    return FAILED;
  }
  MS_LOG(INFO) << "Searching strategy succeeded.";

  if (entire_costgraph->InitSelectedStrategy() == SUCCESS) {
    MS_LOG(INFO) << "Init selected strategy succeeded.";
  } else {
    MS_LOG(EXCEPTION) << "Init selected strategy failed.";
  }

  for (auto &op : entire_costgraph->GetOperators()) {
    // print the selected strategy
    StrategyPtr s_strategy = op->selected_strategy();
    if (s_strategy != nullptr) {
      MS_LOG(INFO) << op->name() << ": The strategy is: " << s_strategy->ToString();
    }
    // Label the cnodes of the op if they were already created
    for (const auto &cnode : op->cnodes()) {
      cnode->AddAttr(OP_INFO_CREATED, MakeValue(true));
    }
  }
  // Remove some operatorInfo from the CNODEs
  (void)IgnoreOperatorsInCostGraph();

  ops_in_a_loop_.clear();
  configured_stra_ops_.clear();
  ignore_candidate_.clear();

  return SUCCESS;
}

std::vector<std::vector<std::string>> RecInputTensorNames(const std::map<std::string, std::string>::iterator &it,
                                                          std::vector<std::vector<std::string>> input_tensor_names) {
  for (size_t j = 0; j < input_tensor_names.size(); j++) {
    for (size_t k = 0; k < input_tensor_names[j].size(); k++) {
      if (it->first == input_tensor_names[j][k]) {
        input_tensor_names[j][k] = it->second;
        break;
      }
    }
  }
  return input_tensor_names;
}

CNodePtr GetInternalOperatorInfo(const CNodePtr &cnode, const ValueNodePtr &prim_anf_node) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  MS_EXCEPTION_IF_NULL(prim);

  if (prim->name() == prim::kPrimTupleGetItem->name() || prim->name() == DEPEND) {
    auto prev_cnode = cnode->input(1)->cast<CNodePtr>();
    if (prev_cnode == nullptr || !IsValueNode<Primitive>(prev_cnode->input(0))) {
      return nullptr;
    }
    if (IsValueNode<FuncGraph>(prev_cnode->input(0))) {
      size_t out_index = 0;
      out_index = LongToSize(GetValue<int64_t>(GetValueNode(prev_cnode->input(INDEX_TWO))));
      auto graph = GetValueNode<FuncGraphPtr>(prev_cnode->input(0));
      MS_EXCEPTION_IF_NULL(graph);
      auto output = graph->output();
      MS_EXCEPTION_IF_NULL(output);
      while (IsPrimitiveCNode(output, prim::kPrimDepend)) {
        auto output_cnode = output->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(output_cnode);
        output = output_cnode->input(1);
      }
      while (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
        auto make_tuple_cnode = output->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(make_tuple_cnode);
        output = make_tuple_cnode->input(out_index + 1);
      }
      prev_cnode = output->cast<CNodePtr>();
    }
    MS_EXCEPTION_IF_NULL(prev_cnode);
    auto prev_prim_value = prev_cnode->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prev_prim_value);
    auto prev_prim = prev_prim_value->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(prev_prim);
    while (prev_prim->name() == prim::kPrimTupleGetItem->name() || prev_prim->name() == DEPEND) {
      prev_cnode = prev_cnode->input(1)->cast<CNodePtr>();
      if (prev_cnode == nullptr || !IsValueNode<Primitive>(prev_cnode->input(0))) {
        return nullptr;
      }
      prev_prim_value = prev_cnode->input(0)->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(prev_prim_value);
      prev_prim = prev_prim_value->value()->cast<PrimitivePtr>();
      MS_EXCEPTION_IF_NULL(prev_prim);
    }
    return prev_cnode;
  }
  return nullptr;
}

void ModifyInputsTensorNameListIfOperatorInfoCreated(const std::string &name, const std::string &uniqueid) {
  size_t iter_ops = 0;
  for (const auto &op : entire_costgraph->GetOperators()) {
    if (op->name() == name) {
      break;
    }
    iter_ops = iter_ops + 1;
  }

  std::vector<std::vector<std::string>> input_tensor_names = entire_costgraph->get_inputs_tensor_name_list();
  for (size_t i = 0; i < input_tensor_names.size(); i++) {
    for (size_t j = 0; j < input_tensor_names[i].size(); j++) {
      if (input_tensor_names[i][j] == uniqueid) {
        input_tensor_names[i][j] = input_tensor_names[iter_ops][0];
      }
    }
  }

  entire_costgraph->set_inputs_tensor_name_list(input_tensor_names);
}

size_t FindOperatorIndexById(const std::string &unique_id,
                             const std::vector<std::vector<std::string>> &input_tensor_names) {
  for (size_t i = 0; i < input_tensor_names.size(); i++) {
    if (input_tensor_names[i][0] == unique_id) {
      return i;
    }
  }
  return SIZE_MAX;
}

std::vector<std::vector<size_t>> GetIndexOfOpsSharingInputTensor(
  const std::vector<std::vector<std::string>> &param_users_uniqueid_list,
  const std::vector<std::vector<std::string>> &input_tensor_names) {
  std::vector<std::vector<size_t>> param_users_ops_index;
  for (const auto &users_uniqueid : param_users_uniqueid_list) {
    std::vector<size_t> users_index;
    for (const auto &user_uniqueid : users_uniqueid) {
      size_t user_index = FindOperatorIndexById(user_uniqueid, input_tensor_names);
      if (user_index != SIZE_MAX) {
        users_index.push_back(user_index);
      }
    }
    param_users_ops_index.push_back(users_index);
  }
  return param_users_ops_index;
}

void CalculateMicroBatchSize(const std::shared_ptr<Graph> &graph, const FuncGraphPtr &root) {
  // The first dimension of an operator is its batch dimension.
  // However, the shape of the first dimension is not the batch_size assigned by users.
  // This function helps to calculate the micro batch size in the pipeline scenario.

  auto manager = root->manager();
  auto ops = entire_costgraph->GetOperators();
  AnfNodePtr virtual_dataset_;
  for (auto &fg : manager->func_graphs()) {
    for (auto &node : fg->nodes()) {
      if (IsPrimitiveCNode(node, prim::kPrimVirtualDataset)) {
        virtual_dataset_ = node;
        break;
      }
    }
  }
  if (!virtual_dataset_) {
    // Normally for auto parallel, virtual dataset is required in order to control the input's parallel strategy.
    // However, in some test cases or NN, there is no input data.
    // This if condition aims to deal with these cases, and return 1.
    graph->micro_batch_size = 1;
    return;
  }
  auto node_user_map = manager->node_users();
  auto node_users = node_user_map[virtual_dataset_];
  int64_t data_user_size = 0;
  int64_t total_batch_size = 0;
  for (auto &node_user : node_users) {
    if (IsPrimitiveCNode(node_user.first, prim::kPrimTupleGetItem)) {
      auto data_users = manager->node_users()[node_user.first];
      auto node_first = data_users.front().first;
      if (!IsPrimitiveCNode(node_first, prim::kPrimStridedSlice)) {
        data_users.clear();
        data_users = node_user_map[node_first];
      }
      data_user_size = int64_t(data_users.size());
    }
  }

  for (auto op : ops) {
    if (op->type() == GET_NEXT) {
      for (auto shape : op->outputs_shape()) {
        if (!shape.empty()) {
          total_batch_size = shape[0];
          break;
        }
      }
      break;
    }
  }
  if (data_user_size != 0) {
    graph->micro_batch_size = total_batch_size / data_user_size;
    MS_LOG(INFO) << "In the pipeline scenario, the micro_batch_size of each stage is " << graph->micro_batch_size;
  } else {
    MS_LOG_WITH_NODE(EXCEPTION, virtual_dataset_)
      << "Data user size equals to 0, which could not be divided by the total batch size";
  }
}

void CreateNodesForCostGraph(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root) {
  if (CostModelContext::GetInstance()->is_multi_subgraphs()) {
    if (ConstructCostGraphNodesByUniqueIdTC(all_nodes, root) == SUCCESS) {
      MS_LOG(INFO) << "Constructing nodes for cost graph succeeded. There are "
                   << entire_costgraph->GetOperators().size() << " operators.";
    } else {
      MS_LOG(EXCEPTION) << "Constructing nodes for cost graph failed.";
    }
  } else {
    if (ConstructCostGraphNodesByUniqueId(all_nodes, root) == SUCCESS) {
      MS_LOG(INFO) << "Constructing nodes for cost graph succeeded. There are "
                   << entire_costgraph->GetOperators().size() << " operators.";
    } else {
      MS_LOG(EXCEPTION) << "Constructing nodes for cost graph failed.";
    }
  }
}

void ReInitCostGraph(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root, bool dyn_shape_tmp_fix) {
  InitCostGraph();
  CreateNodesForCostGraph(all_nodes, root);
  if (!dyn_shape_tmp_fix) {
    ReshapeCostCompute(all_nodes);
  }
}

void WriteStrategiesBackToAnfGraph(const std::vector<std::shared_ptr<OperatorInfo>> &ops) {
  for (auto &op : ops) {
    auto op_type = op->type();
    if (op_type == CAST || op_type == RESHAPE) {
      continue;
    }
    auto op_strategy = op->selected_strategy()->GetInputDim();
    if (!op_strategy.empty()) {
      std::vector<ValuePtr> strategies;
      (void)std::transform(op_strategy.begin(), op_strategy.end(), std::back_inserter(strategies),
                           [](const Dimensions &dim) { return MakeValue(dim); });
      ValueTuplePtr var = std::make_shared<ValueTuple>(strategies);
      op->cnode()->AddPrimalAttr(parallel::IN_STRATEGY, var);
    }
  }
}

void TMpInferBatchMatMul(const std::shared_ptr<Graph> &graph, Graph::NodeType *node) {
  if (node->apply.arguments[0].tensor_shape.shape_c != -1 && node->apply.arguments[1].tensor_shape.shape_c == -1) {
    auto infer_shape = node->apply.arguments[0].tensor_shape.shape_c;
    node->apply.arguments[1].tensor_shape.shape_c = infer_shape;

    if (node->node_out.size() == 0) {
      MS_LOG(EXCEPTION) << "The current BatchMatMul (" << node->name << ") does not have an outgoing node.";
    }
    auto &outgoing_node = graph->nodes[node->node_out[0].idx];
    if (outgoing_node.apply.arguments[0].tensor_shape.shape_c == node->tensor_parm.tensor_shape.shape_c) {
      outgoing_node.apply.arguments[0].tensor_shape.shape_c = infer_shape;
    }

    node->tensor_parm.tensor_shape.shape_c = infer_shape;
  }
}

void TmpInferForDynamicShapeInSAPP(const std::shared_ptr<Graph> &graph) {
  for (size_t index = graph->nodes.size(); index > 0; index--) {
    auto node = graph->nodes[index - 1];
    if (node.apply.op_type == OperatorType::kRecBatchMatMul) {
      TMpInferBatchMatMul(graph, &node);
    }
  }
}

bool HasUserConfiguredStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops) {
  for (auto op : ops) {
    auto prim_anf_node = GetValueNode<PrimitivePtr>(op->cnode()->input(0));
    MS_EXCEPTION_IF_NULL(prim_anf_node);
    bool has_user_configured_strategy = prim_anf_node->HasAttr(parallel::IN_STRATEGY);
    if (has_user_configured_strategy) {
      return true;
    }
  }
  return false;
}

Status ParallelStrategyRecSearch(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root, size_t rank_id,
                                 const size_t device_num) {
  MS_LOG(INFO) << "Now entering Symbolic Automatic Parallel Planner";
  bool dyn_shape_tmp_fix = false;
  if (device_num > 0) {
    dyn_shape_tmp_fix = true;
  }

  ReInitCostGraph(all_nodes, root, dyn_shape_tmp_fix);
  auto ops = entire_costgraph->GetOperators();
  if (dyn_shape_tmp_fix && HasUserConfiguredStrategy(ops)) {
    MS_LOG(WARNING) << "Now the split strategy will be automatically generated through SAPP, which will overwrite "
                       "the strategy that has been manually configured by the user.";
  }

  std::vector<std::vector<std::string>> input_tensor_names = entire_costgraph->get_inputs_tensor_name_list();
  // Needed by rec_parser 2
  auto param_users_uniqueid_list = entire_costgraph->get_param_users_uniqueid_list();
  auto tuple_getitem_list = entire_costgraph->get_tuple_getitem_list();
  for (auto it = tuple_getitem_list.begin(); it != tuple_getitem_list.end();) {
    input_tensor_names = RecInputTensorNames(it++, input_tensor_names);
  }
  std::shared_ptr<Graph> graph = ParseGraph(ops, input_tensor_names);

  std::vector<std::vector<size_t>> param_users_ops_index =
    GetIndexOfOpsSharingInputTensor(param_users_uniqueid_list, input_tensor_names);
  std::shared_ptr<std::vector<std::vector<size_t>>> eli_list = std::make_shared<std::vector<std::vector<size_t>>>();
  std::shared_ptr<std::vector<size_t>> index_list = std::make_shared<std::vector<size_t>>();
  graph = EliminateGraph(graph, eli_list, index_list, dyn_shape_tmp_fix);
  graph->dyn_shape_tmp_fix = dyn_shape_tmp_fix;

  if (graph->dyn_shape_tmp_fix) {
    TmpInferForDynamicShapeInSAPP(graph);
  }

  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    CalculateMicroBatchSize(graph, root);
  }

  size_t num_device = g_device_manager->DeviceNum();
  const auto device_memory = CostModelContext::GetInstance()->device_memory_capacity();
  // To specify the process is training or inference. For training, if optimizer parallel is activated, it requires at
  // least one cut on DP dimension.
  bool isTraining = IsTraining(root->manager());
  if (PartitionForAllDevices(num_device, device_memory, graph, isTraining, root) == SUCCESS) {
    MS_LOG(INFO) << "Partition Success With " << num_device << " devices.";
  } else {
    MS_LOG(ERROR) << "PartitionForAllDevices failed.";
    return FAILED;
  }

  // Needed when changing stage number
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    if (!graph->dyn_shape_tmp_fix) {
      if (ParallelInit() != SUCCESS) {
        MS_LOG(EXCEPTION) << "Parallel init failed after Rec search";
      }
    } else {
      if (ParallelInit(rank_id, device_num) != SUCCESS) {
        MS_LOG(EXCEPTION) << "Parallel init failed";
      }
    }
    if (parallel::ParallelContext::GetInstance()->auto_pipeline()) {
      ReInitCostGraph(all_nodes, root, graph->dyn_shape_tmp_fix);
      ops = entire_costgraph->GetOperators();
    }
  }

  GenerateStrategy(graph, ops, eli_list, input_tensor_names, index_list, isTraining, param_users_ops_index, root);

  // print the selected strategy
  for (auto &op : entire_costgraph->GetOperators()) {
    StrategyPtr s_strategy = op->selected_strategy();
    if (s_strategy != nullptr) {
      MS_LOG(INFO) << op->name() << ": The strategy is: " << s_strategy->ToString();
    }
  }

  if (graph->dyn_shape_tmp_fix) {
    (void)WriteStrategiesBackToAnfGraph(ops);
    (void)IgnoreOperatorsInCostGraph();
    ops_in_a_loop_.clear();
    configured_stra_ops_.clear();
    ignore_candidate_.clear();
    return SUCCESS;
  }

  if (entire_costgraph->InitSelectedStrategy() == SUCCESS) {
    MS_LOG(INFO) << "Init selected strategy succeeded.";
  } else {
    MS_LOG(ERROR) << "Init selected strategy failed.";
    return FAILED;
  }

  (void)IgnoreOperatorsInCostGraph();
  ops_in_a_loop_.clear();
  configured_stra_ops_.clear();
  ignore_candidate_.clear();

  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
