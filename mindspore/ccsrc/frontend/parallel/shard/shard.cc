/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <memory>
#include <set>
#include <queue>
#include <utility>
#include <string>
#include <vector>

#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/shard/shard.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/common/utils/comm_manager.h"
#include "include/common/utils/parallel_context.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace parallel {
namespace {
using ExpectFunc = std::function<bool(const CNodePtr &)>;
}
static void GenerateDefaultStrategy(const ValueNodePtr &axes, const std::vector<AnfNodePtr> &nodes,
                                    const size_t device_num, std::vector<std::vector<int64_t>> *default_strategy) {
  auto strategies = axes->value()->cast<ValueTuplePtr>()->value();
  size_t i = 0;
  for (auto &strategy : strategies) {
    auto node = nodes[i];
    if (strategy->isa<None>()) {
      auto node_shape = common::AnfAlgo::GetOutputInferShape(node, 0);
      auto node_size = node_shape.size();
      std::vector<int64_t> current_d_strategy(node_size, 1);
      if (!node_shape.empty() && device_num > 0 && node_shape[0] % device_num == 0) {
        current_d_strategy[0] = SizeToLong(device_num);
      }
      (void)default_strategy->emplace_back(std::move(current_d_strategy));
    } else {
      (void)default_strategy->emplace_back(GetValue<Shape>(strategy));
    }
    i += 1;
  }
}

static bool CheckOneDimensionalIntTuple(const ValuePtr &value_ptr) {
  if (!value_ptr->isa<ValueTuple>()) {
    return false;
  }
  auto elements = value_ptr->cast<ValueTuplePtr>()->value();
  return std::all_of(elements.begin(), elements.end(),
                     [](const ValuePtr &element) { return element->isa<Int64Imm>(); });
}

static bool CheckLayout(const ValueNodePtr &axes, bool *need_default_strategy, size_t *axes_size) {
  auto strategies = axes->value()->cast<ValueTuplePtr>()->value();
  for (auto &strategy : strategies) {
    *axes_size += 1;
    if (strategy->isa<None>()) {
      *need_default_strategy = true;
      continue;
    }
    if (!CheckOneDimensionalIntTuple(strategy)) {
      return false;
    }
  }
  return true;
}

static void GetInputNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *input_nodes) {
  auto parameters = func_graph->parameters();
  for (auto &parameter : parameters) {
    if (parameter->abstract()->isa<abstract::AbstractMonad>()) {
      continue;
    }
    input_nodes->push_back(parameter);
  }
}

static bool CheckDeviceNum(const std::vector<std::vector<int64_t>> &strategies, const int64_t device_num) {
  for (size_t i = 0; i < strategies.size(); ++i) {
    auto strategy = strategies[i];
    int64_t required_num = 1;
    (void)std::for_each(strategy.begin(), strategy.end(),
                        [&required_num](const int64_t data) { required_num *= data; });
    if (required_num > device_num) {
      MS_LOG(ERROR) << "required device number: " << required_num
                    << " is larger than available device number: " << device_num << " at index: " << i;
      return false;
    }
    if (device_num % required_num != 0) {
      MS_LOG(ERROR) << "required device number: " << required_num
                    << " is not divisible by device number: " << device_num << " at index: " << i;
      return false;
    }
  }
  return true;
}

static bool CheckShapeStrategy(const std::vector<int64_t> &strategy, const std::vector<int64_t> &shapes) {
  for (size_t i = 0; i < strategy.size(); ++i) {
    auto devide_num = strategy[i];
    auto shape = shapes[i];
    if (shape % devide_num != 0) {
      MS_LOG(ERROR) << "dimension length: " << shape << " is not divisible by layout: " << devide_num
                    << " at index: " << i;
      return false;
    }
  }
  return true;
}

static ValueTuplePtr ShapesToValueTuplePtr(const Shapes &shapes) {
  std::vector<ValuePtr> value_list;
  (void)std::transform(shapes.begin(), shapes.end(), std::back_inserter(value_list),
                       [](const Shape &shape) { return MakeValue(shape); });
  return std::make_shared<ValueTuple>(value_list);
}

static Shapes ValueTuplePtrToShapes(const ValueTuplePtr &value_tuple_ptr) {
  Shapes shapes;
  auto value_list = value_tuple_ptr->value();
  (void)std::transform(value_list.begin(), value_list.end(), std::back_inserter(shapes),
                       [](const ValuePtr &value_ptr) { return GetValue<Shape>(value_ptr); });
  return shapes;
}

static AnfNodeIndexSet FindAnfNodeIndexSetToInsertStrategy(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                           const ExpectFunc &filter_func) {
  FuncGraphManagerPtr manager = func_graph->manager();
  AnfNodeIndexSet ret_set;
  auto node_users = manager->node_users()[node];
  std::queue<std::pair<AnfNodePtr, int>> bfs_queuq;
  std::set<CNodePtr> visited_cnodes;
  (void)std::for_each(node_users.begin(), node_users.end(),
                      [&bfs_queuq](const std::pair<AnfNodePtr, int> &user) { bfs_queuq.push(user); });
  while (!bfs_queuq.empty()) {
    auto user = bfs_queuq.front();
    bfs_queuq.pop();
    auto cnode = user.first->cast<CNodePtr>();
    visited_cnodes.insert(cnode);
    if (!filter_func(cnode)) {
      auto tmp_users = manager->node_users()[cnode];
      for (auto &tmp_user : tmp_users) {
        if (visited_cnodes.count(tmp_user.first->cast<CNodePtr>()) == 0) {
          bfs_queuq.push(tmp_user);
        }
      }
      continue;
    }
    ret_set.insert(user);
  }
  return ret_set;
}

static bool IsSettingStrategyByInsertIdentity(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                              const std::string &param_name) {
  FuncGraphManagerPtr manager = func_graph->manager();
  auto node_users = manager->node_users()[cnode];
  for (const auto &user : node_users) {
    auto user_node = user.first;
    if (IsPrimitiveCNode(user_node, prim::kPrimShardIdentity)) {
      auto attrs = GetCNodePrimitive(user_node)->attrs();
      if (StrategyFound(attrs)) {
        auto origin_strategies = ValueTuplePtrToShapes(attrs[parallel::IN_STRATEGY]->cast<ValueTuplePtr>());
        MS_LOG(WARNING) << "For " << param_name << ", its strategy has been set to " << origin_strategies.at(0)
                        << ", the relevant settings in input_strategy will be ignored";
        return true;
      }
    }
  }
  return false;
}

// New a primitive for cnode and set in_strategy to it.
static void SetStrategyToCNode(const CNodePtr &cnode, const Shapes &strategies) {
  auto strategy = ShapesToValueTuplePtr(strategies);
  PrimitivePtr prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  PrimitivePtr new_prim;
  if (prim->isa<PrimitivePy>()) {
    auto prim_py = prim->cast<PrimitivePyPtr>();
    MS_EXCEPTION_IF_NULL(prim_py);
    new_prim = std::make_shared<PrimitivePy>(*prim_py);
  } else {
    new_prim = std::make_shared<Primitive>(*prim);
  }
  auto attrs_temp = prim->attrs();
  attrs_temp[parallel::IN_STRATEGY] = strategy;
  (void)new_prim->SetAttrs(attrs_temp);

  ValuePtr new_prim_value = MakeValue(new_prim);
  ValueNodePtr new_prim_value_node = NewValueNode(new_prim_value);
  auto new_prim_anf_node = new_prim_value_node->cast<AnfNodePtr>();
  MS_EXCEPTION_IF_NULL(new_prim_anf_node);
  cnode->set_input(0, new_prim_anf_node);
}

static void SetInputLayout(const FuncGraphPtr &func_graph, const AnfNodePtr &in_strategy, const int64_t device_num) {
  auto in_strategy_tuple = in_strategy->cast<ValueNodePtr>();
  bool need_default_strategy = false;
  size_t in_strategy_size = 0;
  if (!IsValueNode<ValueTuple>(in_strategy_tuple) ||
      !CheckLayout(in_strategy_tuple, &need_default_strategy, &in_strategy_size)) {
    MS_LOG(EXCEPTION) << "in_strategy should be a two-dimension tuple";
  }
  std::vector<AnfNodePtr> input_nodes;
  GetInputNodes(func_graph, &input_nodes);
  if (input_nodes.size() != in_strategy_size) {
    MS_LOG(EXCEPTION) << "Input numbers: " << input_nodes.size()
                      << " is not equal to in_strategy numbers: " << in_strategy_size;
  }
  std::vector<std::vector<int64_t>> input_strategy;
  if (need_default_strategy) {
    GenerateDefaultStrategy(in_strategy_tuple, input_nodes, device_num, &input_strategy);
  } else {
    input_strategy = GetValue<std::vector<std::vector<int64_t>>>(in_strategy_tuple->value());
  }
  if (!CheckDeviceNum(input_strategy, device_num)) {
    MS_LOG(EXCEPTION) << "check device number failed";
  }

  FuncGraphManagerPtr manager = func_graph->manager();
  auto parameters = func_graph->parameters();

  for (size_t i = 0; i < parameters.size(); ++i) {
    auto parameter = parameters[i];
    if (parameter->abstract()->isa<abstract::AbstractMonad>()) {
      continue;
    }
    // Verify that the user has set the valid layout, if the layout is generated by 'GenareteDefaultStrategy', ignored
    // its check.
    auto param_shape = common::AnfAlgo::GetOutputInferShape(parameter, 0);
    if (!CheckDeviceNum(input_strategy, device_num)) {
      MS_LOG(EXCEPTION) << "check device number failed";
    }
    if (!input_strategy[i].empty() && param_shape.size() != input_strategy[i].size()) {
      MS_LOG(EXCEPTION) << "Input dimension: " << param_shape.size()
                        << " is not equal to in_strategy dimension: " << input_strategy[i].size() << " at index " << i;
    }
    if (!CheckShapeStrategy(input_strategy[i], param_shape)) {
      MS_LOG(EXCEPTION) << "Check conformance between input strategy " << input_strategy[i] << "and tensor shape "
                        << param_shape << "failed";
    }

    auto to_insert_nodes_set = manager->node_users()[parameter];

    auto execution_mode = MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE);
    if (execution_mode == kPynativeMode) {
      // In PyNative mode, to_insert_nodes_set are TupleGetItem Prims.
      to_insert_nodes_set = FindAnfNodeIndexSetToInsertStrategy(
        func_graph, parameter, [](const CNodePtr &cnode) { return IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem); });
    }

    if (to_insert_nodes_set.empty()) {
      MS_LOG(EXCEPTION) << "For input: \"" << parameter->fullname_with_scope()
                        << "\", failed to find node to insert strategy.";
    }
    for (auto &node : to_insert_nodes_set) {
      auto to_insert_cnode = node.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(to_insert_cnode);
      if (IsSettingStrategyByInsertIdentity(func_graph, to_insert_cnode, parameter->fullname_with_scope())) {
        continue;
      }
      int64_t layout_index = static_cast<int64_t>(i);
      CNodePtr identity_cnode = nullptr;
      if (execution_mode == kGraphMode) {
        // Setting strategy by insert identity CNode directly using inputs in GraphMode.
        // e.g TupleGetItem(parameter, index) -> func{identity{input_strategy[i], input_i}}.
        identity_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimShardIdentity), parameter});
        AnfNodePtrList node_inputs_list(to_insert_cnode->inputs().begin(), to_insert_cnode->inputs().end());
        auto input_index =
          std::distance(node_inputs_list.begin(),
                        std::find(node_inputs_list.begin(), node_inputs_list.end(), parameter->cast<AnfNodePtr>()));
        identity_cnode->set_abstract(parameter->abstract());
        FuncGraphManagerPtr local_mng = func_graph->manager();
        local_mng->SetEdge(to_insert_cnode, input_index, identity_cnode);
      }

      if (execution_mode == kPynativeMode) {
        // Setting strategy by insert identity after TupleGetItem in PynativeMode.
        // e.g TupleGetItem(parameter, index) -> identity{in_strategy=[input_strategy[index], TupleGetItem_i}
        identity_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimShardIdentity), to_insert_cnode});
        auto to_insert_cnode_abstract = to_insert_cnode->abstract();
        MS_EXCEPTION_IF_NULL(to_insert_cnode_abstract);
        identity_cnode->set_abstract(to_insert_cnode_abstract->Clone());
        (void)manager->Replace(to_insert_cnode, identity_cnode);

        // Get corresponding param_layout index in PynativeMode.
        auto tuple_index = to_insert_cnode->input(2);
        auto value_node = tuple_index->cast<ValueNodePtr>();
        MS_EXCEPTION_IF_NULL(value_node);
        layout_index = GetValue<int64_t>(value_node->value());
      }
      Shapes current_layout = {input_strategy[layout_index]};
      SetStrategyToCNode(identity_cnode, current_layout);
      MS_LOG(INFO) << "Succeed to set layout " << current_layout << "to node "
                   << to_insert_cnode->fullname_with_scope();
    }
  }
}

static void SetParameterLayout(const FuncGraphPtr &root) {
  FuncGraphManagerPtr manager = root->manager();
  auto root_parameters = root->parameters();
  for (const auto &param : root_parameters) {
    auto parameter = param->cast<ParameterPtr>();
    auto param_info = parameter->param_info();
    if (param_info == nullptr || param_info->param_strategy().empty()) {
      // Do not set param_strategy, skip it.
      continue;
    }
    auto param_strategy = param_info->param_strategy();
    auto param_name = param_info->name();
    AnfNodeIndexSet users = manager->node_users()[parameter];
    auto to_insert_nodes_set = FindAnfNodeIndexSetToInsertStrategy(
      root, parameter, [](const CNodePtr &cnode) { return IsPrimitiveCNode(cnode, prim::kPrimLoad); });

    for (const auto &user : to_insert_nodes_set) {
      auto load_cnode = user.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(load_cnode);

      auto cur_graph = load_cnode->func_graph();
      FuncGraphManagerPtr local_manager = cur_graph->manager();
      if (IsSettingStrategyByInsertIdentity(cur_graph, load_cnode, param_name)) {
        continue;
      }
      // Setting param_layout by insert identity. e.g Load(param) -> identity{in_strategy=[param_layout]}
      auto identity_cnode = cur_graph->NewCNode({NewValueNode(prim::kPrimShardIdentity), load_cnode});
      auto load_cnode_abstract = load_cnode->abstract();
      MS_EXCEPTION_IF_NULL(load_cnode_abstract);
      identity_cnode->set_abstract(load_cnode_abstract->Clone());
      (void)local_manager->Replace(load_cnode, identity_cnode);
      Shapes current_layout = {param_strategy};
      SetStrategyToCNode(identity_cnode, current_layout);
      MS_LOG(INFO) << "The layout of \"" << param_name << "\" has been set to " << load_cnode->fullname_with_scope()
                   << ". Current strategies is " << current_layout;
    }
  }
}

void CheckVmap(AnfNodePtr node) {
  auto cnode = node->cast<CNodePtr>();
  AnfNodePtr value_node = cnode->input(1);
  auto func_graph = GetValueNode<FuncGraphPtr>(value_node);
  if (func_graph == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Unexpected meta function graph node:" << cnode->DebugString();
  }
  if (parallel::IsEmbedShardNode(func_graph)) {
    MS_LOG(EXCEPTION)
      << "The usage of vmap nested shard (e.g vmap(shard)) is not supported currently. Current FuncGraph: "
      << func_graph->ToString();
  }
}

static bool SetStrategyForShard(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                                const int64_t device_num) {
  constexpr size_t kShardFnIndex = 1;
  constexpr size_t kShardInStrategyIndex = 2;
  auto set_success = false;
  auto execution_mode = MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE);
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimVmap)) {
      CheckVmap(node);
    }
    if (IsPrimitiveCNode(node, prim::kPrimShard)) {
      auto cnode = node->cast<CNodePtr>();
      auto vnode = cnode->input(kShardFnIndex)->cast<ValueNodePtr>();
      auto in_strategy = cnode->input(kShardInStrategyIndex);
      ScopeGuard scope_guard(vnode->scope());
      auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
      if (func_graph->has_flag(kSharded)) {
        continue;
      }
      MS_EXCEPTION_IF_NULL(func_graph);
      if (IsEmbedShardNode(func_graph) && execution_mode == kPynativeMode) {
        MS_LOG(EXCEPTION) << "Nested use of shard (e.g shard(shard(...), ...) is not supported in "
                          << "PyNative mode currently, | FuncGraph: " << func_graph->ToString();
      }
      // get input nodes
      std::vector<AnfNodePtr> input_nodes;
      GetInputNodes(func_graph, &input_nodes);
      if (HasNestedMetaFg(func_graph)) {
        return set_success;
      }
      SetInputLayout(func_graph, in_strategy, device_num);
      func_graph->set_flag(kSharded, true);
      set_success = true;
    }
    if (IsPrimitiveCNode(node, prim::kPrimReshard)) {
      // Get reshard attributes, e.g., in_layout/in_strategy.
      auto reshard_cnode = node->cast<CNodePtr>();
      auto attrs = GetCNodePrimitive(reshard_cnode)->attrs();

      // New a identity prim and set the attributes the same as reshard prim.
      auto new_identity_prim = std::make_shared<Primitive>(prim::kPrimShardIdentity->name());
      new_identity_prim->SetAttrs(attrs);

      // Make identity prim into identity node and cast to AnfNode.
      ValuePtr new_identity_value = MakeValue(new_identity_prim);
      ValueNodePtr new_identity_value_node = NewValueNode(new_identity_value);
      auto new_identity_anf_node = new_identity_value_node->cast<AnfNodePtr>();

      // Replace reshard node as identity node.
      constexpr size_t kPrimIndex = 0;
      reshard_cnode->set_input(kPrimIndex, new_identity_anf_node);

      set_success = true;
    }
  }
  if (set_success) {
    SetParameterLayout(root);
    root->set_flag(kSharded, true);
    // If Shard is set under semi auto-parallel mode, change mode to auto_parallel and
    // switch on sharding propagation.
    if (ParallelContext::GetInstance()->parallel_mode() == parallel::kSemiAutoParallel) {
      ParallelContext::GetInstance()->set_parallel_mode(parallel::kAutoParallel);
      ParallelContext::GetInstance()->set_strategy_search_mode(parallel::kShardingPropagation);
      MS_LOG(INFO) << "Shard is set under semi auto-parallel mode, change mode to auto_parallel"
                   << " and switch on sharding propagation.";
    }
  }
  return set_success;
}

bool Shard(const FuncGraphPtr &root, const opt::OptimizerPtr &) {
  MS_EXCEPTION_IF_NULL(root);
  MS_LOG(INFO) << "Shard pass starts.";
  bool change = false;
  if (!root->has_flag(kHasShard)) {
    MS_LOG(INFO) << "Shard Prim don't exist, skip Shard pass";
    return change;
  }
  auto parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kSemiAutoParallel && parallel_mode != kAutoParallel) {
    MS_LOG(INFO) << "Only auto_parallel and semi_auto_parallel support shard";
    return change;
  }

  if (ParallelInit() != SUCCESS) {
    MS_LOG(EXCEPTION) << "parallel init failed.";
  }

  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  CheckGlobalDeviceManager();
  auto device_num_shard = g_device_manager->stage_device_num();
  change = SetStrategyForShard(root, all_nodes, device_num_shard);
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpGraph(root, std::string(STEP_SHARD_END));
  }
#endif
  return change;
}
}  // namespace parallel
}  // namespace mindspore
