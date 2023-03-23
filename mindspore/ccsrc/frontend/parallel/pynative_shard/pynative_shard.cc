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
#include <string>
#include <vector>
#include <set>
#include <memory>

#include "frontend/parallel/pynative_shard/pynative_shard.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel.h"
#include "utils/ms_context.h"
#include "include/common/utils/comm_manager.h"
#include "frontend/parallel/ops_info/ops_utils.h"

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
    if (parameter->cast<ParameterPtr>()->name() == "u" || parameter->cast<ParameterPtr>()->name() == "io") {
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

AnfNodeIndexSet FindAnfNodeIndexSetToInsertStrategy(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const ExpectFunc &filter_func) {
  FuncGraphManagerPtr manager = func_graph->manager();
  AnfNodeIndexSet ret_set;
  auto node_users = manager->node_users()[node];
  std::queue<std::pair<AnfNodePtr, int>> bfs_queuq;
  (void)std::for_each(node_users.begin(), node_users.end(),
                      [&bfs_queuq](const std::pair<AnfNodePtr, int> &user) { bfs_queuq.push(user); });
  while (!bfs_queuq.empty()) {
    auto user = bfs_queuq.front();
    bfs_queuq.pop();
    auto cnode = user.first->cast<CNodePtr>();
    if (!filter_func(cnode)) {
      auto tmp_users = manager->node_users()[cnode];
      (void)std::for_each(tmp_users.begin(), tmp_users.end(),
                          [&bfs_queuq](const std::pair<AnfNodePtr, int> &user) { bfs_queuq.push(user); });
      continue;
    }
    ret_set.insert(user);
  }
  return ret_set;
}

bool IsSettingStrategyByInsertIdentity(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                       const std::string &param_name) {
  FuncGraphManagerPtr manager = func_graph->manager();
  auto node_users = manager->node_users()[cnode];
  for (const auto &user : node_users) {
    auto user_node = user.first;
    if (IsPrimitiveCNode(user_node, prim::kPrimIdentity)) {
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
void SetStrategyToCNode(const CNodePtr &cnode, const Shapes &strategies) {
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

void SetInputLayout(const FuncGraphPtr &func_graph, const AnfNodePtr &in_strategy, const int64_t device_num) {
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
    if (parameter->cast<ParameterPtr>()->name() == "u" || parameter->cast<ParameterPtr>()->name() == "io") {
      continue;
    }
    // Verify that the user has set the valid layout, if the layout is generated by 'GenareteDefaultStrategy', ignored
    // its check.
    auto output_shape = common::AnfAlgo::GetOutputInferShape(parameter, 0);
    if (!input_strategy[i].empty() && output_shape.size() != input_strategy[i].size()) {
      MS_LOG(EXCEPTION) << "Input dimension: " << output_shape.size()
                        << " is not equal to in_strategy dimension: " << input_strategy[i].size() << " at index " << i;
    }
    AnfNodeIndexSet param_sub_set = manager->node_users()[parameter];
    auto to_insert_nodes_set = FindAnfNodeIndexSetToInsertStrategy(
      func_graph, parameter, [](const CNodePtr &cnode) { return IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem); });
    if (to_insert_nodes_set.empty()) {
      MS_LOG(EXCEPTION) << "For input: \"" << parameter->fullname_with_scope()
                        << "\", failed to find node to insert strategy.";
    }
    for (auto &node : to_insert_nodes_set) {
      auto tuple_get_item_cnode = node.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(tuple_get_item_cnode);
      if (IsSettingStrategyByInsertIdentity(func_graph, tuple_get_item_cnode, parameter->fullname_with_scope())) {
        continue;
      }

      // Setting strategy by insert identity.
      // e.g TupleGetItem(parameter, index) -> identity{in_strategy=[input_strategy[index]}
      auto identity_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimIdentity), tuple_get_item_cnode});
      auto tuple_get_item_cnode_abstract = tuple_get_item_cnode->abstract();
      MS_EXCEPTION_IF_NULL(tuple_get_item_cnode_abstract);
      identity_cnode->set_abstract(tuple_get_item_cnode_abstract->Clone());
      (void)manager->Replace(tuple_get_item_cnode, identity_cnode);

      // Get corresponding param_layout
      auto tuple_index = tuple_get_item_cnode->input(2);
      auto value_node = tuple_index->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto index = GetValue<int64_t>(value_node->value());
      Shapes current_strategies = {input_strategy[index]};
      SetStrategyToCNode(identity_cnode, current_strategies);
    }
  }
}

void SetParameterLayout(const FuncGraphPtr &root, const FuncGraphPtr &func_graph) {
  FuncGraphManagerPtr manager = func_graph->manager();
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
      func_graph, parameter, [](const CNodePtr &cnode) { return IsPrimitiveCNode(cnode, prim::kPrimLoad); });
    for (const auto &user : to_insert_nodes_set) {
      auto load_cnode = user.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(load_cnode);
      if (IsSettingStrategyByInsertIdentity(func_graph, load_cnode, param_name)) {
        continue;
      }

      // Setting param_layout by insert identity. e.g Load(param) -> identity{in_strategy=[param_layout]}
      auto identity_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimIdentity), load_cnode});
      auto load_cnode_abstract = load_cnode->abstract();
      MS_EXCEPTION_IF_NULL(load_cnode_abstract);
      identity_cnode->set_abstract(load_cnode_abstract->Clone());
      (void)manager->Replace(load_cnode, identity_cnode);
      Shapes current_strategies = {param_strategy};
      SetStrategyToCNode(identity_cnode, current_strategies);
      MS_LOG(DEBUG) << "The layout of \"" << param_name << "\" has been set to "
                    << identity_cnode->fullname_with_scope() << ". Current strategies is " << current_strategies;
    }
  }
}

static bool SetStrategyForShard(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                                const int64_t device_num) {
  constexpr size_t kShardFnIndex = 1;
  constexpr size_t kShardInStrategyIndex = 2;
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimShard)) {
      auto cnode = node->cast<CNodePtr>();
      auto vnode = cnode->input(kShardFnIndex)->cast<ValueNodePtr>();
      auto in_strategy = cnode->input(kShardInStrategyIndex);
      ScopeGuard scope_guard(vnode->scope());
      auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
      MS_EXCEPTION_IF_NULL(func_graph);
      if (IsEmbedShardNode(func_graph)) {
        MS_LOG(EXCEPTION) << "Nested use of shard (e.g shard(shard(...), ...) is not supported currently."
                          << " | FuncGraph: " << func_graph->ToString();
      }
      if (HasNestedMetaFg(func_graph)) {
        return false;
      }
      SetInputLayout(func_graph, in_strategy, device_num);
      SetParameterLayout(root, func_graph);
      return true;
    }
  }
  return false;
}

bool PynativeShard(const FuncGraphPtr &root, const opt::OptimizerPtr &) {
  bool change = false;
  MS_EXCEPTION_IF_NULL(root);
  auto parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kSemiAutoParallel && parallel_mode != kAutoParallel) {
    MS_LOG(INFO) << "Only auto_parallel and semi_auto_parallel support pynative shard";
    return change;
  }

  auto execution_mode = MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE);
  if (execution_mode != kPynativeMode) {
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
  MS_LOG(INFO) << "Leaving pynative shard";
  return change;
}
}  // namespace parallel
}  // namespace mindspore
