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
static void GenerateDefaultStrategy(const ValueNodePtr &axes, const std::vector<AnfNodePtr> &nodes,
                                    std::vector<std::vector<int64_t>> *default_strategy) {
  auto strategies = axes->value()->cast<ValueTuplePtr>()->value();
  size_t i = 0;
  for (auto &strategy : strategies) {
    auto node = nodes[i];
    if (strategy->isa<None>()) {
      (void)default_strategy->emplace_back(Shape());
    } else {
      (void)default_strategy->emplace_back(GetValue<Shape>(strategy));
    }
    i += 1;
  }
}

// Generate strategies like ((), (), ..., ())
Shapes GenerateEmptyStrategies(const CNodePtr &cnode) {
  size_t input_size = cnode->size() - 1;
  Shapes ret_strategies(input_size, Shape());
  return ret_strategies;
}

static bool CheckOneDimensionalIntTuple(const ValuePtr &value_ptr) {
  if (!value_ptr->isa<ValueTuple>()) {
    return false;
  }
  auto elements = value_ptr->cast<ValueTuplePtr>()->value();
  for (auto &element : elements) {
    if (!element->isa<Int64Imm>()) {
      return false;
    }
  }
  return true;
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

static Shapes GenerateFullStrategy(const Shapes &current_strategy, const CNodePtr &cnode) {
  OperatorInfoPtr op_info = CreateOperatorInfo(cnode);
  MS_EXCEPTION_IF_NULL(op_info);
  return op_info->GenerateFullStrategy(current_strategy);
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

static bool CheckDeviceNum(const std::vector<std::vector<int64_t>> &strategies, const int64_t &device_num) {
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

static Shapes GenerateDefaultStrategyForParam(const CNodePtr &cnode, const Shapes &input_strategy) {
  auto current_inputs = cnode->inputs();
  Shapes elements;
  for (size_t i = 1; i < current_inputs.size(); ++i) {
    auto current_input = current_inputs[i];
    if (current_input->isa<ValueNode>()) {
      auto current_value = current_input->cast<ValueNodePtr>()->value();
      if (!current_value->isa<mindspore::tensor::Tensor>()) {
        continue;
      }
    }
    if (IsPrimitiveCNode(current_input, prim::kPrimTupleGetItem)) {
      auto tuple_getitem_cnode = current_input->cast<CNodePtr>();
      auto tuple_index = tuple_getitem_cnode->input(2);
      auto value_node = tuple_index->cast<ValueNodePtr>();
      auto index = GetValue<int64_t>(value_node->value());
      elements.push_back(input_strategy[index]);
    } else {
      (void)elements.emplace_back(Shape());
    }
  }
  return elements;
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

static std::set<CNodePtr> SetInputLayout(const FuncGraphPtr &func_graph, const AnfNodePtr &in_strategy,
                                         const int64_t &device_num) {
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
    GenerateDefaultStrategy(in_strategy_tuple, input_nodes, &input_strategy);
  } else {
    input_strategy = GetValue<std::vector<std::vector<int64_t>>>(in_strategy_tuple->value());
  }
  if (!CheckDeviceNum(input_strategy, device_num)) {
    MS_LOG(EXCEPTION) << "check device number failed";
  }
  std::set<CNodePtr> concerned_nodes;
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
    for (auto &param_pair : param_sub_set) {
      auto tuple_getitem_nodes = manager->node_users()[param_pair.first];
      for (auto &tuple_getitem_node : tuple_getitem_nodes) {
        auto nodes = manager->node_users()[tuple_getitem_node.first];
        for (auto &node : nodes) {
          CNodePtr param_cnode = node.first->cast<CNodePtr>();
          (void)concerned_nodes.insert(param_cnode);
        }
      }
    }
  }
  for (auto &cnode : concerned_nodes) {
    Shapes ret_strategy = GenerateDefaultStrategyForParam(cnode, input_strategy);
    // Set in_strategy
    auto strategy = ShapesToValueTuplePtr(ret_strategy);
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(prim);
    auto attrs_temp = prim->attrs();
    attrs_temp[parallel::IN_STRATEGY] = strategy;
    PrimitivePyPtr prim_py = dyn_cast<PrimitivePy>(prim);
    PrimitivePtr new_prim = std::make_shared<PrimitivePy>(*prim_py);
    (void)new_prim->SetAttrs(attrs_temp);
    ValuePtr new_prim_value = MakeValue(new_prim);
    ValueNodePtr new_prim_value_node = NewValueNode(new_prim_value);
    AnfNodePtr new_prim_anf_node = new_prim_value_node->cast<AnfNodePtr>();
    MS_EXCEPTION_IF_NULL(new_prim_anf_node);
    cnode->set_input(0, new_prim_anf_node);
  }
  return concerned_nodes;
}

static bool CheckParamLayout(const ValueNodePtr &layout) {
  auto parameter_plan = layout->value()->cast<ValueTuplePtr>()->value();
  for (const auto &p : parameter_plan) {
    if (!p->isa<ValueTuple>()) {
      MS_LOG(EXCEPTION) << "each item in layout must be tuple";
    }
    auto p_tuple = p->cast<ValueTuplePtr>()->value();
    auto param_name = p_tuple.at(kIndex0);
    if (!param_name->isa<StringImm>()) {
      MS_LOG(ERROR) << "param_name must be a string";
      return false;
    }
    auto param_layout = p_tuple.at(kIndex1);
    if (!CheckOneDimensionalIntTuple(param_layout)) {
      MS_LOG(ERROR) << "param_layout must be a one-dimensional integer tuple";
      return false;
    }
  }
  return true;
}

AnfNodePtr SearchParamByName(const std::vector<AnfNodePtr> &parameter_list, std::string param_name) {
  const std::string prefix = "self.";
  if (param_name.rfind(prefix) == 0) {
    param_name = param_name.substr(prefix.size(), param_name.size() - prefix.size());
  }

  for (auto parameter : parameter_list) {
    auto parameter_ptr = parameter->cast<ParameterPtr>();
    if (parameter_ptr != nullptr && parameter_ptr->name() == param_name) {
      return parameter;
    }
  }
  return nullptr;
}

static std::set<CNodePtr> SetParameterLayout(const FuncGraphPtr &root, const FuncGraphPtr &func_graph,
                                             const AnfNodePtr &parameter_plan) {
  auto parameter_plan_vnode = parameter_plan->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(parameter_plan_vnode);
  if (parameter_plan_vnode->value()->isa<None>()) {
    MS_LOG(INFO) << "parameter_plan is none, no need to set layout for parameter";
    return std::set<CNodePtr>();
  }
  if (!IsValueNode<ValueTuple>(parameter_plan_vnode) || !CheckParamLayout(parameter_plan_vnode)) {
    MS_LOG(EXCEPTION) << "parameter_plan should be a tuple while each element must be (string, 1-D tuple)";
  }
  auto parameter_plan_list = parameter_plan_vnode->value()->cast<ValueTuplePtr>()->value();
  FuncGraphManagerPtr manager = func_graph->manager();
  auto root_parameters = root->parameters();
  std::set<CNodePtr> concerned_cnode;
  for (auto p : parameter_plan_list) {
    auto p_tuple = p->cast<ValueTuplePtr>()->value();
    auto param_name = GetValue<std::string>(p_tuple[0]);
    auto parameter = SearchParamByName(root_parameters, param_name);
    if (parameter == nullptr) {
      MS_LOG(WARNING) << "Parameter \'" << param_name << "\' is not exist, ignore it.";
      continue;
    }
    AnfNodeIndexSet users = manager->node_users()[parameter];
    std::queue<std::pair<AnfNodePtr, int>> to_solve_list;
    (void)std::for_each(users.begin(), users.end(),
                        [&to_solve_list](const std::pair<AnfNodePtr, int> &user) { to_solve_list.push(user); });

    while (!to_solve_list.empty()) {
      auto user = to_solve_list.front();
      to_solve_list.pop();
      CNodePtr cnode = user.first->cast<CNodePtr>();
      // If the cnode is not a splittable operator, apply strategy to the next cnode
      if (!IsSplittableOperator(GetPrimName(cnode))) {
        auto tmp_users = manager->node_users()[cnode];
        (void)std::for_each(tmp_users.begin(), tmp_users.end(),
                            [&to_solve_list](const std::pair<AnfNodePtr, int> &user) { to_solve_list.push(user); });
        continue;
      }

      PrimitivePtr prim = GetCNodePrimitive(cnode);
      MS_EXCEPTION_IF_NULL(prim);
      auto attrs = prim->attrs();
      if (attrs.count(parallel::IN_STRATEGY) == 0) {
        auto empty_strategies = GenerateEmptyStrategies(cnode);
        attrs[parallel::IN_STRATEGY] = ShapesToValueTuplePtr(empty_strategies);
      }
      auto current_strategies = ValueTuplePtrToShapes(attrs[parallel::IN_STRATEGY]->cast<ValueTuplePtr>());
      auto param_layout = GetValue<Shape>(p_tuple[kIndex1]);
      // If a layout has been set, skip it.
      if (current_strategies[user.second - 1] != Shape()) {
        MS_LOG(WARNING) << "For " << cnode->fullname_with_scope() << ", the " << user.second
                        << "th strategy has been set to " << current_strategies[user.second - 1] << ", current setting "
                        << param_layout << " will be ignored.";
        continue;
      }
      current_strategies[user.second - 1] = param_layout;
      attrs[parallel::IN_STRATEGY] = ShapesToValueTuplePtr(current_strategies);
      PrimitivePyPtr prim_py = dyn_cast<PrimitivePy>(prim);
      MS_EXCEPTION_IF_NULL(prim_py);
      PrimitivePtr new_prim = std::make_shared<PrimitivePy>(*prim_py);
      (void)new_prim->SetAttrs(attrs);
      ValueNodePtr new_prim_value_node = NewValueNode(MakeValue(new_prim));
      AnfNodePtr new_prim_anf_node = new_prim_value_node->cast<AnfNodePtr>();
      MS_EXCEPTION_IF_NULL(new_prim_anf_node);
      cnode->set_input(0, new_prim_anf_node);
      (void)concerned_cnode.insert(cnode);
      MS_LOG(INFO) << "The layout of \"" << param_name << "\" has been set to the " << user.second << "th of "
                   << cnode->fullname_with_scope() << "'s in_strategy. Current strategies is " << current_strategies;
    }
  }
  return concerned_cnode;
}

void CompleteConcernedCNodeStrategies(std::set<CNodePtr> concerned_cnode) {
  for (auto cnode : concerned_cnode) {
    PrimitivePtr prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    auto attrs = prim->attrs();
    Shapes current_strategies = ValueTuplePtrToShapes(attrs[parallel::IN_STRATEGY]->cast<ValueTuplePtr>());
    Shapes full_strategies = GenerateFullStrategy(current_strategies, cnode);
    attrs[parallel::IN_STRATEGY] = ShapesToValueTuplePtr(full_strategies);
    (void)prim->SetAttrs(attrs);
    MS_LOG(INFO) << cnode->fullname_with_scope() << ": Completion strategies success. " << current_strategies << " -> "
                 << full_strategies << "(origin_strategies -> completion_strategies)";
  }
}

static bool SetStrategyForShard(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                                const int64_t &device_num) {
  constexpr size_t kShardFnIndex = 1;
  constexpr size_t kShardInStrategyIndex = 2;
  constexpr size_t kShardParameterPlanIndex = 4;
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimShard)) {
      auto cnode = node->cast<CNodePtr>();
      auto vnode = cnode->input(kShardFnIndex)->cast<ValueNodePtr>();
      auto in_strategy = cnode->input(kShardInStrategyIndex);
      auto parameter_plan = cnode->input(kShardParameterPlanIndex);
      ScopeGuard scope_guard(vnode->scope());
      auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
      MS_EXCEPTION_IF_NULL(func_graph);
      if (HasNestedMetaFg(func_graph)) {
        return false;
      }
      std::set<CNodePtr> concerned_cnode;
      auto input_concerned_cnode = SetInputLayout(func_graph, in_strategy, device_num);
      auto parameter_concerned_cnode = SetParameterLayout(root, func_graph, parameter_plan);
      (void)std::set_union(input_concerned_cnode.begin(), input_concerned_cnode.end(),
                           parameter_concerned_cnode.begin(), parameter_concerned_cnode.end(),
                           std::inserter(concerned_cnode, concerned_cnode.end()));
      CompleteConcernedCNodeStrategies(concerned_cnode);
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

  if (!ParallelContext::GetInstance()->device_num_is_set()) {
    MS_LOG(EXCEPTION) << "device_num must be set when use shard function";
  }

  if (ParallelInit() != SUCCESS) {
    MS_LOG(EXCEPTION) << "parallel init failed.";
  }

  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  auto device_num_shard = parallel::ParallelContext::GetInstance()->device_num();
  change = SetStrategyForShard(root, all_nodes, device_num_shard);
  MS_LOG(INFO) << "Leaving pynative shard";
  return change;
}
}  // namespace parallel
}  // namespace mindspore
