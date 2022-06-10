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
                                    const int64_t device_num, std::vector<std::vector<int64_t>> *default_strategy) {
  auto strategies = axes->value()->cast<ValueTuplePtr>()->value();
  size_t i = 0;
  for (auto &strategy : strategies) {
    auto node = nodes[i];
    if (strategy->isa<None>()) {
      auto node_size = common::AnfAlgo::GetOutputInferShape(node, 0).size();
      std::vector<int64_t> current_d_strategy(node_size, 1);
      if (node_size >= 1) {
        current_d_strategy[0] = device_num;
      }
      default_strategy->push_back(current_d_strategy);
    } else {
      auto current_strategy = GetValue<std::vector<int64_t>>(strategy);
      default_strategy->push_back(current_strategy);
    }
    i += 1;
  }
}

static bool CheckLayout(const ValueNodePtr &axes, bool *need_default_strategy, size_t *axes_size) {
  auto strategies = axes->value()->cast<ValueTuplePtr>()->value();
  for (auto &strategy : strategies) {
    *axes_size += 1;
    if (strategy->isa<None>()) {
      *need_default_strategy = true;
      continue;
    }
    if (!strategy->isa<ValueTuple>()) {
      return false;
    }
    auto elements = strategy->cast<ValueTuplePtr>()->value();

    for (auto &element : elements) {
      if (!element->isa<Int64Imm>()) {
        return false;
      }
    }
  }
  return true;
}

static Shapes GenerateParamStrategy(const Shapes &default_strategy, const CNodePtr &cnode) {
  OperatorInfoPtr op_info = CreateOperatorInfo(cnode);
  MS_EXCEPTION_IF_NULL(op_info);
  return op_info->GenerateParamStrategy(default_strategy);
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

static void GetOutputNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *output_nodes) {
  auto return_node = func_graph->get_return();
  auto real_return_node = return_node->cast<CNodePtr>()->input(1);
  while (IsPrimitiveCNode(real_return_node, prim::kPrimDepend)) {
    real_return_node = real_return_node->cast<CNodePtr>()->input(1);
  }
  if (!IsPrimitiveCNode(real_return_node, prim::kPrimMakeTuple)) {
    output_nodes->push_back(real_return_node);
  } else {
    auto cnode = real_return_node->cast<CNodePtr>();
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      output_nodes->push_back(cnode->input(i));
    }
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

static void SetOutputLayout(const FuncGraphPtr &func_graph, const AnfNodePtr &out_strategy, const int64_t &device_num) {
  auto out_strategy_tuple = out_strategy->cast<ValueNodePtr>();
  bool need_default_strategy = false;
  size_t out_strategy_size = 0;
  if (!IsValueNode<ValueTuple>(out_strategy_tuple) ||
      !CheckLayout(out_strategy_tuple, &need_default_strategy, &out_strategy_size)) {
    MS_LOG(EXCEPTION) << "out_strategy should be a two-dimension tuple";
  }
  std::vector<AnfNodePtr> output_nodes;
  GetOutputNodes(func_graph, &output_nodes);
  if (output_nodes.size() != out_strategy_size) {
    MS_LOG(EXCEPTION) << "Output number: " << output_nodes.size()
                      << " is not equal to out_strategy number: " << out_strategy_size;
  }

  std::vector<std::vector<int64_t>> output_strategy;
  if (need_default_strategy) {
    GenerateDefaultStrategy(out_strategy_tuple, output_nodes, device_num, &output_strategy);
  } else {
    output_strategy = GetValue<std::vector<std::vector<int64_t>>>(out_strategy_tuple->value());
  }
  MS_LOG(WARNING) << "The output strategy will be overwritten as data-parallel";

  for (size_t i = 0; i < output_nodes.size(); ++i) {
    auto node = output_nodes[i];
    auto output_shape = common::AnfAlgo::GetOutputInferShape(node, 0);
    if (output_shape.size() != output_strategy[i].size()) {
      MS_LOG(EXCEPTION) << "Output dimension: " << output_shape.size()
                        << " is not equal to out_strategy dimension: " << output_strategy[i].size() << " at index "
                        << i;
    }
    std::vector<ValuePtr> elements;
    elements.push_back(MakeValue(output_strategy[i]));
    auto prim = GetCNodePrimitive(node);
    auto attrs_temp = prim->attrs();
    ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
    attrs_temp[parallel::OUT_STRATEGY] = strategy;
  }
}

static Shapes GenerateDefaultStrategyForParam(const CNodePtr &cnode, const std::vector<AnfNodePtr> &parameters,
                                              const Shapes &input_strategy) {
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
      auto shape = current_input->Shape()->cast<abstract::ShapePtr>();
      auto dimension = shape->shape().size();
      std::vector<int64_t> default_strategy(dimension, 1);
      elements.push_back(default_strategy);
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

static void SetInputLayout(const FuncGraphPtr &func_graph, const AnfNodePtr &in_strategy, const int64_t &device_num) {
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
  std::set<CNodePtr> concerned_nodes;
  FuncGraphManagerPtr manager = func_graph->manager();
  auto parameters = func_graph->parameters();
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto parameter = parameters[i];
    if (parameter->cast<ParameterPtr>()->name() == "u" || parameter->cast<ParameterPtr>()->name() == "io") {
      continue;
    }
    auto output_shape = common::AnfAlgo::GetOutputInferShape(parameter, 0);
    if (output_shape.size() != input_strategy[i].size()) {
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
          concerned_nodes.insert(param_cnode);
        }
      }
    }
  }
  for (auto &cnode : concerned_nodes) {
    Shapes default_strategy = GenerateDefaultStrategyForParam(cnode, parameters, input_strategy);
    // Some operators has a special requirements for parallel strategy
    Shapes ret_strategy = GenerateParamStrategy(default_strategy, cnode);
    // Set in_strategy
    auto strategy = ShapesToValueTuplePtr(ret_strategy);
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(prim);
    auto attrs_temp = prim->attrs();
    attrs_temp[parallel::IN_STRATEGY] = strategy;
    (void)prim->SetAttrs(attrs_temp);
  }
}

static bool SetStrategyForShard(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                                const int64_t &device_num) {
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimShard)) {
      root->set_flag(kPynativeShard, true);
      auto cnode = node->cast<CNodePtr>();
      auto vnode = cnode->input(1)->cast<ValueNodePtr>();
      auto in_strategy = cnode->input(2);
      auto out_strategy = cnode->input(3);
      ScopeGuard scope_guard(vnode->scope());
      auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
      MS_EXCEPTION_IF_NULL(func_graph);
      if (HasNestedMetaFg(func_graph)) {
        return false;
      }
      SetInputLayout(func_graph, in_strategy, device_num);
      SetOutputLayout(func_graph, out_strategy, device_num);
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
