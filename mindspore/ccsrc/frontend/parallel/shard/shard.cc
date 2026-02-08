/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/dynamic_shape/dynamic_shape.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/shard/shard.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/utils/comm_manager.h"
#include "include/utils/parallel_context.h"
#include "primitive/framework_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/other_ops.h"
#include "utils/ms_context.h"
#include "ir/tensor.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_v.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
namespace parallel {
namespace {
using ExpectFunc = std::function<bool(const CNodePtr &)>;
}
static std::vector<std::string> layout_keys = {DEVICE_MATRIX, TENSOR_MAP, INTERLEAVED_PARALLEL, ALIAS_NAME};

static void GenerateDefaultStrategy(const ValueNodePtr &axes, const std::vector<AnfNodePtr> &nodes,
                                    const size_t device_num, std::vector<std::vector<int64_t>> *default_strategy) {
  auto axes_value = axes->value()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(axes_value);
  auto strategies = axes_value->value();
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

static void PreCheckStrategy(const ValueNodePtr &axes, bool *need_default_strategy, size_t *axes_size) {
  MS_EXCEPTION_IF_NULL(axes);
  auto axes_value = axes->value()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(axes_value);
  auto strategies = axes_value->value();
  for (auto &strategy : strategies) {
    *axes_size += 1;
    if (strategy->isa<None>()) {
      *need_default_strategy = true;
      continue;
    }
  }
}

static void GetInputNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *input_nodes) {
  auto parameters = func_graph->parameters();
  for (auto &parameter : parameters) {
    if (parameter->abstract()->isa<abstract::AbstractMonad>()) {
      continue;
    }
    if (IsValueNode<mindspore::tensor::Tensor>(parameter) || parameter->isa<Parameter>()) {
      input_nodes->push_back(parameter);
    }
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

static bool CheckShapeStrategy(const std::vector<int64_t> &strategy, const std::vector<int64_t> &shapes,
                               const AnfNodePtr parameter) {
  auto symbols = GetNodeSymbol(parameter);
  auto tensor_symbol = symbols[0];
  for (size_t i = 0; i < strategy.size(); ++i) {
    auto devide_num = strategy[i];
    auto shape = shapes[i];
    if (shape % devide_num != 0 && !(shape == -1 && tensor_symbol[i].divisor % devide_num == 0)) {
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
  MS_EXCEPTION_IF_NULL(value_tuple_ptr);
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
    if (IsPrimitiveCNode(user_node, prim::kPrimAShardIdentity)) {
      auto cnode_primitive = GetCNodePrimitive(user_node);
      MS_EXCEPTION_IF_NULL(cnode_primitive);
      auto attrs = cnode_primitive->attrs();
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

static PrimitivePtr CreateNewPrimitive(const PrimitivePtr &prim) {
  if (prim->isa<PrimitivePy>()) {
    auto prim_py = prim->cast<PrimitivePyPtr>();
    MS_EXCEPTION_IF_NULL(prim_py);
    return std::make_shared<PrimitivePy>(*prim_py);
  } else {
    return std::make_shared<Primitive>(*prim);
  }
}

static void UpdateCNodeInput(const CNodePtr &cnode, const PrimitivePtr &new_prim) {
  ValuePtr new_prim_value = MakeValue(new_prim);
  ValueNodePtr new_prim_value_node = NewValueNode(new_prim_value);
  auto new_prim_anf_node = new_prim_value_node->cast<AnfNodePtr>();
  MS_EXCEPTION_IF_NULL(new_prim_anf_node);
  cnode->set_input(0, new_prim_anf_node);
}

static void SetStrategyToCNode(const CNodePtr &cnode, const Shapes &strategies) {
  PrimitivePtr prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  PrimitivePtr new_prim = CreateNewPrimitive(prim);
  auto attrs_temp = prim->attrs();
  attrs_temp[parallel::IN_STRATEGY] = ShapesToValueTuplePtr(strategies);
  (void)new_prim->SetAttrs(attrs_temp);
  UpdateCNodeInput(cnode, new_prim);
}

static void SetLayoutToCNode(const CNodePtr &cnode, const ValueTuplePtr &current_layout) {
  PrimitivePtr prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  PrimitivePtr new_prim = CreateNewPrimitive(prim);
  auto attrs_temp = prim->attrs();
  attrs_temp[parallel::IN_LAYOUT] = current_layout;
  (void)new_prim->SetAttrs(attrs_temp);
  UpdateCNodeInput(cnode, new_prim);
}

static bool IsItemTypeBool(const ValuePtr &item) { return item->type()->isa<Bool>(); }

static void updateStrategyType(const std::vector<ValuePtr> &layout_value_vector, std::string *strategy_type) {
  bool has_bool = std::any_of(layout_value_vector.begin(), layout_value_vector.end(), [](const ValuePtr &layout_item) {
    auto layout_item_value = layout_item->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(layout_item_value);
    auto layout_item_tuple = layout_item_value->value();
    return std::any_of(layout_item_tuple.begin(), layout_item_tuple.end(),
                       [](const auto &item) { return IsItemTypeBool(item); });
  });
  if (has_bool) {
    *strategy_type = LAYOUT;
  }
}

static CNodePtr InsertIdentityCNode(const AnfNodePtr &parameter, const FuncGraphPtr &func_graph,
                                    const CNodePtr &to_insert_cnode) {
  CNodePtr identity_cnode = nullptr;
  FuncGraphManagerPtr manager = func_graph->manager();
  // Setting strategy by insert identity CNode directly using inputs.
  // e.g TupleGetItem(parameter, index) -> func{identity{input_strategy[i], input_i}}.
  identity_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimAShardIdentity), parameter});
  AnfNodePtrList node_inputs_list(to_insert_cnode->inputs().begin(), to_insert_cnode->inputs().end());
  auto input_index = std::distance(node_inputs_list.begin(), std::find(node_inputs_list.begin(), node_inputs_list.end(),
                                                                       parameter->cast<AnfNodePtr>()));
  identity_cnode->set_abstract(parameter->abstract());
  manager->SetEdge(to_insert_cnode, input_index, identity_cnode);
  return identity_cnode;
}

static void GetInputLayout(std::vector<ValuePtr> *input_layout, const std::vector<ValuePtr> &layout_value_vector) {
  for (size_t i = 0; i < layout_value_vector.size(); ++i) {
    std::vector<std::pair<ValuePtr, ValuePtr>> dim_layout;
    auto layout_item = layout_value_vector[i]->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(layout_item);
    auto layout_item_tuple = layout_item->value();
    for (size_t j = 0; j < layout_item_tuple.size(); ++j) {
      auto item = layout_item_tuple[j];
      dim_layout.emplace_back(std::make_pair(MakeValue(layout_keys[j]), item));
    }
    input_layout->emplace_back(std::make_shared<ValueDictionary>(dim_layout));
  }
}

static void CheckInputStrategy(const std::vector<std::vector<int64_t>> &input_strategy, const AnfNodePtr parameter,
                               const size_t i) {
  if (!input_strategy.empty()) {
    // Verify that the user has set the valid layout, if the layout is generated by 'GenareteDefaultStrategy', ignored
    // its check.
    auto param_shape = common::AnfAlgo::GetOutputInferShape(parameter, 0);
    if (!input_strategy[i].empty() && param_shape.size() != input_strategy[i].size()) {
      MS_LOG_WITH_NODE(EXCEPTION, parameter)
        << "Input dimension: " << param_shape.size()
        << " is not equal to in_strategy dimension: " << input_strategy[i].size() << " at index " << i;
    }
    if (!CheckShapeStrategy(input_strategy[i], param_shape, parameter)) {
      MS_LOG_WITH_NODE(EXCEPTION, parameter) << "Check conformance between input strategy " << input_strategy[i]
                                             << "and tensor shape " << param_shape << "failed";
    }
  }
}

static AnfNodePtr FindCellLastNode(const FuncGraphPtr &func_graph) {
  auto ret = func_graph->get_return();
  auto input_ret = ret->input(1);
  MS_EXCEPTION_IF_NULL(input_ret);
  auto cnode_input_ret = input_ret->cast<CNodePtr>();
  auto filter_func = [](const CNodePtr &cnode) {
    bool filter = IsPrimitiveCNode(cnode, prim::kPrimDepend);
    return std::make_pair(filter, 1);
  };
  auto real_input = GetInputNodeWithFilter(cnode_input_ret, filter_func);
  MS_EXCEPTION_IF_NULL(real_input);
  auto cnode_real_input = real_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode_real_input);
  if (utils::isa<CNodePtr>(cnode_real_input)) {
    auto input0 = cnode_real_input->input(0);
    auto input0_cnode = input0->cast<CNodePtr>();
    if (input0_cnode != nullptr && input0_cnode->size() > 1) {
      auto input1 = input0_cnode->input(1);
      if (AnfAlgo::NodeValueIsFuncGraph(input1)) {
        return nullptr;
      }
    }
    return cnode_real_input;
  }
  return nullptr;
}

static void SetOutputLayout(const FuncGraphPtr &func_graph, const AnfNodePtr &out_strategy, const int64_t device_num) {
  auto out_strategy_tuple = out_strategy->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(out_strategy_tuple);
  bool need_default_strategy = false;
  size_t out_strategy_size = 0;
  PreCheckStrategy(out_strategy_tuple, &need_default_strategy, &out_strategy_size);

  if (out_strategy_size != 1) {
    MS_LOG(WARNING) << "Numbers of out strategy should be 1.";
    return;
  }

  // Get strategy in ValueTuple.
  std::vector<ValuePtr> layout_value_vector;
  auto &out_strategy_value = out_strategy_tuple->value();
  if (!out_strategy_value->isa<ValueTuple>()) {
    MS_LOG_WITH_NODE(EXCEPTION, out_strategy)
      << "Parse out_strategy to ValueType failed. Please check out_strategy format.";
  }
  auto layout_value_vector_ptr = out_strategy_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(layout_value_vector_ptr);
  layout_value_vector = layout_value_vector_ptr->value();

  // Check strategy type, it can only be "tuple" or "layout".
  std::string strategy_type = TUPLE;
  if (!need_default_strategy) {
    updateStrategyType(layout_value_vector, &strategy_type);
  }

  auto cell_last_node = FindCellLastNode(func_graph);
  if (cell_last_node == nullptr) {
    MS_LOG(WARNING) << "Out strategy set failed.";
    return;
  }
  // Get strategy either in input_strategy (given tuple) or input_layout (given layout).
  std::vector<std::vector<int64_t>> output_strategy;
  std::vector<ValuePtr> output_layout;
  if (strategy_type == TUPLE) {
    output_strategy = GetValue<std::vector<std::vector<int64_t>>>(out_strategy_tuple->value());
    if (!CheckDeviceNum(output_strategy, device_num)) {
      MS_LOG_WITH_NODE(EXCEPTION, out_strategy) << "check device number failed";
    }
    common::AnfAlgo::SetNodeAttr(parallel::OUT_STRATEGY, out_strategy_value, cell_last_node);
  }
  if (strategy_type == LAYOUT) {
    GetInputLayout(&output_layout, layout_value_vector);
    auto current_layout = std::make_shared<ValueTuple>(std::vector<ValuePtr>(output_layout));
    common::AnfAlgo::SetNodeAttr(parallel::OUT_LAYOUT, current_layout, cell_last_node);
  }
  common::AnfAlgo::SetNodeAttr(parallel::CELL_SHARD_OP, MakeValue(True), cell_last_node);
}

static void SetInputLayout(const FuncGraphPtr &func_graph, const AnfNodePtr &in_strategy, const int64_t device_num) {
  auto in_strategy_tuple = in_strategy->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(in_strategy_tuple);
  bool need_default_strategy = false;
  size_t in_strategy_size = 0;
  PreCheckStrategy(in_strategy_tuple, &need_default_strategy, &in_strategy_size);

  // Get input nodes.
  std::vector<AnfNodePtr> input_nodes;
  GetInputNodes(func_graph, &input_nodes);
  if (input_nodes.size() != in_strategy_size) {
    MS_LOG_WITH_NODE(EXCEPTION, in_strategy)
      << "Input numbers: " << input_nodes.size() << " is not equal to in_strategy numbers: " << in_strategy_size;
  }

  // Get strategy in ValueTuple.
  std::vector<ValuePtr> layout_value_vector;
  auto &in_strategy_value = in_strategy_tuple->value();
  if (!in_strategy_value->isa<ValueTuple>()) {
    MS_LOG_WITH_NODE(EXCEPTION, in_strategy)
      << "Parse in_strategy to ValueType failed. Please check in_strategy format.";
  }
  auto layout_value_vector_ptr = in_strategy_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(layout_value_vector_ptr);
  layout_value_vector = layout_value_vector_ptr->value();

  // Check strategy type, it can only be "tuple" or "layout".
  std::string strategy_type = TUPLE;
  if (!need_default_strategy) {
    updateStrategyType(layout_value_vector, &strategy_type);
  }

  // Get strategy either in input_strategy (given tuple) or input_layout (given layout).
  std::vector<std::vector<int64_t>> input_strategy;
  std::vector<ValuePtr> input_layout;
  if (strategy_type == TUPLE) {
    if (need_default_strategy) {
      GenerateDefaultStrategy(in_strategy_tuple, input_nodes, device_num, &input_strategy);
    } else {
      input_strategy = GetValue<std::vector<std::vector<int64_t>>>(in_strategy_tuple->value());
    }
    if (!CheckDeviceNum(input_strategy, device_num)) {
      MS_LOG_WITH_NODE(EXCEPTION, in_strategy) << "check device number failed";
    }
  }
  if (strategy_type == LAYOUT) {
    GetInputLayout(&input_layout, layout_value_vector);
  }

  // Insert strategy into function graph.
  FuncGraphManagerPtr manager = func_graph->manager();
  auto parameters = func_graph->parameters();

  for (size_t i = 0; i < parameters.size(); ++i) {
    auto parameter = parameters[i];
    if (parameter->abstract()->isa<abstract::AbstractMonad>()) {
      continue;
    }
    CheckInputStrategy(input_strategy, parameter, i);

    auto to_insert_nodes_set = manager->node_users()[parameter];

    if (to_insert_nodes_set.empty()) {
      MS_LOG(INFO) << "For input: \"" << parameter->fullname_with_scope()
                   << "\", failed to find node to insert strategy. This input may not be used in computation, skip it.";
      continue;
    }
    for (auto &node : to_insert_nodes_set) {
      auto to_insert_cnode = node.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(to_insert_cnode);
      if (IsSettingStrategyByInsertIdentity(func_graph, to_insert_cnode, parameter->fullname_with_scope())) {
        continue;
      }
      CNodePtr identity_cnode = InsertIdentityCNode(parameter, func_graph, to_insert_cnode);
      int64_t layout_index = static_cast<int64_t>(i);
      if (!input_strategy.empty()) {
        Shapes current_layout = {input_strategy[layout_index]};
        SetStrategyToCNode(identity_cnode, current_layout);
        MS_LOG(INFO) << "Succeed to set strategy " << current_layout << "to node "
                     << to_insert_cnode->fullname_with_scope();
      }
      if (!input_layout.empty()) {
        auto current_layout = std::make_shared<ValueTuple>(std::vector<ValuePtr>({input_layout[layout_index]}));
        SetLayoutToCNode(identity_cnode, current_layout);
        MS_LOG(INFO) << "Succeed to set layout " << current_layout->ToString() << "to node "
                     << to_insert_cnode->fullname_with_scope();
      }
    }
  }
}

static void SetParameterLayout(const FuncGraphPtr &root) {
  FuncGraphManagerPtr manager = root->manager();
  auto root_parameters = root->parameters();
  for (const auto &param : root_parameters) {
    auto parameter = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(parameter);
    auto param_info = parameter->param_info();
    if (param_info == nullptr) {
      // Do not set param_strategy, skip it.
      continue;
    }
    std::vector<int64_t> param_strategy = param_info->param_strategy();
    std::vector<int64_t> device_matrix = param_info->device_matrix();
    std::vector<int64_t> tensor_map = param_info->tensor_map();
    bool interleaved_parallel = param_info->interleaved_parallel();
    std::vector<std::string> alias_name = param_info->alias_name();
    auto param_name = param_info->name();
    std::string param_type;
    if (!param_strategy.empty()) {
      param_type = TUPLE;
    } else if (!device_matrix.empty()) {
      param_type = LAYOUT;
    } else {
      continue;
    }
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
      auto identity_cnode = cur_graph->NewCNode({NewValueNode(prim::kPrimAShardIdentity), load_cnode});
      auto load_cnode_abstract = load_cnode->abstract();
      MS_EXCEPTION_IF_NULL(load_cnode_abstract);
      identity_cnode->set_abstract(load_cnode_abstract->Clone());
      (void)local_manager->Replace(load_cnode, identity_cnode);
      if (param_type == TUPLE) {
        Shapes current_layout = {param_strategy};
        SetStrategyToCNode(identity_cnode, current_layout);
        MS_LOG(INFO) << "The layout of \"" << param_name << "\" has been set to " << load_cnode->fullname_with_scope()
                     << ". Current strategies is " << current_layout;
      }
      if (param_type == LAYOUT) {
        std::vector<std::pair<ValuePtr, ValuePtr>> layout_map;
        layout_map.emplace_back(std::make_pair(MakeValue(DEVICE_MATRIX), MakeValue(device_matrix)));
        layout_map.emplace_back(std::make_pair(MakeValue(TENSOR_MAP), MakeValue(tensor_map)));
        layout_map.emplace_back(std::make_pair(MakeValue(INTERLEAVED_PARALLEL), MakeValue(interleaved_parallel)));
        layout_map.emplace_back(std::make_pair(MakeValue(ALIAS_NAME), MakeValue(alias_name)));
        ValuePtr layout_dict = std::make_shared<ValueDictionary>(layout_map);
        auto current_layout = std::make_shared<ValueTuple>(std::vector<ValuePtr>({layout_dict}));
        SetLayoutToCNode(identity_cnode, current_layout);
        MS_LOG(INFO) << "The layout of \"" << param_name << "\" has been set to " << load_cnode->fullname_with_scope()
                     << ". Current strategies is " << current_layout->ToString();
      }
    }
  }
}

void CheckVmap(AnfNodePtr node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AnfNodePtr value_node = cnode->input(1);
  auto func_graph = GetValueNode<FuncGraphPtr>(value_node);
  if (func_graph == nullptr) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode) << "Unexpected meta function graph node:" << cnode->DebugString();
  }
  if (parallel::IsEmbedShardNode(func_graph)) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode)
      << "The usage of vmap nested shard (e.g vmap(shard)) is not supported currently. Current FuncGraph: "
      << func_graph->ToString();
  }
}

static bool SetStrategyForShard(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                                const int64_t device_num) {
  constexpr size_t kShardFnIndex = 1;
  constexpr size_t kShardInStrategyIndex = 2;
  constexpr size_t kShardOutStrategyIndex = 3;
  auto set_success = false;
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimVmap)) {
      CheckVmap(node);
    }
    if (IsPrimitiveCNode(node, prim::kPrimShard)) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto vnode = cnode->input(kShardFnIndex)->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(vnode);
      auto in_strategy = cnode->input(kShardInStrategyIndex);
      auto out_strategy = cnode->input(kShardOutStrategyIndex);
      ScopeGuard scope_guard(vnode->scope());
      auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
      MS_EXCEPTION_IF_NULL(func_graph);
      if (func_graph->has_flag(kSharded)) {
        continue;
      }
      // get input nodes
      SetInputLayout(func_graph, in_strategy, device_num);
      auto output_value_ptr = out_strategy->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(output_value_ptr);
      auto output_value = output_value_ptr->value();
      if (!output_value->isa<None>()) {
        SetOutputLayout(func_graph, out_strategy, device_num);
      }
      func_graph->set_flag(kSharded, true);
      set_success = true;
    }
    if (IsPrimitiveCNode(node, prim::kPrimReshard)) {
      // Get reshard attributes, e.g., in_layout/in_strategy.
      auto reshard_cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(reshard_cnode);
      auto reshard_prim = GetCNodePrimitive(reshard_cnode);
      MS_EXCEPTION_IF_NULL(reshard_prim);
      auto attrs = reshard_prim->attrs();

      // New a identity prim and set the attributes the same as reshard prim.
      auto new_identity_prim = std::make_shared<Primitive>(prim::kPrimAShardIdentity->name());
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

  // assertion
  auto parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kSemiAutoParallel && parallel_mode != kAutoParallel) {
    MS_LOG(DEBUG) << "Only auto_parallel and semi_auto_parallel support shard.";
    return change;
  }

  auto device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target != kAscendDevice && device_target != kGPUDevice) {
    MS_LOG(EXCEPTION) << "'Shard' now only supports 'Ascend' and 'GPU'";
  }

  auto strategy_search_mode = ParallelContext::GetInstance()->strategy_search_mode();
  if (parallel_mode == kAutoParallel && strategy_search_mode != kShardingPropagation) {
    MS_LOG(EXCEPTION)
      << "'search_mode' must be 'sharding_propagation' for 'Shard' when the 'parallel_mode' is 'auto_parallel.'";
  }

  if (IsParallelDynamicShape(root)) {
    MS_LOG(WARNING) << "Sharding does not support dynamic shape, will be ignored in the following procedures.";
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

static std::vector<std::pair<std::string, ValuePtr>> GetPrimAttrFromAddAttrNode(const AnfNodePtr &addattr_node) {
  const size_t add_attr_attr_pair_index = 2;
  std::vector<std::pair<std::string, ValuePtr>> kv_pair_vec;
  if (!addattr_node || !addattr_node->isa<CNode>() || !IsPrimitiveCNode(addattr_node, prim::kPrimAddAttr)) {
    return kv_pair_vec;
  }
  CNodePtr addattr_cnode = addattr_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(addattr_cnode);
  AnfNodePtr attr_pair_node = addattr_cnode->input(add_attr_attr_pair_index);
  MS_EXCEPTION_IF_NULL(attr_pair_node);
  ValueNodePtr attr_pair_vnode = attr_pair_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(attr_pair_vnode);
  MS_EXCEPTION_IF_NULL(attr_pair_vnode->value());
  if (!attr_pair_vnode->value()->isa<ValueTuple>()) {
    MS_LOG_WITH_NODE(EXCEPTION, addattr_node) << "Parse attr_pair to ValueTuple failed. Please check attr_pair format.";
  }
  ValueTuplePtr attr_value_tuple = attr_pair_vnode->value()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(attr_value_tuple);
  const std::vector<ValuePtr> attr_pair_vals = attr_value_tuple->value();
  MS_LOG(INFO) << "In HandleAddAttr, Parse attr pairs success.";
  for (const ValuePtr &val : attr_pair_vals) {
    if (!val->isa<ValueTuple>()) {
      continue;
    }
    auto pair_value_tuple = GetValueSequence(val);
    if (pair_value_tuple.size() != kSizeTwo) {
      MS_LOG(EXCEPTION) << "attr key-value pair length should be 2, but got: " << pair_value_tuple.size();
    }
    auto primitive_key_value = pair_value_tuple.at(kIndex0);
    auto primitive_value_value = pair_value_tuple.at(kIndex1);
    std::string prim_key = GetValue<std::string>(primitive_key_value);
    kv_pair_vec.emplace_back(prim_key, primitive_value_value);
    MS_LOG(INFO) << "Obtain key: " << prim_key << ", value: " << primitive_value_value->ToString();
  }
  return kv_pair_vec;
}

static void GetNestedAddAttrPrims(const AnfNodePtr &addattr_node,
                                  std::vector<std::pair<std::string, ValuePtr>> *kv_pairs) {
  if (!addattr_node || !addattr_node->isa<CNode>() || !IsPrimitiveCNode(addattr_node, prim::kPrimAddAttr)) {
    return;
  }
  const size_t add_attr_fn_index = 1;
  CNodePtr addattr_cnode = addattr_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(addattr_cnode);
  AnfNodePtr fn_node = addattr_cnode->input(add_attr_fn_index);
  MS_EXCEPTION_IF_NULL(fn_node);
  ValueNodePtr fn_vnode = fn_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(fn_vnode);
  FuncGraphPtr graph_need_inline = GetValueNode<FuncGraphPtr>(fn_vnode);
  MS_EXCEPTION_IF_NULL(graph_need_inline);
  MS_LOG(DEBUG) << "Find graph need inline: " << graph_need_inline->ToString();
  // Collect prim attr from addattr_node into kv_pairs
  auto kv_vec = GetPrimAttrFromAddAttrNode(addattr_node);
  std::copy(kv_vec.begin(), kv_vec.end(), std::back_inserter(*kv_pairs));
  auto addattr_iter = std::find_if(graph_need_inline->nodes().begin(), graph_need_inline->nodes().end(),
                                   [](const AnfNodePtr &node) { return IsPrimitiveCNode(node, prim::kPrimAddAttr); });
  if (addattr_iter != graph_need_inline->nodes().end()) {
    GetNestedAddAttrPrims(*addattr_iter, kv_pairs);
  }
  return;
}

static void GetNodesToTagAttr(const AnfNodePtr &addattr_node, AnfNodePtrList *nodes_to_tag) {
  if (!addattr_node || !addattr_node->isa<CNode>() || !IsPrimitiveCNode(addattr_node, prim::kPrimAddAttr)) {
    return;
  }
  const size_t add_attr_fn_index = 1;
  CNodePtr addattr_cnode = addattr_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(addattr_cnode);
  AnfNodePtr fn_node = addattr_cnode->input(add_attr_fn_index);
  MS_EXCEPTION_IF_NULL(fn_node);
  ValueNodePtr fn_vnode = fn_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(fn_vnode);
  FuncGraphPtr graph_need_inline = GetValueNode<FuncGraphPtr>(fn_vnode);
  MS_EXCEPTION_IF_NULL(graph_need_inline);
  auto addattr_iter = std::find_if(graph_need_inline->nodes().begin(), graph_need_inline->nodes().end(),
                                   [](const AnfNodePtr &node) { return IsPrimitiveCNode(node, prim::kPrimAddAttr); });
  graph_need_inline->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, false);
  if (addattr_iter != graph_need_inline->nodes().end()) {
    GetNodesToTagAttr(*addattr_iter, nodes_to_tag);
  } else {
    // collect nodes to tag
    std::copy_if(graph_need_inline->nodes().begin(), graph_need_inline->nodes().end(),
                 std::back_inserter(*nodes_to_tag), [](const AnfNodePtr &node) {
                   return !IsPrimitiveCNode(node, prim::kPrimDepend) &&
                          !IsPrimitiveCNode(node, prim::kPrimUpdateState) && !IsPrimitiveCNode(node, prim::kPrimReturn);
                 });
  }
  return;
}

bool HandleAddAttr(const FuncGraphPtr &root, const opt::OptimizerPtr &) {
  MS_EXCEPTION_IF_NULL(root);
  MS_LOG(INFO) << "HandleAddAttr pass start.";
  bool change = false;
  AnfNodePtr ret = root->get_return();
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimAddAttr)) {
      continue;
    }
    change = true;
    std::vector<std::pair<std::string, ValuePtr>> kv_pair_vec;
    AnfNodePtrList nodes_to_tag;
    (void)GetNestedAddAttrPrims(node, &kv_pair_vec);
    (void)GetNodesToTagAttr(node, &nodes_to_tag);
    for (AnfNodePtr &node_to_tag : nodes_to_tag) {
      if (!node_to_tag->isa<CNode>()) {
        continue;
      }
      CNodePtr cnode_to_tag = node_to_tag->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode_to_tag);
      PrimitivePtr prim = GetCNodePrimitive(cnode_to_tag);
      if (!prim) {
        continue;
      }
      prim = prim->Clone();
      MS_EXCEPTION_IF_NULL(prim);
      for (const auto &p : kv_pair_vec) {
        prim->AddAttr(p.first, p.second);
      }
      FuncGraphPtr fg = node->func_graph();
      MS_EXCEPTION_IF_NULL(fg);
      FuncGraphManagerPtr fg_mng = fg->manager();
      MS_EXCEPTION_IF_NULL(fg_mng);
      (void)fg_mng->SetEdge(cnode_to_tag, kIndex0, NewValueNode(prim));
    }
  }
  return change;
}

}  // namespace parallel
}  // namespace mindspore
