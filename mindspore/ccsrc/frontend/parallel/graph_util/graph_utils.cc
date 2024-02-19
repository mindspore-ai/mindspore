/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <vector>
#include <string>
#include "frontend/parallel/graph_util/graph_utils.h"
#include "include/common/utils/anfalgo.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/tensor_layout/prime_generator.h"
#include "mindspore/core/ir/primitive.h"
#include "mindspore/core/ir/func_graph.h"

namespace mindspore::parallel {
std::set<FuncGraphPtr> FindForwardGraphByRootNodes(const AnfNodeSet &root_all_nodes) {
  // J->CNode->Graph
  std::set<FuncGraphPtr> graph_set;
  for (auto &node : root_all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    if ((cnode->size() < 2) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    auto expect_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (expect_prim->name() != J && expect_prim->name() != SHARD) {
      continue;
    }
    if (IsValueNode<FuncGraph>(cnode->input(1))) {
      auto graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      MS_LOG(DEBUG) << "Find the forward graph success";
      (void)graph_set.insert(graph);
      auto manager = graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      auto graph_used = manager->func_graphs_used_total(graph);
      for (auto iter = graph_used.cbegin(); iter != graph_used.cend(); ++iter) {
        (void)graph_set.insert(*iter);
      }
    }
  }
  return graph_set;
}

AnfNodePtr GetAccuGrad(const std::vector<AnfNodePtr> &parameters, const std::string &weight_name) {
  for (auto &param : parameters) {
    if (!ParameterIsCloned(param)) {
      continue;
    }

    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (param_ptr->name().find(weight_name) != std::string::npos &&
        param_ptr->name().find(ACCU_GRADS) != std::string::npos) {
      MS_LOG(INFO) << "Find the accumulation grad node: " << param_ptr->name();
      return param;
    }
  }
  return nullptr;
}

std::vector<AnfNodePtr> CreateMirrorInput(const FuncGraphPtr &root, const Operator &op, const AnfNodePtr &node,
                                          const std::string &instance_name, const std::string &weight_name) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(root->manager());

  std::string op_name = op.first;
  OperatorArgs arg_forward = op.second;
  AnfNodePtr grad_accu = nullptr;

  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  int64_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();

  if (grad_accumulation_step > 1 || split_stage_num > 1) {
    auto parameters = root->parameters();
    grad_accu = GetAccuGrad(parameters, weight_name);
    if (!grad_accu && op_name == MICRO_STEP_ALL_GATHER) {
      MS_LOG(EXCEPTION) << "You should define `accu_grads` when use " << op_name << " parameter:" << weight_name;
    }
  }

  OperatorParams params = arg_forward.second;

  std::vector<AnfNodePtr> new_node_input;
  if (op_name == MIRROR_MINI_STEP_OPERATOR || op_name == MINI_STEP_ALL_GATHER ||
      op_name == MIRROR_MICRO_STEP_OPERATOR || op_name == MICRO_STEP_ALL_GATHER) {
    MS_EXCEPTION_IF_NULL(grad_accu);
    new_node_input = {node, grad_accu};
    MS_LOG(INFO) << "Insert the grad accumulation node as the mirror op's input";
  } else {
    new_node_input = {node};
  }

  if (!params.empty()) {
    for (auto &param : params) {
      AnfNodePtr val = NewValueNode(param.first.second);
      MS_EXCEPTION_IF_NULL(val);
      int64_t position = param.second;
      (void)new_node_input.insert(new_node_input.cbegin() + position - 1, val);
    }
  }

  new_node_input = ConvertToRealInputs(op_name, instance_name, new_node_input, arg_forward.first);
  // if the op have 'group' attr, set the rank list name for the op
  SetCommunicationOpGroupLabel(new_node_input);
  return new_node_input;
}

std::vector<AnfNodePtr> CreateInput(const Operator &op, const AnfNodePtr &node, const std::string &instance_name,
                                    const TensorRedistribution &tensor_redistribution) {
  MS_EXCEPTION_IF_NULL(node);
  OperatorArgs arg_forward = op.second;
  OperatorParams params = arg_forward.second;

  std::vector<AnfNodePtr> new_node_input = {node};
  if (!params.empty()) {
    if (tensor_redistribution.IsInited() && tensor_redistribution.IsAssembledStaticShape() && (op.first == RESHAPE)) {
      // TODO(liuchongming74): liuchongming74, Add input mapping to arg_forward.
      //  Should call `CreateInputsAccordingToAttrAndDynamicDimsMapping` here.
      // TODO(liuchongming74): liuchongming74, Traverse the second input of the reshape,
      //  and insert TupleGetItem to replace fake constant value.
      // AssembledDynamicDimsMapping is alias of std::map<int64_t, AnfNodePtr>.
      // Pair of AssembledDynamicDimsMapping is (fake_prime_dim_value, created_tuple_get_item_cnode).
      AssembledDynamicDimsMapping dyn_dims_mapping = tensor_redistribution.GetDynamicDimsMapping();
      MS_LOG(DEBUG) << "Insert TupleGetItem to replace fake constant value.";
      // 1. replace constant value by TupleGetItem.
      // 2. create MakeTuple to assemble TupleGetItem and constant value.
      // 3. push MakeTuple to new_node_input.
      std::vector<int64_t> prime_set(dyn_dims_mapping.size());
      size_t cnt = 0;
      for (auto &iter : dyn_dims_mapping) {
        prime_set[cnt++] = iter.first;
      }
      // Get shape attr from params.
      for (auto &param : params) {
        if (param.first.first != SHAPE) {
          continue;
        }
        AnfNodePtr val = NewValueNode(param.first.second);
        val->set_abstract(param.first.second->ToAbstract());
        Shape shape_vec = GetValue<Shape>(param.first.second);
        for (const auto v : shape_vec) {
          DecomposeDim decompose = DecomposeDim::Decompose(v, prime_set);
          // TODO(liuchongming74): liuchongming74, Add TupleGetItem to shape_inputs.
        }
        int64_t position = param.second;
        (void)new_node_input.insert(new_node_input.cbegin() + position - 1, val);
        // TODO(liuchongming74): liuchongming74, Insert MakeTuple to construct target shape of reshape.
        // InsertMakeTuple() can be used here.
      }
    } else {
      for (auto &param : params) {
        AnfNodePtr val = NewValueNode(param.first.second);
        MS_EXCEPTION_IF_NULL(val);
        val->set_abstract(param.first.second->ToAbstract());
        int64_t position = param.second;
        (void)new_node_input.insert(new_node_input.cbegin() + position - 1, val);
      }
    }
  }

  new_node_input = ConvertToRealInputs(op.first, instance_name, new_node_input, arg_forward.first);

  // if the op have 'group' attr, set the rank list name for the op
  SetCommunicationOpGroupLabel(new_node_input);
  return new_node_input;
}

void InsertNode(const Operator &op, const CNodePtr &node, size_t index, const AnfNodePtr &pre_node,
                const FuncGraphPtr &func_graph, const std::string &instance_name, const std::string &param_name,
                const FuncGraphPtr &root, const TensorRedistribution &tensor_redistribution) {
  // insert new node before the node
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  ScopePtr scope = node->scope();
  MS_EXCEPTION_IF_NULL(scope);
  std::vector<AnfNodePtr> node_input;
  if (root && !param_name.empty()) {
    node_input = CreateMirrorInput(root, op, pre_node, instance_name, param_name);
  } else {
    node_input = CreateInput(op, pre_node, instance_name, tensor_redistribution);
  }

  CNodePtr new_node = func_graph->NewCNode(node_input);
  MS_EXCEPTION_IF_NULL(new_node);
  if (instance_name.find(SPLIT_SENS) == std::string::npos) {
    new_node->set_in_forward_flag(true);  // mark forward flag
  }
  auto new_node_value = node_input[0]->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(new_node_value);
  PrimitivePtr new_node_prim = new_node_value->value()->cast<PrimitivePtr>();
  new_node_prim->set_instance_name(instance_name);
  new_node_prim->set_attr("keep_value_node_input", MakeValue(true));
  if (instance_name.find(NOT_RECOMPUTE) != std::string::npos) {
    new_node_prim->set_attr("recompute", MakeValue(false));
  } else if (instance_name.find(RECOMPUTE) != std::string::npos) {
    new_node_prim->set_attr("recompute", MakeValue(true));
  }

  auto primitive = common::AnfAlgo::GetCNodePrimitive(new_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (node->HasPrimalAttr(SEGMENT)) {
    primitive->AddAttr(SEGMENT, node->GetPrimalAttr(SEGMENT));
    new_node->AddPrimalAttr(SEGMENT, node->GetPrimalAttr(SEGMENT));
  }
  if (node->HasPrimalAttr(MICRO)) {
    new_node->AddPrimalAttr(MICRO, node->GetPrimalAttr(MICRO));
  }
  new_node->set_scope(scope);
  node_input[0]->set_scope(scope);
  if (instance_name.find(REDISTRIBUTION_OP) != std::string::npos) {
    new_node->AddPrimalAttr(kPrimalAttrForwardCommNodeUniqueId, MakeValue<std::string>(new_node->UniqueId()));
    if (node->HasPrimalAttr(MICRO)) {
      new_node->AddPrimalAttr(MICRO, node->GetPrimalAttr(MICRO));
    }
  }
  manager->SetEdge(node, SizeToInt(index), new_node);
  MS_LOG(INFO) << "Insert " << instance_name << " success";
}
}  // namespace mindspore::parallel
