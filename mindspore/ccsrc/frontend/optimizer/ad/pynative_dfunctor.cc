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

#include "frontend/optimizer/ad/pynative_dfunctor.h"
#include "ir/func_graph_cloner.h"

namespace mindspore {
namespace ad {
tensor::TensorPtr PynativeDFunctor::GenNewTensorInner(const TypePtr &type_elem, const BaseShapePtr &shape_elem) {
  MS_EXCEPTION_IF_NULL(type_elem);
  MS_EXCEPTION_IF_NULL(shape_elem);
  auto shape = shape_elem->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto tensor_type = type_elem->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(type);
  return std::make_shared<tensor::Tensor>(type->type_id(), shape->shape());
}

ValueNodePtr PynativeDFunctor::GenNewTensor(const CNodePtr &cnode_morph) {
  MS_EXCEPTION_IF_NULL(cnode_morph);
  if (cnode_morph->forward().first != nullptr) {
    return cnode_morph->forward().first;
  }
  if (IsPrimitiveCNode(cnode_morph, prim::kPrimUpdateState)) {
    ValueNodePtr out_vnode = NewValueNode(std::make_shared<UMonad>());
    out_vnode->set_abstract(std::make_shared<abstract::AbstractUMonad>());
    return out_vnode;
  }
  // Function used to generate value node
  auto gen_output_value_node = [](const ValuePtr &value) -> ValueNodePtr {
    MS_EXCEPTION_IF_NULL(value);
    auto v_node = NewValueNode(value);
    v_node->set_abstract(value->ToAbstract()->Broaden());
    return v_node;
  };
  // Create output value node for CNode
  auto cnode_shape = cnode_morph->Shape();
  MS_EXCEPTION_IF_NULL(cnode_shape);
  auto cnode_type = cnode_morph->Type();
  MS_EXCEPTION_IF_NULL(cnode_type);
  if (cnode_type->isa<Tuple>()) {
    auto tuple_shape = cnode_shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape);
    auto tuple_type = cnode_type->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_type);
    size_t output_num = tuple_type->elements().size();
    MS_EXCEPTION_IF_CHECK_FAIL(output_num != 0, "No output value.");
    std::vector<ValuePtr> output_values;
    for (size_t i = 0; i < output_num; ++i) {
      auto shape_elem = tuple_shape->shape()[i];
      auto type_elem = tuple_type->elements()[i];
      output_values.push_back(GenNewTensorInner(type_elem, shape_elem));
    }
    auto value_tuple = std::make_shared<ValueTuple>(output_values);
    return gen_output_value_node(value_tuple);
  } else if (cnode_type->isa<TensorType>()) {
    auto tensor_value = GenNewTensorInner(cnode_type, cnode_shape);
    return gen_output_value_node(tensor_value);
  } else if (cnode_shape->isa<abstract::NoShape>()) {
    ShapeVector NoShape;
    auto tensor_value = std::make_shared<tensor::Tensor>(cnode_type->type_id(), NoShape);
    return gen_output_value_node(tensor_value);
  }
  MS_LOG(EXCEPTION) << "Unknown shape: " << cnode_shape->ToString() << ", type: " << cnode_type->ToString();
}

void PynativeDFunctor::GetForwardOutNodeAndBpropGraph(const CNodePtr &k_app, CNodePtr *forward_node,
                                                      FuncGraphPtr *bprop_graph, FuncGraphPtr *fprop_graph) {
  MS_EXCEPTION_IF_NULL(k_app);
  MS_EXCEPTION_IF_NULL(fprop_graph);
  const auto &prim = k_app->input(0);
  if (!IsValueNode<FuncGraph>(prim)) {
    return;
  }
  // Clone a new fprop graph for different k_app.
  auto original_fprop = GetValueNode<FuncGraphPtr>(prim);
  MS_EXCEPTION_IF_NULL(original_fprop);
  *fprop_graph = BasicClone(original_fprop);
  k_app->set_input(0, NewValueNode(*fprop_graph));

  // {prim::maketuple, forward_output, bprop_graph}
  auto output = (*fprop_graph)->output();
  MS_EXCEPTION_IF_NULL(output);
  if (!output->isa<CNode>()) {
    return;
  }
  auto make_tuple_node = output->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(make_tuple_node);
  constexpr size_t input_size = 3;
  if (make_tuple_node->size() != input_size) {
    MS_LOG(EXCEPTION) << "The inputs size of make tuple node " << make_tuple_node->DebugString() << " is not equal to "
                      << input_size;
  }

  // Get forward CNode.
  const size_t forward_output_index = 1;
  const auto &output_node = make_tuple_node->input(forward_output_index);
  MS_EXCEPTION_IF_NULL(output_node);
  if (!output_node->isa<CNode>()) {
    return;
  }

  // Get bprop graph of forward CNode.
  const size_t bprop_graph_index = 2;
  const auto &bprop_vnode = make_tuple_node->input(bprop_graph_index);
  if (!IsValueNode<FuncGraph>(bprop_vnode)) {
    return;
  }

  MS_EXCEPTION_IF_NULL(forward_node);
  MS_EXCEPTION_IF_NULL(bprop_graph);
  *forward_node = output_node->cast<CNodePtr>();
  *bprop_graph = GetValueNode<FuncGraphPtr>(bprop_vnode);
}

std::vector<AnfNodePtr> PynativeDFunctor::RunOutputReplace(const CNodePtr &forward_node,
                                                           const FuncGraphPtr &bprop_graph,
                                                           const FuncGraphPtr &fprop_graph,
                                                           const CNodePtr &cnode_morph) {
  MS_EXCEPTION_IF_NULL(cnode_morph);
  if (IsPrimitiveCNode(cnode_morph, prim::kPrimStopGradient) || IsPrimitiveCNode(cnode_morph, prim::kPrimMirror)) {
    return {};
  }
  // Use manager to get the link relation among nodes.
  MS_EXCEPTION_IF_NULL(bprop_graph);
  MS_EXCEPTION_IF_NULL(fprop_graph);
  auto manager = Manage({fprop_graph, bprop_graph}, false);

  // Replace output node.
  MS_EXCEPTION_IF_NULL(forward_node);
  auto ref_size = manager->node_users().at(forward_node).size();
  MS_LOG(DEBUG) << "Ref size: " << ref_size;
  auto output_vnode = GenNewTensor(cnode_morph);
  MS_EXCEPTION_IF_NULL(output_vnode);
  output_vnode->set_has_new_value(true);
  manager->Replace(forward_node, output_vnode);
  MS_LOG(DEBUG) << "Replace: " << forward_node->DebugString() << " with " << output_vnode->ToString();

  // Save forward output node when it used in its bprop graph.
  std::vector<AnfNodePtr> used_forward_nodes;
  if (ref_size > 1) {
    cnode_morph->set_forward(output_vnode, "");
    used_forward_nodes.push_back(cnode_morph);
    MS_LOG(DEBUG) << "node has been used in grad graph: " << cnode_morph->DebugString()
                  << ", its output value: " << output_vnode->ToString();
  }
  return used_forward_nodes;
}

std::vector<AnfNodePtr> PynativeDFunctor::RunInputReplace(const FuncGraphPtr &bprop_graph,
                                                          const FuncGraphPtr &fprop_graph,
                                                          const CNodePtr &cnode_morph) {
  // Use manager to get the link relation among nodes.
  MS_EXCEPTION_IF_NULL(bprop_graph);
  MS_EXCEPTION_IF_NULL(fprop_graph);
  auto manager = Manage({fprop_graph, bprop_graph}, false);

  MS_EXCEPTION_IF_NULL(cnode_morph);
  const auto &paras = fprop_graph->parameters();
  if (cnode_morph->size() - 1 != paras.size() && !IsPrimitiveCNode(cnode_morph, prim::kPrimUpdateState)) {
    MS_LOG(EXCEPTION) << "The size of parameters in fprop graph:" << paras.size()
                      << ", but the size of input tensors of forward node: " << cnode_morph->inputs().size() - 1;
  }

  std::vector<AnfNodePtr> used_input_nodes;
  for (size_t i = 0; i < paras.size(); ++i) {
    const auto &input_node = cnode_morph->input(i + 1);
    MS_EXCEPTION_IF_NULL(input_node);
    // Parameter, ValueNode and StopGradient CNode no need to replace.
    if (input_node->isa<Parameter>() || input_node->isa<ValueNode>() ||
        IsPrimitiveCNode(input_node, prim::kPrimStopGradient) || IsPrimitiveCNode(input_node, prim::kPrimMirror)) {
      continue;
    }
    // Replace forward input node by its output value.
    auto para_ref_size = manager->node_users()[paras[i]].size();
    CNodePtr cnode_i = input_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode_i);
    auto output_vnode_i = GenNewTensor(cnode_i);
    MS_EXCEPTION_IF_NULL(output_vnode_i);
    output_vnode_i->set_has_new_value(true);
    manager->Replace(paras[i], output_vnode_i);
    if (IsPrimitiveCNode(cnode_i, prim::kPrimLoad)) {
      para_ref_size += 1;
    }
    MS_LOG(DEBUG) << "Replace: " << paras[i]->DebugString() << " with " << output_vnode_i->ToString();
    // Save forward input node when it used in bprop graph.
    if (para_ref_size > 0 && !IsPrimitiveCNode(input_node, prim::kPrimUpdateState)) {
      cnode_i->set_forward(output_vnode_i, "");
      used_input_nodes.push_back(cnode_i);
      MS_LOG(DEBUG) << "Input CNode has been used in grad graph: " << cnode_i->DebugString()
                    << ", its output value: " << output_vnode_i->ToString();
    }
  }

  return used_input_nodes;
}

void PynativeDFunctor::ReplaceEquivdout(const CNodePtr &k_app, const CNodePtr &cnode_morph) {
  // The process of replacing forward node only works in pynative mode, when @jit is used.
  MS_EXCEPTION_IF_NULL(cnode_morph);
  MS_LOG(DEBUG) << "Run replace for cnode morph: " << cnode_morph->DebugString(2);
  // Get forward node and its fprop graph, bprop graph.
  MS_EXCEPTION_IF_NULL(k_app);
  CNodePtr forward_node = nullptr;
  FuncGraphPtr bprop_graph = nullptr;
  FuncGraphPtr fprop_graph = nullptr;
  GetForwardOutNodeAndBpropGraph(k_app, &forward_node, &bprop_graph, &fprop_graph);
  if (forward_node == nullptr || bprop_graph == nullptr || fprop_graph == nullptr) {
    return;
  }

  // Replace forward node used in bprop graph by its output tensors. The same process for its input node.
  auto used_forward_nodes = RunOutputReplace(forward_node, bprop_graph, fprop_graph, cnode_morph);
  auto used_input_nodes = RunInputReplace(bprop_graph, fprop_graph, cnode_morph);

  // Save used forward input and output nodes to func_graph.
  auto ms_func_graph = cnode_morph->func_graph();
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  ms_func_graph->set_used_forward_nodes(used_forward_nodes);
  ms_func_graph->set_used_forward_nodes(used_input_nodes);
}
}  // namespace ad
}  // namespace mindspore
