/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/common/helper.h"
#include <string>
#include <utility>
#include <unordered_set>
#include <algorithm>
#include <map>
#include <set>
#include <deque>
#include "utils/utils.h"
#include "base/base_ref.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "base/core_ops.h"
#include "backend/kernel_compiler/tbe/tbe_dynaminc_shape_util.h"
#include "frontend/operator/ops.h"
#include "utils/ms_utils.h"
#include "runtime/device/kernel_info.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
constexpr size_t kType32Len = 4;
std::vector<int64_t> Convert2Int(const std::vector<size_t> &v) {
  std::vector<int64_t> result;
  (void)std::transform(v.begin(), v.end(), std::back_inserter(result), SizeToInt);
  return result;
}

std::vector<int64_t> Convert2Long(const std::vector<size_t> &v) {
  std::vector<int64_t> result;
  (void)std::transform(v.begin(), v.end(), std::back_inserter(result), SizeToLong);
  return result;
}

bool IsDepend(const FuncGraph &graph, const AnfNodePtr &node, const std::vector<AnfNodePtr> &nodes) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<AnfNodePtr> node_list = TopoSort(graph.get_return());
  std::map<AnfNodePtr, std::set<AnfNodePtr>> control_depend_map;
  for (auto &nd : node_list) {
    MS_EXCEPTION_IF_NULL(nd);
    if (AnfAlgo::CheckPrimitiveType(nd, prim::kPrimControlDepend)) {
      auto control_depend = nd->cast<CNodePtr>();
      auto prior_node = control_depend->input(kControlDependPriorIndex);
      auto behind_node = control_depend->input(kControlDependBehindIndex);
      auto it = control_depend_map.find(behind_node);
      if (it == control_depend_map.end()) {
        control_depend_map[behind_node] = std::set<AnfNodePtr>{prior_node};
      } else {
        it->second.insert(prior_node);
      }
    }
  }

  FuncGraphManagerPtr manager = graph.manager();
  MS_EXCEPTION_IF_NULL(manager);

  std::unordered_set<AnfNodePtr> seen_node;
  std::deque<AnfNodePtr> todo{node};
  while (!todo.empty()) {
    AnfNodePtr nd = todo.front();
    todo.pop_front();
    if (seen_node.count(nd) > 0 || !manager->all_nodes().contains(nd)) {
      continue;
    }
    (void)seen_node.insert(nd);

    if (std::any_of(nodes.begin(), nodes.end(), [&nd](const AnfNodePtr &item) { return nd == item; })) {
      return true;
    }
    if (nd->isa<CNode>()) {
      auto cnode = nd->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto inputs = cnode->inputs();
      (void)todo.insert(todo.end(), inputs.begin(), inputs.end());
    }
    auto it = control_depend_map.find(nd);
    if (it != control_depend_map.end()) {
      (void)todo.insert(todo.end(), it->second.begin(), it->second.end());
    }
  }
  return false;
}

bool UnVisited(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    AnfNodePtr in = utils::cast<AnfNodePtr>(n);
    MS_EXCEPTION_IF_NULL(in);
    if (IsValueNode<Primitive>(in)) {
      auto value_node = in->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto value = value_node->value();
      MS_EXCEPTION_IF_NULL(value);
      auto prim_py = value->cast<PrimitivePtr>();
      MS_EXCEPTION_IF_NULL(prim_py);
      return !prim_py->HasAttr(kAttrVisited);
    } else if (IsValueNode<FuncGraph>(in)) {
      auto func_graph = GetValueNode<FuncGraphPtr>(in);
      MS_EXCEPTION_IF_NULL(func_graph);
      return !func_graph->has_flag(kAttrVisited);
    }
    return false;
  }
  return false;
}

CNodePtr CheckAnfNodeIfCNodeAndInputSize(const AnfNodePtr &node, size_t input_size) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "The node is expected to be a cnode";
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() != input_size) {
    auto op_name = AnfAlgo::GetCNodeName(cnode);
    MS_LOG(EXCEPTION) << "op[" + op_name + "] has less than " << input_size << " inputs.";
  }
  return cnode;
}

void CheckCNodeInputSize(const CNodePtr &cnode, size_t input_size) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() != input_size) {
    MS_LOG(EXCEPTION) << "The input size of node " + cnode->DebugString() + " is not equal to " << input_size;
  }
}

bool HasSymmetricalKernelInfo(const AnfNodePtr &node_x, const AnfNodePtr &node_y) {
  MS_EXCEPTION_IF_NULL(node_x);
  MS_EXCEPTION_IF_NULL(node_y);
  return (AnfAlgo::GetInputDeviceDataType(node_x, 0) == AnfAlgo::GetOutputDeviceDataType(node_y, 0) &&
          AnfAlgo::GetOutputDeviceDataType(node_x, 0) == AnfAlgo::GetInputDeviceDataType(node_y, 0));
}

const AnfNodePtr EliminateDependTransop(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);

  auto transop_cnode = CheckAnfNodeIfCNodeAndInputSize(node, kTransOpInputNum);
  MS_EXCEPTION_IF_NULL(transop_cnode);
  auto depend_cnode = CheckAnfNodeIfCNodeAndInputSize(transop_cnode->input(kCastInputNum - 1), kDependInputNum);
  auto prev_transop_cnode = CheckAnfNodeIfCNodeAndInputSize(depend_cnode->input(1), kTransOpInputNum);
  MS_EXCEPTION_IF_NULL(depend_cnode->input(kDependInputNum - 1));
  MS_EXCEPTION_IF_NULL(prev_transop_cnode->input(kTransOpInputNum - 1));
  auto transed_node = prev_transop_cnode->input(kTransOpInputNum - 1);
  MS_EXCEPTION_IF_NULL(transed_node);

  std::vector<AnfNodePtr> replace_depend_inputs{NewValueNode(prim::kPrimDepend), transed_node,
                                                depend_cnode->input(kDependInputNum - 1)};
  AnfNodePtr replace_depend = func_graph->NewCNode(replace_depend_inputs);
  MS_EXCEPTION_IF_NULL(replace_depend);
  auto transed_abstract = transed_node->abstract();
  replace_depend->set_abstract(transed_abstract);
  return replace_depend;
}

bool Visited(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    AnfNodePtr in = utils::cast<AnfNodePtr>(n);
    MS_EXCEPTION_IF_NULL(in);
    if (IsValueNode<Primitive>(in)) {
      auto value_node = in->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto value = value_node->value();
      MS_EXCEPTION_IF_NULL(value);
      auto prim_py = value->cast<PrimitivePtr>();
      MS_EXCEPTION_IF_NULL(prim_py);
      return prim_py->HasAttr(kAttrVisited);
    } else if (IsValueNode<FuncGraph>(in)) {
      auto func_graph = GetValueNode<FuncGraphPtr>(in);
      MS_EXCEPTION_IF_NULL(func_graph);
      return func_graph->has_flag(kAttrVisited);
    }
    return false;
  }
  return false;
}

void CreateMultipleOutputsOfAnfNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t output_num,
                                    std::vector<AnfNodePtr> *outputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(outputs);
  for (size_t i = 0; i < output_num; i++) {
    int64_t temp = SizeToLong(i);
    auto idx = NewValueNode(temp);
    MS_EXCEPTION_IF_NULL(idx);
    auto imm = std::make_shared<Int64Imm>(temp);
    auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
    idx->set_abstract(abstract_scalar);
    auto tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
    MS_EXCEPTION_IF_NULL(tuple_getitem);
    AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(node, i)},
                                        {AnfAlgo::GetOutputInferShape(node, i)}, tuple_getitem.get());
    (*outputs).push_back(tuple_getitem);
  }
}

template <typename T>
tensor::TensorPtr CreateTensorWithValueTuple(const ValueTuplePtr &value_tuple_ptr, const TypePtr &type_ptr,
                                             size_t data_length) {
  MS_EXCEPTION_IF_NULL(value_tuple_ptr);
  MS_EXCEPTION_IF_NULL(type_ptr);
  std::vector<T> values;
  for (const auto &v : value_tuple_ptr->value()) {
    MS_EXCEPTION_IF_NULL(v);
    if (v->isa<Scalar>()) {
      ScalarPtr scalar = v->cast<ScalarPtr>();
      values.push_back(GetValue<T>(scalar));
    } else {
      MS_LOG(WARNING) << "The value " << v << "of tuple is not a scalar";
      return nullptr;
    }
  }
  std::vector<int64_t> tensor_shape = {SizeToLong(values.size())};
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_ptr->type_id(), tensor_shape);
  MS_EXCEPTION_IF_NULL(tensor);
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, type_ptr};
  tensor->set_device_info(device_info);
  auto data_ptr = tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  auto elem_num = values.size() * data_length;
  auto ret_code = memcpy_s(data_ptr, static_cast<size_t>(tensor->data().nbytes()), values.data(), elem_num);
  if (ret_code != 0) {
    MS_LOG(EXCEPTION) << "Failed to copy data into Tensor.";
  }
  return tensor;
}

tensor::TensorPtr CreateTupleTensor(const ValueTuplePtr &value_tuple) {
  MS_EXCEPTION_IF_NULL(value_tuple);
  tensor::TensorPtr tensor = nullptr;
  if (value_tuple->value().empty()) {
    MS_LOG(WARNING) << "The value tuple is empty.";
    return nullptr;
  }
  ValuePtr v = *(value_tuple->value().begin());
  MS_EXCEPTION_IF_NULL(v);
  // Currently we only deal with the scalar tuple
  if (!v->isa<Scalar>()) {
    MS_LOG(WARNING) << "The value " << v << "of tuple is not a scalar";
    return nullptr;
  }
  ScalarPtr scalar = v->cast<ScalarPtr>();
  MS_EXCEPTION_IF_NULL(scalar);
  if (scalar->isa<Int32Imm>()) {
    tensor = CreateTensorWithValueTuple<int32_t>(value_tuple, kInt32, sizeof(int32_t));
  } else if (scalar->isa<Int64Imm>()) {
    tensor = CreateTensorWithValueTuple<int64_t>(value_tuple, kInt64, sizeof(int64_t));
  } else if (scalar->isa<FloatImm>()) {
    tensor = CreateTensorWithValueTuple<float>(value_tuple, kFloat32, sizeof(float));
  } else {
    auto type = scalar->type();
    auto type_str = (type == nullptr) ? "nullptr" : type->ToString();
    MS_LOG(ERROR) << "Invalid scalar type: " << type_str;
    return nullptr;
  }
  return tensor;
}

bool IsNopNode(const AnfNodePtr &node) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto target = GetCNodeTarget(node);
  if (target == kCPUDevice) {
    return false;
  }
  if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice &&
      context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kGPUDevice) {
    return false;
  }

  static std::unordered_set<std::string> nop_nodes = {prim::kPrimReshape->name(), kExpandDimsOpName,
                                                      prim::kPrimSqueeze->name(), prim::kPrimFlatten->name(),
                                                      kFlattenGradOpName,         prim::kPrimReformat->name()};
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().empty()) {
    return false;
  }
  auto input0 = cnode->input(0);
  MS_EXCEPTION_IF_NULL(input0);
  if (!input0->isa<ValueNode>()) {
    return false;
  }
  bool is_nop_node = false;
  if (AnfAlgo::HasNodeAttr("nop_op", cnode)) {
    is_nop_node = AnfAlgo::GetNodeAttr<bool>(cnode, "nop_op");
  }
  if (nop_nodes.find(AnfAlgo::GetCNodeName(cnode)) == nop_nodes.end() && !is_nop_node) {
    return false;
  }
  return true;
}

bool IsAllNopNode(const session::KernelGraph *const graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto execution_order = graph->execution_order();
  for (auto &cnode : execution_order) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (!IsNopNode(cnode)) {
      return false;
    }
  }
  return true;
}

bool CheckNopNodeIsOutputNode(const std::vector<AnfNodePtr> &outputs, const AnfNodePtr &node, bool is_dynamic_graph) {
  MS_EXCEPTION_IF_NULL(node);
  // if node is not a nop node, keep it in execution order
  if (!IsNopNode(node)) {
    return true;
  }
  // if node is nop node and the graph is dynamic graph, check if the nop node is graph's output.
  if (is_dynamic_graph) {
    auto iter = find(outputs.begin(), outputs.end(), node);
    if (iter != outputs.end()) {
      return true;
    }
  }
  return false;
}

void HideNopNode(session::KernelGraph *const graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (IsAllNopNode(graph) == true) {
    return;
  }
  auto execution_order = graph->execution_order();
  auto outputs = graph->outputs();
  bool is_dynamic_graph = graph->is_dynamic_shape();
  MS_LOG(INFO) << "nop node info (Before Remove) size: " << execution_order.size();
  std::vector<CNodePtr> new_nodes;
  for (auto &cnode : execution_order) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (CheckNopNodeIsOutputNode(outputs, cnode, is_dynamic_graph)) {
      new_nodes.push_back(cnode);
    }
  }
  graph->set_execution_order(new_nodes);
  MS_LOG(INFO) << "nop node info (After Remove) size: " << graph->execution_order().size();
}

void RemoveNopNode(session::KernelGraph *const graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (IsAllNopNode(graph) == true) {
    return;
  }
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<CNodePtr> new_nodes;
    auto outputs = graph->outputs();
    bool is_dynamic_graph = graph->is_dynamic_shape();
    for (auto &cnode : graph->execution_order()) {
      MS_EXCEPTION_IF_NULL(cnode);
      // ignore nop node itself
      if (!CheckNopNodeIsOutputNode(outputs, cnode, is_dynamic_graph)) {
        continue;
      }
      // Replace the input which is nop node
      std::vector<AnfNodePtr> new_inputs;
      new_inputs.push_back(cnode->input(0));
      bool need_update = false;
      for (size_t i = 1; i < cnode->inputs().size(); ++i) {
        auto input = cnode->input(i);
        MS_EXCEPTION_IF_NULL(input);
        auto cinput = input->cast<CNodePtr>();
        if (cinput == nullptr || !IsNopNode(cinput)) {
          new_inputs.push_back(input);
          continue;
        }
        if (cinput->inputs().size() == 2) {
          new_inputs.push_back(cinput->input(1));
          need_update = true;
          changed = true;
        } else {
          new_inputs.push_back(input);
        }
      }
      if (need_update) {
        cnode->set_inputs(new_inputs);
      }
      // push into new execution list
      new_nodes.push_back(cnode);
    }
    graph->set_execution_order(new_nodes);
  }
}

size_t GetRealNodeNum(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  auto out_list = GetRealNodeUsedList(graph, node);
  MS_EXCEPTION_IF_NULL(out_list);
  return out_list->size();
}

std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetRealNodeUsedList(const FuncGraphPtr &graph,
                                                                             const AnfNodePtr &node) {
  auto output_node_list = std::make_shared<std::vector<std::pair<AnfNodePtr, int>>>();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto iter = manager->node_users().find(node);
  if (iter == manager->node_users().end()) {
    MS_LOG(EXCEPTION) << "node has no output in manager";
  }
  auto output_info_list = iter->second;
  for (const auto &output_info : output_info_list) {
    if (AnfAlgo::GetCNodeName(output_info.first) == prim::kPrimControlDepend->name()) {
      continue;
    }
    if (AnfAlgo::GetCNodeName(output_info.first) == prim::kPrimDepend->name() &&
        output_info.second == kDependAttachNodeIndex) {
      continue;
    }
    output_node_list->push_back(output_info);
  }
  return output_node_list;
}

std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetRealNodeUsedListByOutputIdx(const FuncGraphPtr &graph,
                                                                                        const AnfNodePtr &node,
                                                                                        size_t output_index) {
  auto output_node_list = std::make_shared<std::vector<std::pair<AnfNodePtr, int>>>();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto iter = manager->node_users().find(node);
  if (iter == manager->node_users().end()) {
    MS_LOG(EXCEPTION) << "node has no output in manager";
  }
  auto output_info_list = iter->second;
  for (const auto &output_info : output_info_list) {
    if (AnfAlgo::GetCNodeName(output_info.first) == prim::kPrimControlDepend->name()) {
      continue;
    }
    if (AnfAlgo::GetCNodeName(output_info.first) == prim::kPrimDepend->name() &&
        output_info.second == kDependAttachNodeIndex) {
      continue;
    }
    size_t used_output_index;
    if (AnfAlgo::GetCNodeName(output_info.first) == prim::kPrimTupleGetItem->name()) {
      used_output_index = AnfAlgo::GetTupleGetItemOutIndex(utils::cast<CNodePtr>(output_info.first));
    } else if (AnfAlgo::GetCNodeName(node) == prim::kPrimTupleGetItem->name()) {
      used_output_index = output_index;
    } else {
      auto kernel_with_index = AnfAlgo::GetPrevNodeOutput(output_info.first, output_info.second - 1);
      if (kernel_with_index.first.get() != node.get()) {
        MS_LOG(EXCEPTION) << "Get used node failed for op[" << AnfAlgo::GetCNodeName(node) << "]";
      }
      used_output_index = kernel_with_index.second;
    }
    if (used_output_index == output_index) {
      output_node_list->push_back(output_info);
    }
  }
  return output_node_list;
}

bool IsUsedByOthers(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto output_node_list = GetRealNodeUsedList(graph, node);
  MS_EXCEPTION_IF_NULL(output_node_list);
  return output_node_list->size() > 1;
}

bool IsNotRealUsedByOthers(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto output_node_list = GetRealNodeUsedList(graph, node);
  MS_EXCEPTION_IF_NULL(output_node_list);
  if (output_node_list->empty()) {
    return true;
  }
  for (const auto &output : *output_node_list) {
    auto out_node = output.first;
    auto name = AnfAlgo::GetCNodeName(out_node);
    if (name == prim::kPrimDepend->name() || name == prim::kPrimMakeTuple->name() ||
        name == prim::kPrimTupleGetItem->name()) {
      auto result = IsNotRealUsedByOthers(graph, out_node);
      if (!result) {
        return result;
      }
      continue;
    }
    return false;
  }
  return true;
}

CNodePtr CreatTupleGetItemNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t output_idx) {
  auto idx = NewValueNode(SizeToLong(output_idx));
  MS_EXCEPTION_IF_NULL(idx);
  auto imm = std::make_shared<Int64Imm>(SizeToLong(output_idx));
  auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
  idx->set_abstract(abstract_scalar);
  CNodePtr tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  tuple_getitem->set_scope(node->scope());
  std::vector<size_t> origin_shape = AnfAlgo::GetOutputInferShape(node, output_idx);
  TypeId origin_type = AnfAlgo::GetOutputInferDataType(node, output_idx);
  AnfAlgo::SetOutputInferTypeAndShape({origin_type}, {origin_shape}, tuple_getitem.get());
  return tuple_getitem;
}

void ConstInputToAttr(const CNodePtr &cnode, const std::unordered_set<size_t> &input_attrs) {
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> new_inputs;
  std::vector<std::string> new_input_names;
  auto primitive = AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive = primitive->Clone();
  auto input_names = primitive->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr in cnode[" + cnode->DebugString() + "]";
    return;
  }
  auto input_names_vec = GetValue<std::vector<std::string>>(input_names);
  auto inputs = cnode->inputs();
  new_inputs.push_back(inputs[0]);
  bool need_update = false;
  for (size_t i = 0; i < inputs.size() - 1; ++i) {
    auto input_node = inputs[i + 1];
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_attrs.find(i) != input_attrs.end() && input_node->isa<ValueNode>()) {
      auto value_node = input_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      MS_LOG(DEBUG) << "start erase input[" << i << "] of cnode[" + cnode->DebugString() + "]";
      if (i >= input_names_vec.size()) {
        MS_LOG(EXCEPTION) << "index " << i << " is larger than input names size [" << input_names_vec.size() << "]";
      }
      primitive->set_attr(input_names_vec[i], value_node->value());
      need_update = true;
    } else {
      new_inputs.push_back(input_node);
      if (i < input_names_vec.size()) {
        new_input_names.push_back(input_names_vec[i]);
      }
    }
  }
  if (need_update) {
    // Update cnode's inputs
    new_inputs[0] = NewValueNode(primitive);
    cnode->set_inputs(new_inputs);
    // Update cnode's input_names attr
    primitive->set_attr(kAttrInputNames, MakeValue(new_input_names));
  }
}

bool AnfEqual(const BaseRef &a, const BaseRef &b) {
  if (utils::isa<AnfNodePtr>(a) && utils::isa<AnfNodePtr>(b)) {
    auto a_node = utils::cast<AnfNodePtr>(a);
    auto b_node = utils::cast<AnfNodePtr>(b);
    MS_EXCEPTION_IF_NULL(a_node);
    MS_EXCEPTION_IF_NULL(b_node);
    if (IsValueNode<Primitive>(a_node) && IsValueNode<Primitive>(b_node)) {
      auto a_value_node = a_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(a_value_node);
      auto a_value = a_value_node->value();
      MS_EXCEPTION_IF_NULL(a_value);
      auto a_prim = a_value->cast<PrimitivePtr>();
      MS_EXCEPTION_IF_NULL(a_prim);

      auto b_value_node = b_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(b_value_node);
      auto b_value = b_value_node->value();
      MS_EXCEPTION_IF_NULL(b_value);
      auto b_prim = b_value->cast<PrimitivePtr>();
      MS_EXCEPTION_IF_NULL(b_prim);

      return a_prim->name() == b_prim->name();
    } else if (a_node->isa<ValueNode>() && b_node->isa<ValueNode>()) {
      auto a_value_node_ptr = a_node->cast<ValueNodePtr>();
      if (a_value_node_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "cast value node ptr fail";
      }
      auto a_value_ptr = a_value_node_ptr->value();
      if (a_value_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "value ptr is nullptr";
      }

      auto b_value_node_ptr = b_node->cast<ValueNodePtr>();
      if (b_value_node_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "cast value node ptr fail";
      }
      auto b_value_ptr = b_value_node_ptr->value();
      if (b_value_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "value ptr is nullptr";
      }

      return (*a_value_ptr) == (*b_value_ptr);
    }
    MS_LOG(DEBUG) << "check AnfNodePtr equal";
  }
  if (utils::isa<FuncGraphPtr>(a) && utils::isa<FuncGraphPtr>(b)) {
    MS_LOG(DEBUG) << "check GraphPtr equal";
  }
  return a == b;
}

bool CNodeTypeEqual(const BaseRef &a, const BaseRef &b) {
  // To matchCNode and Kernel's type
  if (utils::isa<CNode>(a) && utils::isa<CNode>(b)) {
    return true;
  }
  return a.type() == b.type();
}

namespace {
ValueNodePtr CreateValueNodeWithSexp(const BaseRef &sexp) {
  if (utils::isa<int>(sexp)) {
    return NewValueNode(utils::cast<int>(sexp));
  }
  if (utils::isa<int64_t>(sexp)) {
    return NewValueNode(utils::cast<int64_t>(sexp));
  }
  if (utils::isa<float>(sexp)) {
    return NewValueNode(utils::cast<float>(sexp));
  }
  if (utils::isa<bool>(sexp)) {
    return NewValueNode(utils::cast<bool>(sexp));
  }
  if (utils::isa<ValuePtr>(sexp)) {
    return NewValueNode(utils::cast<ValuePtr>(sexp));
  }
  return nullptr;
}

CNodePtr CreateCNodeWithGraph(const std::vector<AnfNodePtr> &input_nodes, const BaseRef &graph) {
  if (utils::isa<FuncGraphPtr>(graph)) {
    return std::make_shared<CNode>(input_nodes, utils::cast<FuncGraphPtr>(graph));
  }
  if (utils::isa<VarPtr>(graph)) {
    return std::make_shared<CNode>(input_nodes, utils::cast<VarPtr>(graph));
  }
  return nullptr;
}

VarNodePtr CreateVarNodeWithSexp(const BaseRef &sexp, const BaseRef &graph) {
  if (utils::isa<VarPtr>(graph)) {
    MS_LOG(DEBUG) << "make VarPtr " + graph.ToString();
    return std::make_shared<VarNode>(utils::cast<VarPtr>(sexp), nullptr);
  }
  if (utils::isa<FuncGraphPtr>(graph)) {
    MS_LOG(DEBUG) << "VarNode, should input a Var in graph. It's GraphPtr: " + graph.ToString();
    return std::make_shared<VarNode>(utils::cast<VarPtr>(sexp), utils::cast<FuncGraphPtr>(graph));
  }
  MS_LOG(ERROR) << "VarNode, should input a Var in graph. It's " + graph.ToString();
  return nullptr;
}

AnfNodePtr HandleSexpVector(const BaseRef &sexp, const BaseRef &graph, PrimitiveVarMap *primitive_vars,
                            bool multigraph) {
  MS_LOG(DEBUG) << "HandleSexpVector sexp: " + sexp.ToString() + ", graph " + graph.ToString();
  std::vector<AnfNodePtr> input_nodes;
  const auto &tuple = utils::cast<VectorRef>(sexp);
  if (multigraph && utils::isa<VarPtr>(graph)) {
    for (auto &x : tuple) {
      AnfNodePtr node = SexpToNode(x, std::make_shared<Var>("G"), primitive_vars, true);
      input_nodes.push_back(node);
    }
    VarPtr var_ptr = utils::cast<VarPtr>(graph);
    return std::make_shared<CNode>(input_nodes, var_ptr);
  }

  for (auto &x : tuple) {
    AnfNodePtr node = SexpToNode(x, graph, primitive_vars, multigraph);
    input_nodes.push_back(node);
  }
  return CreateCNodeWithGraph(input_nodes, graph);
}
}  // namespace

AnfNodePtr SexpToNode(const BaseRef &sexp, const BaseRef &graph, PrimitiveVarMap *primitive_vars, bool multigraph) {
  MS_LOG(DEBUG) << "SexpToNode sexp: " + sexp.ToString() + ", graph " + graph.ToString();
  MS_EXCEPTION_IF_NULL(primitive_vars);
  if (utils::isa<VectorRef>(sexp)) {
    return HandleSexpVector(sexp, graph, primitive_vars, multigraph);
  }
  if (utils::isa<VarPtr>(sexp)) {
    auto var_ptr = utils::cast<VarPtr>(sexp);
    MS_EXCEPTION_IF_NULL(var_ptr);
    if (var_ptr->primitive()) {
      (*primitive_vars)[var_ptr->primitive()] = var_ptr;
      return NewValueNode(var_ptr->primitive());
    }
    return CreateVarNodeWithSexp(sexp, graph);
  }
  if (utils::isa<AnfNodePtr>(sexp)) {
    return utils::cast<AnfNodePtr>(sexp);
  }
  auto value_node = CreateValueNodeWithSexp(sexp);
  if (value_node == nullptr) {
    MS_LOG(EXCEPTION) << "sexp cannot converted. sexp: " + sexp.ToString();
  }
  return value_node;
}

bool IsSameNode(const EquivPtr &equiv1, const EquivPtr &equiv2, const VarPtr &var_node) {
  MS_EXCEPTION_IF_NULL(equiv1);
  MS_EXCEPTION_IF_NULL(equiv2);
  MS_EXCEPTION_IF_NULL(var_node);
  auto equiv1_node = GetAnfNodeByVar(equiv1, var_node);
  MS_EXCEPTION_IF_NULL(equiv1_node);
  auto equiv2_node = GetAnfNodeByVar(equiv2, var_node);
  MS_EXCEPTION_IF_NULL(equiv2_node);
  return *equiv1_node == *equiv2_node;
}

AnfNodePtr GetAnfNodeByVar(const EquivPtr &equiv, const VarPtr &var_node) {
  MS_EXCEPTION_IF_NULL(equiv);
  MS_EXCEPTION_IF_NULL(var_node);
  auto iter = (*equiv).find(var_node);
  if (iter == (*equiv).end()) {
    MS_LOG(INFO) << "The equiv map doesn't contain the var_node after matched.";
    return nullptr;
  }
  auto res = utils::cast<AnfNodePtr>(iter->second);
  if (res == nullptr) {
    MS_LOG(EXCEPTION) << "Cast fail! Maybe var is not a anf node";
  }
  return res;
}

bool CompareTupleGetitem(const AnfNodePtr &n1, const AnfNodePtr &n2) {
  MS_EXCEPTION_IF_NULL(n1);
  MS_EXCEPTION_IF_NULL(n2);
  auto n1_cnode = n1->cast<CNodePtr>();
  auto n2_cnode = n2->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(n1_cnode);
  MS_EXCEPTION_IF_NULL(n2_cnode);
  auto index_input1 = n1_cnode->input(kInputNodeOutputIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(index_input1);
  auto value_node1 = index_input1->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node1);
  auto index_input2 = n2_cnode->input(kInputNodeOutputIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(index_input2);
  auto value_node2 = index_input2->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node2);
  return GetValue<int64_t>(value_node1->value()) < GetValue<int64_t>(value_node2->value());
}

bool GetBoolAttr(const AnfNodePtr &node, const std::string &attr_name) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(INFO) << "node is not a cnode";
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  return AnfAlgo::HasNodeAttr(attr_name, cnode) && AnfAlgo::GetNodeAttr<bool>(node, attr_name);
}

bool CheckSupportDataType(const AnfNodePtr &node, const std::set<TypeId> &supported_data_type_set) {
  MS_EXCEPTION_IF_NULL(node);
  TypeId data_type = AnfAlgo::GetOutputInferDataType(node, 0);
  if (supported_data_type_set.find(data_type) != supported_data_type_set.end()) {
    return true;
  }
  MS_LOG(DEBUG) << "Not supported data type. Node:" << node->DebugString();
  return false;
}

ValueNodePtr MakeValueNode(const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  ValueNodePtr new_value_node = std::make_shared<ValueNode>(value_node->value());
  new_value_node->set_abstract(value_node->abstract());
  // create kernel_info fo new value node
  auto kernel_info = std::make_shared<device::KernelInfo>();
  new_value_node->set_kernel_info(kernel_info);
  // create kernel_build_info for new value node
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  // set the format of value_node to DEFAULT_FORMAT
  kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
  // set value node initial device data type = infer data type
  std::vector<TypeId> types;
  for (size_t index = 0; index < AnfAlgo::GetOutputTensorNum(value_node); ++index) {
    types.push_back(kTypeUnknown);
  }
  kernel_build_info_builder->SetOutputsDeviceType(types);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), new_value_node.get());
  return new_value_node;
}

void TransferDepend(const CNodePtr &old_node, const FuncGraphPtr &graph, const CNodePtr &new_node) {
  MS_EXCEPTION_IF_NULL(old_node);
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  // find BatchNorm's output which is a Depend or ControlDepend
  for (const auto &node_index : manager->node_users()[old_node]) {
    AnfNodePtr output = node_index.first;
    size_t index = IntToSize(node_index.second);
    MS_EXCEPTION_IF_NULL(output);
    if (AnfAlgo::CheckPrimitiveType(output, prim::kPrimControlDepend)) {
      auto control_depend = output->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(control_depend);
      control_depend->set_input(index, new_node);
    } else if (AnfAlgo::CheckPrimitiveType(output, prim::kPrimDepend)) {
      auto depend = output->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(depend);
      depend->set_input(index, new_node);
    }
  }
}
}  // namespace opt
}  // namespace mindspore
