/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "backend/common/optimizer/helper.h"
#include <string>
#include <utility>
#include <algorithm>
#include <map>
#include <set>
#include <deque>
#include "utils/hash_set.h"
#include "include/common/utils/utils.h"
#include "base/base_ref.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/core_ops.h"
#include "utils/ms_utils.h"
#include "include/common/utils/convert_utils.h"
#include "runtime/device/kernel_info.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"
#include "backend/common/optimizer/const_input_to_attr.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kType32Len = 4;
constexpr size_t kType64Len = 8;
constexpr auto kNopNodeRealInputIndex = 1;

void UpdateDumpFlagAndDebugInfo(const CNodePtr &node, const std::vector<AnfNodePtr> &orig_nodes) {
  std::vector<AnfNodePtr> orig_real_cnodes;
  for (auto &orig_node : orig_nodes) {
    if (AnfUtils::IsRealCNodeKernel(orig_node)) {
      auto orig_cnode = orig_node->cast<CNodePtr>();
      if (common::AnfAlgo::HasNodeAttr(kAttrDump, orig_cnode)) {
        common::AnfAlgo::CopyNodeAttr(kAttrDump, orig_cnode, node);
      }
      orig_real_cnodes.push_back(orig_node);
    }
  }

  node->AddFusedDebugInfoList(orig_real_cnodes);
}
}  // namespace

bool IsDepend(const FuncGraph &graph, const AnfNodePtr &node, const std::vector<AnfNodePtr> &nodes) {
  mindspore::HashSet<AnfNodePtr> visited_nodes;
  return IsDepend(graph, node, nodes, &visited_nodes);
}

bool IsDepend(const FuncGraph &graph, const AnfNodePtr &node, const std::vector<AnfNodePtr> &nodes,
              mindspore::HashSet<AnfNodePtr> *visited_nodes) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(visited_nodes);
  FuncGraphManagerPtr manager = graph.manager();
  MS_EXCEPTION_IF_NULL(manager);

  std::deque<AnfNodePtr> todo{node};
  while (!todo.empty()) {
    AnfNodePtr nd = todo.front();
    todo.pop_front();
    if (visited_nodes->count(nd) > 0 || !manager->all_nodes().contains(nd)) {
      continue;
    }
    (void)visited_nodes->insert(nd);

    if (std::any_of(nodes.begin(), nodes.end(), [&nd](const AnfNodePtr &item) { return nd == item; })) {
      return true;
    }
    if (nd->isa<CNode>()) {
      auto cnode = nd->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto inputs = cnode->inputs();
      (void)todo.insert(todo.cend(), inputs.cbegin(), inputs.cend());
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

CNodePtr NewCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &fg,
                  const std::vector<AnfNodePtr> &orig_nodes) {
  MS_EXCEPTION_IF_NULL(fg);
  auto node = fg->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(node);
  UpdateDumpFlagAndDebugInfo(node, orig_nodes);
  return node;
}

CNodePtr NewCNode(const CNodePtr &cnode, const KernelGraphPtr &fg, const std::vector<AnfNodePtr> &orig_nodes) {
  MS_EXCEPTION_IF_NULL(fg);
  auto node = fg->NewCNode(cnode);
  MS_EXCEPTION_IF_NULL(node);
  UpdateDumpFlagAndDebugInfo(node, orig_nodes);
  return node;
}

CNodePtr CheckAnfNodeIfCNodeAndInputSize(const AnfNodePtr &node, size_t input_size) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "The node is expected to be a cnode";
  }
  auto cnode = node->cast<CNodePtr>();
  CheckCNodeInputSize(cnode, input_size);
  return cnode;
}

void CheckCNodeInputSize(const CNodePtr &cnode, size_t input_tensor_size) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto real_input_tensor_num = common::AnfAlgo::GetInputTensorNum(cnode);
  if (real_input_tensor_num != input_tensor_size) {
    MS_LOG(EXCEPTION) << "The input tensor size[" << real_input_tensor_num
                      << "] of node [" + cnode->DebugString() + "] is not equal to " << input_tensor_size
                      << trace::DumpSourceLines(cnode);
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

  auto transop_cnode = CheckAnfNodeIfCNodeAndInputSize(node, kTransOpInputTensorNum);
  MS_EXCEPTION_IF_NULL(transop_cnode);
  auto depend_cnode = CheckAnfNodeIfCNodeAndInputSize(transop_cnode->input(1), kDependInputTensorNum);
  auto prev_transop_cnode = CheckAnfNodeIfCNodeAndInputSize(depend_cnode->input(1), kTransOpInputTensorNum);
  auto transed_node = prev_transop_cnode->input(1);
  MS_EXCEPTION_IF_NULL(transed_node);

  std::vector<AnfNodePtr> replace_depend_inputs{NewValueNode(prim::kPrimDepend), transed_node,
                                                depend_cnode->input(kDependAttachNodeIndex)};
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
  auto type_ptr = node->Type();
  auto shape_ptr = node->Shape();
  for (size_t i = 0; i < output_num; i++) {
    int64_t temp = SizeToLong(i);
    auto idx = NewValueNode(temp);
    MS_EXCEPTION_IF_NULL(idx);
    auto imm = std::make_shared<Int64Imm>(temp);
    auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
    idx->set_abstract(abstract_scalar);
    auto tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
    MS_EXCEPTION_IF_NULL(tuple_getitem);
    common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(type_ptr, i)},
                                                {common::AnfAlgo::GetOutputInferShape(node, shape_ptr, i)},
                                                tuple_getitem.get());
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
    MS_LOG(EXCEPTION) << "Failed to copy data into tensor, memcpy_s errorno: " << ret_code;
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
    MS_LOG(DEBUG) << "The value " << v << "of tuple is not a scalar";
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

AnfNodePtr CreateTensorMoveOp(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto prim = std::make_shared<Primitive>(kTensorMoveOpName);
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(prim), node};
  auto new_node = graph->NewCNode(new_node_inputs);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_abstract(node->abstract());
  new_node->set_scope(node->scope());
  common::AnfAlgo::SetNodeAttr(kAttrDatadumpOriginalNames, MakeValue<std::vector<std::string>>({}), new_node);
  return new_node;
}

std::vector<AnfNodePtr> InsertTensorMoveForGraphOutput(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  std::vector<AnfNodePtr> ret;
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto iter = node_users.find(node);
  if (iter == node_users.end()) {
    return ret;
  }
  for (auto &item : iter->second) {
    MS_EXCEPTION_IF_NULL(item.first);
    auto next_node = item.first->cast<CNodePtr>();
    bool find = false;
    auto graph_outputs_pair =
      common::AnfAlgo::GetAllOutputIndexByReturnTypes(graph->output(), {prim::kPrimTupleGetItem});
    for (auto output_pair : graph_outputs_pair) {
      while (AnfUtils::IsRealCNodeKernel(output_pair.first)) {
        auto output_kernel = output_pair.first;
        MS_EXCEPTION_IF_NULL(output_kernel);
        auto cnode = output_kernel->cast<CNodePtr>();
        // nop node
        if (common::AnfAlgo::IsNopNode(cnode)) {
          output_pair = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(kNopNodeRealInputIndex), 0, true);
          continue;
        }
        // ref node
        if (kernel_graph->IsInRefOutputMap(output_pair)) {
          output_pair = kernel_graph->GetRefCorrespondOutput(output_pair);
          continue;
        }
        break;
      }
      if (next_node == output_pair.first->cast<CNodePtr>()) {
        find = true;
        break;
      }
    }
    if (!find) {
      continue;
    }
    auto tensor_move = CreateTensorMoveOp(graph, next_node);
    auto kernel_info = std::make_shared<device::KernelInfo>();
    tensor_move->set_kernel_info(kernel_info);
    (void)manager->Replace(next_node, tensor_move);
    ret.push_back(tensor_move);
    MS_LOG(DEBUG) << "Insert Output TensorMove for op " << node->fullname_with_scope();
  }
  return ret;
}

bool IsAllNopNode(const session::KernelGraph *const graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto execution_order = graph->execution_order();
  for (auto &cnode : execution_order) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (!common::AnfAlgo::IsNopNode(cnode)) {
      return false;
    }
  }
  return true;
}

bool NeedHideNode(const std::vector<AnfNodePtr> &outputs, const AnfNodePtr &node, bool is_dynamic_graph) {
  MS_EXCEPTION_IF_NULL(node);
  // if node is not a nop node, keep it in execution order
  if (!common::AnfAlgo::IsNopNode(node)) {
    return false;
  }
  // if node is nop node and the graph is dynamic graph, check if the nop node is graph's output.
  if (is_dynamic_graph) {
    auto iter = find(outputs.begin(), outputs.end(), node);
    if (iter != outputs.end()) {
      return false;
    }
  }
  return true;
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
    if (NeedHideNode(outputs, cnode, is_dynamic_graph)) {
      common::AnfAlgo::SetNodeAttr(kAttrSkipNopOpAddr, MakeValue(true), cnode);
      common::AnfAlgo::SetNodeAttr(kAttrSkipNopOpExecution, MakeValue(true), cnode);
    } else {
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
      if (NeedHideNode(outputs, cnode, is_dynamic_graph)) {
        common::AnfAlgo::SetNodeAttr(kAttrSkipNopOpAddr, MakeValue(true), cnode);
        common::AnfAlgo::SetNodeAttr(kAttrSkipNopOpExecution, MakeValue(true), cnode);
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
        if (cinput == nullptr || !common::AnfAlgo::IsNopNode(cinput)) {
          new_inputs.push_back(input);
          continue;
        }
        constexpr auto kInputSize = 2;
        if (cinput->inputs().size() == kInputSize) {
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
    return output_node_list;
  }
  auto output_info_list = iter->second;
  for (const auto &output_info : output_info_list) {
    auto cnode_name = common::AnfAlgo::GetCNodeName(output_info.first);
    if ((cnode_name == prim::kPrimDepend->name() && output_info.second == kDependAttachNodeIndex) ||
        (cnode_name == prim::kPrimUpdateState->name())) {
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
    auto cnode_name = common::AnfAlgo::GetCNodeName(output_info.first);
    if ((cnode_name == prim::kPrimDepend->name() && output_info.second == kDependAttachNodeIndex) ||
        (cnode_name == prim::kPrimUpdateState->name())) {
      continue;
    }
    size_t used_output_index;
    if (cnode_name == prim::kPrimTupleGetItem->name()) {
      used_output_index = common::AnfAlgo::GetTupleGetItemOutIndex(utils::cast<CNodePtr>(output_info.first));
    } else if (common::AnfAlgo::GetCNodeName(node) == prim::kPrimTupleGetItem->name()) {
      used_output_index = output_index;
    } else {
      auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(output_info.first, IntToSize(output_info.second - 1));
      if (kernel_with_index.first.get() != node.get()) {
        MS_LOG(EXCEPTION) << "Get used node failed for op[" << common::AnfAlgo::GetCNodeName(node) << "]";
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
    auto name = common::AnfAlgo::GetCNodeName(out_node);
    if (name == prim::kPrimDepend->name() || name == prim::kPrimMakeTuple->name() ||
        name == prim::kPrimTupleGetItem->name() || name == prim::kPrimLoad->name()) {
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
  MS_EXCEPTION_IF_NULL(func_graph);
  auto idx = NewValueNode(SizeToLong(output_idx));
  MS_EXCEPTION_IF_NULL(idx);
  auto imm = std::make_shared<Int64Imm>(SizeToLong(output_idx));
  auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
  idx->set_abstract(abstract_scalar);
  CNodePtr tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  tuple_getitem->set_scope(node->scope());
  auto abs = node->abstract()->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abs);
  auto abs_i = abs->elements()[output_idx];
  MS_EXCEPTION_IF_NULL(abs_i);
  tuple_getitem->set_abstract(abs_i);
  return tuple_getitem;
}

ValueNodePtr CreateShapeValueNode(const FuncGraphPtr &func_graph, const std::vector<int64_t> &shape, bool to_tensor) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  ValuePtr shape_value = nullptr;
  AbstractBasePtr abstract = nullptr;
  if (to_tensor) {
    // create Tensor
    int64_t shape_dim = SizeToLong(shape.size());
    std::vector<int64_t> shape_vec_shape = {shape_dim};
    auto shape_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, shape_vec_shape);
    MS_EXCEPTION_IF_NULL(shape_tensor);
    auto data_ptr = shape_tensor->data_c();
    MS_EXCEPTION_IF_NULL(data_ptr);
    auto elem_num = shape.size() * kType64Len;
    auto ret_code = memcpy_s(data_ptr, static_cast<size_t>(shape_tensor->data().nbytes()), &shape[0], elem_num);
    if (ret_code != 0) {
      MS_LOG(EXCEPTION) << "Failed to copy data into tensor, memcpy_s errorno: " << ret_code;
    }
    shape_value = shape_tensor;
    abstract = std::make_shared<abstract::AbstractTensor>(kInt64, shape_vec_shape);
  } else {
    // create ValueTuple
    std::vector<ValuePtr> dim_values{};
    abstract::AbstractBasePtrList abs{};
    for (const auto &dim : shape) {
      dim_values.push_back(MakeValue(dim));
      abs.push_back(std::make_shared<abstract::AbstractScalar>(dim));
    }
    shape_value = std::make_shared<ValueTuple>(dim_values);
    abstract = std::make_shared<abstract::AbstractTuple>(abs);
  }
  MS_EXCEPTION_IF_NULL(shape_value);
  MS_EXCEPTION_IF_NULL(abstract);
  auto shape_value_node = kernel_graph->NewValueNode(abstract, shape_value);
  MS_EXCEPTION_IF_NULL(shape_value_node);
  kernel_graph->AddValueNodeToGraph(shape_value_node);
  return shape_value_node;
}

CNodePtr AddCastNode(const FuncGraphPtr &func_graph, const TypeId dst_type, const CNodePtr &node, const bool is_input) {
  std::vector<AnfNodePtr> new_cast_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name()))};
  BaseShapePtr shape;
  if (is_input) {
    auto node_input = common::AnfAlgo::GetInputNode(node, 0);
    (void)new_cast_inputs.emplace_back(node_input);
    shape = common::AnfAlgo::GetOutputDetailShape(node_input, 0);
  } else {
    (void)new_cast_inputs.emplace_back(node);
    shape = common::AnfAlgo::GetOutputDetailShape(node, 0);
  }
  CNodePtr new_cast = NewCNode(new_cast_inputs, func_graph, {node});
  new_cast->set_scope(node->scope());
  new_cast->set_abstract(node->abstract());
  common::AnfAlgo::SetNodeAttr(kAttrDstType, MakeValue(static_cast<size_t>(dst_type)), new_cast);
  common::AnfAlgo::SetOutputTypeAndDetailShape({dst_type}, {shape}, new_cast.get());
  return new_cast;
}

AnfNodePtr CreateNodeBase(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &new_node_inputs,
                          const AnfNodePtr &node) {
  auto new_node = graph->NewCNode(new_node_inputs);
  MS_EXCEPTION_IF_NULL(new_node);

  new_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  new_node->set_scope(node->scope());
  new_node->set_abstract(node->abstract());

  auto types = {common::AnfAlgo::GetOutputInferDataType(node, 0)};
  auto shapes = {common::AnfAlgo::GetOutputInferShape(node, 0)};
  common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, new_node.get());

  return new_node;
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
        MS_LOG(EXCEPTION) << "Cast value node ptr fail.";
      }
      auto a_value_ptr = a_value_node_ptr->value();
      if (a_value_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "Value ptr is nullptr.";
      }

      auto b_value_node_ptr = b_node->cast<ValueNodePtr>();
      if (b_value_node_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "Cast value node ptr fail.";
      }
      auto b_value_ptr = b_value_node_ptr->value();
      if (b_value_ptr == nullptr) {
        MS_LOG(EXCEPTION) << "Value ptr is nullptr.";
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
ValueNodePtr CreateValueNodeWithSexp(const BaseRef &sexp, PrimitiveVarMap *primitive_vars) {
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
    auto value = utils::cast<ValuePtr>(sexp);
    if (utils::isa<PrimitivePtr>(sexp)) {
      auto prim = utils::cast<PrimitivePtr>(sexp);
      if (primitive_vars->find(prim) != primitive_vars->end()) {
        prim = std::make_shared<Primitive>(prim->name());
        value = prim;
      }
      (*primitive_vars)[prim] = std::make_shared<Var>(prim);
    }
    return NewValueNode(value);
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

// rectify absttract if the input has been converted to the attr
AbstractBasePtrList RectifyAbstractFromRegAttr(const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &input_abstract) {
  MS_EXCEPTION_IF_NULL(primitive);
  opt::ConstInputToAttrInfoRegister reg;
  if (!opt::ConstInputToAttrInfoRegistry::Instance().GetRegisterByOpName(primitive->name(), &reg)) {
    return input_abstract;
  }
  if (common::AnfAlgo::HasDynamicShapeFlag(primitive)) {
    return input_abstract;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device == kGPUDevice) {
    if (IsOneOfDynamicShapeConstInputToAttrGPU(primitive->name())) {
      return input_abstract;
    }
  } else if (IsOneOfDynamicShapeConstInputToAttr(primitive->name())) {
    return input_abstract;
  }
  auto convert_input_list = reg.GetConstInputAttrInfo();
  auto input_names = primitive->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    return input_abstract;
  }
  auto input_names_vec = GetValue<std::vector<std::string>>(input_names);
  AbstractBasePtrList rectify_abs_list;
  size_t ori_index = 0;
  rectify_abs_list.resize(input_names_vec.size());
  for (size_t index = 0; index < rectify_abs_list.size(); ++index) {
    // if convert input list find the index it means the input has been converted to the attr
    if (convert_input_list.find(index) != convert_input_list.end()) {
      AbstractBasePtr rectify_abs = nullptr;
      auto input_name = input_names_vec[index];
      auto attr = primitive->GetAttr(input_name);
      if (attr != nullptr) {
        rectify_abs = attr->ToAbstract();
      } else {
        MS_LOG(DEBUG) << "the node prim name :" << primitive->name() << "input index :" << index
                      << " input name :" << input_name << "has not been converted to the attr";
        rectify_abs = input_abstract[ori_index++];
      }
      rectify_abs_list[index] = rectify_abs;
      continue;
    }
    if (ori_index > input_abstract.size()) {
      MS_LOG(EXCEPTION) << "Index " << ori_index << " is out of range in input abstract size " << input_abstract.size();
    }
    rectify_abs_list[index] = input_abstract[ori_index++];
  }
  return rectify_abs_list;
}

AbstractBasePtrList RectifyAbstractFromDynamicInput(const PrimitivePtr &primitive,
                                                    const AbstractBasePtrList &input_abstract) {
  auto dynamic_inputs_list = primitive->GetAttr(kAttrDynInputSizes);
  if (dynamic_inputs_list == nullptr) {
    return input_abstract;
  }
  AbstractBasePtrList rectifyed_abs_list;
  const int kNotDynamicFlag = -1;
  auto dynamic_inputs_index = GetValue<std::vector<int64_t>>(dynamic_inputs_list);
  size_t input_index = 0;
  for (auto item : dynamic_inputs_index) {
    if (item == kNotDynamicFlag) {
      if (input_index >= input_abstract.size()) {
        MS_LOG(EXCEPTION) << "Index " << input_index << " is out of range in input abstract " << input_abstract.size();
      }
      (void)rectifyed_abs_list.emplace_back(input_abstract[input_index++]);
    } else {
      if (item < 0) {
        MS_LOG(EXCEPTION) << "The dynamic input size check error the index should be -1 or positive number but got "
                          << item;
      }
      AbstractBasePtrList dynamic_inputs_abs;
      for (auto index = item; index > 0; --index) {
        if (input_index >= input_abstract.size()) {
          MS_LOG(EXCEPTION) << "Index " << input_index << " is out of range in input abstract "
                            << input_abstract.size();
        }
        (void)dynamic_inputs_abs.emplace_back(input_abstract[input_index++]);
      }
      (void)rectifyed_abs_list.emplace_back(std::make_shared<abstract::AbstractTuple>(dynamic_inputs_abs));
    }
  }
  return rectifyed_abs_list;
}

AbstractBasePtrList RectifyAbstract(const PrimitivePtr &primitive, const AbstractBasePtrList &input_abstract) {
  auto rectify_abs_list = RectifyAbstractFromRegAttr(primitive, input_abstract);
  return RectifyAbstractFromDynamicInput(primitive, rectify_abs_list);
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
  auto value_node = CreateValueNodeWithSexp(sexp, primitive_vars);
  if (value_node == nullptr) {
    MS_LOG(EXCEPTION) << "Sexp cannot converted, sexp: " + sexp.ToString();
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
  if (iter == (*equiv).cend()) {
    MS_LOG(INFO) << "The equiv map doesn't contain the var_node after matched.";
    return nullptr;
  }
  auto res = utils::cast<AnfNodePtr>(iter->second);
  if (res == nullptr) {
    MS_LOG(EXCEPTION) << "Cast fail! Maybe var is not a anf node";
  }
  return res;
}

int64_t GetGetitemIndex(const AnfNodePtr &getitem) {
  if (!getitem->isa<CNode>() || IsPrimitive(getitem, prim::kPrimTupleGetItem)) {
    MS_LOG(EXCEPTION) << "Expect TupleGetItem, but got " << getitem->DebugString();
  }
  auto vnode = GetValueNode(getitem->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem));
  return GetValue<int64_t>(vnode);
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
  return common::AnfAlgo::HasNodeAttr(attr_name, cnode) && common::AnfAlgo::GetNodeAttr<bool>(node, attr_name);
}

bool CheckSupportDataType(const AnfNodePtr &node, const std::set<TypeId> &supported_data_type_set) {
  MS_EXCEPTION_IF_NULL(node);
  TypeId data_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  if (supported_data_type_set.find(data_type) != supported_data_type_set.end()) {
    return true;
  }
  MS_LOG(DEBUG) << "Not supported data type. Node:" << node->DebugString();
  return false;
}

ValueNodePtr MakeValueNode(const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  ValueNodePtr new_value_node = std::make_shared<ValueNode>(value_node->value());
  MS_EXCEPTION_IF_NULL(new_value_node);
  new_value_node->set_abstract(value_node->abstract());
  // create kernel_info fo new value node
  auto kernel_info = std::make_shared<device::KernelInfo>();
  new_value_node->set_kernel_info(kernel_info);
  // create kernel_build_info for new value node
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(kernel_build_info_builder);
  // set the format of value_node to DEFAULT_FORMAT
  kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
  // set value node initial device data type = infer data type
  std::vector<TypeId> types;
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(value_node);
  for (size_t index = 0; index < output_num; ++index) {
    types.push_back(kTypeUnknown);
  }
  kernel_build_info_builder->SetOutputsDeviceType(types);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), new_value_node.get());
  return new_value_node;
}

void TransferDependOrUpdateState(const CNodePtr &old_node, const FuncGraphPtr &graph, const CNodePtr &new_node) {
  MS_EXCEPTION_IF_NULL(old_node);
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  // Find BatchNorm's output which is a Depend or UpdateState.
  auto node_users = manager->node_users()[old_node];
  for (const auto &node_index : node_users) {
    AnfNodePtr output = node_index.first;
    MS_EXCEPTION_IF_NULL(output);
    if (common::AnfAlgo::CheckPrimitiveType(output, prim::kPrimDepend) ||
        common::AnfAlgo::CheckPrimitiveType(output, prim::kPrimUpdateState)) {
      auto depend = output->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(depend);
      manager->SetEdge(depend, node_index.second, new_node);
    }
  }
}

void CppInferShape(const PrimitivePtr &prim, const AbstractBasePtrList &args_spec_list,
                   const AbstractBasePtr &out_abs) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(out_abs);
  auto &prim_eval_implement_map = abstract::GetPrimitiveToEvalImplMap();
  auto ret = prim_eval_implement_map.find(prim);
  if (ret != prim_eval_implement_map.end()) {
    // fing infer function in the front infer map and restore input abastract form dynamic inputs and reg attr
    MS_EXCEPTION_IF_CHECK_FAIL(ret->second.IsImplInferShapeAndType(),
                               "There is no infer-shape implement for frontend!");
    auto infer_spec_list = RectifyAbstract(prim, args_spec_list);
    auto shape = ret->second.InferShape(prim, infer_spec_list);
    if (shape == nullptr) {
      MS_LOG(EXCEPTION) << "Infer shape with frontend function failed.";
    }
    out_abs->set_shape(shape);
    return;
  } else {
    // if the infer function has been not founded in the front infer map find it in the backend infer map instead
    auto &prim_backend_eval_impl_map = abstract::GetPrimitiveToBackendEvalImplMap();
    auto ret_backend = prim_backend_eval_impl_map.find(prim);
    if (ret_backend != prim_backend_eval_impl_map.end()) {
      MS_EXCEPTION_IF_CHECK_FAIL(ret_backend->second.IsImplInferShapeAndType(),
                                 "There is no infer-shape implement for backend!");
      auto infer_spec_list = args_spec_list;
      if (!ret_backend->second.IsInWhileList()) {
        infer_spec_list = RectifyAbstract(prim, args_spec_list);
      }
      auto shape = ret_backend->second.InferShape(prim, infer_spec_list);
      if (shape == nullptr) {
        MS_LOG(EXCEPTION) << "Infer shape with backend function failed";
      }
      out_abs->set_shape(shape);
      return;
    }
  }

  MS_LOG(EXCEPTION) << "Get infer functions failed, the operator is not support dynamic shape yet, primitive name:"
                    << prim->name() << " primitive type:" << prim->type_name();
}

AbstractBasePtr CppInferShapeAndType(const PrimitivePtr &prim, const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(prim);
  auto &prim_eval_implement_map = abstract::GetPrimitiveToEvalImplMap();
  auto ret = prim_eval_implement_map.find(prim);
  if (ret != prim_eval_implement_map.end()) {
    // fing infer function in the front infer map and restore input abastract form dynamic inputs and reg attr
    MS_EXCEPTION_IF_CHECK_FAIL(ret->second.IsImplInferShapeAndType(), "There is no infer-abstract implement!");
    auto infer_spec_list = RectifyAbstract(prim, args_spec_list);
    return ret->second.InferShapeAndType(nullptr, prim, infer_spec_list);
  } else {
    // if the infer function has been not founded in the front infer map find it in the backend infer map instead
    auto &prim_backend_eval_impl_map = abstract::GetPrimitiveToBackendEvalImplMap();
    auto ret_backend = prim_backend_eval_impl_map.find(prim);
    if (ret_backend != prim_backend_eval_impl_map.end()) {
      MS_EXCEPTION_IF_CHECK_FAIL(ret_backend->second.IsImplInferShapeAndType(),
                                 "There is no infer-abstract implement!");
      auto infer_spec_list = args_spec_list;
      if (!ret_backend->second.IsInWhileList()) {
        infer_spec_list = RectifyAbstract(prim, args_spec_list);
      }
      return ret_backend->second.InferShapeAndType(nullptr, prim, infer_spec_list);
    }
  }
  MS_LOG(EXCEPTION) << "Get infer shape function failed, the operator is not support dynamic shape yet, primitive name:"
                    << prim->name() << " primitive type:" << prim->type_name();
}

kernel::KernelBuildInfoPtr GenerateKernelBuildInfo(const std::vector<AnfNodePtr> &node_list) {
  std::vector<std::string> inputs_device_format;
  std::vector<std::string> outputs_device_format;
  std::vector<TypeId> inputs_device_type;
  std::vector<TypeId> outputs_device_type;
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  for (size_t idx = 0; idx < node_list.size(); ++idx) {
    auto cnode = utils::cast<CNodePtr>(node_list[idx]);
    MS_EXCEPTION_IF_NULL(cnode);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
    for (size_t input_index = 0; input_index < input_num; ++input_index) {
      (void)inputs_device_format.emplace_back(kOpFormat_DEFAULT);
      (void)inputs_device_type.emplace_back(common::AnfAlgo::GetPrevNodeOutputInferDataType(cnode, input_index));
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(cnode);
    for (size_t output_index = 0; output_index < output_num; ++output_index) {
      (void)outputs_device_format.emplace_back(kOpFormat_DEFAULT);
      (void)outputs_device_type.emplace_back(common::AnfAlgo::GetOutputInferDataType(cnode, output_index));
    }
  }
  builder.SetInputsFormat(inputs_device_format);
  builder.SetOutputsFormat(outputs_device_format);
  builder.SetInputsDeviceType(inputs_device_type);
  builder.SetOutputsDeviceType(outputs_device_type);
  return builder.Build();
}

std::vector<int64_t> GetNodeOutputUsedNum(const session::KernelGraph &kernel_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto output_num = common::AnfAlgo::GetOutputTensorNum(node);
  std::vector<int64_t> output_used_num(output_num, 0);
  if (output_num == 1) {
    output_used_num[0] = SizeToLong(manager->node_users()[node].size());
  } else {
    for (auto out_getitem : manager->node_users()[node]) {
      MS_EXCEPTION_IF_NULL(out_getitem.first);
      if (!common::AnfAlgo::CheckPrimitiveType(out_getitem.first, prim::kPrimTupleGetItem)) {
        continue;
      }
      auto out_getitem_ptr = out_getitem.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(out_getitem_ptr);
      auto getitem_input2 = out_getitem_ptr->input(kInputNodeOutputIndexInTupleGetItem);
      auto output_idx = LongToSize(GetValue<int64_t>(GetValueNode(getitem_input2)));
      output_used_num[output_idx] = SizeToLong(manager->node_users()[out_getitem.first].size());
    }
  }
  return output_used_num;
}

int64_t GetNodeOutputTotalUsedNum(const session::KernelGraph &kernel_graph, const AnfNodePtr &node) {
  auto output_used_num = GetNodeOutputUsedNum(kernel_graph, node);
  return std::accumulate(output_used_num.begin(), output_used_num.end(), int64_t(0));
}

void GetCustomOpAttrIndex(const PrimitivePtr &primitive, mindspore::HashSet<size_t> *indexes) {
  if (primitive == nullptr || primitive->name() != prim::kPrimCustom->name()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(indexes);
  auto input_names = primitive->GetAttr(kAttrInputNames);
  auto attr_names = primitive->GetAttr(kAttrAttrNames);
  if (input_names == nullptr || attr_names == nullptr) {
    return;
  }
  auto input_names_vec = GetValue<std::vector<std::string>>(input_names);
  auto attr_names_vec = GetValue<std::vector<std::string>>(attr_names);
  if (input_names_vec.size() >= attr_names_vec.size()) {
    size_t offset = input_names_vec.size() - attr_names_vec.size();
    for (size_t i = offset; i < input_names_vec.size(); ++i) {
      if (input_names_vec[i] != attr_names_vec[i - offset]) {
        MS_LOG(EXCEPTION) << primitive->name() << " found mismatching attr name " << input_names_vec[i]
                          << "in input_names and " << attr_names_vec[i - offset] << " in attr_names";
      }
      (void)indexes->insert(i);
    }
  }
}
}  // namespace opt
}  // namespace mindspore
