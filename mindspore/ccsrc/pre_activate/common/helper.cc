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

#include "pre_activate/common/helper.h"
#include <string>
#include <unordered_set>
#include <algorithm>
#include "utils/utils.h"
#include "utils/base_ref.h"
#include "session/anf_runtime_algorithm.h"
#include "operator/ops.h"
#include "common/utils.h"
#include "device/kernel_info.h"
#include "utils/context/ms_context.h"

namespace mindspore {
namespace opt {
constexpr size_t kType32Len = 4;
std::vector<int> Convert2Int(const std::vector<size_t> &v) {
  std::vector<int> result;
  (void)std::transform(v.begin(), v.end(), std::back_inserter(result), SizeToInt);
  return result;
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
    } else {
      return false;
    }
  }
  return false;
}

bool CheckIfCNodeAndInputSize(const AnfNodePtr &node, int input_size, CNodePtr *cnode) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(ERROR) << "The node is expected to be a cnode";
    return false;
  }
  *cnode = node->cast<CNodePtr>();
  if (*cnode == nullptr) {
    return false;
  }
  if ((*cnode)->inputs().size() < IntToSize(input_size)) {
    auto op_name = AnfAlgo::GetCNodeName(*cnode);
    MS_LOG(ERROR) << "op[" + op_name + "] has less than " << input_size << " inputs.";
    return false;
  }
  return true;
}

CNodePtr CheckAnfNodeIfCNodeAndInputSize(const AnfNodePtr &node, int input_size) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "The node is expected to be a cnode";
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() != IntToSize(input_size)) {
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
    } else {
      return false;
    }
  }
  return false;
}

void CreateOutputsOfConvBn1(const FuncGraphPtr &func_graph, const CNodePtr &conv_cnode, const CNodePtr &bn_cnode,
                            std::vector<AnfNodePtr> *conv_bn1_outputs) {
  auto prim = std::make_shared<Primitive>(kConvBN1OpName);
  std::vector<AnfNodePtr> conv_bn1_inputs = {NewValueNode(prim)};
  MS_EXCEPTION_IF_NULL(conv_cnode);
  // All the inputs of conv_bn1 are from the inputs of conv
  for (size_t i = 1; i < conv_cnode->inputs().size(); i++) {
    conv_bn1_inputs.push_back(conv_cnode->input(i));
  }
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr conv_bn1_cnode = func_graph->NewCNode(conv_bn1_inputs);
  MS_EXCEPTION_IF_NULL(conv_bn1_cnode);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  conv_bn1_cnode->set_kernel_info(kernel_info);
  // Set attr for conv_bn1
  AnfAlgo::CopyNodeAttrs(conv_cnode, conv_bn1_cnode);
  // Set abstract of conv_bn1
  MS_EXCEPTION_IF_NULL(bn_cnode);
  auto bn_abstract_tuple = dyn_cast<abstract::AbstractTuple>(bn_cnode->abstract());
  MS_EXCEPTION_IF_NULL(bn_abstract_tuple);
  AbstractBasePtrList conv_bn1_abstract_list;
  conv_bn1_abstract_list.push_back(conv_cnode->abstract());
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(
    kFloat32, Convert2Int(AnfAlgo::GetPrevNodeOutputInferShape(bn_cnode, kVariance - 1)));
  conv_bn1_abstract_list.push_back(abstract_tensor);
  conv_bn1_abstract_list.push_back(bn_abstract_tuple->elements()[kSaveMean]);
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(conv_bn1_abstract_list);
  conv_bn1_cnode->set_abstract(abstract_tuple);

  CreateMultipleOutputsOfAnfNode(func_graph, conv_bn1_cnode, kConvBn1OutputNum, conv_bn1_outputs);
}

void CreateOutputsOfFusedBn2(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &fused_bn1_outputs,
                             const CNodePtr &bn_node, std::vector<AnfNodePtr> *fused_bn2_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(bn_node);
  MS_EXCEPTION_IF_NULL(fused_bn2_outputs);
  if (bn_node->inputs().size() != kBnInputNum) {
    MS_LOG(EXCEPTION) << "BN node has wrong input size";
  }
  if (fused_bn1_outputs.size() != kBN1OutputNum) {
    MS_LOG(EXCEPTION) << "BN1 outputs has wrong input size";
  }

  // the inputs of fused_bn2 are from the outputs of fused_bn1 and the inputs of bn
  std::vector<AnfNodePtr> fused_bn2_inputs = {NewValueNode(std::make_shared<Primitive>(kFusedBN2OpName))};
  fused_bn2_inputs.push_back(fused_bn1_outputs[0]);
  fused_bn2_inputs.push_back(fused_bn1_outputs[1]);
  fused_bn2_inputs.push_back(bn_node->input(4));
  fused_bn2_inputs.push_back(bn_node->input(5));
  auto fused_bn2 = graph->NewCNode(fused_bn2_inputs);
  MS_EXCEPTION_IF_NULL(fused_bn2);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  fused_bn2->set_kernel_info(kernel_info);
  auto types = {AnfAlgo::GetOutputInferDataType(bn_node, 4), AnfAlgo::GetOutputInferDataType(bn_node, 1),
                AnfAlgo::GetOutputInferDataType(bn_node, 2)};
  auto shapes = {AnfAlgo::GetOutputInferShape(bn_node, 4), AnfAlgo::GetOutputInferShape(bn_node, 1),
                 AnfAlgo::GetOutputInferShape(bn_node, 2)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, fused_bn2.get());
  fused_bn2->set_scope(bn_node->scope());
  AnfAlgo::CopyNodeAttr(kAttrMomentum, bn_node, fused_bn2);

  CreateMultipleOutputsOfAnfNode(graph, fused_bn2, kBN2OutputNum, fused_bn2_outputs);
}

void CreateOutputsOfFusedBn3(const FuncGraphPtr &graph, const AnfNodePtr &data_input,
                             const std::vector<AnfNodePtr> &fused_bn1_outputs,
                             const std::vector<AnfNodePtr> &fused_bn2_outputs, const CNodePtr &bn_node,
                             std::vector<AnfNodePtr> *fused_bn3_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(data_input);
  MS_EXCEPTION_IF_NULL(bn_node);
  MS_EXCEPTION_IF_NULL(fused_bn3_outputs);
  if (bn_node->inputs().size() != kBnInputNum) {
    MS_LOG(EXCEPTION) << "BN node has wrong input size";
  }

  if (fused_bn1_outputs.size() != kBN1OutputNum) {
    MS_LOG(EXCEPTION) << "BN1 outputs has wrong input size";
  }

  if (fused_bn2_outputs.size() != kBN2OutputNum) {
    MS_LOG(EXCEPTION) << "BN2 outputs has wrong input size";
  }

  // the inputs of fused_bn3 are from the outputs of fused_bn1 and the inputs of bn
  std::vector<AnfNodePtr> fused_bn3_inputs = {NewValueNode(std::make_shared<Primitive>(kFusedBN3OpName))};
  fused_bn3_inputs.push_back(data_input);
  fused_bn3_inputs.push_back(fused_bn1_outputs[0]);
  fused_bn3_inputs.push_back(fused_bn2_outputs[0]);
  fused_bn3_inputs.push_back(bn_node->input(2));
  fused_bn3_inputs.push_back(bn_node->input(3));
  auto fused_bn3 = graph->NewCNode(fused_bn3_inputs);
  MS_EXCEPTION_IF_NULL(fused_bn3);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  fused_bn3->set_kernel_info(kernel_info);
  auto types = {AnfAlgo::GetOutputInferDataType(bn_node, 0)};
  auto shapes = {AnfAlgo::GetOutputInferShape(bn_node, 0)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, fused_bn3.get());

  fused_bn3->set_scope(bn_node->scope());
  AnfAlgo::CopyNodeAttr(kAttrEpsilon, kAttrEps, bn_node, fused_bn3);

  (*fused_bn3_outputs).push_back(fused_bn3);
}

void CreateMultipleOutputsOfAnfNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t output_num,
                                    std::vector<AnfNodePtr> *outputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(outputs);
  for (size_t i = 0; i < output_num; i++) {
    auto idx = NewValueNode(SizeToInt(i));
    MS_EXCEPTION_IF_NULL(idx);
    int temp = SizeToInt(i);
    auto imm = std::make_shared<Int32Imm>(temp);
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
  std::vector<int> tensor_shape = {SizeToInt(values.size())};
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_ptr->type_id(), tensor_shape);
  MS_EXCEPTION_IF_NULL(tensor);
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, type_ptr};
  tensor->set_device_info(device_info);
  auto data_ptr = tensor->data_c(true);
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
  ValuePtr v = *(value_tuple->value().begin());
  MS_EXCEPTION_IF_NULL(v);
  // Currently we only deal with the scalar tuple
  if (!v->isa<Scalar>()) {
    MS_LOG(WARNING) << "The value " << v << "of tuple is not a scalar";
    return nullptr;
  }
  ScalarPtr scalar = v->cast<ScalarPtr>();
  MS_EXCEPTION_IF_NULL(scalar);
  if (scalar->isa<IntergerImm>()) {
    tensor = CreateTensorWithValueTuple<int>(value_tuple, kInt32, kType32Len);
  } else if (scalar->isa<FloatImm>()) {
    tensor = CreateTensorWithValueTuple<float>(value_tuple, kFloat32, kType32Len);
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
  if (context_ptr->device_target() != kAscendDevice) {
    return false;
  }
  static std::unordered_set<std::string> nop_nodes = {prim::kPrimReshape->name(), kExpandDimsOpName,
                                                      prim::kPrimSqueeze->name(), prim::kPrimFlatten->name()};
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (nop_nodes.find(AnfAlgo::GetCNodeName(cnode)) == nop_nodes.end()) {
    return false;
  }
  return true;
}

void HideNopNode(session::KernelGraph *const graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto execution_order = graph->execution_order();
  MS_LOG(INFO) << "nop node info (Before Remove) size: " << execution_order.size();
  std::vector<CNodePtr> new_nodes;
  for (auto &cnode : execution_order) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (!IsNopNode(cnode)) {
      new_nodes.push_back(cnode);
    }
  }
  graph->set_execution_order(new_nodes);
  MS_LOG(INFO) << "nop node info (After Remove) size: " << graph->execution_order().size();
}

void RemoveNopNode(session::KernelGraph *const graph) {
  MS_EXCEPTION_IF_NULL(graph);
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<CNodePtr> new_nodes;
    for (auto &cnode : graph->execution_order()) {
      MS_EXCEPTION_IF_NULL(cnode);
      // ignore nop node itself
      if (IsNopNode(cnode)) {
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

bool IsUsedByOthers(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(node) == manager->node_users().end()) {
    MS_LOG(EXCEPTION) << "node has no output in manager";
  }
  return manager->node_users()[node].size() > 1;
}

AnfNodePtr CreatTupleGetItemNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t output_idx) {
  auto idx = NewValueNode(SizeToInt(output_idx));
  MS_EXCEPTION_IF_NULL(idx);
  auto imm = std::make_shared<Int32Imm>(SizeToInt(output_idx));
  auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
  idx->set_abstract(abstract_scalar);
  AnfNodePtr tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
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
    cnode->set_inputs(new_inputs);
    // Update cnode's input_names attr
    primitive->set_attr(kAttrInputNames, MakeValue(new_input_names));
  }
}
}  // namespace opt
}  // namespace mindspore
