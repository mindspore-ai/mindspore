/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "session/anf_runtime_algorithm.h"
#include <memory>
#include <algorithm>
#include <map>
#include <set>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "operator/ops.h"
#include "utils/utils.h"
#include "device/kernel_info.h"
#include "device/device_address.h"
#include "pre_activate/common/helper.h"
#include "kernel/kernel.h"
#include "kernel/kernel_build_info.h"
#include "common/utils.h"
#include "common/trans.h"

namespace mindspore {
namespace session {
using abstract::AbstractTensor;
using abstract::AbstractTuple;
using device::KernelInfo;
using device::ascend::AscendDeviceAddress;
using kernel::KernelBuildInfoPtr;
using kernel::KernelMod;
using kernel::KernelModPtr;
namespace {
std::vector<size_t> TransShapeToSizet(const abstract::ShapePtr &shape) {
  MS_EXCEPTION_IF_NULL(shape);
  std::vector<size_t> shape_size_t;
  std::transform(shape->shape().begin(), shape->shape().end(), std::back_inserter(shape_size_t), IntToSize);
  return shape_size_t;
}
}  // namespace

KernelWithIndex AnfRuntimeAlgorithm::VisitKernel(const AnfNodePtr &anf_node, size_t index) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (anf_node->isa<ValueNode>()) {
    return std::make_pair(anf_node, 0);
  } else if (anf_node->isa<Parameter>()) {
    return std::make_pair(anf_node, 0);
  } else if (anf_node->isa<CNode>()) {
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto input0 = cnode->input(0);
    MS_EXCEPTION_IF_NULL(input0);
    if (IsPrimitive(input0, prim::kPrimMakeTuple)) {
      auto node = cnode->input(index + IntToSize(1));
      MS_EXCEPTION_IF_NULL(node);
      return VisitKernel(node, 0);
    } else if (IsPrimitive(input0, prim::kPrimTupleGetItem)) {
      if (cnode->inputs().size() != kTupleGetItemInputSize) {
        MS_LOG(EXCEPTION) << "The node tuple_get_item must have 2 inputs!";
      }
      auto input2 = cnode->input(kInputNodeOutputIndexInTupleGetItem);
      MS_EXCEPTION_IF_NULL(input2);
      auto value_node = input2->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      int item_idx = GetValue<int>(value_node->value());
      return VisitKernel(cnode->input(kRealInputNodeIndexInTupleGetItem), IntToSize(item_idx));
    } else if (IsPrimitive(input0, prim::kPrimDepend) || IsPrimitive(input0, prim::kPrimControlDepend)) {
      return VisitKernel(cnode->input(kRealInputIndexInDepend), 0);
    } else {
      return std::make_pair(anf_node, index);
    }
  } else {
    MS_LOG(EXCEPTION) << "The input is invalid";
  }
}

KernelWithIndex AnfRuntimeAlgorithm::VisitKernelWithReturnType(const AnfNodePtr &anf_node, size_t index,
                                                               bool visit_nop_node,
                                                               const std::vector<PrimitivePtr> &return_types) {
  MS_EXCEPTION_IF_NULL(anf_node);
  for (const auto &prim_type : return_types) {
    if (CheckPrimitiveType(anf_node, prim_type)) {
      return std::make_pair(anf_node, index);
    }
  }
  if (anf_node->isa<ValueNode>()) {
    return std::make_pair(anf_node, 0);
  } else if (anf_node->isa<Parameter>()) {
    return std::make_pair(anf_node, 0);
  } else if (anf_node->isa<CNode>()) {
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto input0 = cnode->input(0);
    MS_EXCEPTION_IF_NULL(input0);
    if (IsPrimitive(input0, prim::kPrimTupleGetItem)) {
      if (cnode->inputs().size() != kTupleGetItemInputSize) {
        MS_LOG(EXCEPTION) << "The node tuple_get_item must have 2 inputs!";
      }
      auto input2 = cnode->input(kInputNodeOutputIndexInTupleGetItem);
      MS_EXCEPTION_IF_NULL(input2);
      auto value_node = input2->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      int item_idx = GetValue<int>(value_node->value());
      return VisitKernelWithReturnType(cnode->input(kRealInputNodeIndexInTupleGetItem), IntToSize(item_idx),
                                       visit_nop_node);
    } else if (IsPrimitive(input0, prim::kPrimDepend) || IsPrimitive(input0, prim::kPrimControlDepend)) {
      return VisitKernelWithReturnType(cnode->input(kRealInputIndexInDepend), 0, visit_nop_node);
    } else if (opt::IsNopNode(cnode) && visit_nop_node) {
      if (cnode->inputs().size() == 2) {
        return VisitKernelWithReturnType(cnode->input(1), 0, visit_nop_node);
      } else {
        MS_LOG(EXCEPTION) << cnode->DebugString() << "Invalid nop node";
      }
    } else {
      return std::make_pair(anf_node, index);
    }
  } else {
    MS_LOG(EXCEPTION) << "The input is invalid";
  }
}

std::vector<AnfNodePtr> AnfRuntimeAlgorithm::GetAllOutput(const AnfNodePtr &node,
                                                          const std::vector<PrimitivePtr> &return_types) {
  std::vector<AnfNodePtr> ret;
  auto return_prim_type = return_types;
  // if visited make_tuple should return back
  return_prim_type.push_back(prim::kPrimMakeTuple);
  auto item_with_index = AnfAlgo::VisitKernelWithReturnType(node, 0, false, return_prim_type);
  if (AnfAlgo::CheckPrimitiveType(item_with_index.first, prim::kPrimMakeTuple)) {
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    auto make_tuple = item_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    for (size_t i = 1; i < make_tuple->inputs().size(); i++) {
      auto input_i_vector = GetAllOutput(make_tuple->input(i), return_types);
      (void)std::copy(input_i_vector.begin(), input_i_vector.end(), std::back_inserter(ret));
    }
    return ret;
  }
  ret.push_back(item_with_index.first);
  return ret;
}

AnfNodePtr AnfRuntimeAlgorithm::GetCNodePrimitiveNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->input(kAnfPrimitiveIndex);
}

PrimitivePtr AnfRuntimeAlgorithm::GetCNodePrimitive(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto attr_input = GetCNodePrimitiveNode(cnode);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto value_node = attr_input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  auto primitive = value->cast<PrimitivePtr>();
  return primitive;
}

bool AnfRuntimeAlgorithm::CheckPrimitiveType(const AnfNodePtr &node, const PrimitivePtr &primitive_type) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  return IsPrimitive(cnode->input(kAnfPrimitiveIndex), primitive_type);
}

std::string AnfRuntimeAlgorithm::GetCNodeName(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>()) {
    auto primitive = AnfAlgo::GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(primitive);
    return primitive->name();
  }
  MS_LOG(EXCEPTION) << "Unknown anf node type " << node->DebugString();
}

std::string AnfRuntimeAlgorithm::GetNodeDebugString(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->DebugString();
}

void AnfRuntimeAlgorithm::SetNodeAttr(const std::string &key, const ValuePtr &value, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Only cnode has attr, but this anf is " << node->DebugString();
  }
  auto primitive = AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive->set_attr(key, value);
}

void AnfRuntimeAlgorithm::CopyNodeAttr(const std::string &key, const AnfNodePtr &from, const AnfNodePtr &to) {
  CopyNodeAttr(key, key, from, to);
}

void AnfRuntimeAlgorithm::CopyNodeAttr(const std::string &old_key, const std::string &new_key, const AnfNodePtr &from,
                                       const AnfNodePtr &to) {
  MS_EXCEPTION_IF_NULL(from);
  MS_EXCEPTION_IF_NULL(to);
  if (!from->isa<CNode>() || !to->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Only cnode has attr, but this from_anf is " << from->DebugString() << " ,to_node is "
                      << to->DebugString();
  }
  auto from_primitive = AnfAlgo::GetCNodePrimitive(from);
  MS_EXCEPTION_IF_NULL(from_primitive);
  auto to_primitive = AnfAlgo::GetCNodePrimitive(to);
  MS_EXCEPTION_IF_NULL(to_primitive);
  to_primitive->set_attr(new_key, from_primitive->GetAttr(old_key));
}

void AnfRuntimeAlgorithm::CopyNodeAttrs(const AnfNodePtr &from, const AnfNodePtr &to) {
  MS_EXCEPTION_IF_NULL(from);
  MS_EXCEPTION_IF_NULL(to);
  if (!from->isa<CNode>() || !to->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Only cnode has attr, but this from_anf is " << from->DebugString() << ",to_node is "
                      << from->DebugString();
  }
  auto from_primitive = AnfAlgo::GetCNodePrimitive(from);
  MS_EXCEPTION_IF_NULL(from_primitive);
  auto to_primitive = AnfAlgo::GetCNodePrimitive(to);
  MS_EXCEPTION_IF_NULL(to_primitive);
  (void)to_primitive->SetAttrs(from_primitive->attrs());
}

void AnfRuntimeAlgorithm::EraseNodeAttr(const std::string &key, const AnfNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Only cnode has attr, but this anf is " << node->DebugString();
  }
  auto primitive = AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive->EraseAttr(key);
}

bool AnfRuntimeAlgorithm::HasNodeAttr(const std::string &key, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(WARNING) << "Only cnode has attr, but this anf is " << node->DebugString();
    return false;
  }
  auto primitive = AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  return primitive->HasAttr(key);
}

size_t AnfRuntimeAlgorithm::GetInputTensorNum(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Only cnode has real input, but this anf is " << node->DebugString();
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  size_t input_num = cnode->inputs().size();
  if (input_num == 0) {
    MS_LOG(EXCEPTION) << "cnode inputs size can't be zero";
  }
  // exclude intputs[0],which is value_node storing attr,inputs left are real input
  return input_num - 1;
}

size_t AnfRuntimeAlgorithm::GetOutputTensorNum(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  TypePtr type = node->Type();
  MS_EXCEPTION_IF_NULL(type);
  if (type->isa<Tuple>()) {
    auto tuple_type = type->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_type);
    return tuple_type->size();
  } else if (type->isa<TensorType>() || type->isa<Number>()) {
    return 1;
  } else if (type->isa<TypeNone>()) {
    return 0;
  } else {
    return 1;
  }
}

std::string AnfRuntimeAlgorithm::GetOutputFormat(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_idx > GetOutputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "Output index:" << output_idx
                      << " is out of the node output range :" << GetOutputTensorNum(node) << " #node ["
                      << node->DebugString() << "]";
  }
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->GetOutputFormat(output_idx);
}

std::string AnfRuntimeAlgorithm::GetInputFormat(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (input_idx > GetInputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "Input index :" << input_idx
                      << " is out of the number node Input range :" << GetInputTensorNum(node) << "#node ["
                      << node->DebugString() << "]";
  }
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->GetInputFormat(input_idx);
}

KernelWithIndex AnfRuntimeAlgorithm::GetPrevNodeOutput(const AnfNodePtr &anf_node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (!anf_node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << anf_node->DebugString() << "anf_node is not CNode.";
  }
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (input_idx + 1 >= cnode->inputs().size()) {
    MS_LOG(EXCEPTION) << "Input index " << input_idx << " is larger than input number " << GetInputTensorNum(cnode);
  }
  auto node = cnode->input(input_idx + 1);
  MS_EXCEPTION_IF_NULL(node);
  return VisitKernel(node, 0);
}

std::string AnfRuntimeAlgorithm::GetPrevNodeOutputFormat(const AnfNodePtr &anf_node, size_t input_idx) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
}

std::vector<size_t> AnfRuntimeAlgorithm::GetOutputInferShape(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  abstract::BaseShapePtr base_shape = node->Shape();
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::Shape>() && output_idx == 0) {
    return TransShapeToSizet(base_shape->cast<abstract::ShapePtr>());
  } else if (base_shape->isa<abstract::TupleShape>()) {
    auto tuple_shape = base_shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape);
    if (output_idx >= tuple_shape->size()) {
      MS_LOG(EXCEPTION) << "Output index " << output_idx << "is larger than output number " << tuple_shape->size()
                        << ".";
    }
    auto b_shp = (*tuple_shape)[output_idx];
    if (b_shp->isa<abstract::Shape>()) {
      return TransShapeToSizet(b_shp->cast<abstract::ShapePtr>());
    } else if (b_shp->isa<abstract::NoShape>()) {
      return std::vector<size_t>();
    } else {
      MS_LOG(EXCEPTION) << "The output type of ApplyKernel should be a NoShape , ArrayShape or a TupleShape, but it is "
                        << base_shape->ToString();
    }
  } else if (base_shape->isa<abstract::NoShape>()) {
    return std::vector<size_t>();
  }
  MS_LOG(EXCEPTION) << "The output type of ApplyKernel should be a NoShape , ArrayShape or a TupleShape, but it is "
                    << base_shape->ToString();
}

std::vector<size_t> AnfRuntimeAlgorithm::GetPrevNodeOutputInferShape(const AnfNodePtr &node, size_t input_idx) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputInferShape(kernel_with_index.first, kernel_with_index.second);
}

std::vector<size_t> AnfRuntimeAlgorithm::GetOutputDeviceShape(const AnfNodePtr &node, size_t output_idx) {
  auto format = GetOutputFormat(node, output_idx);
  auto infer_shape = GetOutputInferShape(node, output_idx);
  if (infer_shape.empty()) {
    return infer_shape;
  }
  // if format is default_format or NC1KHKWHWC0,device shape = original shape
  if (trans::IsNeedPadding(format, infer_shape.size())) {
    infer_shape = trans::PaddingShapeTo4d(infer_shape, GetOutputReshapeType(node, output_idx));
  }
  return trans::TransShapeToDevice(infer_shape, format);
}

std::vector<size_t> AnfRuntimeAlgorithm::GetInputDeviceShape(const AnfNodePtr &node, size_t input_idx) {
  auto format = GetInputFormat(node, input_idx);
  auto infer_shape = GetPrevNodeOutputInferShape(node, input_idx);
  if (infer_shape.empty()) {
    return infer_shape;
  }
  // if format is default_format or NC1KHKWHWC0,device shape = original shape
  if (trans::IsNeedPadding(format, infer_shape.size())) {
    infer_shape = trans::PaddingShapeTo4d(infer_shape, GetInputReshapeType(node, input_idx));
  }
  return trans::TransShapeToDevice(infer_shape, format);
}

std::vector<kernel::Axis> AnfRuntimeAlgorithm::GetInputReshapeType(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (input_idx > GetInputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "The index:" << input_idx
                      << " is out of range of the node's input size : " << GetInputTensorNum(node) << "#node["
                      << node->DebugString() << "]";
  }
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  if (build_info->IsInputDefaultPadding()) {
    return {};
  }
  return build_info->GetInputReshapeType(input_idx);
}

std::vector<kernel::Axis> AnfRuntimeAlgorithm::GetOutputReshapeType(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_idx > GetOutputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "The index [" << output_idx << "] is out of range of the node's output size [ "
                      << GetOutputTensorNum(node) << "#node[ " << node->DebugString() << "]";
  }
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  if (build_info->IsOutputDefaultPadding()) {
    return {};
  }
  return build_info->GetOutputReshapeType(output_idx);
}

TypeId AnfRuntimeAlgorithm::GetOutputInferDataType(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  TypePtr type_ptr = node->Type();
  MS_EXCEPTION_IF_NULL(type_ptr);
  if (type_ptr->isa<TensorType>() && output_idx == 0) {
    auto tensor_ptr = type_ptr->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    TypePtr elem = tensor_ptr->element();
    MS_EXCEPTION_IF_NULL(elem);
    return elem->type_id();
  } else if (type_ptr->isa<Tuple>()) {
    auto tuple_ptr = type_ptr->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_ptr);
    if (output_idx >= tuple_ptr->size()) {
      MS_LOG(EXCEPTION) << "Output index " << output_idx << " must be less than output number " << tuple_ptr->size();
    }
    auto tuple_i = (*tuple_ptr)[output_idx];
    MS_EXCEPTION_IF_NULL(tuple_i);
    if (tuple_i->isa<TensorType>()) {
      auto tensor_ptr = tuple_i->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(tensor_ptr);
      TypePtr elem = tensor_ptr->element();
      MS_EXCEPTION_IF_NULL(elem);
      return elem->type_id();
    } else if (tuple_i->isa<Number>()) {
      return tuple_i->type_id();
    } else {
      MS_LOG(WARNING) << "Not support type " << tuple_i->ToString();
      return tuple_i->type_id();
    }
  } else if (type_ptr->isa<Number>()) {
    return type_ptr->type_id();
  }
  return type_ptr->type_id();
}

TypeId AnfRuntimeAlgorithm::GetPrevNodeOutputInferDataType(const AnfNodePtr &node, size_t input_idx) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputInferDataType(kernel_with_index.first, kernel_with_index.second);
}

TypeId AnfRuntimeAlgorithm::GetOutputDeviceDataType(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_idx > GetOutputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "The index [" << output_idx << "] is out of range of the node's output size [ "
                      << GetOutputTensorNum(node) << "#node [ " << node->DebugString() << "]";
  }
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->GetOutputDeviceType(output_idx);
}

TypeId AnfRuntimeAlgorithm::GetInputDeviceDataType(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (input_idx > GetInputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "The index [" << input_idx << "] is out of range of the node's input size [ "
                      << GetInputTensorNum(node) << "#node [ " << node->DebugString() << "]";
  }
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->GetInputDeviceType(input_idx);
}

TypeId AnfRuntimeAlgorithm::GetPrevNodeOutputDeviceDataType(const AnfNodePtr &anf_node, size_t input_idx) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
}

// get output device addr of anf_node
const DeviceAddress *AnfRuntimeAlgorithm::GetOutputAddr(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (opt::IsNopNode(node)) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->inputs().size() == 2) {
      return AnfRuntimeAlgorithm::GetPrevNodeOutputAddr(cnode, 0);
    } else {
      MS_LOG(EXCEPTION) << node->DebugString() << "Invalid nop node";
    }
  }
  if (output_idx > GetOutputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "The index [" << output_idx << "] is out of range of the node's output size [ "
                      << GetOutputTensorNum(node) << "#node:[ " << node->DebugString() << "]";
  }
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetOutputAddr(output_idx);
  if (addr == nullptr) {
    MS_LOG(EXCEPTION) << "Output_idx " << output_idx << " of node " << node->DebugString()
                      << " output addr is not exist";
  }
  return addr;
}

DeviceAddressPtr AnfRuntimeAlgorithm::GetMutableOutputAddr(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (opt::IsNopNode(node)) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->inputs().size() == 2) {
      return AnfRuntimeAlgorithm::GetPrevNodeMutableOutputAddr(cnode, 0);
    } else {
      MS_LOG(EXCEPTION) << node->DebugString() << "Invalid nop node.";
    }
  }
  if (output_idx > GetOutputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "The index [" << output_idx << "] is out of range of the node's output size [ "
                      << GetOutputTensorNum(node) << "#node:[ " << node->DebugString() << "]";
  }
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetMutableOutputAddr(output_idx);
  if (addr == nullptr) {
    MS_LOG(EXCEPTION) << "Output_idx" << output_idx << " of node " << node->DebugString()
                      << " output addr is not exist";
  }
  return addr;
}

// get output device addr of anf_node
bool AnfRuntimeAlgorithm::OutputAddrExist(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_idx > GetOutputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "The index [" << output_idx << "] is out of range of the node's output size [ "
                      << GetOutputTensorNum(node) << "#node:[ " << node->DebugString() << "]";
  }
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->OutputAddrExist(output_idx);
}

const DeviceAddress *AnfRuntimeAlgorithm::GetPrevNodeOutputAddr(const AnfNodePtr &anf_node, size_t input_idx) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputAddr(kernel_with_index.first, kernel_with_index.second);
}

DeviceAddressPtr AnfRuntimeAlgorithm::GetPrevNodeMutableOutputAddr(const AnfNodePtr &anf_node, size_t input_idx) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetMutableOutputAddr(kernel_with_index.first, kernel_with_index.second);
}

// set output device addr of anf_node
void AnfRuntimeAlgorithm::SetOutputAddr(const DeviceAddressPtr &addr, size_t output_idx, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (!kernel_info->SetOutputAddr(addr, output_idx)) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << "set adr" << output_idx << " fail";
  }
}

// set workspace device addr of anf_node
void AnfRuntimeAlgorithm::SetWorkspaceAddr(const DeviceAddressPtr &addr, size_t output_idx, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (!kernel_info->SetWorkspaceAddr(addr, output_idx)) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << "set adr" << output_idx << " fail";
  }
}

// get workspace device addr of anf_node
DeviceAddress *AnfRuntimeAlgorithm::GetWorkspaceAddr(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetWorkspaceAddr(output_idx);
  if (addr == nullptr) {
    MS_LOG(EXCEPTION) << "Output_idx " << output_idx << " of node " << node->DebugString()
                      << "] workspace addr is not exist";
  }
  return addr;
}

// set infer shapes and types of anf node
void AnfRuntimeAlgorithm::SetOutputInferTypeAndShape(const std::vector<TypeId> &types,
                                                     const std::vector<std::vector<size_t>> &shapes, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  if (types.size() != shapes.size()) {
    MS_LOG(EXCEPTION) << "Types size " << types.size() << "should be same with shapes size " << shapes.size();
  }
  if (shapes.empty()) {
    MS_LOG(EXCEPTION) << "Illegal empty output_types_shapes";
  } else if (shapes.size() == 1) {
    // single output handle
    std::vector<int> shape_int;
    std::transform(shapes[0].begin(), shapes[0].end(), std::back_inserter(shape_int), SizeToInt);
    auto abstract = std::make_shared<AbstractTensor>(TypeIdToType(types[0]), shape_int);
    node->set_abstract(abstract);
  } else {
    // multiple output handle
    std::vector<AbstractBasePtr> abstract_list;
    for (size_t i = 0; i < types.size(); ++i) {
      std::vector<int> shape_int;
      std::transform(shapes[i].begin(), shapes[i].end(), std::back_inserter(shape_int), SizeToInt);
      abstract_list.push_back(std::make_shared<AbstractTensor>(TypeIdToType(types[i]), shape_int));
    }
    auto abstract_tuple = std::make_shared<AbstractTuple>(abstract_list);
    node->set_abstract(abstract_tuple);
  }
}
// copy an abstract of a node to another node
void AnfRuntimeAlgorithm::CopyAbstract(const AnfNodePtr &from_node, AnfNode *to_node) {
  to_node->set_abstract(from_node->abstract());
}

// get KernelBuildType of node, such as ATT,RT,FWK and so on
KernelType AnfRuntimeAlgorithm::GetKernelType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  // select_kernel_build_info() has checked whether return pointer is null
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->kernel_type();
}

kernel::Processor AnfRuntimeAlgorithm::GetProcessor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->processor();
}

kernel::FusionType AnfRuntimeAlgorithm::GetFusionType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->fusion_type();
}

// set select kernel_build_info
void AnfRuntimeAlgorithm::SetSelectKernelBuildInfo(const KernelBuildInfoPtr &select_kernel_build_info, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->set_select_kernel_build_info(select_kernel_build_info);
}

// get select kernel_build_info
KernelBuildInfoPtr AnfRuntimeAlgorithm::GetSelectKernelBuildInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->GetMutableSelectKernelBuildInfo();
}

// get kernelMode
KernelMod *AnfRuntimeAlgorithm::GetKernelMod(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->MutableKernelMod();
}

// set kernel mod
void AnfRuntimeAlgorithm::SetKernelMod(const KernelModPtr &kernel_mod, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_kernel_mod(kernel_mod);
}

bool AnfRuntimeAlgorithm::IsRealKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // parameter and value node is not a real kernel too
  if (!node->isa<CNode>()) {
    return true;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Illegal null input of cnode(%s)" << node->DebugString();
  }
  auto input = cnode->inputs()[0];
  bool is_virtual_node = IsPrimitive(input, prim::kPrimImageSummary) || IsPrimitive(input, prim::kPrimScalarSummary) ||
                         IsPrimitive(input, prim::kPrimTensorSummary) ||
                         IsPrimitive(input, prim::kPrimHistogramSummary) || IsPrimitive(input, prim::kPrimMakeTuple) ||
                         IsPrimitive(input, prim::kPrimStateSetItem) || IsPrimitive(input, prim::kPrimDepend) ||
                         IsPrimitive(input, prim::kPrimTupleGetItem) || IsPrimitive(input, prim::kPrimControlDepend) ||
                         IsPrimitive(input, prim::kPrimReturn);
  return !is_virtual_node;
}

bool AnfRuntimeAlgorithm::IsRealCNodeKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // parameter and value node is not a real cnode kernel
  if (!node->isa<CNode>()) {
    return false;
  }
  // return considered as a real node
  if (CheckPrimitiveType(node, prim::kPrimReturn)) {
    return true;
  }
  return IsRealKernel(node);
}

bool AnfRuntimeAlgorithm::IsParameterWeight(const ParameterPtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->has_default();
}

void AnfRuntimeAlgorithm::SetStreamId(uint32_t stream_id, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_stream_id(stream_id);
}

uint32_t AnfRuntimeAlgorithm::GetStreamId(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->stream_id();
}

void AnfRuntimeAlgorithm::SetStreamDistinctionLabel(uint32_t stream_label, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_stream_distinction_label(stream_label);
}

uint32_t AnfRuntimeAlgorithm::GetStreamDistinctionLabel(const AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->stream_distinction_label();
}

void AnfRuntimeAlgorithm::SetGraphId(uint32_t graph_id, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_graph_id(graph_id);
}

uint32_t AnfRuntimeAlgorithm::GetGraphId(const AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->graph_id();
}

bool AnfRuntimeAlgorithm::IsTupleOutput(const AnfNodePtr &anf) {
  MS_EXCEPTION_IF_NULL(anf);
  TypePtr type = anf->Type();
  MS_EXCEPTION_IF_NULL(type);
  return type->isa<Tuple>();
}

AnfNodePtr AnfRuntimeAlgorithm::GetInputNode(const CNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  auto get_input_index = index + 1;
  if (index + 1 > node->inputs().size()) {
    MS_LOG(EXCEPTION) << "Input index size " << get_input_index << "but the node input size just"
                      << node->inputs().size();
  }
  // input 0 is primitive node
  return node->input(get_input_index);
}

bool AnfRuntimeAlgorithm::IsFeatureMapOutput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>()) {
    return false;
  }
  auto kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->is_feature_map();
}

bool AnfRuntimeAlgorithm::IsFeatureMapInput(const AnfNodePtr &node, size_t input_index) {
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Cannot input a parameter or a valuenode to charge it's input if is a feature map";
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_node = cnode->input(input_index + 1);
  return IsFeatureMapOutput(input_node);
}

size_t AnfRuntimeAlgorithm::GetRealInputIndex(const mindspore::AnfNodePtr &anf_node, const size_t cur_index) {
  MS_EXCEPTION_IF_NULL(anf_node);
  static std::map<std::string, std::map<size_t, size_t>> spec_node_list = {
    {prim::kPrimConv2DBackpropInput->name(), {{0, 1}, {1, 0}}},
    {prim::kPrimConv2DBackpropFilter->name(), {{0, 1}, {1, 0}}},
    {prim::kPrimLogSoftmaxGrad->name(), {{0, 1}, {1, 0}}},
    {prim::kPrimLayerNormGrad->name(), {{0, 1}, {1, 0}, {2, 2}, {3, 3}, {4, 4}}},
    {prim::kPrimLayerNormBetaGammaBackprop->name(), {{0, 1}, {1, 0}, {2, 2}, {3, 3}}},
    {prim::kPrimLayerNormXBackprop->name(), {{0, 1}, {1, 0}, {2, 2}, {3, 3}, {4, 4}}},
    {prim::kPrimMinimumGrad->name(), {{0, 2}, {1, 0}, {2, 1}}},
    {prim::kPrimMaximumGrad->name(), {{0, 2}, {1, 0}, {2, 1}}}};
  size_t ret = cur_index;
  auto node_name = AnfAlgo::GetCNodeName(anf_node);
  if (AnfAlgo::GetKernelType(anf_node) == TBE_KERNEL) {
    auto find = spec_node_list.find(node_name);
    if (find != spec_node_list.end()) {
      ret = find->second[cur_index];
      MS_LOG(INFO) << "Real input index change to" << ret << ", node name:" << node_name;
    }
  }
  return ret;
}

void AnfRuntimeAlgorithm::SetNodeInput(const CNodePtr &node, const AnfNodePtr &input_node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(input_node);
  node->set_input(index + 1, input_node);
}

bool AnfRuntimeAlgorithm::IsCommunicationOp(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_name = AnfAlgo::GetCNodeName(node);
  auto kernel_type = AnfAlgo::GetKernelType(node);
  if (kernel_name == kAllReduceOpName || kernel_type == HCCL_KERNEL) {
    return true;
  }
  return false;
}

bool AnfRuntimeAlgorithm::IsAllReduceOp(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>() && AnfAlgo::GetCNodeName(node) == kAllReduceOpName) {
    return true;
  }
  return false;
}

bool AnfRuntimeAlgorithm::IsGetNext(const NotNull<AnfNodePtr> &node) {
  auto kernel_name = AnfAlgo::GetCNodeName(node);
  return kernel_name == kGetNextOpName;
}
}  // namespace session
}  // namespace mindspore
