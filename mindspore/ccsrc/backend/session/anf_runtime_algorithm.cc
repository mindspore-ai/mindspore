/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "backend/session/anf_runtime_algorithm.h"
#include <memory>
#include <algorithm>
#include <map>
#include <set>
#include <unordered_set>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "base/core_ops.h"
#include "utils/utils.h"
#include "utils/shape_utils.h"
#include "runtime/device/kernel_info.h"
#include "runtime/device/device_address.h"
#include "backend/optimizer/common/helper.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/kernel_build_info.h"
#include "common/trans.h"
#include "abstract/param_validator.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "utils/trace_base.h"

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
constexpr size_t kNopNodeInputSize = 2;
constexpr size_t kNopNodeRealInputIndex = 1;

using PrimitiveSet = std::unordered_set<PrimitivePtr, PrimitiveHasher, PrimitiveEqual>;

PrimitiveSet follow_first_input_prims = {prim::kPrimDepend, prim::kPrimLoad};

bool IsShapeDynamic(const abstract::ShapePtr &shape) {
  MS_EXCEPTION_IF_NULL(shape);
  return std::any_of(shape->shape().begin(), shape->shape().end(), [](int64_t s) { return s < 0; });
}

bool IsShapeDynamic(const std::vector<size_t> &shape) {
  return std::any_of(shape.begin(), shape.end(), [](int64_t s) { return s < 0; });
}

bool IsOneOfPrimitive(const AnfNodePtr &node, const PrimitiveSet &prim_set) {
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(node);
  return (prim && prim_set.find(prim) != prim_set.end());
}

bool IsOneOfPrimitiveCNode(const AnfNodePtr &node, const PrimitiveSet &prim_set) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr || cnode->size() == 0) {
    return false;
  }
  return IsOneOfPrimitive(cnode->inputs().at(kAnfPrimitiveIndex), prim_set);
}

bool IsRealKernelCNode(const CNodePtr &cnode) {
  static const PrimitiveSet virtual_prims = {
    prim::kPrimImageSummary, prim::kPrimScalarSummary, prim::kPrimTensorSummary, prim::kPrimHistogramSummary,
    prim::kPrimMakeTuple,    prim::kPrimStateSetItem,  prim::kPrimTupleGetItem,  prim::kPrimReturn,
    prim::kPrimPartial,      prim::kPrimDepend,        prim::kPrimUpdateState,   prim::kPrimLoad};
  if (cnode->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Illegal null input of cnode(%s)" << cnode->DebugString();
  }
  const auto &input = cnode->inputs().at(0);
  bool is_virtual_node = IsOneOfPrimitive(input, virtual_prims);
  return !is_virtual_node;
}

std::vector<size_t> TransShapeToSizet(const abstract::ShapePtr &shape) {
  MS_EXCEPTION_IF_NULL(shape);
  std::vector<size_t> shape_size_t;
  if (IsShapeDynamic(shape)) {
    if (std::all_of(shape->max_shape().begin(), shape->max_shape().end(), [](int64_t s) { return s >= 0; })) {
      std::transform(shape->max_shape().begin(), shape->max_shape().end(), std::back_inserter(shape_size_t),
                     LongToSize);
    } else {
      MS_LOG(EXCEPTION) << "Invalid Max Shape";
    }
  } else {
    std::transform(shape->shape().begin(), shape->shape().end(), std::back_inserter(shape_size_t), LongToSize);
  }
  return shape_size_t;
}

enum ShapeType { kMaxShape, kMinShape };
}  // namespace

AnfNodePtr AnfRuntimeAlgorithm::GetTupleGetItemRealInput(const CNodePtr &tuple_get_item) {
  MS_EXCEPTION_IF_NULL(tuple_get_item);
  if (tuple_get_item->size() != kTupleGetItemInputSize) {
    MS_LOG(EXCEPTION) << "The node tuple_get_item must have 2 inputs!";
  }
  return tuple_get_item->input(kRealInputNodeIndexInTupleGetItem);
}

size_t AnfRuntimeAlgorithm::GetTupleGetItemOutIndex(const CNodePtr &tuple_get_item) {
  MS_EXCEPTION_IF_NULL(tuple_get_item);
  if (tuple_get_item->size() != kTupleGetItemInputSize) {
    MS_LOG(EXCEPTION) << "The node tuple_get_item must have 2 inputs!";
  }
  auto output_index_value_node = tuple_get_item->input(kInputNodeOutputIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(output_index_value_node);
  auto value_node = output_index_value_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  return LongToSize(GetValue<int64_t>(value_node->value()));
}

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
      auto item_idx = GetValue<int64_t>(value_node->value());
      return VisitKernel(cnode->input(kRealInputNodeIndexInTupleGetItem), LongToSize(item_idx));
    } else if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimUpdateState)) {
      return VisitKernel(cnode->input(kUpdateStateRealInput), 0);
    } else if (IsOneOfPrimitive(input0, follow_first_input_prims)) {
      return VisitKernel(cnode->input(kRealInputIndexInDepend), 0);
    } else {
      return std::make_pair(anf_node, index);
    }
  } else {
    MS_LOG(EXCEPTION) << "The input is invalid";
  }
}

KernelWithIndex AnfRuntimeAlgorithm::VisitKernelWithReturnType(const AnfNodePtr &anf_node, int index,
                                                               bool visit_nop_node,
                                                               const std::vector<PrimitivePtr> &return_types) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (std::any_of(return_types.begin(), return_types.end(), [&anf_node](const PrimitivePtr &prim_type) -> bool {
        return CheckPrimitiveType(anf_node, prim_type);
      })) {
    return KernelWithIndex(anf_node, index);
  }
  if (!anf_node->isa<CNode>()) {
    return KernelWithIndex(anf_node, 0);
  }
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (CheckPrimitiveType(cnode, prim::kPrimTupleGetItem)) {
    auto item_with_index_tmp = VisitKernelWithReturnType(GetTupleGetItemRealInput(cnode),
                                                         GetTupleGetItemOutIndex(cnode), visit_nop_node, return_types);
    if (CheckPrimitiveType(item_with_index_tmp.first, prim::kPrimMakeTuple)) {
      MS_EXCEPTION_IF_NULL(item_with_index_tmp.first);
      auto make_tuple = item_with_index_tmp.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(make_tuple);
      const std::vector<AnfNodePtr> &make_tuple_inputs = make_tuple->inputs();
      size_t make_tuple_input_index = item_with_index_tmp.second + 1;
      if (make_tuple_input_index >= make_tuple_inputs.size()) {
        MS_LOG(EXCEPTION) << "Index[" << make_tuple_input_index << "] out of range[" << make_tuple_inputs.size()
                          << "].";
      }
      return VisitKernelWithReturnType(make_tuple_inputs[make_tuple_input_index], 0, visit_nop_node, return_types);
    }
    return item_with_index_tmp;
  }
  if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimUpdateState)) {
    return VisitKernelWithReturnType(cnode->input(kUpdateStateStateInput), index, visit_nop_node, return_types);
  }
  if (IsOneOfPrimitiveCNode(cnode, follow_first_input_prims)) {
    return VisitKernelWithReturnType(cnode->input(kRealInputIndexInDepend), index, visit_nop_node, return_types);
  }
  if (opt::IsNopNode(cnode) && visit_nop_node) {
    if (cnode->size() != kNopNodeInputSize) {
      MS_LOG(EXCEPTION) << "Invalid nop node " << cnode->DebugString() << " trace: " << trace::DumpSourceLines(cnode);
    }
    return VisitKernelWithReturnType(cnode->input(kNopNodeRealInputIndex), 0, visit_nop_node, return_types);
  }
  return KernelWithIndex(anf_node, index);
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

FuncGraphPtr AnfRuntimeAlgorithm::GetCNodeFuncGraphPtr(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto value_node = attr_input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  return value->cast<FuncGraphPtr>();
}

std::string AnfRuntimeAlgorithm::GetCNodeName(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>()) {
    auto primitive = AnfAlgo::GetCNodePrimitive(node);
    if (primitive != nullptr) {
      return primitive->name();
    }
    auto func_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
    MS_EXCEPTION_IF_NULL(func_graph);
    if (func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
      std::string fg_name = "GraphKernel_";
      fg_name += GetValue<std::string>(func_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
      return fg_name;
    }
    return func_graph->ToString();
  }
  MS_LOG(EXCEPTION) << "Unknown anf node type " << node->DebugString() << " trace: " << trace::DumpSourceLines(node);
}

std::string AnfRuntimeAlgorithm::GetNodeDebugString(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->DebugString();
}

void AnfRuntimeAlgorithm::SetNodeAttr(const std::string &key, const ValuePtr &value, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Only cnode has attr, but this anf is " << node->DebugString()
                      << " trace: " << trace::DumpSourceLines(node);
  }
  // single op cnode.
  auto primitive = AnfAlgo::GetCNodePrimitive(node);
  if (primitive != nullptr) {
    primitive->set_attr(key, value);
    return;
  }
  // graph kernel cnode.
  auto fg = AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(fg);
  fg->set_attr(key, value);
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
                      << to->DebugString() << " trace: " << trace::DumpSourceLines(from);
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
                      << from->DebugString() << " trace: " << trace::DumpSourceLines(from);
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
    MS_LOG(EXCEPTION) << "Only cnode has attr, but this anf is " << node->DebugString()
                      << " trace: " << trace::DumpSourceLines(node);
  }
  // single op cnode.
  auto primitive = AnfAlgo::GetCNodePrimitive(node);
  if (primitive != nullptr) {
    primitive->EraseAttr(key);
    return;
  }
  // graph kernel cnode.
  auto fg = AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(fg);
  fg->erase_flag(key);
}

bool AnfRuntimeAlgorithm::HasNodeAttr(const std::string &key, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(WARNING) << "Only cnode has attr, but this anf is " << node->DebugString();
    return false;
  }
  // single op cnode.
  auto primitive = AnfAlgo::GetCNodePrimitive(node);
  if (primitive != nullptr) {
    return primitive->HasAttr(key);
  }
  // graph kernel cnode.
  auto fg = AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(fg);
  return fg->has_attr(key);
}

size_t AnfRuntimeAlgorithm::GetInputNum(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  size_t input_num = cnode->size();
  if (input_num == 0) {
    MS_LOG(EXCEPTION) << "Cnode inputs size can't be zero";
  }
  return input_num - 1;
}

size_t AnfRuntimeAlgorithm::GetInputTensorNum(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    MS_LOG(EXCEPTION) << "Only cnode has real input, but this anf is " << node->DebugString()
                      << " trace: " << trace::DumpSourceLines(node);
  }
  ssize_t input_tensor_num = cnode->input_tensor_num();
  if (input_tensor_num >= 0) {
    return static_cast<size_t>(input_tensor_num);
  }
  size_t input_num = cnode->inputs().size();
  if (input_num == 0) {
    MS_LOG(EXCEPTION) << "Cnode inputs size can't be zero"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  // Exclude inputs[0].
  --input_num;

  // Exclude monad inputs for real cnodes.
  if (input_num > 0 && IsRealKernelCNode(cnode)) {
    auto &inputs = cnode->inputs();
    // Search monad inputs, backward.
    for (auto iter = inputs.rbegin(); iter != inputs.rend(); ++iter) {
      if (!HasAbstractMonad(*iter)) {
        // Stop count if we encounter a non-monad input.
        break;
      }
      --input_num;
    }
  }
  cnode->set_input_tensor_num(static_cast<ssize_t>(input_num));
  return input_num;
}

size_t AnfRuntimeAlgorithm::GetOutputTensorNum(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  TypePtr type = node->Type();
  if (type == nullptr) {
    return 0;
  }
  if (type->isa<Tuple>()) {
    auto tuple_type = type->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_type);
    return tuple_type->size();
  }
  if (type->isa<TypeNone>()) {
    return 0;
  }
  return 1;
}

std::vector<std::string> AnfRuntimeAlgorithm::GetAllOutputFormats(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfAlgo::IsRealKernel(node)) {
    MS_LOG(EXCEPTION) << "Not real kernel:"
                      << "#node [" << node->DebugString() << "]"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto format = build_info->GetAllOutputFormats();
  return format;
}

std::vector<std::string> AnfRuntimeAlgorithm::GetAllInputFormats(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfAlgo::IsRealKernel(node)) {
    MS_LOG(EXCEPTION) << "Not real kernel:"
                      << "#node [" << node->DebugString() << "]"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto format = build_info->GetAllInputFormats();
  return format;
}

std::vector<TypeId> AnfRuntimeAlgorithm::GetAllInputDeviceTypes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfAlgo::IsRealKernel(node)) {
    MS_LOG(EXCEPTION) << "Not real kernel:"
                      << "#node [" << node->DebugString() << "]"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto types = build_info->GetAllInputDeviceTypes();
  return types;
}

std::vector<TypeId> AnfRuntimeAlgorithm::GetAllOutputDeviceTypes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfAlgo::IsRealKernel(node)) {
    MS_LOG(EXCEPTION) << "Not real kernel:"
                      << "#node [" << node->DebugString() << "]"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto types = build_info->GetAllOutputDeviceTypes();
  return types;
}

std::string AnfRuntimeAlgorithm::GetOriginDataFormat(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfAlgo::IsRealKernel(node)) {
    MS_LOG(EXCEPTION) << "Not real kernel:"
                      << "#node [" << node->DebugString() << "]"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto format = build_info->GetOriginDataFormat();
  return format;
}

std::string AnfRuntimeAlgorithm::GetOutputFormat(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_idx > GetOutputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "Output index:" << output_idx
                      << " is out of the node output range :" << GetOutputTensorNum(node) << " #node ["
                      << node->DebugString() << "]"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  if (!AnfAlgo::IsRealKernel(node)) {
    return AnfAlgo::GetPrevNodeOutputFormat(node, output_idx);
  }
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto format = build_info->GetOutputFormat(output_idx);
  if (format == kernel::KernelBuildInfo::kInvalidFormat) {
    MS_LOG(EXCEPTION) << "Node [" << node->DebugString() << "]"
                      << " has a invalid output format"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  return format;
}

std::string AnfRuntimeAlgorithm::GetInputFormat(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (input_idx > GetInputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "Input index :" << input_idx
                      << " is out of the number node Input range :" << GetInputTensorNum(node) << "#node ["
                      << node->DebugString() << "]"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  if (!IsRealKernel(node)) {
    return GetPrevNodeOutputFormat(node, input_idx);
  }
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto format = build_info->GetInputFormat(input_idx);
  if (format == kernel::KernelBuildInfo::kInvalidFormat) {
    MS_LOG(EXCEPTION) << "Node [" << node->DebugString() << "]"
                      << " has a invalid input format"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  return format;
}

KernelWithIndex AnfRuntimeAlgorithm::GetPrevNodeOutput(const AnfNodePtr &anf_node, size_t input_idx,
                                                       bool visit_nop_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (!anf_node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << anf_node->DebugString() << "anf_node is not CNode."
                      << " trace: " << trace::DumpSourceLines(anf_node);
  }
  if (CheckPrimitiveType(anf_node, prim::kPrimTupleGetItem)) {
    return VisitKernelWithReturnType(anf_node, 0, visit_nop_node);
  }
  auto input_node = AnfAlgo::GetInputNode(anf_node->cast<CNodePtr>(), input_idx);
  MS_EXCEPTION_IF_NULL(input_node);
  return VisitKernelWithReturnType(input_node, 0, visit_nop_node);
}

std::string AnfRuntimeAlgorithm::GetPrevNodeOutputFormat(const AnfNodePtr &anf_node, size_t input_idx) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
}

std::string AnfRuntimeAlgorithm::GetPrevNodeOutputReshapeType(const AnfNodePtr &node, size_t input_idx) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(node, input_idx);
  return GetOutputReshapeType(kernel_with_index.first, kernel_with_index.second);
}

std::vector<size_t> AnfRuntimeAlgorithm::GetOutputInferShape(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  abstract::BaseShapePtr base_shape = node->Shape();
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::Shape>()) {
    if (output_idx == 0) {
      return TransShapeToSizet(base_shape->cast<abstract::ShapePtr>());
    }
    MS_LOG(EXCEPTION) << "The node " << node->DebugString() << "is a single output node but got index [" << output_idx
                      << "."
                      << " trace: " << trace::DumpSourceLines(node);
  } else if (base_shape->isa<abstract::TupleShape>()) {
    auto tuple_shape = base_shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape);
    if (output_idx >= tuple_shape->size()) {
      MS_LOG(EXCEPTION) << "Output index " << output_idx << "is larger than output number " << tuple_shape->size()
                        << " node:" << node->DebugString() << "."
                        << " trace: " << trace::DumpSourceLines(node);
    }
    auto b_shp = (*tuple_shape)[output_idx];
    if (b_shp->isa<abstract::Shape>()) {
      return TransShapeToSizet(b_shp->cast<abstract::ShapePtr>());
    } else if (b_shp->isa<abstract::NoShape>()) {
      return std::vector<size_t>();
    } else {
      MS_LOG(EXCEPTION) << "The output type of ApplyKernel index:" << output_idx
                        << " should be a NoShape , ArrayShape or a TupleShape, but it is " << base_shape->ToString()
                        << "node :" << node->DebugString() << "."
                        << " trace: " << trace::DumpSourceLines(node);
    }
  } else if (base_shape->isa<abstract::NoShape>()) {
    return std::vector<size_t>();
  }
  MS_LOG(EXCEPTION) << "The output type of ApplyKernel should be a NoShape , ArrayShape or a TupleShape, but it is "
                    << base_shape->ToString() << " node : " << node->DebugString()
                    << " trace: " << trace::DumpSourceLines(node);
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
    infer_shape = trans::PaddingShape(infer_shape, format, GetOutputReshapeType(node, output_idx));
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
    infer_shape = trans::PaddingShape(infer_shape, format, GetInputReshapeType(node, input_idx));
  }
  return trans::TransShapeToDevice(infer_shape, format);
}

std::string AnfRuntimeAlgorithm::GetInputReshapeType(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (input_idx > GetInputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "The index:" << input_idx
                      << " is out of range of the node's input size : " << GetInputTensorNum(node) << "#node["
                      << node->DebugString() << "]"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  if (!IsRealKernel(node)) {
    return GetPrevNodeOutputReshapeType(node, input_idx);
  }
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  if (build_info->IsInputDefaultPadding()) {
    return "";
  }
  return build_info->GetInputReshapeType(input_idx);
}

std::string AnfRuntimeAlgorithm::GetOutputReshapeType(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_idx > GetOutputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "The index [" << output_idx << "] is out of range of the node's output size [ "
                      << GetOutputTensorNum(node) << "#node[ " << node->DebugString() << "]"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  if (!IsRealKernel(node)) {
    return GetPrevNodeOutputReshapeType(node, output_idx);
  }
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  if (build_info->IsOutputDefaultPadding()) {
    return "";
  }
  return build_info->GetOutputReshapeType(output_idx);
}

TypeId AnfRuntimeAlgorithm::GetOutputInferDataType(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto get_single_type = [](const TypePtr &type_ptr) -> TypeId {
    MS_EXCEPTION_IF_NULL(type_ptr);
    if (type_ptr->isa<TensorType>()) {
      auto tensor_ptr = type_ptr->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(tensor_ptr);
      TypePtr elem = tensor_ptr->element();
      MS_EXCEPTION_IF_NULL(elem);
      return elem->type_id();
    }
    if (type_ptr->isa<Number>()) {
      return type_ptr->type_id();
    }
    return type_ptr->type_id();
  };
  auto get_tuple_type = [get_single_type](const TypePtr &type_ptr, size_t output_idx) -> TypeId {
    MS_EXCEPTION_IF_NULL(type_ptr);
    if (!type_ptr->isa<Tuple>()) {
      return get_single_type(type_ptr);
    }
    auto tuple_ptr = type_ptr->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_ptr);
    if (output_idx >= tuple_ptr->size()) {
      MS_LOG(EXCEPTION) << "Output index " << output_idx << " must be less than output number " << tuple_ptr->size();
    }
    return get_single_type((*tuple_ptr)[output_idx]);
  };
  TypePtr type_ptr = node->Type();
  return get_tuple_type(type_ptr, output_idx);
}

TypeId AnfRuntimeAlgorithm::GetPrevNodeOutputInferDataType(const AnfNodePtr &node, size_t input_idx) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputInferDataType(kernel_with_index.first, kernel_with_index.second);
}

TypeId AnfRuntimeAlgorithm::GetOutputDeviceDataType(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_idx > GetOutputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "The index [" << output_idx << "] is out of range of the node's output size [ "
                      << GetOutputTensorNum(node) << "#node [ " << node->DebugString() << "]"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  if (!IsRealKernel(node)) {
    return GetPrevNodeOutputDeviceDataType(node, output_idx);
  }
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto dtype = build_info->GetOutputDeviceType(output_idx);
  if (dtype == TypeId::kNumberTypeEnd) {
    MS_LOG(EXCEPTION) << "Node [" << node->DebugString() << "]"
                      << " has a invalid dtype"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  return dtype;
}

TypeId AnfRuntimeAlgorithm::GetInputDeviceDataType(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (input_idx > GetInputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "The index [" << input_idx << "] is out of range of the node's input size [ "
                      << GetInputTensorNum(node) << "#node [ " << node->DebugString() << "]"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  if (!IsRealKernel(node)) {
    return GetPrevNodeOutputDeviceDataType(node, 0);
  }
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto dtype = build_info->GetInputDeviceType(input_idx);
  if (dtype == TypeId::kNumberTypeEnd) {
    MS_LOG(EXCEPTION) << "Node [" << node->DebugString() << "]"
                      << " has a invalid dtype"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  return dtype;
}

TypeId AnfRuntimeAlgorithm::GetPrevNodeOutputDeviceDataType(const AnfNodePtr &anf_node, size_t input_idx) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
}

// get output device addr of anf_node
const DeviceAddress *AnfRuntimeAlgorithm::GetOutputAddr(const AnfNodePtr &node, size_t output_idx,
                                                        bool visit_nop_node) {
  MS_EXCEPTION_IF_NULL(node);
  if (opt::IsNopNode(node) && visit_nop_node) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->size() == kNopNodeInputSize) {
      return AnfRuntimeAlgorithm::GetPrevNodeOutputAddr(cnode, 0);
    } else {
      MS_LOG(EXCEPTION) << node->DebugString() << "Invalid nop node"
                        << " trace: " << trace::DumpSourceLines(node);
    }
  }
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetOutputAddr(output_idx);
  if (addr == nullptr) {
    MS_LOG(EXCEPTION) << "Output_idx " << output_idx << " of node " << node->DebugString()
                      << " output addr is not exist"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  return addr;
}

DeviceAddressPtr AnfRuntimeAlgorithm::GetMutableOutputAddr(const AnfNodePtr &node, size_t output_idx,
                                                           bool visit_nop_node) {
  MS_EXCEPTION_IF_NULL(node);
  if (opt::IsNopNode(node) && visit_nop_node) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->inputs().size() == kNopNodeInputSize) {
      return AnfRuntimeAlgorithm::GetPrevNodeMutableOutputAddr(cnode, 0);
    } else {
      MS_LOG(EXCEPTION) << node->DebugString() << "Invalid nop node."
                        << " trace: " << trace::DumpSourceLines(node);
    }
  }
  // Critical path performance optimization: `KernelInfo` is unique subclass of `KernelInfoDevice`
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetMutableOutputAddr(output_idx);
  if (addr == nullptr) {
    MS_LOG(EXCEPTION) << "Output_idx" << output_idx << " of node " << node->DebugString() << " output addr is not exist"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  return addr;
}

// get output device addr of anf_node
bool AnfRuntimeAlgorithm::OutputAddrExist(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  // Critical path performance optimization: `KernelInfo` is unique subclass of `KernelInfoDevice`
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->OutputAddrExist(output_idx);
}

const DeviceAddress *AnfRuntimeAlgorithm::GetPrevNodeOutputAddr(const AnfNodePtr &anf_node, size_t input_idx,
                                                                bool visit_nop_node) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputAddr(kernel_with_index.first, kernel_with_index.second, visit_nop_node);
}

DeviceAddressPtr AnfRuntimeAlgorithm::GetPrevNodeMutableOutputAddr(const AnfNodePtr &anf_node, size_t input_idx,
                                                                   bool visit_nop_node) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetMutableOutputAddr(kernel_with_index.first, kernel_with_index.second, visit_nop_node);
}

// set output device addr of anf_node
void AnfRuntimeAlgorithm::SetOutputAddr(const DeviceAddressPtr &addr, size_t output_idx, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (!kernel_info->SetOutputAddr(addr, output_idx)) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << "set adr" << output_idx << " fail."
                      << " trace: " << trace::DumpSourceLines(node);
  }
}

// set workspace device addr of anf_node
void AnfRuntimeAlgorithm::SetWorkspaceAddr(const DeviceAddressPtr &addr, size_t output_idx, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (!kernel_info->SetWorkspaceAddr(addr, output_idx)) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << "set adr" << output_idx << " fail。"
                      << " trace: " << trace::DumpSourceLines(node);
  }
}

// get workspace device addr of anf_node
DeviceAddress *AnfRuntimeAlgorithm::GetWorkspaceAddr(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetWorkspaceAddr(output_idx);
  if (addr == nullptr) {
    MS_LOG(EXCEPTION) << "Output_idx " << output_idx << " of node " << node->DebugString()
                      << "] workspace addr is not exist"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  return addr;
}

// get workspace device mutable addr of anf_node
DeviceAddressPtr AnfRuntimeAlgorithm::GetMutableWorkspaceAddr(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetMutableWorkspaceAddr(index);
  if (addr == nullptr) {
    MS_LOG(EXCEPTION) << "Index " << index << " of node " << node->DebugString() << "] workspace addr is not exist"
                      << " trace: " << trace::DumpSourceLines(node);
  }
  return addr;
}

// set infer shapes and types of anf node
void AnfRuntimeAlgorithm::SetOutputInferTypeAndShape(const std::vector<TypeId> &types,
                                                     const std::vector<std::vector<size_t>> &shapes, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto node_ptr = node->cast<AnfNodePtr>();
  MS_EXCEPTION_IF_NULL(node_ptr);
  if (types.size() != shapes.size()) {
    MS_LOG(EXCEPTION) << "Types size " << types.size() << "should be same with shapes size " << shapes.size()
                      << " trace: " << trace::DumpSourceLines(node);
  }
  if (shapes.empty()) {
    node->set_abstract(std::make_shared<abstract::AbstractNone>());
  } else if (shapes.size() == 1) {
    // single output handle
    ShapeVector shape_int;
    auto abstract_ptr = node_ptr->abstract();
    abstract::AbstractTensorPtr abstract = nullptr;
    if (abstract_ptr != nullptr) {
      auto max_shape = GetOutputMaxShape(node_ptr, 0);
      auto min_shape = GetOutputMinShape(node_ptr, 0);
      std::transform(shapes[0].begin(), shapes[0].end(), std::back_inserter(shape_int), SizeToLong);
      abstract = std::make_shared<AbstractTensor>(TypeIdToType(types[0]),
                                                  std::make_shared<abstract::Shape>(shape_int, min_shape, max_shape));
    } else {
      abstract = std::make_shared<AbstractTensor>(TypeIdToType(types[0]), shape_int);
    }
    node->set_abstract(abstract);
  } else {
    // multiple output handle
    std::vector<AbstractBasePtr> abstract_list;
    for (size_t i = 0; i < types.size(); ++i) {
      ShapeVector shape_int;
      auto abstract_ptr = node_ptr->abstract();
      abstract::AbstractTensorPtr abstract = nullptr;
      if (abstract_ptr != nullptr) {
        auto max_shape = GetOutputMaxShape(node_ptr, i);
        auto min_shape = GetOutputMinShape(node_ptr, i);
        std::transform(shapes[i].begin(), shapes[i].end(), std::back_inserter(shape_int), SizeToLong);
        abstract = std::make_shared<AbstractTensor>(TypeIdToType(types[i]),
                                                    std::make_shared<abstract::Shape>(shape_int, min_shape, max_shape));
      } else {
        abstract =
          std::make_shared<AbstractTensor>(TypeIdToType(types[i]), std::make_shared<abstract::Shape>(shape_int));
      }
      abstract_list.emplace_back(abstract);
    }
    auto abstract_tuple = std::make_shared<AbstractTuple>(abstract_list);
    node->set_abstract(abstract_tuple);
  }
}
// copy an abstract of a node to another node
void AnfRuntimeAlgorithm::CopyAbstract(const AnfNodePtr &from_node, AnfNode *to_node) {
  to_node->set_abstract(from_node->abstract());
}

kernel::OpPattern AnfRuntimeAlgorithm::GetOpPattern(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  // select_kernel_build_info() has checked whether return pointer is null
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->op_pattern();
}

// get KernelBuildType of node, such as ATT,RT,FWK and so on
KernelType AnfRuntimeAlgorithm::GetKernelType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  // select_kernel_build_info() has checked whether return pointer is null
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->kernel_type();
}

kernel::Processor AnfRuntimeAlgorithm::GetProcessor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->processor();
}

kernel::FusionType AnfRuntimeAlgorithm::GetFusionType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->fusion_type();
}

// set select kernel_build_info
void AnfRuntimeAlgorithm::SetSelectKernelBuildInfo(const KernelBuildInfoPtr &select_kernel_build_info, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->set_select_kernel_build_info(select_kernel_build_info);
}

// get select kernel_build_info
KernelBuildInfoPtr AnfRuntimeAlgorithm::GetSelectKernelBuildInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->GetMutableSelectKernelBuildInfo();
}

// get kernelMode
KernelMod *AnfRuntimeAlgorithm::GetKernelMod(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->MutableKernelMod();
}

// set kernel mod
void AnfRuntimeAlgorithm::SetKernelMod(const KernelModPtr &kernel_mod, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_kernel_mod(kernel_mod);
}

bool AnfRuntimeAlgorithm::IsRealKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // parameter and value node is a real kernel too
  if (!node->isa<CNode>()) {
    return true;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Illegal null input of cnode(%s)" << node->DebugString()
                      << " trace: " << trace::DumpSourceLines(node);
  }
  return IsRealKernelCNode(cnode);
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

bool AnfRuntimeAlgorithm::IsGraphKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // graph kernel should be a real cnode kernel.
  if (!IsRealCNodeKernel(node)) {
    return false;
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input = cnode->input(kAnfPrimitiveIndex);
  // graph kernel should has func_graph as first input.
  if (!IsValueNode<FuncGraph>(input)) {
    return false;
  }

  auto func_graph = GetValueNode<FuncGraphPtr>(input);
  MS_EXCEPTION_IF_NULL(func_graph);
  return func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
}

bool AnfRuntimeAlgorithm::IsNodeInGraphKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->func_graph() != nullptr && node->func_graph()->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
}

bool AnfRuntimeAlgorithm::IsParameterWeight(const ParameterPtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->has_default();
}

bool AnfRuntimeAlgorithm::IsLabelIndexInNode(const AnfNodePtr &node, size_t label_index) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetCNodeName(cnode) == kLabelGotoOpName &&
      (AnfAlgo::GetNodeAttr<uint32_t>(cnode, kAttrLabelIndex) == label_index)) {
    return true;
  } else if (AnfAlgo::GetCNodeName(cnode) == kLabelSwitchOpName) {
    auto label_list = AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cnode, kAttrLabelSwitchList);
    if (std::find(label_list.begin(), label_list.end(), label_index) != label_list.end()) {
      return true;
    }
  }
  return false;
}

void AnfRuntimeAlgorithm::SetStreamId(uint32_t stream_id, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_stream_id(stream_id);
}

uint32_t AnfRuntimeAlgorithm::GetStreamId(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->stream_id();
}

void AnfRuntimeAlgorithm::SetStreamDistinctionLabel(uint32_t stream_label, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_stream_distinction_label(stream_label);
}

uint32_t AnfRuntimeAlgorithm::GetStreamDistinctionLabel(const AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<const device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->stream_distinction_label();
}

void AnfRuntimeAlgorithm::SetGraphId(uint32_t graph_id, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_graph_id(graph_id);
}

uint32_t AnfRuntimeAlgorithm::GetGraphId(const AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = static_cast<const device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->graph_id();
}

bool AnfRuntimeAlgorithm::IsTupleOutput(const AnfNodePtr &anf) {
  MS_EXCEPTION_IF_NULL(anf);
  TypePtr type = anf->Type();
  if (type == nullptr) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(type);
  return type->isa<Tuple>();
}

AnfNodePtr AnfRuntimeAlgorithm::GetInputNode(const CNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  auto get_input_index = index + 1;
  if (index + 1 >= node->inputs().size()) {
    MS_LOG(EXCEPTION) << "Input index size " << get_input_index << "but the node input size just"
                      << node->inputs().size() << " trace: " << trace::DumpSourceLines(node);
  }
  // input 0 is primitive node
  return node->input(get_input_index);
}

bool AnfRuntimeAlgorithm::IsFeatureMapOutput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>()) {
    return false;
  }
  if (IsPrimitiveCNode(node, prim::kPrimLoad)) {
    return IsFeatureMapOutput(node->cast<CNodePtr>()->input(1));
  }
  auto kernel_info = static_cast<const device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->is_feature_map();
}

bool AnfRuntimeAlgorithm::IsFeatureMapInput(const AnfNodePtr &node, size_t input_index) {
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Cannot input a parameter or a valuenode to charge it's input if is a feature map"
                      << " trace: " << trace::DumpSourceLines(node);
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
    {kFusionOpConv2DBackpropInputReluGradV2Name, {{0, 1}, {1, 0}, {2, 2}}},
    {kFusionOpConv2DBackpropInputAddNReluGradV2Name, {{0, 1}, {1, 0}, {2, 2}, {3, 3}}},
    {prim::kPrimConv2DBackpropFilter->name(), {{0, 1}, {1, 0}}},
    {prim::kPrimLogSoftmaxGrad->name(), {{0, 1}, {1, 0}}},
    {prim::kPrimLayerNormGrad->name(), {{0, 1}, {1, 0}, {2, 2}, {3, 3}, {4, 4}}},
    {prim::kPrimLayerNormBetaGammaBackprop->name(), {{0, 1}, {1, 0}, {2, 2}, {3, 3}}},
    {prim::kPrimLayerNormXBackprop->name(), {{0, 1}, {1, 0}, {2, 2}, {3, 3}, {4, 4}}},
    {prim::kPrimMinimumGrad->name(), {{0, 2}, {1, 0}, {2, 1}}},
    {prim::kPrimMaximumGrad->name(), {{0, 2}, {1, 0}, {2, 1}}},
    {prim::kPrimApplyCenteredRMSProp->name(),
     {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 4}}}};
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

bool AnfRuntimeAlgorithm::IsInplaceNode(const mindspore::AnfNodePtr &kernel, const string &type) {
  MS_EXCEPTION_IF_NULL(kernel);
  auto primitive = AnfAlgo::GetCNodePrimitive(kernel);
  if (!primitive) {
    return false;
  }

  auto inplace_attr = primitive->GetAttr(type);
  if (inplace_attr == nullptr) {
    return false;
  }

  return true;
}

bool AnfRuntimeAlgorithm::IsCommunicationOp(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto kernel_name = AnfAlgo::GetCNodeName(node);
  if (kernel_name == kAllReduceOpName || kernel_name == kAllGatherOpName || kernel_name == kBroadcastOpName ||
      kernel_name == kReduceScatterOpName || kernel_name == kHcomSendOpName || kernel_name == kReceiveOpName) {
    return true;
  }
  return false;
}

bool AnfRuntimeAlgorithm::IsFusedCommunicationOp(const AnfNodePtr &node) {
  if (!IsCommunicationOp(node)) {
    return false;
  }
  auto primitive = AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  ValuePtr attr_fusion = primitive->GetAttr(kAttrFusion);
  if (attr_fusion == nullptr) {
    return false;
  }
  auto fusion = GetValue<int64_t>(attr_fusion);
  if (fusion == 0) {
    return false;
  }
  return true;
}

bool AnfRuntimeAlgorithm::IsGetNext(const NotNull<AnfNodePtr> &node) {
  auto kernel_name = AnfAlgo::GetCNodeName(node);
  return kernel_name == kGetNextOpName;
}

FuncGraphPtr AnfRuntimeAlgorithm::GetValueNodeFuncGraph(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    return nullptr;
  }
  auto value = value_node->value();
  if (value == nullptr) {
    return nullptr;
  }
  auto func_graph = value->cast<FuncGraphPtr>();
  return func_graph;
}

std::vector<KernelGraphPtr> AnfRuntimeAlgorithm::GetCallSwitchKernelGraph(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!(AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall) || AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch) ||
        AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitchLayer))) {
    MS_LOG(EXCEPTION) << "Node: " << cnode->DebugString() << "is not a call or switch or switch_layer node."
                      << " trace: " << trace::DumpSourceLines(cnode);
  }
  auto get_switch_kernel_graph = [cnode](size_t input_index) -> KernelGraphPtr {
    auto partial = cnode->input(input_index);
    MS_EXCEPTION_IF_NULL(partial);
    if (IsValueNode<KernelGraph>(partial)) {
      return GetValueNode<KernelGraphPtr>(partial);
    }
    auto partial_cnode = partial->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(partial_cnode);
    auto graph_node = partial_cnode->input(kCallKernelGraphIndex);
    MS_EXCEPTION_IF_NULL(graph_node);
    auto graph_value_node = graph_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(graph_value_node);
    auto graph_value = graph_value_node->value();
    MS_EXCEPTION_IF_NULL(graph_value);
    auto child_graph = graph_value->cast<KernelGraphPtr>();
    return child_graph;
  };
  if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall)) {
    auto input1 = cnode->input(kCallKernelGraphIndex);
    MS_EXCEPTION_IF_NULL(input1);
    auto value_node = input1->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto kernel_graph = value_node->value();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    return {kernel_graph->cast<KernelGraphPtr>()};
  } else if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch)) {
    return {get_switch_kernel_graph(kSwitchTrueKernelGraphIndex),
            get_switch_kernel_graph(kSwitchFalseKernelGraphIndex)};
  } else if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitchLayer)) {
    std::vector<KernelGraphPtr> child_graphs;
    for (size_t idx = kMakeTupleInSwitchLayerIndex; idx < cnode->inputs().size(); idx++) {
      auto child_graph = get_switch_kernel_graph(idx);
      child_graphs.emplace_back(child_graph);
    }
    return child_graphs;
  }
  return {};
}

bool AnfRuntimeAlgorithm::IsSwitchCall(const CNodePtr &call_node) {
  MS_EXCEPTION_IF_NULL(call_node);
  if (!CheckPrimitiveType(call_node, prim::kPrimCall)) {
    MS_LOG(EXCEPTION) << "Call node should be a 'call', but is a " << call_node->DebugString()
                      << " trace: " << trace::DumpSourceLines(call_node);
  }
  auto input1 = call_node->input(1);
  if (input1->isa<ValueNode>()) {
    return false;
  } else if (input1->isa<CNode>() && AnfAlgo::CheckPrimitiveType(input1, prim::kPrimSwitch)) {
    return true;
  }
  MS_LOG(EXCEPTION) << "Unexpected input1 of call node,input1:" << input1->DebugString()
                    << " trace: " << trace::DumpSourceLines(call_node);
}

bool AnfRuntimeAlgorithm::IsScalarInput(const CNodePtr &cnode, size_t index) {
  auto shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, index);
  if (shape.empty()) {
    return true;
  }
  return shape.size() == kShape1dDims && shape[0] == 1;
}

bool AnfRuntimeAlgorithm::IsScalarOutput(const CNodePtr &cnode, size_t index) {
  auto shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, index);
  if (shape.empty()) {
    return true;
  }
  return shape.size() == kShape1dDims && shape[0] == 1;
}

void AnfRuntimeAlgorithm::ReorderOptimizerExecList(NotNull<std::vector<CNodePtr> *> node_list) {
  std::vector<CNodePtr> all_opt_list;
  std::vector<CNodePtr> non_opt_list;
  std::vector<CNodePtr> trans_list;
  std::vector<CNodePtr> transpose_list;
  std::vector<CNodePtr> cast_list;
  for (const auto &node : *node_list) {
    MS_EXCEPTION_IF_NULL(node);
    auto trans_pose_func = [&](const CNodePtr &node) -> bool {
      MS_EXCEPTION_IF_NULL(node);
      if (AnfAlgo::GetCNodeName(node) == prim::kPrimTranspose->name()) {
        auto kernel_index = AnfAlgo::VisitKernelWithReturnType(AnfAlgo::GetInputNode(node, 0), 0);
        MS_EXCEPTION_IF_NULL(kernel_index.first);
        if (kernel_index.first->isa<CNode>() && kOptOperatorSet.find(AnfAlgo::GetCNodeName(
                                                  kernel_index.first->cast<CNodePtr>())) != kOptOperatorSet.end()) {
          return true;
        }
      }
      return false;
    };

    auto trans_data_func = [&](const CNodePtr &node) -> bool {
      MS_EXCEPTION_IF_NULL(node);
      if (AnfAlgo::GetCNodeName(node) == prim::KPrimTransData->name()) {
        auto kernel_index = AnfAlgo::VisitKernelWithReturnType(AnfAlgo::GetInputNode(node, 0), 0);
        MS_EXCEPTION_IF_NULL(kernel_index.first);
        if (kernel_index.first->isa<CNode>() && kOptOperatorSet.find(AnfAlgo::GetCNodeName(
                                                  kernel_index.first->cast<CNodePtr>())) != kOptOperatorSet.end()) {
          return true;
        }
        if (!kernel_index.first->isa<CNode>()) {
          return false;
        }
        return trans_pose_func(kernel_index.first->cast<CNodePtr>());
      }
      return false;
    };

    auto cast_func = [&](const CNodePtr &node) -> bool {
      MS_EXCEPTION_IF_NULL(node);
      if (AnfAlgo::GetCNodeName(node) == prim::kPrimCast->name()) {
        auto kernel_index = AnfAlgo::VisitKernelWithReturnType(AnfAlgo::GetInputNode(node, 0), 0);
        MS_EXCEPTION_IF_NULL(kernel_index.first);
        if (kernel_index.first->isa<CNode>() && kOptOperatorSet.find(AnfAlgo::GetCNodeName(
                                                  kernel_index.first->cast<CNodePtr>())) != kOptOperatorSet.end()) {
          return true;
        }
        if (!kernel_index.first->isa<CNode>()) {
          return false;
        }
        return trans_data_func(kernel_index.first->cast<CNodePtr>());
      }
      return false;
    };

    if (trans_pose_func(node)) {
      transpose_list.emplace_back(node);
    } else if (trans_data_func(node)) {
      trans_list.emplace_back(node);
    } else if (cast_func(node)) {
      cast_list.emplace_back(node);
    } else if (kOptOperatorSet.find(AnfAlgo::GetCNodeName(node)) != kOptOperatorSet.end()) {
      all_opt_list.emplace_back(node);
    } else {
      non_opt_list.emplace_back(node);
    }
  }
  node_list->clear();
  std::copy(non_opt_list.begin(), non_opt_list.end(), std::back_inserter(*node_list));
  std::copy(all_opt_list.begin(), all_opt_list.end(), std::back_inserter(*node_list));
  std::copy(transpose_list.begin(), transpose_list.end(), std::back_inserter(*node_list));
  std::copy(trans_list.begin(), trans_list.end(), std::back_inserter(*node_list));
  std::copy(cast_list.begin(), cast_list.end(), std::back_inserter(*node_list));
}

void AnfRuntimeAlgorithm::ReorderPosteriorExecList(NotNull<std::vector<CNodePtr> *> node_list) {
  std::vector<CNodePtr> ordinary_node_list;
  std::vector<CNodePtr> posterior_node_list;

  for (const auto &node : *node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (kPosteriorOperatorSet.find(AnfAlgo::GetCNodeName(node)) != kPosteriorOperatorSet.end()) {
      posterior_node_list.emplace_back(node);
    } else {
      ordinary_node_list.emplace_back(node);
    }
  }
  node_list->clear();
  std::copy(ordinary_node_list.begin(), ordinary_node_list.end(), std::back_inserter(*node_list));
  std::copy(posterior_node_list.begin(), posterior_node_list.end(), std::back_inserter(*node_list));
}

TypeId AnfRuntimeAlgorithm::GetCNodeOutputPrecision(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto prim = AnfAlgo::GetCNodePrimitive(node);
  if (prim == nullptr) {
    return kTypeUnknown;
  }

  TypeId except_type = kTypeUnknown;
  if (prim->GetAttr(kAttrOutputPrecision) != nullptr) {
    auto output_type_str = GetValue<std::string>(prim->GetAttr(kAttrOutputPrecision));
    if (output_type_str == "float16") {
      except_type = kNumberTypeFloat16;
    } else if (output_type_str == "float32") {
      except_type = kNumberTypeFloat32;
    } else {
      MS_LOG(EXCEPTION) << "The fix precision must be float16 or float32, but got " << output_type_str
                        << " trace: " << trace::DumpSourceLines(node);
    }
  }

  return except_type;
}

TypeId AnfRuntimeAlgorithm::GetPrevNodeOutputPrecision(const AnfNodePtr &node, size_t input_idx) {
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << node->DebugString() << ", input node is not CNode."
                      << " trace: " << trace::DumpSourceLines(node);
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (input_idx + 1 >= cnode->inputs().size()) {
    MS_LOG(EXCEPTION) << "Input index " << input_idx << " is larger than input number " << GetInputTensorNum(cnode)
                      << " trace: " << trace::DumpSourceLines(node);
  }
  auto input_node = cnode->input(input_idx + 1);
  MS_EXCEPTION_IF_NULL(input_node);
  auto kernel_with_index = VisitKernel(input_node, 0);
  if (!kernel_with_index.first->isa<CNode>()) {
    return kTypeUnknown;
  }
  return GetCNodeOutputPrecision(kernel_with_index.first);
}

bool AnfRuntimeAlgorithm::IsCondControlKernel(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Illegal null input of cnode."
                      << " trace: " << trace::DumpSourceLines(node);
  }
  auto input = node->input(kAnfPrimitiveIndex);
  return IsPrimitive(input, prim::kPrimLabelGoto) || IsPrimitive(input, prim::kPrimLabelSwitch);
}

bool AnfRuntimeAlgorithm::IsIndependentNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (AnfAlgo::GetKernelType(node) != AICPU_KERNEL) {
    return false;
  }

  if (AnfAlgo::GetCNodeName(node) == kGetNextOpName) {
    MS_LOG(INFO) << "GetNext should not be independent node";
    return false;
  }

  size_t input_nums = AnfAlgo::GetInputTensorNum(node);
  if (input_nums == 0) {
    return true;
  }

  auto inputs = node->inputs();
  for (size_t i = 1; i < inputs.size(); i++) {
    if (!inputs[i]->isa<ValueNode>()) {
      return false;
    }
  }
  return true;
}

bool AnfRuntimeAlgorithm::GetBooleanAttr(const AnfNodePtr &node, const std::string &attr) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto has_attr = AnfAlgo::HasNodeAttr(attr, cnode);
  if (!has_attr) {
    return false;
  }
  return AnfAlgo::GetNodeAttr<bool>(node, attr);
}

bool AnfRuntimeAlgorithm::HasDynamicShapeFlag(const PrimitivePtr &prim) {
  auto get_bool_attr = [](const PrimitivePtr &primitive, const std::string &attr_name) -> bool {
    MS_EXCEPTION_IF_NULL(primitive);
    if (!primitive->HasAttr(attr_name)) {
      return false;
    }
    return GetValue<bool>(primitive->GetAttr(attr_name));
  };
  return get_bool_attr(prim, kAttrInputIsDynamicShape) || get_bool_attr(prim, kAttrOutputIsDynamicShape) ||
         get_bool_attr(prim, kAttrIsDynamicShape);
}

bool AnfRuntimeAlgorithm::IsDynamicShape(const AnfNodePtr &node) {
  return GetBooleanAttr(node, kAttrInputIsDynamicShape) || GetBooleanAttr(node, kAttrOutputIsDynamicShape) ||
         GetBooleanAttr(node, kAttrIsDynamicShape);
}

void AnfRuntimeAlgorithm::GetRealDynamicShape(const std::vector<size_t> &shape,
                                              NotNull<std::vector<int64_t> *> dynamic_shape) {
  for (auto size : shape) {
    if (size == SIZE_MAX) {
      dynamic_shape->push_back(-1);
    } else {
      dynamic_shape->push_back(SizeToLong(size));
    }
  }
}

std::vector<int64_t> GetShapeFromSequeueShape(const abstract::SequeueShapePtr &sequeue_shape_ptr, size_t index,
                                              ShapeType type) {
  MS_EXCEPTION_IF_NULL(sequeue_shape_ptr);
  auto shape_list = sequeue_shape_ptr->shape();
  if (index >= shape_list.size()) {
    MS_LOG(EXCEPTION) << "Output Index:" << index << " >= " << shape_list.size();
  }

  auto shape = shape_list[index];
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->isa<abstract::Shape>()) {
    auto shape_ptr = shape->cast<abstract::ShapePtr>();
    if (type == kMaxShape) {
      return shape_ptr->max_shape().empty() ? shape_ptr->shape() : shape_ptr->max_shape();
    } else {
      return shape_ptr->min_shape().empty() ? shape_ptr->shape() : shape_ptr->min_shape();
    }
  } else {
    MS_LOG(EXCEPTION) << "Invalid Shape Type In Shape List";
  }
}

std::vector<int64_t> AnfRuntimeAlgorithm::GetInputMaxShape(const AnfNodePtr &anf_node, size_t index) {
  auto input_node_with_index = AnfAlgo::GetPrevNodeOutput(anf_node, index);
  return GetOutputMaxShape(input_node_with_index.first, input_node_with_index.second);
}

std::vector<int64_t> AnfRuntimeAlgorithm::GetInputMinShape(const AnfNodePtr &anf_node, size_t index) {
  auto input_node_with_index = AnfAlgo::GetPrevNodeOutput(anf_node, index);
  return GetOutputMinShape(input_node_with_index.first, input_node_with_index.second);
}

std::vector<int64_t> AnfRuntimeAlgorithm::GetOutputMaxShape(const AnfNodePtr &anf_node, size_t index) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto shape = anf_node->Shape();
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->isa<abstract::Shape>()) {
    auto shape_ptr = shape->cast<abstract::ShapePtr>();
    return shape_ptr->max_shape().empty() ? shape_ptr->shape() : shape_ptr->max_shape();
  } else if (shape->isa<abstract::SequeueShape>()) {
    auto shape_ptr = shape->cast<abstract::SequeueShapePtr>();
    return GetShapeFromSequeueShape(shape_ptr, index, kMaxShape);
  } else if (shape->isa<abstract::NoShape>()) {
    return {};
  } else {
    MS_LOG(EXCEPTION) << "Invalid Shape Type"
                      << " trace: " << trace::DumpSourceLines(anf_node);
  }
}

std::vector<int64_t> AnfRuntimeAlgorithm::GetOutputMinShape(const AnfNodePtr &anf_node, size_t index) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto shape = anf_node->Shape();
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->isa<abstract::Shape>()) {
    auto shape_ptr = shape->cast<abstract::ShapePtr>();
    return shape_ptr->min_shape().empty() ? shape_ptr->shape() : shape_ptr->min_shape();
  } else if (shape->isa<abstract::SequeueShape>()) {
    auto shape_ptr = shape->cast<abstract::SequeueShapePtr>();
    return GetShapeFromSequeueShape(shape_ptr, index, kMinShape);
  } else if (shape->isa<abstract::NoShape>()) {
    return {};
  } else {
    MS_LOG(EXCEPTION) << "Invalid Shape Type"
                      << " trace: " << trace::DumpSourceLines(anf_node);
  }
}

bool IsNodeOutputDynamicShape(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto base_shape = node->Shape();
  if (base_shape == nullptr) {
    MS_LOG(INFO) << "Invalid base shape, node: " << node->fullname_with_scope();
    return false;
  }
  if (base_shape->isa<abstract::Shape>()) {
    if (IsShapeDynamic(base_shape->cast<abstract::ShapePtr>())) {
      return true;
    }
  } else if (base_shape->isa<abstract::TupleShape>()) {
    auto tuple_shape = base_shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape);
    for (size_t i = 0; i < tuple_shape->size(); i++) {
      auto b_shape = (*tuple_shape)[i];
      if (!b_shape->isa<abstract::Shape>()) {
        continue;
      }
      if (IsShapeDynamic(b_shape->cast<abstract::ShapePtr>())) {
        return true;
      }
    }
  }
  return false;
}

bool IsNodeInputDynamicShape(const CNodePtr &anf_node_ptr) {
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  auto input_num = AnfAlgo::GetInputTensorNum(anf_node_ptr);
  for (size_t i = 0; i < input_num; ++i) {
    auto input_with_index = AnfAlgo::GetPrevNodeOutput(anf_node_ptr, i);
    auto input = input_with_index.first;
    auto index = input_with_index.second;
    MS_EXCEPTION_IF_NULL(input);

    auto base_shape = input->Shape();
    if (base_shape == nullptr) {
      MS_LOG(INFO) << "Invalid shape ptr, node:" << input->fullname_with_scope();
      continue;
    }
    if (base_shape->isa<abstract::Shape>()) {
      if (IsShapeDynamic(base_shape->cast<abstract::ShapePtr>())) {
        return true;
      }
    } else if (base_shape->isa<abstract::TupleShape>()) {
      auto tuple_shape = base_shape->cast<abstract::TupleShapePtr>();
      MS_EXCEPTION_IF_NULL(tuple_shape);

      if (index >= tuple_shape->size()) {
        MS_LOG(INFO) << "Node:" << anf_node_ptr->fullname_with_scope() << "Invalid index:" << index
                     << " and tuple_shape size:" << tuple_shape->size();
        continue;
      }

      auto b_shp = (*tuple_shape)[index];
      if (!b_shp->isa<abstract::Shape>()) {
        continue;
      }
      if (IsShapeDynamic(b_shp->cast<abstract::ShapePtr>())) {
        return true;
      }
    }
  }
  return false;
}

bool AnfRuntimeAlgorithm::IsNodeDynamicShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(DEBUG) << "Node is not a cnode";
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  auto in_dynamic = IsNodeInputDynamicShape(cnode);
  auto out_dynamic = IsNodeOutputDynamicShape(cnode);
  if (in_dynamic && !AnfAlgo::HasNodeAttr(kAttrInputIsDynamicShape, cnode)) {
    AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), cnode);
    MS_LOG(INFO) << "Set Input Dynamic Shape Attr to Node:" << cnode->fullname_with_scope();
  }
  if (out_dynamic && !AnfAlgo::HasNodeAttr(kAttrOutputIsDynamicShape, cnode)) {
    AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), cnode);
    MS_LOG(INFO) << "Set Output Dynamic Shape Attr to Node:" << cnode->fullname_with_scope();
  }
  return in_dynamic || out_dynamic;
}

std::vector<size_t> AnfRuntimeAlgorithm::GetInputRealDeviceShapeIfExist(const AnfNodePtr &anf_node, size_t index) {
  auto device_shape = GetInputDeviceShape(anf_node, index);
  // Initialize GPUKernel with max shape to fit 'InitDynamicOutputKernelRef()' for memory reuse.
  if (IsShapeDynamic(device_shape)) {
    auto max_shape = GetInputMaxShape(anf_node, index);
    std::transform(max_shape.begin(), max_shape.end(), device_shape.begin(), IntToSize);
    auto format = GetInputFormat(anf_node, index);
    trans::TransShapeToDevice(device_shape, format);
  }
  return device_shape;
}

std::vector<size_t> AnfRuntimeAlgorithm::GetOutputRealDeviceShapeIfExist(const AnfNodePtr &anf_node, size_t index) {
  auto device_shape = GetOutputDeviceShape(anf_node, index);
  // Initialize GPUKernel with max shape to fit 'InitDynamicOutputKernelRef()' for memory reuse.
  if (IsShapeDynamic(device_shape)) {
    auto max_shape = GetOutputMaxShape(anf_node, index);
    std::transform(max_shape.begin(), max_shape.end(), device_shape.begin(), IntToSize);
    auto format = GetOutputFormat(anf_node, index);
    trans::TransShapeToDevice(device_shape, format);
  }
  return device_shape;
}

void AnfRuntimeAlgorithm::GetAllFatherRealNode(const AnfNodePtr &anf_node, std::vector<AnfNodePtr> *result,
                                               std::set<AnfNodePtr> *visited) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(result);
  MS_EXCEPTION_IF_NULL(visited);
  if (visited->find(anf_node) != visited->end()) {
    MS_LOG(INFO) << "Node:" << anf_node->fullname_with_scope() << " has already been visited";
    return;
  }
  visited->insert(anf_node);
  if (AnfAlgo::IsRealKernel(anf_node)) {
    result->emplace_back(anf_node);
    return;
  }
  if (!anf_node->isa<CNode>()) {
    return;
  }
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Illegal null input of cnode(%s)" << anf_node->DebugString();
  }
  auto input0 = cnode->input(0);
  if (IsPrimitive(input0, prim::kPrimMakeTuple)) {
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      GetAllFatherRealNode(cnode->input(i), result, visited);
    }
  } else if (IsPrimitive(input0, prim::kPrimTupleGetItem)) {
    if (cnode->inputs().size() != kTupleGetItemInputSize) {
      MS_LOG(EXCEPTION) << "The node tuple_get_item must have 2 inputs!";
    }
    GetAllFatherRealNode(cnode->input(kRealInputNodeIndexInTupleGetItem), result, visited);
  } else if (IsPrimitive(input0, prim::kPrimDepend)) {
    if (cnode->inputs().size() != kDependInputSize) {
      MS_LOG(EXCEPTION) << "Depend node must have 2 inputs!";
    }
    GetAllFatherRealNode(cnode->input(kRealInputIndexInDepend), result, visited);
    GetAllFatherRealNode(cnode->input(kDependAttachNodeIndex), result, visited);
  }
}

void AnfRuntimeAlgorithm::InferShape(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(INFO) << "InferShape start, node:" << node->DebugString();
  auto inputs = node->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid inputs";
  }
  AbstractBasePtrList args_spec_list;
  auto primitive = GetValueNode<PrimitivePtr>(inputs[0]);
  auto input_size = AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < input_size; ++i) {
    auto input_with_index = AnfAlgo::GetPrevNodeOutput(node, i);
    auto real_input = input_with_index.first;
    MS_EXCEPTION_IF_NULL(real_input);
    auto cnode_input = node->input(i + 1);
    MS_EXCEPTION_IF_NULL(cnode_input);
    if (AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimTupleGetItem)) {
      auto base_shape = real_input->Shape();
      if (!base_shape->isa<abstract::TupleShape>()) {
        MS_LOG(EXCEPTION) << "Node:" << node->DebugString()
                          << " input is a tuple_get_item but real input node shape is not a TupleShape";
      }
      auto tuple_ptr = base_shape->cast<abstract::TupleShapePtr>();
      MS_EXCEPTION_IF_NULL(tuple_ptr);
      auto tuple_get_item_index = AnfAlgo::GetTupleGetItemOutIndex(cnode_input->cast<CNodePtr>());
      auto real_shape = tuple_ptr->shape().at(tuple_get_item_index);
      auto abstract_tensor = cnode_input->abstract()->cast<abstract::AbstractTensorPtr>();
      MS_EXCEPTION_IF_NULL(abstract_tensor);
      args_spec_list.emplace_back(std::make_shared<abstract::AbstractTensor>(abstract_tensor->element(), real_shape));
    } else if (cnode_input->isa<CNode>() && AnfAlgo::GetCNodeName(cnode_input) == prim::kPrimReshape->name()) {
      args_spec_list.emplace_back(cnode_input->abstract());
    } else {
      args_spec_list.emplace_back(real_input->abstract());
    }
  }
  auto eval_result = opt::CppInferShape(primitive, args_spec_list);
  node->set_abstract(eval_result);
}
}  // namespace session
}  // namespace mindspore
