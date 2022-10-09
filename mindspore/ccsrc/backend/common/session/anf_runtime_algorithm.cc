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
#include "backend/common/session/anf_runtime_algorithm.h"

#include <memory>
#include <algorithm>
#include <map>
#include <set>
#include <functional>
#include <numeric>
#include "ir/anf.h"
#include "mindspore/core/ops/core_ops.h"
#include "utils/shape_utils.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_dump_utils.h"
#include "runtime/device/kernel_info.h"
#include "runtime/device/device_address.h"
#include "backend/common/optimizer/helper.h"
#include "kernel/kernel.h"
#include "kernel/kernel_build_info.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "utils/trace_base.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"
#include "kernel/oplib/oplib.h"

namespace mindspore {
namespace session {
using abstract::AbstractTensor;
using abstract::AbstractTuple;
using device::KernelInfo;
using kernel::KernelBuildInfoPtr;
using kernel::KernelMod;
using kernel::KernelModPtr;
namespace {
constexpr size_t kReturnDataIndex = 1;
constexpr size_t kSwitchTrueBranchIndex = 2;

// ops pair that dynamic input order is differ from the fixed shape ops
// pair: <input_index_in_kernel->input_index_in_graph, input_index_in_graph->input_index_in_kernel>
static std::map<std::string, std::pair<std::map<size_t, size_t>, std::map<size_t, size_t>>> spec_dynamic_node_list = {
  {kStridedSliceGradOpName, {{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}}, {{1, 0}, {2, 1}, {3, 2}, {4, 3}, {0, 4}}}},
  {kConv2DBackpropInputOpName, {{{0, 2}, {1, 1}, {2, 0}}, {{0, 2}, {1, 1}, {2, 0}}}},
  {kConv2DBackpropFilterOpName, {{{0, 1}, {1, 2}, {2, 0}}, {{1, 0}, {2, 1}, {0, 2}}}}};

// pair: <input_index_in_kernel->input_index_in_graph, input_index_in_graph->input_index_in_kernel>
static std::map<std::string, std::pair<std::map<size_t, size_t>, std::map<size_t, size_t>>> spec_node_list = {
  {kConv2DBackpropInputOpName, {{{0, 1}, {1, 0}}, {{0, 1}, {1, 0}}}},
  {kFusionOpConv2DBackpropInputReluGradV2Name, {{{0, 1}, {1, 0}, {2, 2}}, {{0, 1}, {1, 0}, {2, 2}}}},
  {kFusionOpConv2DBackpropInputAddNReluGradV2Name,
   {{{0, 1}, {1, 0}, {2, 2}, {3, 3}}, {{0, 1}, {1, 0}, {2, 2}, {3, 3}}}},
  {kConv2DBackpropFilterOpName, {{{0, 1}, {1, 0}}, {{0, 1}, {1, 0}}}},
  {kLogSoftmaxGradOpName, {{{0, 1}, {1, 0}}, {{0, 1}, {1, 0}}}},
  {kLayerNormGradOpName, {{{0, 1}, {1, 0}, {2, 2}, {3, 3}, {4, 4}}, {{0, 1}, {1, 0}, {2, 2}, {3, 3}, {4, 4}}}},
  {kLayerNormBetaGammaBackpropOpName, {{{0, 1}, {1, 0}, {2, 2}, {3, 3}}, {{0, 1}, {1, 0}, {2, 2}, {3, 3}}}},
  {kLayerNormXBackpropOpName, {{{0, 1}, {1, 0}, {2, 2}, {3, 3}, {4, 4}}, {{0, 1}, {1, 0}, {2, 2}, {3, 3}, {4, 4}}}},
  {kLayerNormXBackpropV2OpName, {{{0, 1}, {1, 0}, {2, 2}, {3, 3}, {4, 4}}, {{0, 1}, {1, 0}, {2, 2}, {3, 3}, {4, 4}}}},
  {kMinimumGradOpName, {{{0, 2}, {1, 0}, {2, 1}}, {{2, 0}, {0, 1}, {1, 2}}}},
  {kMaximumGradOpName, {{{0, 2}, {1, 0}, {2, 1}}, {{2, 0}, {0, 1}, {1, 2}}}},
  {kApplyCenteredRMSPropOpName,
   {{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 4}},
    {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {5, 4}, {6, 5}, {7, 6}, {8, 7}, {4, 8}}}},
  {kStridedSliceGradOpName, {{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}}, {{1, 0}, {2, 1}, {3, 2}, {4, 3}, {0, 4}}}}};

std::string PrintKernelFormatAndType(const std::string &fmt, const TypeId &type, const std::vector<int64_t> &shape) {
  std::ostringstream buffer;
  buffer << "<" << TypeIdLabel(type);
  if (!fmt.empty()) {
    buffer << "x" << fmt << shape;
  }
  buffer << ">";
  return buffer.str();
}

struct AnfDumpHandlerRegister {
  AnfDumpHandlerRegister() {
    AnfDumpHandler::SetPrintInputTypeShapeFormatHandler(
      [](const std::shared_ptr<AnfNode> &node, size_t idx) -> std::string {
        if (node == nullptr) {
          return "";
        }
        auto format = AnfAlgo::GetInputFormat(node, idx);
        auto type = AnfAlgo::GetInputDeviceDataType(node, idx);
        auto shape = AnfAlgo::GetInputDeviceShape(node, idx);
        return PrintKernelFormatAndType(format, type, shape);
      });
    AnfDumpHandler::SetPrintOutputTypeShapeFormatHandler(
      [](const std::shared_ptr<AnfNode> &node, size_t idx) -> std::string {
        if (node == nullptr) {
          return "";
        }
        auto format = AnfAlgo::GetOutputFormat(node, idx);
        auto type = AnfAlgo::GetOutputDeviceDataType(node, idx);
        auto shape = AnfAlgo::GetOutputDeviceShape(node, idx);
        return PrintKernelFormatAndType(format, type, shape);
      });
  }
} callback_register;

tensor::TensorPtr GetForwardOutputTensor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<tensor::Tensor>()) {
      auto tensor = value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      if (tensor->is_forward_output()) {
        return tensor;
      }
    }
  }
  return nullptr;
}
}  // namespace

AnfNodePtr AnfRuntimeAlgorithm::MakeMonadValueNode(const KernelGraphPtr &kg) {
  return kg->NewValueNode(kUMonad->ToAbstract(), kUMonad);
}

// Convert: a = former(xxx)
//          b = latter(x, xxx)
// To:      a = former(xxx)
//          d1 = Depend(x, a)
//          b = latter(d1, xxx)
//          ...
//          out = Depend(out, latter)
void AnfRuntimeAlgorithm::KeepOrder(const KernelGraphPtr &kg, const AnfNodePtr &former, const AnfNodePtr &latter) {
  MS_EXCEPTION_IF_NULL(kg);
  MS_EXCEPTION_IF_NULL(latter);
  if (latter->isa<CNode>()) {
    auto latter_cnode = latter->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(latter_cnode);
    constexpr size_t inputsize = 2;
    constexpr size_t kFirstDataInputIndex = 1;
    if (latter_cnode->inputs().size() < inputsize) {
      return;
    }
    auto latter_input = latter_cnode->input(kFirstDataInputIndex);
    auto depend1 = kg->NewCNode({NewValueNode(prim::kPrimDepend), latter_input, former});
    MS_EXCEPTION_IF_NULL(depend1);
    depend1->set_abstract(latter_input->abstract());
    latter_cnode->set_input(kFirstDataInputIndex, depend1);

    auto return_node = kg->get_return();
    MS_EXCEPTION_IF_NULL(return_node);
    auto depend2 = kg->NewCNode(
      {NewValueNode(prim::kPrimDepend), return_node->cast<CNodePtr>()->input(kFirstDataInputIndex), latter});
    MS_EXCEPTION_IF_NULL(depend2);
    depend2->set_abstract(return_node->cast<CNodePtr>()->input(kFirstDataInputIndex)->abstract());
    kg->set_output(depend2);
    MS_LOG(DEBUG) << "former: " << former->DebugString() << ", latter: " << latter->DebugString()
                  << ", depend1: " << depend1->DebugString() << ", depend2: " << depend2->DebugString();
  }
}

size_t AnfRuntimeAlgorithm::GetOutputTensorMemSize(const AnfNodePtr &node, size_t output_index) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_index >= common::AnfAlgo::GetOutputTensorNum(node)) {
    MS_EXCEPTION(ArgumentError) << "output index [" << output_index << "] large than the output size ["
                                << common::AnfAlgo::GetOutputTensorNum(node) << "] of node!";
  }
  TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(node, output_index);
  if (output_type_id == kTypeUnknown) {
    output_type_id = common::AnfAlgo::GetOutputInferDataType(node, output_index);
  }
  size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
  auto shape = AnfAlgo::GetOutputDeviceShape(node, output_index);
  if (IsDynamic(shape)) {
    auto max_shape = common::AnfAlgo::GetOutputMaxShape(node, output_index);
    if (!max_shape.empty()) {
      shape = max_shape;
      MS_LOG(DEBUG) << "shape[" << shape << "] is dynamic, using max_shape[" << max_shape << "] instead.";
    } else {
      shape = {1};
      MS_LOG(DEBUG) << "shape[" << shape << "] is dynamic, set default to {1}";
    }
  }
  auto format = AnfAlgo::GetOutputFormat(node, output_index);
  auto dtype = AnfAlgo::GetOutputDeviceDataType(node, output_index);
  if (shape.empty() && format != kOpFormat_DEFAULT) {
    shape = trans::PaddingShape(shape, format, AnfAlgo::GetOutputReshapeType(node, output_index), node);
    shape = trans::TransShapeToDevice(shape, format, node, output_index, dtype);
  }
  // scalar's output shape is a empty vector
  size_t tensor_size = type_size * SizeOf(shape);
  return tensor_size;
}

std::vector<std::string> AnfRuntimeAlgorithm::GetAllOutputFormats(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    MS_LOG(EXCEPTION) << "Not real kernel:"
                      << "#node [" << node->DebugString() << "]" << trace::DumpSourceLines(node);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto format = build_info->GetAllOutputFormats();
  return format;
}

std::vector<std::string> AnfRuntimeAlgorithm::GetAllInputFormats(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    MS_LOG(EXCEPTION) << "Not real kernel:"
                      << "#node [" << node->DebugString() << "]" << trace::DumpSourceLines(node);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto format = build_info->GetAllInputFormats();
  return format;
}

std::vector<TypeId> AnfRuntimeAlgorithm::GetAllInputDeviceTypes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    MS_LOG(EXCEPTION) << "Not real kernel:"
                      << "#node [" << node->DebugString() << "]" << trace::DumpSourceLines(node);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto types = build_info->GetAllInputDeviceTypes();
  return types;
}

std::vector<TypeId> AnfRuntimeAlgorithm::GetAllOutputDeviceTypes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    MS_LOG(EXCEPTION) << "Not real kernel:"
                      << "#node [" << node->DebugString() << "]" << trace::DumpSourceLines(node);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto types = build_info->GetAllOutputDeviceTypes();
  return types;
}

std::string AnfRuntimeAlgorithm::GetOriginDataFormat(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    MS_LOG(EXCEPTION) << "Not real kernel:"
                      << "#node [" << node->DebugString() << "]" << trace::DumpSourceLines(node);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto format = build_info->GetOriginDataFormat();
  return format;
}

std::string AnfRuntimeAlgorithm::GetOutputFormat(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_idx > common::AnfAlgo::GetOutputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "Output index:" << output_idx
                      << " is out of the node output range :" << common::AnfAlgo::GetOutputTensorNum(node) << " #node ["
                      << node->DebugString() << "]" << trace::DumpSourceLines(node);
  }
  if (common::AnfAlgo::CheckAbsSparseTensor(node)) {
    return kOpFormat_DEFAULT;
  }
  if (!AnfUtils::IsRealKernel(node)) {
    return AnfAlgo::GetPrevNodeOutputFormat(node, output_idx);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto format = build_info->GetOutputFormat(output_idx);
  if (format == kernel::KernelBuildInfo::kInvalidFormat) {
    MS_LOG(EXCEPTION) << "Node [" << node->DebugString() << "]"
                      << " has a invalid output format" << trace::DumpSourceLines(node);
  }
  return format;
}

std::string AnfRuntimeAlgorithm::GetInputFormat(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (input_idx > common::AnfAlgo::GetInputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "Input index :" << input_idx
                      << " is out of the number node Input range :" << common::AnfAlgo::GetInputTensorNum(node)
                      << "#node [" << node->DebugString() << "]" << trace::DumpSourceLines(node);
  }
  if (!AnfUtils::IsRealKernel(node)) {
    return GetPrevNodeOutputFormat(node, input_idx);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto format = build_info->GetInputFormat(input_idx);
  if (format == kernel::KernelBuildInfo::kInvalidFormat) {
    MS_LOG(EXCEPTION) << "Node [" << node->DebugString() << "]"
                      << " has a invalid input format" << trace::DumpSourceLines(node);
  }
  return format;
}

std::string AnfRuntimeAlgorithm::GetPrevNodeOutputFormat(const AnfNodePtr &anf_node, size_t input_idx) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
}

std::string AnfRuntimeAlgorithm::GetPrevNodeOutputReshapeType(const AnfNodePtr &node, size_t input_idx) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, input_idx);
  return GetOutputReshapeType(kernel_with_index.first, kernel_with_index.second);
}

std::vector<int64_t> AnfRuntimeAlgorithm::GetOutputDeviceShapeForTbeBuild(const AnfNodePtr &node,
                                                                          const size_t output_idx,
                                                                          const std::string &format) {
  auto output_shape = common::AnfAlgo::GetOutputDetailShape(node, output_idx);
  std::vector<int64_t> infer_shape;
  if (output_shape->isa<abstract::Shape>()) {
    auto shape_ptr = output_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    infer_shape = shape_ptr->shape();
  }
  if (infer_shape.empty()) {
    return infer_shape;
  }

  // if format is default_format or NC1KHKWHWC0,device shape = original shape
  if (trans::IsNeedPadding(format, infer_shape.size())) {
    infer_shape = trans::PaddingShape(infer_shape, format, GetOutputReshapeType(node, output_idx), node);
  }
  auto dtype = GetOutputDeviceDataType(node, output_idx);
  return trans::TransShapeToDevice(infer_shape, format, node, output_idx, dtype);
}

bool AnfRuntimeAlgorithm::IsShapesDynamic(const std::vector<ShapeVector> &shapes) {
  for (const auto &shape : shapes) {
    if (IsDynamic(shape)) {
      return true;
    }
  }

  return false;
}

ShapeVector AnfRuntimeAlgorithm::GetOutputDeviceShape(const AnfNodePtr &node, size_t output_idx) {
  auto format = GetOutputFormat(node, output_idx);
  auto infer_shape = common::AnfAlgo::GetOutputInferShape(node, output_idx);
  if (infer_shape.empty()) {
    return infer_shape;
  }

  // if format is default_format or NC1KHKWHWC0,device shape = original shape
  if (trans::IsNeedPadding(format, infer_shape.size())) {
    infer_shape = trans::PaddingShape(infer_shape, format, GetOutputReshapeType(node, output_idx), node);
  }
  auto dtype = GetOutputDeviceDataType(node, output_idx);
  return trans::TransShapeToDevice(infer_shape, format, node, output_idx, dtype);
}

std::vector<int64_t> AnfRuntimeAlgorithm::GetInputDeviceShapeForTbeBuild(const AnfNodePtr &node, const size_t input_idx,
                                                                         const std::string &format) {
  auto output_shape = common::AnfAlgo::GetPrevNodeOutputDetailShape(node, input_idx);
  std::vector<int64_t> infer_shape;
  if (output_shape->isa<abstract::Shape>()) {
    auto shape_ptr = output_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    infer_shape = shape_ptr->shape();
  }
  if (infer_shape.empty()) {
    return infer_shape;
  }

  // if format is default_format or NC1KHKWHWC0,device shape = original shape
  if (trans::IsNeedPadding(format, infer_shape.size())) {
    infer_shape = trans::PaddingShape(infer_shape, format, GetInputReshapeType(node, input_idx), node);
  }
  auto dtype = GetInputDeviceDataType(node, input_idx);
  return trans::TransShapeToDevice(infer_shape, format, node, input_idx, dtype, false);
}

std::vector<int64_t> AnfRuntimeAlgorithm::GetInputDeviceShape(const AnfNodePtr &node, size_t input_idx) {
  auto format = GetInputFormat(node, input_idx);
  auto infer_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, input_idx);
  if (infer_shape.empty()) {
    return infer_shape;
  }
  // if format is default_format or NC1KHKWHWC0,device shape = original shape
  if (trans::IsNeedPadding(format, infer_shape.size())) {
    infer_shape = trans::PaddingShape(infer_shape, format, GetInputReshapeType(node, input_idx), node);
  }
  auto dtype = GetInputDeviceDataType(node, input_idx);
  return trans::TransShapeToDevice(infer_shape, format, node, input_idx, dtype, false);
}

std::string AnfRuntimeAlgorithm::GetInputReshapeType(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (input_idx > common::AnfAlgo::GetInputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "The index:" << input_idx
                      << " is out of range of the node's input size : " << common::AnfAlgo::GetInputTensorNum(node)
                      << "#node[" << node->DebugString() << "]" << trace::DumpSourceLines(node);
  }
  if (!AnfUtils::IsRealKernel(node)) {
    return GetPrevNodeOutputReshapeType(node, input_idx);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr || build_info->IsInputDefaultPadding()) {
    return "";
  }
  return build_info->GetInputReshapeType(input_idx);
}

std::string AnfRuntimeAlgorithm::GetOutputReshapeType(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_idx > common::AnfAlgo::GetOutputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "The index [" << output_idx << "] is out of range of the node's output size [ "
                      << common::AnfAlgo::GetOutputTensorNum(node) << "#node[ " << node->DebugString() << "]"
                      << trace::DumpSourceLines(node);
  }
  if (!AnfUtils::IsRealKernel(node)) {
    return GetPrevNodeOutputReshapeType(node, output_idx);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr || build_info->IsOutputDefaultPadding()) {
    return "";
  }
  return build_info->GetOutputReshapeType(output_idx);
}

TypeId AnfRuntimeAlgorithm::GetOutputDeviceDataType(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_idx > common::AnfAlgo::GetOutputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "The index [" << output_idx << "] is out of range of the node's output size [ "
                      << common::AnfAlgo::GetOutputTensorNum(node) << "#node [ " << node->DebugString() << "]"
                      << trace::DumpSourceLines(node);
  }
  if (common::AnfAlgo::CheckAbsSparseTensor(node)) {
    return common::AnfAlgo::GetSparseTypeIdAt(node, output_idx);
  }
  if (!AnfUtils::IsRealKernel(node)) {
    return GetPrevNodeOutputDeviceDataType(node, output_idx);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto dtype = build_info->GetOutputDeviceType(output_idx);
  if (dtype == TypeId::kNumberTypeEnd) {
    MS_LOG(EXCEPTION) << "Node [" << node->DebugString() << "] has a invalid dtype" << trace::DumpSourceLines(node);
  }
  return dtype;
}

TypeId AnfRuntimeAlgorithm::GetInputDeviceDataType(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (input_idx > common::AnfAlgo::GetInputTensorNum(node)) {
    MS_LOG(EXCEPTION) << "The index [" << input_idx << "] is out of range of the node's input size [ "
                      << common::AnfAlgo::GetInputTensorNum(node) << "#node [ " << node->DebugString() << "]"
                      << trace::DumpSourceLines(node);
  }
  if (!AnfUtils::IsRealKernel(node)) {
    return GetPrevNodeOutputDeviceDataType(node, 0);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  auto dtype = build_info->GetInputDeviceType(input_idx);
  if (dtype == TypeId::kNumberTypeEnd) {
    MS_LOG(EXCEPTION) << "Node [" << node->DebugString() << "]"
                      << " has a invalid dtype." << trace::DumpSourceLines(node);
  }
  return dtype;
}

TypeId AnfRuntimeAlgorithm::GetPrevNodeOutputDeviceDataType(const AnfNodePtr &anf_node, size_t input_idx) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
}

// get output device addr of anf_node
const DeviceAddress *AnfRuntimeAlgorithm::GetOutputAddr(const AnfNodePtr &node, size_t output_idx, bool skip_nop_node) {
  MS_EXCEPTION_IF_NULL(node);
  auto tensor = GetForwardOutputTensor(node);
  if (tensor != nullptr) {
    return dynamic_cast<const DeviceAddress *>(tensor->device_address().get());
  }

  if (common::AnfAlgo::IsNopNode(node) && (skip_nop_node || common::AnfAlgo::IsNeedSkipNopOpAddr(node))) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    return AnfRuntimeAlgorithm::GetPrevNodeOutputAddr(cnode, 0);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetOutputAddr(output_idx);
  if (addr == nullptr) {
    MS_LOG(EXCEPTION) << "Output_idx " << output_idx << " of node " << node->DebugString()
                      << " output addr is not exist." << trace::DumpSourceLines(node);
  }
  return addr;
}

DeviceAddressPtr AnfRuntimeAlgorithm::GetMutableOutputAddr(const AnfNodePtr &node, size_t output_idx,
                                                           bool skip_nop_node) {
  MS_EXCEPTION_IF_NULL(node);
  auto tensor = GetForwardOutputTensor(node);
  if (tensor != nullptr) {
    return std::dynamic_pointer_cast<DeviceAddress>(tensor->device_address());
  }

  if (common::AnfAlgo::IsNopNode(node) && (skip_nop_node || common::AnfAlgo::IsNeedSkipNopOpAddr(node))) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    return AnfRuntimeAlgorithm::GetPrevNodeMutableOutputAddr(cnode, 0);
  }
  // Critical path performance optimization: `KernelInfo` is unique subclass of `KernelInfoDevice`
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetMutableOutputAddr(output_idx);
  if (addr == nullptr) {
    MS_LOG(EXCEPTION) << "Output_idx" << output_idx << " of node " << node->DebugString()
                      << " output addr is not exist." << trace::DumpSourceLines(node);
  }
  return addr;
}

// get output device addr of anf_node
bool AnfRuntimeAlgorithm::OutputAddrExist(const AnfNodePtr &node, size_t output_idx, bool skip_nop_node) {
  MS_EXCEPTION_IF_NULL(node);
  if (common::AnfAlgo::IsNopNode(node) && (skip_nop_node || common::AnfAlgo::IsNeedSkipNopOpAddr(node))) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->inputs().size() > 1) {
      auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, 0);
      return OutputAddrExist(kernel_with_index.first, kernel_with_index.second, skip_nop_node);
    }
    return false;
  }
  // Critical path performance optimization: `KernelInfo` is unique subclass of `KernelInfoDevice`
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->OutputAddrExist(output_idx);
}

bool AnfRuntimeAlgorithm::WorkspaceAddrExist(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  // Critical path performance optimization: `KernelInfo` is unique subclass of `KernelInfoDevice`
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->WorkspaceAddrExist(output_idx);
}

const DeviceAddress *AnfRuntimeAlgorithm::GetPrevNodeOutputAddr(const AnfNodePtr &anf_node, size_t input_idx,
                                                                bool skip_nop_node) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputAddr(kernel_with_index.first, kernel_with_index.second, skip_nop_node);
}

DeviceAddressPtr AnfRuntimeAlgorithm::GetPrevNodeMutableOutputAddr(const AnfNodePtr &anf_node, size_t input_idx,
                                                                   bool skip_nop_node) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetMutableOutputAddr(kernel_with_index.first, kernel_with_index.second, skip_nop_node);
}

size_t AnfRuntimeAlgorithm::GetOutputAddressNum(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->GetOutputNumWithoutMonad();
}

// set output device addr of anf_node
void AnfRuntimeAlgorithm::SetOutputAddr(const DeviceAddressPtr &addr, size_t output_idx, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (!kernel_info->SetOutputAddr(addr, output_idx)) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << "set output index:" << output_idx << " fail."
                      << trace::DumpSourceLines(node);
  }
}

// set workspace device addr of anf_node
void AnfRuntimeAlgorithm::SetWorkspaceAddr(const DeviceAddressPtr &addr, size_t output_idx, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (!kernel_info->SetWorkspaceAddr(addr, output_idx)) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << "set output index:" << output_idx << " fail."
                      << trace::DumpSourceLines(node);
  }
}

// get workspace device addr of anf_node
DeviceAddress *AnfRuntimeAlgorithm::GetWorkspaceAddr(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetWorkspaceAddr(output_idx);
  if (addr == nullptr) {
    MS_LOG(EXCEPTION) << "Output_idx " << output_idx << " of node " << node->DebugString()
                      << "] workspace addr is not exist." << trace::DumpSourceLines(node);
  }
  return addr;
}

// get workspace device mutable addr of anf_node
DeviceAddressPtr AnfRuntimeAlgorithm::GetMutableWorkspaceAddr(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetMutableWorkspaceAddr(index);
  if (addr == nullptr) {
    MS_LOG(EXCEPTION) << "Index " << index << " of node " << node->DebugString() << "] workspace addr is not exist."
                      << trace::DumpSourceLines(node);
  }
  return addr;
}

kernel::OpPattern AnfRuntimeAlgorithm::GetOpPattern(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  // select_kernel_build_info() has checked whether return pointer is null
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->op_pattern();
}

// get KernelBuildType of node, such as ATT,RT,FWK and so on
KernelType AnfRuntimeAlgorithm::GetKernelType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  // select_kernel_build_info() has checked whether return pointer is null
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->kernel_type();
}

void AnfRuntimeAlgorithm::SetFusionType(const AnfNodePtr &node, const kernel::FusionType &type) {
  MS_EXCEPTION_IF_NULL(node);
  auto builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(node));
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetFusionType(type);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), node.get());
}

void AnfRuntimeAlgorithm::SetCoreType(const AnfNodePtr &node, const std::string &core_type) {
  MS_EXCEPTION_IF_NULL(node);
  auto builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(node));
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetCoreType(core_type);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), node.get());
}

void AnfRuntimeAlgorithm::SetOutputDataDesc(const AnfNodePtr &node, const std::vector<nlohmann::json> &desc) {
  MS_EXCEPTION_IF_NULL(node);
  auto builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(node));
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetOutputDataDesc(desc);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), node.get());
}

std::vector<nlohmann::json> AnfRuntimeAlgorithm::GetOutputDataDesc(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  if (kernel_info == nullptr) {
    return {};
  }
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    return {};
  }
  return build_info->output_data_desc();
}

kernel::Processor AnfRuntimeAlgorithm::GetProcessor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->processor();
}

kernel::FusionType AnfRuntimeAlgorithm::GetFusionType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    return kernel::FusionType::UNKNOWN_FUSION_TYPE;
  }
  return build_info->fusion_type();
}

// set select kernel_build_info
void AnfRuntimeAlgorithm::SetSelectKernelBuildInfo(const KernelBuildInfoPtr &select_kernel_build_info, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->set_select_kernel_build_info(select_kernel_build_info);
}

// get select kernel_build_info
KernelBuildInfoPtr AnfRuntimeAlgorithm::GetSelectKernelBuildInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->GetMutableSelectKernelBuildInfo();
}

// get kernelMode
KernelMod *AnfRuntimeAlgorithm::GetKernelMod(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->MutableKernelMod();
}

// set kernel mod
void AnfRuntimeAlgorithm::SetKernelMod(const KernelModPtr &kernel_mod, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_kernel_mod(kernel_mod);
}

void AnfRuntimeAlgorithm::SetStreamId(uint32_t stream_id, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_stream_id(stream_id);
}

uint32_t AnfRuntimeAlgorithm::GetStreamId(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->stream_id();
}

void AnfRuntimeAlgorithm::SetStreamDistinctionLabel(uint32_t stream_label, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_stream_distinction_label(stream_label);
}

uint32_t AnfRuntimeAlgorithm::GetStreamDistinctionLabel(const AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<const device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->stream_distinction_label();
}

void AnfRuntimeAlgorithm::SetGraphId(uint32_t graph_id, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_graph_id(graph_id);
}

uint32_t AnfRuntimeAlgorithm::GetGraphId(const AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<const device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->graph_id();
}

bool AnfRuntimeAlgorithm::IsFeatureMapOutput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    ValuePtr value = value_node->value();
    std::vector<tensor::TensorPtr> tensors;
    TensorValueToTensor(value, &tensors);
    auto ret = false;
    if (!tensors.empty()) {
      auto all_tensor_have_address = true;
      for (const auto &tensor : tensors) {
        MS_EXCEPTION_IF_NULL(tensor);
        if (tensor->device_address() == nullptr) {
          all_tensor_have_address = false;
          break;
        }
      }
      ret = all_tensor_have_address;
    }
    return ret;
  }
  if (IsPrimitiveCNode(node, prim::kPrimLoad)) {
    return IsFeatureMapOutput(node->cast<CNodePtr>()->input(1));
  }
  auto kernel_info = dynamic_cast<const device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  return kernel_info->is_feature_map();
}

bool AnfRuntimeAlgorithm::IsFeatureMapInput(const AnfNodePtr &node, size_t input_index) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Cannot input a parameter or a valuenode to charge it's input if is a feature map."
                      << trace::DumpSourceLines(node);
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_node = cnode->input(input_index + 1);
  return IsFeatureMapOutput(input_node);
}

size_t AnfRuntimeAlgorithm::GetInputIndexInGraph(const mindspore::AnfNodePtr &anf_node,
                                                 const size_t input_index_in_kernel) {
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t ret = input_index_in_kernel;
  auto node_name = common::AnfAlgo::GetCNodeName(anf_node);
  if (AnfAlgo::GetKernelType(anf_node) == TBE_KERNEL || AnfAlgo::GetKernelType(anf_node) == ACL_KERNEL) {
    if (common::AnfAlgo::IsDynamicShape(anf_node)) {
      auto find_dynamic = spec_dynamic_node_list.find(node_name);
      if (find_dynamic != spec_dynamic_node_list.cend()) {
        auto dyn_index_converter = find_dynamic->second;
        ret = dyn_index_converter.first[input_index_in_kernel];
        MS_LOG(DEBUG) << "Real input index change to " << ret << ", node name:" << node_name;
        return ret;
      }
      auto op_info = kernel::OpLib::FindOp(node_name, kernel::kTBE, true);
      if (op_info != nullptr) {
        auto real_input_index = op_info->real_input_index();
        if (!real_input_index.first.empty()) {
          ret = real_input_index.first[input_index_in_kernel];
          return ret;
        }
      }
    }
    auto find = spec_node_list.find(node_name);
    if (find != spec_node_list.cend()) {
      auto index_converter = find->second;
      ret = index_converter.first[input_index_in_kernel];
      MS_LOG(DEBUG) << "Real input index change to " << ret << ", node name:" << node_name;
      return ret;
    }
    auto op_info = kernel::OpLib::FindOp(node_name, kernel::kTBE);
    if (op_info != nullptr) {
      auto real_input_index = op_info->real_input_index();
      if (!real_input_index.first.empty()) {
        ret = real_input_index.first[input_index_in_kernel];
        return ret;
      }
    }
  }
  return ret;
}

size_t AnfRuntimeAlgorithm::GetInputIndexInKernel(const mindspore::AnfNodePtr &anf_node,
                                                  const size_t input_index_in_graph) {
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t ret = input_index_in_graph;
  auto node_name = common::AnfAlgo::GetCNodeName(anf_node);
  if (AnfAlgo::GetKernelType(anf_node) == TBE_KERNEL) {
    if (common::AnfAlgo::IsDynamicShape(anf_node)) {
      auto find_dynamic = spec_dynamic_node_list.find(node_name);
      if (find_dynamic != spec_dynamic_node_list.cend()) {
        auto dyn_index_converter = find_dynamic->second;
        ret = dyn_index_converter.second[input_index_in_graph];
        MS_LOG(DEBUG) << "Get original input index " << ret << ", node name:" << node_name;
        return ret;
      }
      auto op_info = kernel::OpLib::FindOp(node_name, kernel::kTBE, true);
      if (op_info != nullptr) {
        auto real_input_index = op_info->real_input_index();
        if (!real_input_index.second.empty()) {
          ret = real_input_index.second[input_index_in_graph];
          return ret;
        }
      }
    }
    auto find = spec_node_list.find(node_name);
    if (find != spec_node_list.cend()) {
      auto index_converter = find->second;
      ret = index_converter.second[input_index_in_graph];
      MS_LOG(DEBUG) << "Get original input index " << ret << ", node name:" << node_name;
    }
    auto op_info = kernel::OpLib::FindOp(node_name, kernel::kTBE);
    if (op_info != nullptr) {
      auto real_input_index = op_info->real_input_index();
      if (!real_input_index.second.empty()) {
        ret = real_input_index.second[input_index_in_graph];
        return ret;
      }
    }
  }
  return ret;
}

std::vector<KernelGraphPtr> AnfRuntimeAlgorithm::GetCallSwitchKernelGraph(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!(common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall) ||
        common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch) ||
        common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitchLayer))) {
    MS_LOG(EXCEPTION) << "Node: " << cnode->DebugString() << "is not a call or switch or switch_layer node."
                      << trace::DumpSourceLines(cnode);
  }
  auto get_switch_kernel_graph = [cnode](size_t input_index) -> KernelGraphPtr {
    auto partial = cnode->input(input_index);
    MS_EXCEPTION_IF_NULL(partial);
    if (IsValueNode<KernelGraph>(partial)) {
      return GetValueNode<KernelGraphPtr>(partial);
    }
    auto partial_cnode = partial->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(partial_cnode);
    auto graph_node = partial_cnode->input(kPartialGraphIndex);
    MS_EXCEPTION_IF_NULL(graph_node);
    auto graph_value_node = graph_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(graph_value_node);
    auto graph_value = graph_value_node->value();
    MS_EXCEPTION_IF_NULL(graph_value);
    auto child_graph = graph_value->cast<KernelGraphPtr>();
    return child_graph;
  };
  if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall)) {
    auto input1 = cnode->input(kPartialGraphIndex);
    MS_EXCEPTION_IF_NULL(input1);
    auto value_node = input1->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto kernel_graph = value_node->value();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    return {kernel_graph->cast<KernelGraphPtr>()};
  } else if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch)) {
    return {get_switch_kernel_graph(kSwitchTrueBranchIndex), get_switch_kernel_graph(kSwitchFalseBranchIndex)};
  } else if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitchLayer)) {
    std::vector<KernelGraphPtr> child_graphs;
    for (size_t idx = kSwitchLayerBranchesIndex; idx < cnode->inputs().size(); idx++) {
      auto child_graph = get_switch_kernel_graph(idx);
      child_graphs.emplace_back(child_graph);
    }
    return child_graphs;
  }
  return {};
}

bool AnfRuntimeAlgorithm::IsIndependentNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (AnfAlgo::GetKernelType(node) != AICPU_KERNEL) {
    return false;
  }

  if (common::AnfAlgo::GetCNodeName(node) == kGetNextOpName) {
    MS_LOG(INFO) << "GetNext should not be independent node";
    return false;
  }

  // aicpu stack ops are not independent nodes.
  if (common::AnfAlgo::GetCNodeName(node) == kStackInitOpName ||
      common::AnfAlgo::GetCNodeName(node) == kStackDestroyOpName ||
      common::AnfAlgo::GetCNodeName(node) == kStackPopOpName ||
      common::AnfAlgo::GetCNodeName(node) == kStackPushOpName) {
    MS_LOG(INFO) << "AICPU stack ops should not be independent node";
    return false;
  }

  size_t input_nums = common::AnfAlgo::GetInputTensorNum(node);
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

static inline void GetMaxOrDefaultShape(const std::vector<int64_t> &max_shape, ShapeVector *device_shape) {
  constexpr int64_t kDefaultValueForDynamicDim = 16;
  auto ConvertNegOneToDefault = [&kDefaultValueForDynamicDim](int64_t size) {
    return static_cast<int64_t>(size) < 0 ? kDefaultValueForDynamicDim : size;
  };
  if (!max_shape.empty()) {
    if (device_shape->empty()) {
      (void)std::transform(max_shape.begin(), max_shape.end(), std::back_inserter(*device_shape),
                           ConvertNegOneToDefault);
    } else {
      *device_shape = max_shape;
    }
  } else {
    auto tmp_shape = *device_shape;
    (void)std::transform(tmp_shape.begin(), tmp_shape.end(), device_shape->begin(), ConvertNegOneToDefault);
  }
}

// This function get input device shape adaptively in case of dynamic shape and static shape.
// when shape is dynamic, it firstly get shape value from max_shape. If max_shape is empty, it
// just return default shape value to avoid calculating error in init of kernels.
// why do we do this? Because in dynamic shape case, the input shape is unknown when the `init`
// function executes at the very first time, but we still need to  some helpful shape to make
// sure the `init` executes correctly.
ShapeVector AnfRuntimeAlgorithm::GetInputDeviceShapeAdaptively(const AnfNodePtr &anf_node, size_t index) {
  auto device_shape = GetInputDeviceShape(anf_node, index);
  // Initialize GPUKernel with max shape to fit 'InitDynamicOutputKernelRef()' for memory reuse.
  if (IsDynamic(device_shape) || device_shape.empty()) {
    auto max_shape = common::AnfAlgo::GetInputMaxShape(anf_node, index);
    GetMaxOrDefaultShape(max_shape, &device_shape);
    auto format = GetInputFormat(anf_node, index);
    auto dtype = GetInputDeviceDataType(anf_node, index);
    (void)trans::TransShapeToDevice(device_shape, format, anf_node, index, dtype, false);
  }

  if (device_shape.empty()) {
    KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(anf_node, index);
    auto shape = common::AnfAlgo::GetOutputInferShape(kernel_with_index.first, kernel_with_index.second);
    ShapeVector ret_shape;
    constexpr int64_t kDefaultValueForDynamicDim = 1;
    auto ConvertNegOneToDefault = [&kDefaultValueForDynamicDim](int64_t size) {
      return size < 0 ? kDefaultValueForDynamicDim : size;
    };
    std::transform(shape.begin(), shape.end(), std::back_inserter(ret_shape), ConvertNegOneToDefault);
    auto format = GetInputFormat(anf_node, index);
    auto dtype = GetInputDeviceDataType(anf_node, index);
    (void)trans::TransShapeToDevice(ret_shape, format, anf_node, index, dtype, false);
    return ret_shape;
  }

  return device_shape;
}

// The same to GetInputDeviceShapeAdaptively
ShapeVector AnfRuntimeAlgorithm::GetOutputDeviceShapeAdaptively(const AnfNodePtr &anf_node, size_t index) {
  auto device_shape = GetOutputDeviceShape(anf_node, index);
  // Initialize GPUKernel with max shape to fit 'InitDynamicOutputKernelRef()' for memory reuse.
  if (IsDynamic(device_shape) || device_shape.empty()) {
    auto max_shape = common::AnfAlgo::GetOutputMaxShape(anf_node, index);
    GetMaxOrDefaultShape(max_shape, &device_shape);
    auto format = GetOutputFormat(anf_node, index);
    auto dtype = GetOutputDeviceDataType(anf_node, index);
    (void)trans::TransShapeToDevice(device_shape, format, anf_node, index, dtype);
  }

  if (device_shape.empty()) {
    auto shape = common::AnfAlgo::GetOutputInferShape(anf_node, index);
    ShapeVector ret_shape;
    constexpr int64_t kDefaultValueForDynamicDim = 1;
    auto ConvertNegOneToOne = [&kDefaultValueForDynamicDim](int64_t size) {
      return size < 0 ? kDefaultValueForDynamicDim : size;
    };
    (void)std::transform(shape.cbegin(), shape.cend(), std::back_inserter(ret_shape), ConvertNegOneToOne);
    auto format = GetOutputFormat(anf_node, index);
    auto dtype = GetOutputDeviceDataType(anf_node, index);
    (void)trans::TransShapeToDevice(ret_shape, format, anf_node, index, dtype, false);
    return ret_shape;
  }

  return device_shape;
}

KernelGraphPtr AnfRuntimeAlgorithm::FetchKernelGraph(const AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &func_graph = node->func_graph();
  if (func_graph == nullptr) {
    return nullptr;
  } else {
    return func_graph->cast<KernelGraphPtr>();
  }
}

AnfNodePtr AnfRuntimeAlgorithm::FetchFrontNodeByBackendNode(const AnfNodePtr &backend_node, const KernelGraph &graph) {
  MS_EXCEPTION_IF_NULL(backend_node);
  auto front_node_with_index = graph.GetFrontNodeByInternalParameter(backend_node);
  if (front_node_with_index.first != nullptr) {
    return front_node_with_index.first;
  }

  auto front_node = graph.GetFrontAnfByBackendAnf(backend_node);
  // PyNative forward graph does not has front node, using backend node instead.
  if (front_node == nullptr) {
    front_node = backend_node;
  }
  return front_node;
}

namespace {
// Host kernel with inputs on host
bool SkipDataSync(const CNodePtr &node, const std::map<uint32_t, tensor::TensorPtr> &depend_tensors) {
  if (!common::AnfAlgo::IsHostKernel(node)) {
    return false;
  }
  auto input_size = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < input_size; ++i) {
    auto input_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i);
    auto real_input = input_with_index.first;
    auto iter_tensor = depend_tensors.find(i);
    if (iter_tensor != depend_tensors.end()) {
      auto output_addr = AnfAlgo::GetOutputAddr(real_input, 0);
      MS_EXCEPTION_IF_NULL(output_addr);
      if (output_addr->GetDeviceType() != device::DeviceType::kCPU) {
        return false;
      }
    }
  }
  return true;
}
}  // namespace

void AnfRuntimeAlgorithm::InferShape(const CNodePtr &node, std::map<uint32_t, tensor::TensorPtr> *depend_tensors) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(INFO) << "InferShape start, node:" << node->DebugString();
  auto inputs = node->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Inputs should not be empty! Cnode: " << node->DebugString() << "."
                      << trace::DumpSourceLines(node);
  }
  AbstractBasePtrList args_spec_list;
  auto primitive = GetValueNode<PrimitivePtr>(inputs[0]);
  auto input_size = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < input_size; ++i) {
    auto input_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i);
    auto real_input = input_with_index.first;
    MS_EXCEPTION_IF_NULL(real_input);
    auto cnode_input = node->input(i + 1);
    MS_EXCEPTION_IF_NULL(cnode_input);
    if (depend_tensors != nullptr) {
      auto iter_tensor = depend_tensors->find(i);
      if (iter_tensor != depend_tensors->cend()) {
        auto tensor_ptr = iter_tensor->second;
        MS_EXCEPTION_IF_NULL(tensor_ptr);
        if (!SkipDataSync(node, *depend_tensors)) {
          // sync data from device to host
          tensor_ptr->data_sync();
        }
        // cppcheck-suppress unreadVariable
        auto lock = AnfUtils::GetAbstractLock(real_input.get());
        auto real_abs = real_input->abstract();
        if (real_abs->isa<abstract::AbstractTensor>()) {
          real_abs->set_value(tensor_ptr);
        } else if (real_abs->isa<abstract::AbstractTuple>()) {
          auto tuple_get_item_index = common::AnfAlgo::GetTupleGetItemOutIndex(cnode_input->cast<CNodePtr>());
          auto abstract_tuple = real_abs->cast<abstract::AbstractTuplePtr>();
          MS_EXCEPTION_IF_NULL(abstract_tuple);
          auto tuple_elements = abstract_tuple->elements()[tuple_get_item_index];
          tuple_elements->set_value(tensor_ptr);
        }
      }
    }
    common::AnfAlgo::AddArgList(&args_spec_list, real_input, input_with_index.second);
  }
  auto eval_result = opt::CppInferShapeAndType(primitive, args_spec_list);
  node->set_abstract(eval_result);
}

void AnfRuntimeAlgorithm::InsertMakeTupleForOutput(const NotNull<KernelGraphPtr> &root_graph) {
  auto return_node = root_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->size() <= kReturnDataIndex) {
    return;
  }
  auto make_tuple = root_graph->NewCNode(
    {NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name())), root_graph->output()});
  MS_EXCEPTION_IF_NULL(root_graph->output());
  make_tuple->set_abstract({root_graph->output()->abstract()});
  root_graph->set_output(make_tuple);
}

void AnfRuntimeAlgorithm::CacheAddrForGraph(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode &&
      ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK) == true) {
    return;
  }
  auto nodes = kernel_graph->execution_order();
  for (auto &kernel : nodes) {
    // Skip transpose kernel with "nop_op" attr which is not hidden or removed in PyNative infer scenario. Transpose
    // kernel, which is not supposed to be executed, is generated in TransDataSplit to support specific Transdata.
    // And hard code here should be removed after new Transdata programme is implemented in the foreseeable future.
    if (common::AnfAlgo::HasNodeAttr(kAttrNopOp, kernel)) {
      for (size_t idx = 0; idx < common::AnfAlgo::GetOutputTensorNum(kernel); idx += 1) {
        auto real_input = GetInputIndexInGraph(kernel, idx);
        auto device_address = GetPrevNodeMutableOutputAddr(kernel, real_input);
        SetOutputAddr(device_address, idx, kernel.get());
      }
      continue;
    }
    auto kernel_mod = GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    if (common::AnfAlgo::GetCNodeName(kernel) == kAtomicAddrCleanOpName) {
      CacheAddrForAtomicClean(kernel, kernel_mod);
      continue;
    }
    CacheAddrForKernel(kernel, kernel_mod);
  }
}

void AnfRuntimeAlgorithm::CacheAddrForKernel(const AnfNodePtr &node, kernel::KernelMod *kernel_mod) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  std::vector<AddressPtr> kernel_inputs;
  std::vector<AddressPtr> kernel_workspaces;
  std::vector<AddressPtr> kernel_outputs;
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto skip_nop_node = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < input_num; ++i) {
    if (common::AnfAlgo::IsNoneInput(node, i)) {
      continue;
    }
    auto real_input = GetInputIndexInGraph(node, i);
    auto device_address = GetPrevNodeOutputAddr(node, real_input, skip_nop_node);
    MS_EXCEPTION_IF_NULL(device_address);
    kernel::AddressPtr input = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(input);
    input->addr = const_cast<void *>(device_address->GetPtr());
    MS_EXCEPTION_IF_NULL(input->addr);
    input->size = device_address->GetSize();
    kernel_inputs.emplace_back(input);
  }
  for (size_t i = 0; i < kernel_mod->GetOutputSizeList().size(); ++i) {
    auto device_address = GetOutputAddr(node, i, skip_nop_node);
    kernel::AddressPtr output = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(output);
    output->addr = const_cast<void *>(device_address->GetPtr());
    MS_EXCEPTION_IF_NULL(output->addr);
    output->size = device_address->GetSize();
    kernel_outputs.emplace_back(output);
  }
  for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
    auto device_address = GetWorkspaceAddr(node, i);
    kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(workspace);
    workspace->addr = const_cast<void *>(device_address->GetPtr());
    MS_EXCEPTION_IF_NULL(workspace->addr);
    workspace->size = device_address->GetSize();
    kernel_workspaces.emplace_back(workspace);
  }
  kernel_mod->set_inputs_addr(kernel_inputs);
  kernel_mod->set_workspaces_addr(kernel_workspaces);
  kernel_mod->set_outputs_addr(kernel_outputs);
}

void AnfRuntimeAlgorithm::CacheAddrForAtomicClean(const AnfNodePtr &node, kernel::KernelMod *kernel_mod) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  std::vector<AddressPtr> kernel_inputs;
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() != kIndex2) {
    MS_LOG(EXCEPTION) << "Atomic Addr clean Node Input nodes not equal 2.";
  }
  MS_EXCEPTION_IF_NULL(cnode->inputs()[1]);
  auto pre_node = (cnode->inputs()[1])->cast<CNodePtr>();
  // set clean output address
  if (common::AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, pre_node)) {
    auto clean_output_indexes = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicOutputIndexs);
    for (auto index : clean_output_indexes) {
      auto device_address = GetOutputAddr(pre_node, index);
      kernel::AddressPtr input = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(input);
      input->addr = const_cast<void *>(device_address->GetPtr());
      MS_EXCEPTION_IF_NULL(input->addr);
      input->size = device_address->GetSize();
      kernel_inputs.emplace_back(input);
    }
    MS_LOG(DEBUG) << "AtomicAddClean clean output size:" << clean_output_indexes.size();
  }
  // set clean workspace address
  if (common::AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, pre_node)) {
    auto clean_workspaces_indexes =
      common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicWorkspaceIndexs);
    for (const auto &index : clean_workspaces_indexes) {
      auto device_address = GetWorkspaceAddr(pre_node, index);
      kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(workspace);
      workspace->addr = const_cast<void *>(device_address->GetPtr());
      MS_EXCEPTION_IF_NULL(workspace->addr);
      workspace->size = device_address->GetSize();
      kernel_inputs.emplace_back(workspace);
    }
  }
  kernel_mod->set_inputs_addr(kernel_inputs);
}

void AnfRuntimeAlgorithm::UpdateGraphValidRefPair(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &origin_ref_map = graph->GetRefMap();
  std::map<AnfWithOutIndex, AnfWithOutIndex> new_ref_map;
  for (const auto &node : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(node);
    auto output_num = common::AnfAlgo::GetOutputTensorNum(node);
    if (output_num == 0) {
      MS_LOG(DEBUG) << "This kernel has no output size.";
      continue;
    }
    for (size_t i = 0; i < output_num; ++i) {
      session::AnfWithOutIndex out_pair(node, i);
      auto iter = origin_ref_map.find(out_pair);
      if (iter != origin_ref_map.end()) {
        auto ret = new_ref_map.try_emplace(iter->first, iter->second);
        if (!ret.second) {
          MS_LOG(WARNING) << "Duplicate ref_map key, node:" << node->fullname_with_scope() << " index:" << i;
        }
      }
    }
  }
  graph->set_ref_out_in_map(new_ref_map);
}

bool AnfRuntimeAlgorithm::IsDynamicShapeSkipExecute(const std::string &op_name, const ShapeVector &axes_shape) {
  // Skip run ReduceSum when axis is a Empty Tensor
  if (op_name != kReduceSumOpName) {
    return false;
  }
  if (std::any_of(axes_shape.begin(), axes_shape.end(), [](int64_t shape) { return shape == 0; })) {
    return true;
  }
  return false;
}

bool AnfRuntimeAlgorithm::IsDynamicShapeSkipExecute(const CNodePtr &cnode) {
  // Skip run ReduceSum when axis is a Empty Tensor
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  if (op_name != kReduceSumOpName) {
    return false;
  }

  const size_t axes_index = 1;
  if (cnode->inputs().size() <= axes_index + 1) {
    return false;
  }
  auto input_axes = cnode->input(axes_index + 1);
  // cppcheck-suppress unreadVariable
  auto lock = AnfUtils::GetAbstractLock(input_axes.get());
  auto axes_abs = input_axes->abstract()->Clone();
  MS_EXCEPTION_IF_NULL(axes_abs);
  auto axes_shape = AnfAlgo::GetInputDeviceShape(cnode, axes_index);
  if (axes_abs->isa<abstract::AbstractTensor>()) {
    if (std::any_of(axes_shape.begin(), axes_shape.end(), [](int64_t shape) { return shape == 0; })) {
      return true;
    }
  }
  return false;
}

bool AnfRuntimeAlgorithm::IsNeedUpdateShapeAndTypeAfterLaunch(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_mod = GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  return kernel_mod->IsNeedRetrieveOutputShape();
}

void AnfRuntimeAlgorithm::UpdateOutputAddrSize(device::KernelInfo const *kernel_info, const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto &output_addresses = kernel_info->output_address_list();
  for (size_t i = 0; i < output_addresses.size(); ++i) {
    auto output_address = output_addresses[i].get();
    MS_EXCEPTION_IF_NULL(output_address);
    auto output_addr_size = AnfAlgo::GetOutputTensorMemSize(kernel, i);
    if (output_addr_size != output_address->GetSize()) {
      output_address->SetSize(output_addr_size);
    }
  }
}

void AnfRuntimeAlgorithm::UpdateInternalParameterShape(
  const std::map<size_t, std::vector<AnfNodeWeakPtr>> &internal_parameters, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  for (auto &internal_parameter_iter : internal_parameters) {
    for (auto &internal_parameter_weakptr : internal_parameter_iter.second) {
      auto internal_parameter = internal_parameter_weakptr.lock();
      MS_EXCEPTION_IF_NULL(internal_parameter);
      common::AnfAlgo::SetOutputInferTypeAndShape(
        {common::AnfAlgo::GetOutputInferDataType(cnode, internal_parameter_iter.first)},
        {common::AnfAlgo::GetOutputInferShape(cnode, internal_parameter_iter.first)}, internal_parameter.get());
    }
  }
}

void AnfRuntimeAlgorithm::AddOutInRefToGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &cnode : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    auto kernel_info = dynamic_cast<device::KernelInfo *>(cnode->kernel_info());
    MS_EXCEPTION_IF_NULL(kernel_info);
    for (const auto &ref : kernel_info->out_in_ref_map()) {
      size_t output_index = ref.first;
      size_t input_index = ref.second;
      auto final_pair = std::make_pair(cnode, output_index);
      auto origin_pair = common::AnfAlgo::VisitKernel(common::AnfAlgo::GetInputNode(cnode, input_index), 0);
      MS_LOG(INFO) << "The reference relation output " << final_pair.first->fullname_with_scope()
                   << ", output index: " << final_pair.second << " to input "
                   << origin_pair.first->fullname_with_scope() << ", output index: " << origin_pair.second;
      // Add to graph only if the input is not a monad.
      if (!HasAbstractUMonad(origin_pair.first) && !HasAbstractIOMonad(origin_pair.first)) {
        graph->AddRefCorrespondPairs(final_pair, origin_pair);
      }
    }
  }
}
}  // namespace session
}  // namespace mindspore
