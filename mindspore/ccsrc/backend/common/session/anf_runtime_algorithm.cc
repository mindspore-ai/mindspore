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
#include "kernel/oplib/super_bar.h"

namespace mindspore::session {
using abstract::AbstractTensor;
using abstract::AbstractTuple;
using device::KernelInfo;
using kernel::KernelBuildInfoPtr;
using kernel::KernelMod;
using kernel::KernelModPtr;
constexpr char kDisableKernelBackoff[] = "MS_DISABLE_KERNEL_BACKOFF";

namespace {
constexpr size_t kReturnDataIndex = 1;
constexpr size_t kSwitchTrueBranchIndex = 2;

std::string PrintKernelFormatAndType(const std::string &fmt, const TypeId &type, const std::vector<int64_t> &shape) {
  std::ostringstream buffer;
  buffer << "<" << TypeIdLabel(type);
  if (!fmt.empty()) {
    buffer << "x" << fmt << shape;
  }
  buffer << ">";
  return buffer.str();
}

[[maybe_unused]] struct AnfDumpHandlerRegister {
  AnfDumpHandlerRegister() {
    AnfDumpHandler::SetPrintInputTypeShapeFormatHandler([](const std::shared_ptr<AnfNode> &node) -> std::string {
      if (node == nullptr) {
        return "";
      }
      std::ostringstream buffer;
      size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
      for (size_t i = 0; i < input_num; ++i) {
        if (i != 0) {
          buffer << ", ";
        }
        auto format = AnfAlgo::GetInputFormat(node, i);
        auto type = AnfAlgo::GetInputDeviceDataType(node, i);
        auto shape = AnfAlgo::GetInputDeviceShape(node, i);
        buffer << PrintKernelFormatAndType(format, type, shape);
      }
      return buffer.str();
    });
    AnfDumpHandler::SetPrintOutputTypeShapeFormatHandler([](const std::shared_ptr<AnfNode> &node) -> std::string {
      if (node == nullptr) {
        return "";
      }
      std::ostringstream buffer;
      size_t output_num = AnfAlgo::GetOutputTensorNum(node);
      for (size_t i = 0; i < output_num; ++i) {
        if (i != 0) {
          buffer << ", ";
        }
        auto format = AnfAlgo::GetOutputFormat(node, (node->isa<Parameter>() ? 0 : i));
        auto type = AnfAlgo::GetOutputDeviceDataType(node, (node->isa<Parameter>() ? 0 : i));
        auto shape = AnfAlgo::GetOutputDeviceShape(node, (node->isa<Parameter>() ? 0 : i));
        buffer << PrintKernelFormatAndType(format, type, shape);
      }
      return buffer.str();
    });
    AnfDumpHandler::SetPrintInputKernelObjectTypesHandler([](const std::shared_ptr<AnfNode> &node) -> std::string {
      if (node == nullptr) {
        return "";
      }
      auto input_obj_types = AnfAlgo::GetInputKernelObjectTypes(node);
      return std::accumulate(
        input_obj_types.begin(), input_obj_types.end(), std::string(), [](std::string &a, const KernelObjectType &b) {
          return a.empty() ? kernel::KernelObjectTypeLabel(b) : a + ", " + kernel::KernelObjectTypeLabel(b);
        });
    });
    AnfDumpHandler::SetPrintOutputKernelObjectTypesHandler([](const std::shared_ptr<AnfNode> &node) -> std::string {
      if (node == nullptr) {
        return "";
      }
      auto output_obj_types = AnfAlgo::GetOutputKernelObjectTypes(node);
      return std::accumulate(
        output_obj_types.begin(), output_obj_types.end(), std::string(), [](std::string &a, const KernelObjectType &b) {
          return a.empty() ? kernel::KernelObjectTypeLabel(b) : a + ", " + kernel::KernelObjectTypeLabel(b);
        });
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

size_t GetOutputTensorNumByKernelInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->kernel_info());
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  MS_EXCEPTION_IF_NULL(build_info);
  return build_info->GetAllOutputDeviceTypes().size();
}
}  // namespace

AnfNodePtr AnfRuntimeAlgorithm::MakeMonadValueNode(const KernelGraphPtr &kg) {
  MS_EXCEPTION_IF_NULL(kg);
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
  MS_EXCEPTION_IF_NULL(former);
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
    MS_EXCEPTION_IF_NULL(latter_input);
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

size_t AnfRuntimeAlgorithm::GetOutputTensorNum(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  size_t res;
  TypePtr type = node->Type();
  if (type == nullptr) {
    res = 0;
  } else if (type->isa<Tuple>()) {
    const auto &kernel_info = node->kernel_info();
    if (kernel_info == nullptr || (!kernel_info->has_build_info())) {
      return 1;
    }
    res = GetOutputTensorNumByKernelInfo(node);
  } else if (type->isa<TypeNone>()) {
    res = 0;
  } else if (type->isa<CSRTensorType>()) {
    // Currently, CSRTensor only supports 2-D matrix (shape has 2 values). 5 outputs = 3 Tensors + 2 shape values.
    constexpr size_t kCSRTensorOutputNum = 5;
    res = kCSRTensorOutputNum;
  } else if (type->isa<COOTensorType>()) {
    // Currently, COOTensor only supports 2-D matrix (shape has 2 values). 4 outputs = 2 Tensors + 2 shape values.
    constexpr size_t kCOOTensorOutputNum = 4;
    res = kCOOTensorOutputNum;
  } else if (AnfUtils::NeedJumpMonadOutput(node) && type->isa<MonadType>()) {
    // Some nodes could have monad outputs like RpcRecv. We need to jump these outputs.
    res = 0;
  } else {
    res = 1;
  }
  return res;
}

size_t AnfRuntimeAlgorithm::GetOutputNumWithoutKernelInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &kernel_info = node->kernel_info();
  if (kernel_info != nullptr) {
    MS_LOG(EXCEPTION) << "Kernel info is not null for node:" << node->DebugString();
  }

  size_t res;
  TypePtr type = node->Type();
  if (type == nullptr) {
    res = 0;
  } else if (type->isa<Tuple>()) {
    res = 1;
  } else if (type->isa<TypeNone>()) {
    res = 0;
  } else if (type->isa<CSRTensorType>()) {
    // Currently, CSRTensor only supports 2-D matrix (shape has 2 values). 5 outputs = 3 Tensors + 2 shape values.
    constexpr size_t kCSRTensorOutputNum = 5;
    res = kCSRTensorOutputNum;
  } else if (type->isa<COOTensorType>()) {
    // Currently, COOTensor only supports 2-D matrix (shape has 2 values). 4 outputs = 2 Tensors + 2 shape values.
    constexpr size_t kCOOTensorOutputNum = 4;
    res = kCOOTensorOutputNum;
  } else if (AnfUtils::NeedJumpMonadOutput(node) && type->isa<MonadType>()) {
    // Some nodes could have monad outputs like RpcRecv. We need to jump these outputs.
    res = 0;
  } else {
    res = 1;
  }
  return res;
}

namespace {
bool IsTupleHasDynamicSequence(const abstract::AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (!abstract->isa<abstract::AbstractSequence>()) {
    return false;
  }
  const auto &sequence_abs = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(sequence_abs);
  if (sequence_abs->dynamic_len() || sequence_abs->dynamic_len_element_abs() != nullptr) {
    return true;
  }
  if (std::any_of(sequence_abs->elements().begin(), sequence_abs->elements().end(),
                  [](const abstract::AbstractBasePtr &abs) { return IsTupleHasDynamicSequence(abs); })) {
    return true;
  }
  return false;
}
}  // namespace

size_t AnfRuntimeAlgorithm::GetOutputElementNum(const AnfNodePtr &node) {
  if (node->abstract() != nullptr && IsTupleHasDynamicSequence(node->abstract())) {
    return common::AnfAlgo::GetOutputNumByAbstract(node->abstract());
  }
  return AnfUtils::GetOutputTensorNum(node);
}

size_t AnfRuntimeAlgorithm::GetOutputTensorMemSize(const AnfNodePtr &node, size_t output_index) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_index >= AnfAlgo::GetOutputTensorNum(node)) {
    MS_EXCEPTION(ArgumentError) << "output index [" << output_index << "] large than the output size ["
                                << AnfAlgo::GetOutputTensorNum(node) << "] of node!";
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
  if (output_idx > AnfAlgo::GetOutputElementNum(node)) {
    MS_LOG(EXCEPTION) << "Output index:" << output_idx
                      << " is out of the node output range :" << AnfAlgo::GetOutputElementNum(node) << " #node ["
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
  std::string format;
  // If the output is TUPLE, output format list's size is 1. So we use the first element as the output format.
  // This scenario could happen before 'insert_type_transform_op' pass.
  auto output_obj_types = build_info->GetAllOutputKernelObjectTypes();
  if (!output_obj_types.empty() && output_obj_types[kIndex0] == KernelObjectType::TUPLE) {
    MS_LOG(DEBUG) << "TUPLE only has one output. So use index 0 output format.";
    format = build_info->GetOutputFormat(kIndex0);
  } else {
    format = build_info->GetOutputFormat(output_idx);
  }
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
                      << " has a invalid input format\n"
                      << trace::DumpSourceLines(node);
  }
  return format;
}

bool AnfRuntimeAlgorithm::IsEquivalentFormat(const std::string &src_format, const std::string &dst_format) {
  if (src_format == dst_format) {
    return true;
  }

  // Equivalent default format.
  if (((src_format == kOpFormat_DEFAULT) || (src_format == kOpFormat_NCHW) || (src_format == kOpFormat_ND)) &&
      ((dst_format == kOpFormat_DEFAULT) || (dst_format == kOpFormat_NCHW) || (dst_format == kOpFormat_ND))) {
    return true;
  }

  return false;
}

std::string AnfRuntimeAlgorithm::GetPrevNodeOutputFormat(const AnfNodePtr &anf_node, size_t input_idx) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(anf_node, input_idx);
  return AnfRuntimeAlgorithm::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
}

std::string AnfRuntimeAlgorithm::GetPrevNodeOutputReshapeType(const AnfNodePtr &node, size_t input_idx) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, input_idx);
  return GetOutputReshapeType(kernel_with_index.first, kernel_with_index.second);
}

std::vector<KernelObjectType> AnfRuntimeAlgorithm::GetInputKernelObjectTypes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    MS_LOG(EXCEPTION) << "Empty build info for node:" << node->fullname_with_scope()
                      << ", debug name:" << node->DebugString();
  }
  return build_info->GetAllInputKernelObjectTypes();
}

KernelObjectType AnfRuntimeAlgorithm::GetInputKernelObjectType(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    MS_LOG(EXCEPTION) << "Empty build info for node:" << node->fullname_with_scope()
                      << ", debug name:" << node->DebugString();
  }
  const auto &input_kernel_obj_types = build_info->GetAllInputKernelObjectTypes();
  if (input_idx >= input_kernel_obj_types.size()) {
    MS_LOG(EXCEPTION) << "Input index " << input_idx << ", but the node input kernel object types size just "
                      << input_kernel_obj_types.size() << ". node: " << node->fullname_with_scope()
                      << ", debug name:" << node->DebugString() << "." << trace::DumpSourceLines(node);
  }
  return input_kernel_obj_types[input_idx];
}

std::vector<KernelObjectType> AnfRuntimeAlgorithm::GetOutputKernelObjectTypes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    return {};
  }
  return build_info->GetAllOutputKernelObjectTypes();
}

KernelObjectType AnfRuntimeAlgorithm::GetOutputKernelObjectType(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    MS_LOG(EXCEPTION) << "Empty build info for node:" << node->fullname_with_scope()
                      << ", debug name:" << node->DebugString();
  }
  const auto &output_kernel_obj_types = build_info->GetAllOutputKernelObjectTypes();
  if (output_idx >= output_kernel_obj_types.size()) {
    MS_LOG(EXCEPTION) << "Output index " << output_idx << ", but the node output kernel object types size just "
                      << output_kernel_obj_types.size() << ". node: " << node->fullname_with_scope()
                      << ", debug name:" << node->DebugString() << "." << trace::DumpSourceLines(node);
  }
  return output_kernel_obj_types[output_idx];
}

bool AnfRuntimeAlgorithm::IsRealSquenceOutput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<KernelObjectType> objects = GetOutputKernelObjectTypes(node);
  bool is_real_tuple = false;
  if (objects.empty()) {
    return false;
  } else {
    is_real_tuple = (objects[0] == KernelObjectType::TUPLE);
  }
  return is_real_tuple;
}

std::vector<int64_t> AnfRuntimeAlgorithm::GetOutputDeviceShapeForTbeBuild(const AnfNodePtr &node, size_t output_idx,
                                                                          const std::string &format) {
  auto output_shape = AnfAlgo::GetOutputDetailShape(node, output_idx);
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
  return std::any_of(shapes.cbegin(), shapes.cend(), [](const auto &shape) { return IsDynamic(shape); });
}

ShapeVector AnfRuntimeAlgorithm::GetOutputDeviceShape(const AnfNodePtr &node, size_t output_idx) {
  auto format = GetOutputFormat(node, output_idx);
  auto infer_shape = common::AnfAlgo::GetOutputInferShape(node, output_idx, IsRealSquenceOutput(node));
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

std::vector<int64_t> AnfRuntimeAlgorithm::GetInputDeviceShapeForTbeBuild(const AnfNodePtr &node, size_t input_idx,
                                                                         const std::string &format) {
  auto output_shape = AnfAlgo::GetPrevNodeOutputDetailShape(node, input_idx);
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
  if (output_idx > AnfAlgo::GetOutputElementNum(node)) {
    MS_LOG(EXCEPTION) << "The index [" << output_idx << "] is out of range of the node's output size [ "
                      << AnfAlgo::GetOutputElementNum(node) << "#node[ " << node->DebugString() << "]"
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
  if (output_idx > AnfAlgo::GetOutputElementNum(node)) {
    MS_LOG(EXCEPTION) << "The index [" << output_idx << "] is out of range of the node's output size [ "
                      << AnfAlgo::GetOutputElementNum(node) << "#node [ " << node->DebugString() << "]"
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

  // If node has only one output and it is Tuple, in build_info, it only has one same dtype, so set output_dix as zero.
  if (build_info->GetOutputNum() == 1 && build_info->GetOutputKernelObjectType(0) == kernel::KernelObjectType::TUPLE) {
    output_idx = 0;
  }

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
    MS_LOG(EXCEPTION) << "Output_idx " << output_idx << " of node " << node->DebugString()
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
  if (build_info == nullptr) {
    MS_LOG(DEBUG) << "Node: " << node->fullname_with_scope() << " has no kernel build info, using UNKNOWN_KERNEL_TYPE";
    return KernelType::UNKNOWN_KERNEL_TYPE;
  }
  return build_info->kernel_type();
}

void AnfRuntimeAlgorithm::SetFusionType(const AnfNodePtr &node, const std::string &type) {
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

std::string AnfRuntimeAlgorithm::GetFusionType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  if (build_info == nullptr) {
    return kernel::kPatternUnknown;
  }
  return build_info->fusion_type();
}

// set select kernel_build_info
void AnfRuntimeAlgorithm::SetSelectKernelBuildInfo(const KernelBuildInfoPtr &select_kernel_build_info, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (kernel_info->has_build_info() && (select_kernel_build_info != nullptr)) {
    auto ori_kernel_build_info = kernel_info->GetMutableSelectKernelBuildInfo();
    auto input_object_types = ori_kernel_build_info->GetAllInputKernelObjectTypes();
    auto output_object_types = ori_kernel_build_info->GetAllOutputKernelObjectTypes();
    if (!input_object_types.empty() && select_kernel_build_info->GetAllInputKernelObjectTypes().empty()) {
      select_kernel_build_info->SetInputsKernelObjectType(input_object_types);
    }
    if (!output_object_types.empty() && select_kernel_build_info->GetAllOutputKernelObjectTypes().empty()) {
      select_kernel_build_info->SetOutputsKernelObjectType(output_object_types);
    }
  }
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
  if (IsPrimitiveCNode(node, prim::kPrimLoad) || IsPrimitiveCNode(node, prim::kPrimDepend)) {
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

size_t AnfRuntimeAlgorithm::GetInputGraphIdxByKernelIdx(const mindspore::AnfNodePtr &anf_node,
                                                        size_t input_index_in_kernel) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto node_name = common::AnfAlgo::GetCNodeName(anf_node);
  auto kernel_type = AnfAlgo::GetKernelType(anf_node);
  if (kernel_type != TBE_KERNEL && kernel_type != ACL_KERNEL) {
    return input_index_in_kernel;
  }
  auto orders = kernel::SuperBar::GetKernelIdxToGraphIdx(node_name);
  if (!orders.has_value()) {
    return input_index_in_kernel;
  }
  auto input_orders = orders.value();
  auto find_iter = input_orders.find(input_index_in_kernel);
  if (find_iter == input_orders.end()) {
    MS_LOG(EXCEPTION) << "Get input order failed. input_idx: " << input_index_in_kernel << ", op_name: " << node_name;
  }
  return find_iter->second;
}

size_t AnfRuntimeAlgorithm::GetInputKernelIdxByGraphIdx(const mindspore::AnfNodePtr &anf_node,
                                                        size_t input_index_in_graph) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto node_name = common::AnfAlgo::GetCNodeName(anf_node);
  auto kernel_type = AnfAlgo::GetKernelType(anf_node);
  if (kernel_type != TBE_KERNEL && kernel_type != ACL_KERNEL) {
    return input_index_in_graph;
  }
  auto orders = kernel::SuperBar::GetGraphIdxToKernelIdx(node_name);
  if (!orders.has_value()) {
    return input_index_in_graph;
  }
  auto input_orders = orders.value();
  auto find_iter = input_orders.find(input_index_in_graph);
  if (find_iter == input_orders.end()) {
    MS_LOG(EXCEPTION) << "Get input order failed. input_idx: " << input_index_in_graph << ", op_name: " << node_name;
  }
  return find_iter->second;
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

static inline void GetDefaultShape(ShapeVector *device_shape) {
  auto ConvertNegOneToDefault = [](int64_t size) {
    constexpr int64_t kDefaultValueForDynamicDim = 16;
    return static_cast<int64_t>(size) < 0 ? kDefaultValueForDynamicDim : size;
  };
  auto tmp_shape = *device_shape;
  (void)std::transform(tmp_shape.begin(), tmp_shape.end(), device_shape->begin(), ConvertNegOneToDefault);
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
  if (IsDynamic(device_shape)) {
    GetDefaultShape(&device_shape);
  }

  return device_shape;
}

// The same to GetInputDeviceShapeAdaptively
ShapeVector AnfRuntimeAlgorithm::GetOutputDeviceShapeAdaptively(const AnfNodePtr &anf_node, size_t index) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto device_shape = GetOutputDeviceShape(anf_node, index);
  // Initialize GPUKernel with max shape to fit 'InitDynamicOutputKernelRef()' for memory reuse.
  if (IsDynamic(device_shape)) {
    GetDefaultShape(&device_shape);
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
        } else if (real_abs->isa<abstract::AbstractTuple>() && (!common::AnfAlgo::IsDynamicSequence(real_input))) {
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
  MS_EXCEPTION_IF_NULL(make_tuple);
  make_tuple->set_abstract({root_graph->output()->abstract()});
  root_graph->set_output(make_tuple);
}

void AnfRuntimeAlgorithm::CacheAddrForGraph(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode &&
      ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK)) {
    return;
  }
  auto nodes = kernel_graph->execution_order();
  for (auto &kernel : nodes) {
    // Skip transpose kernel with "nop_op" attr which is not hidden or removed in PyNative infer scenario. Transpose
    // kernel, which is not supposed to be executed, is generated in TransDataSplit to support specific Transdata.
    // And hard code here should be removed after new Transdata programme is implemented in the foreseeable future.
    if (common::AnfAlgo::HasNodeAttr(kAttrNopOp, kernel)) {
      for (size_t idx = 0; idx < AnfAlgo::GetOutputTensorNum(kernel); idx += 1) {
        auto real_input = GetInputGraphIdxByKernelIdx(kernel, idx);
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
    auto real_input = GetInputGraphIdxByKernelIdx(node, i);
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
    auto output_num = AnfAlgo::GetOutputTensorNum(node);
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

bool AnfRuntimeAlgorithm::IsDynamicShapeSkipExecute(bool skip_mode, const ShapeVector &axes_shape) {
  // Skip run ReduceSum when axis is a Empty Tensor
  if (std::any_of(axes_shape.begin(), axes_shape.end(), [](int64_t shape) { return shape == 0; }) && skip_mode) {
    return true;
  }
  return false;
}

bool AnfRuntimeAlgorithm::IsDynamicShapeSkipExecute(const CNodePtr &cnode) {
  // Skip run ReduceSum when axis is a Empty Tensor
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  if ((op_name != kReduceSumOpName) && (op_name != kReduceSumDOpName)) {
    return false;
  }

  bool skip_mode = false;
  if (common::AnfAlgo::HasNodeAttr(kAttrSkipMode, cnode)) {
    skip_mode = common::AnfAlgo::GetNodeAttr<bool>(cnode, kAttrSkipMode);
  }

  const size_t axes_index = 1;
  if (cnode->inputs().size() <= axes_index + 1) {
    return false;
  }
  auto input_axes = cnode->input(axes_index + 1);
  // cppcheck-suppress unreadVariable
  auto lock = AnfUtils::GetAbstractLock(input_axes.get());
  auto abs = input_axes->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto axes_abs = abs->Clone();
  MS_EXCEPTION_IF_NULL(axes_abs);
  auto axes_shape = AnfAlgo::GetInputDeviceShape(cnode, axes_index);
  if (axes_abs->isa<abstract::AbstractTensor>()) {
    if (std::any_of(axes_shape.begin(), axes_shape.end(), [](int64_t shape) { return shape == 0; }) && skip_mode) {
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

bool AnfRuntimeAlgorithm::HasOriginFormat(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  return anf_node->isa<CNode>() && common::AnfAlgo::HasNodeAttr(kAttrOriginFormat, anf_node->cast<CNodePtr>());
}

std::string AnfRuntimeAlgorithm::GetOriginFormat(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (anf_node->isa<CNode>() && common::AnfAlgo::HasNodeAttr(kAttrOriginFormat, anf_node->cast<CNodePtr>())) {
    return common::AnfAlgo::GetNodeAttr<std::string>(anf_node, kAttrOriginFormat);
  }
  return {};
}

bool AnfRuntimeAlgorithm::NodeValueIsFuncGraph(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value().get();
  MS_EXCEPTION_IF_NULL(value);
  return value->isa<FuncGraph>();
}

bool AnfRuntimeAlgorithm::IsEnableKernelSelectBackoff() {
  static std::string disable_kernel_backoff;
  static bool first_get_backoff_env = true;
  if (first_get_backoff_env) {
    disable_kernel_backoff = common::GetEnv(kDisableKernelBackoff);
    first_get_backoff_env = false;
  }

  return (disable_kernel_backoff != "1");
}

void AnfRuntimeAlgorithm::SetKernelSelectBackoffInfo(const CNodePtr &node,
                                                     const std::pair<std::string, ExceptionType> &failure_info) {
  MS_EXCEPTION_IF_NULL(node);
  common::AnfAlgo::SetNodeAttr(kAttrKernelBackoffWithFailureInfo, MakeValue(failure_info.first), node);
  common::AnfAlgo::SetNodeAttr(kAttrKernelBackoffWithFailureType, MakeValue(static_cast<int32_t>(failure_info.second)),
                               node);
}

std::pair<std::string, ExceptionType> AnfRuntimeAlgorithm::GetKernelSelectBackoffInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsKernelSelectBackoffOp(node)) {
    return {"", NoExceptionType};
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto failure_info = common::AnfAlgo::GetNodeAttr<std::string>(node, kAttrKernelBackoffWithFailureInfo);
  auto failure_type =
    static_cast<ExceptionType>(common::AnfAlgo::GetNodeAttr<int32_t>(node, kAttrKernelBackoffWithFailureType));
  return {failure_info, failure_type};
}

bool AnfRuntimeAlgorithm::IsKernelSelectBackoffOp(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::HasNodeAttr(kAttrKernelBackoffWithFailureInfo, cnode) &&
      common::AnfAlgo::HasNodeAttr(kAttrKernelBackoffWithFailureType, cnode)) {
    return true;
  }
  return false;
}

std::string AnfRuntimeAlgorithm::FetchDeviceTarget(const AnfNodePtr &node, const KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  if (!node->isa<CNode>()) {
    return device::GetDeviceNameByType(graph->device_target());
  }

  // Only the CPU support kernel backoff.
  if (AnfAlgo::IsKernelSelectBackoffOp(node)) {
    return kCPUDevice;
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto ud_target = cnode->user_data<std::string>(kAttrPrimitiveTarget);
  if (ud_target != nullptr) {
    return *ud_target;
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrPrimitiveTarget, cnode)) {
    return common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrPrimitiveTarget);
  }

  return device::GetDeviceNameByType(graph->device_target());
}

TypeId AnfRuntimeAlgorithm::GetAbstractObjectType(const AbstractBasePtr &abstract) {
  if (abstract == nullptr) {
    return kTypeUnknown;
  }
  if (abstract->isa<AbstractTensor>()) {
    return kObjectTypeTensorType;
  }
  if (abstract->isa<AbstractTuple>()) {
    return kObjectTypeTuple;
  }
  if (abstract->isa<abstract::AbstractList>()) {
    return kObjectTypeList;
  }
  if (abstract->isa<abstract::AbstractScalar>()) {
    // scalar input may not converted to tensor
    return kObjectTypeNumber;
  }
  return kTypeUnknown;
}

TypeId AnfRuntimeAlgorithm::GetOutputObjectType(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto abstract = node->abstract();
  if (abstract->isa<AbstractTuple>()) {
    auto tuple_abs = abstract->cast<abstract::AbstractTuplePtr>();
    auto items = tuple_abs->elements();
    MS_EXCEPTION_IF_CHECK_FAIL(output_idx < items.size(), "invalid output_idx");
    return AnfAlgo::GetAbstractObjectType(items[output_idx]);
  }
  if (output_idx != 0) {
    MS_LOG(EXCEPTION) << node->DebugString() << "invalid output_idx" << trace::DumpSourceLines(node);
  }
  return AnfAlgo::GetAbstractObjectType(abstract);
}

TypeId AnfRuntimeAlgorithm::GetInputObjectType(const CNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto input_node = common::AnfAlgo::GetInputNode(node, input_idx);
  const std::vector<PrimitivePtr> need_handled_prims = {prim::kPrimMakeTuple, prim::kPrimTupleGetItem};
  auto real_input_node = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, false, need_handled_prims).first;
  return AnfAlgo::GetAbstractObjectType(real_input_node->abstract());
}

std::vector<TypeId> AnfRuntimeAlgorithm::GetAllInputObjectType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << node->DebugString() << "anf_node is not CNode." << trace::DumpSourceLines(node);
  }
  auto cnode = node->cast<CNodePtr>();
  std::vector<TypeId> obj_types;
  auto input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t index = 0; index < input_num; ++index) {
    obj_types.push_back(AnfAlgo::GetInputObjectType(cnode, index));
  }
  return obj_types;
}

std::vector<TypeId> AnfRuntimeAlgorithm::GetAllOutputObjectType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (AnfAlgo::GetOutputElementNum(node) == 0 && !node->abstract()->isa<abstract::AbstractSequence>()) {
    return {};
  }
  return {AnfAlgo::GetAbstractObjectType(node->abstract())};
}

abstract::BaseShapePtr AnfRuntimeAlgorithm::GetOutputDetailShape(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto base_shape = node->Shape();
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::Shape>()) {
    if (output_idx == 0) {
      return base_shape;
    }
    MS_LOG(EXCEPTION) << "The node " << node->DebugString() << "is a single output node but got index [" << output_idx
                      << "]." << trace::DumpSourceLines(node);
  } else if (base_shape->isa<abstract::TupleShape>()) {
    auto tuple_shape = base_shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape);
    if (IsRealSquenceOutput(node)) {
      return tuple_shape;
    }
    if (output_idx >= tuple_shape->size()) {
      MS_LOG(EXCEPTION) << "Output index " << output_idx << "is larger than output number " << tuple_shape->size()
                        << " node:" << node->DebugString() << "." << trace::DumpSourceLines(node);
    }
    auto b_shp = (*tuple_shape)[output_idx];
    if (b_shp->isa<abstract::Shape>() || b_shp->isa<abstract::NoShape>()) {
      return b_shp;
    } else {
      MS_LOG(EXCEPTION) << "The output type of ApplyKernel index:" << output_idx
                        << " should be a NoShape , ArrayShape or a TupleShape, but it is " << base_shape->ToString()
                        << "node :" << node->DebugString() << "." << trace::DumpSourceLines(node);
    }
  } else if (base_shape->isa<abstract::NoShape>()) {
    return base_shape;
  } else if (base_shape->isa<abstract::DynamicSequenceShape>()) {
    return common::AnfAlgo::GetDynamicSequenceShape(node, output_idx);
  }
  MS_LOG(EXCEPTION) << "The output type of ApplyKernel should be a NoShape , ArrayShape or a TupleShape, but it is "
                    << base_shape->ToString() << " node : " << node->DebugString() << trace::DumpSourceLines(node);
}

abstract::BaseShapePtr AnfRuntimeAlgorithm::GetPrevNodeOutputDetailShape(const AnfNodePtr &node, size_t input_idx) {
  KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, input_idx);
  return AnfAlgo::GetOutputDetailShape(kernel_with_index.first, kernel_with_index.second);
}

std::vector<TypeId> AnfRuntimeAlgorithm::GetAllOutputInferDataTypes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<TypeId> outputs;
  auto out_nums = AnfAlgo::GetOutputElementNum(node);
  for (size_t i = 0; i < out_nums; i++) {
    auto type = common::AnfAlgo::GetOutputInferDataType(node, i);
    outputs.push_back(type);
  }
  return outputs;
}

// if input node is MakeTuple, find the PrevNodeNum recursively;
// The monad node in the end is not included in the num;
size_t AnfRuntimeAlgorithm::GetInputElementNum(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  size_t element_num = 0;
  size_t input_num = cnode->inputs().size() - 1;
  bool cal_monad_flag = false;
  for (size_t i = input_num; i > 0; --i) {
    auto input_node = common::AnfAlgo::GetInputNode(cnode, i - 1);
    if (!cal_monad_flag && HasAbstractMonad(input_node)) {
      continue;
    } else if (common::AnfAlgo::CheckPrimitiveType(input_node, prim::kPrimMakeTuple)) {
      element_num += GetInputElementNum(input_node);
      cal_monad_flag = true;
    } else if (common::AnfAlgo::IsTupleOutput(input_node)) {
      element_num += AnfAlgo::GetOutputElementNum(input_node);
      cal_monad_flag = true;
    } else {
      ++element_num;
      cal_monad_flag = true;
    }
  }

  return element_num;
}

void AnfRuntimeAlgorithm::SetDynamicAttrToPrim(const PrimitivePtr &prim) {
  prim->AddAttr(kAttrMutableKernel, MakeValue(true));
  prim->AddAttr(kAttrInputIsDynamicShape, MakeValue(true));
  prim->AddAttr(kAttrOutputIsDynamicShape, MakeValue(true));
}

bool AnfRuntimeAlgorithm::IsScalarConvertToTensor(const AnfNodePtr &input_node, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(node);
  if (!input_node->isa<ValueNode>()) {
    return false;
  }

  auto value_node = input_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<Scalar>()) {
    return false;
  }

  bool need_converted = true;
  const auto &abs = node->abstract();
  // Check the output abstract of node whether is scalar.
  if ((abs != nullptr) && (abs->isa<abstract::AbstractScalar>())) {
    need_converted = false;
  }
  // Check the output abstracts of node whether have scalar.
  if ((abs != nullptr) && (abs->isa<abstract::AbstractSequence>())) {
    auto abs_seq = abs->cast_ptr<abstract::AbstractSequence>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    if (abs_seq->dynamic_len()) {
      const auto &element_abs = abs_seq->dynamic_len_element_abs();
      need_converted = (element_abs == nullptr) || (!element_abs->isa<abstract::AbstractScalar>());
    } else {
      const auto &elements = abs_seq->elements();
      need_converted = !std::any_of(elements.begin(), elements.end(), [](const AbstractBasePtr &element) {
        return (element != nullptr) && element->isa<abstract::AbstractScalar>();
      });
    }
  }

  if (!need_converted) {
    MS_LOG(INFO) << "The input scalar value node:" << input_node->fullname_with_scope()
                 << " of cnode:" << node->fullname_with_scope() << " doesn't need convert to tensor.";
  }
  return need_converted;
}
}  // namespace mindspore::session
