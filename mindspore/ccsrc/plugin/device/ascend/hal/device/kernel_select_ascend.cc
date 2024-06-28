/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"
#include <vector>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <set>
#include "mindspore/core/ops/array_ops.h"
#include "plugin/device/ascend/kernel/hccl/hccl_kernel_metadata.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"

#ifndef ENABLE_SECURITY
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/optimizer/helper.h"
#include "mindspore/core/ops/framework_ops.h"
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_build.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_build.h"
#include "plugin/device/ascend/kernel/host/host_kernel_build.h"
#include "plugin/device/ascend/kernel/host/host_kernel_metadata.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_build.h"
#include "kernel/kernel_build_info.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/op_api_util.h"
#include "transform/acl_ir/op_api_exec.h"
#include "transform/acl_ir/ge_adapter_info.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/backend/debug/data_dump/overflow_dumper.h"
#include "include/backend/debug/profiler/profiling.h"
#include "backend/common/pass/insert_type_transform_op.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "plugin/device/cpu/hal/device/kernel_select_cpu.h"
#include "utils/anf_utils.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr uint32_t kFirstItem = 0;
constexpr size_t kOpTypeNumber = 6;
constexpr size_t kAclnnOpSelect = 0;
constexpr size_t kAclOpSelect = 1;
constexpr size_t kHcclOpSelect = 2;
constexpr size_t kHostOpSelect = 3;
constexpr size_t kInternalOpSelect = 4;

std::string KernelSelectDebugString(const kernel::KernelBuildInfo *build_info,
                                    const std::vector<std::shared_ptr<kernel::KernelBuildInfo>> &kernel_info_list) {
  MS_EXCEPTION_IF_NULL(build_info);
  std::ostringstream output_buffer;
  output_buffer << std::endl;
  output_buffer << "need build info: " << std::endl;
  output_buffer << build_info->ToString() << std::endl;
  output_buffer << "candidate build info list: " << std::endl;
  for (const auto &info : kernel_info_list) {
    output_buffer << info->ToString() << std::endl;
  }
  return output_buffer.str();
}

using AclKernelFormatList = std::vector<std::pair<std::vector<string>, std::vector<string>>>;
AclKernelFormatList GetValidFormat(size_t input_num, size_t output_num) {
  std::vector<std::string> inputs_format(input_num, kOpFormat_DEFAULT);
  std::vector<std::string> outputs_format(output_num, kOpFormat_DEFAULT);
  return {std::make_pair(inputs_format, outputs_format)};
}

void SetWeightFormat(const AnfNodePtr &real_input_node, std::vector<string> output_format, const CNodePtr &kernel_node,
                     size_t input_index, bool force_fresh = false) {
  MS_EXCEPTION_IF_NULL(real_input_node);
  if (real_input_node->isa<CNode>()) {
    return;
  }

  if (AnfAlgo::OutputAddrExist(real_input_node, 0) &&
      AnfAlgo::GetOutputDeviceDataType(real_input_node, 0) != kTypeUnknown) {
    return;
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool disable_convert = real_input_node->isa<Parameter>() || real_input_node->isa<ValueNode>();
  // In PyNative mode, the weight data will be copied to the device in the first step,
  // and there will be no HostToDeviceCopy in the follow-up. If host format conversion is disabled,
  // the TransData operator will be executed in each subsequent step, resulting in poor performance.
  if (disable_convert && (context_ptr->get_param<bool>(MS_CTX_ENABLE_LOOP_SINK) ||
                          context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode)) {
    disable_convert = trans::kFormatWithTransFunc.find(output_format[0]) == trans::kFormatWithTransFunc.end();
  }
  // if not find in host convert format map means the host has not registered the convert function of this format
  if (output_format[0] != kOpFormat_DEFAULT && disable_convert) {
    output_format = {AnfAlgo::GetOutputFormat(real_input_node, 0)};
  }
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(builder);
  auto selected_kernel_info = AnfAlgo::GetSelectKernelBuildInfo(kernel_node);
  MS_EXCEPTION_IF_NULL(selected_kernel_info);
  // tensor id -> type id
  static std::unordered_map<std::string, TypeId> format_type;
  if (AnfAlgo::GetOutputDeviceDataType(real_input_node, 0) == kTypeUnknown || force_fresh) {
    if (IsValueNode<tensor::Tensor>(real_input_node)) {
      auto host_tensor_ptr = GetValueNode<tensor::TensorPtr>(real_input_node);
      MS_EXCEPTION_IF_NULL(host_tensor_ptr);
      std::vector<string> format = {host_tensor_ptr->device_info().host_format_};
      output_format = format[0] == kOpFormat_DEFAULT ? output_format : format;
      builder->SetOutputsFormat(output_format);
      auto iter = format_type.find(host_tensor_ptr->id());
      if (iter != format_type.end()) {
        std::vector<TypeId> output_type = {iter->second};
        builder->SetOutputsDeviceType(output_type);
        AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), real_input_node.get());
      } else {
        std::vector<TypeId> output_type = {selected_kernel_info->GetInputDeviceType(input_index)};
        builder->SetOutputsDeviceType(output_type);
        AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), real_input_node.get());
        format_type[host_tensor_ptr->id()] = output_type[0];
      }
    } else {
      builder->SetOutputsFormat(output_format);
      std::vector<TypeId> output_type = {common::AnfAlgo::GetOutputInferDataType(real_input_node, 0)};
      builder->SetOutputsDeviceType(output_type);
      AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), real_input_node.get());
    }
  }
}

bool RefreshCastAndParamWeightFormat(const AnfNodePtr &input_node, const string &format) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    return false;
  }
  if (!input_node->isa<CNode>()) {
    return false;
  }
  auto cast_node = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cast_node);
  if (common::AnfAlgo::GetCNodeName(cast_node) != prim::kPrimCast->name()) {
    return true;
  }
  if (AnfAlgo::IsFeatureMapOutput(cast_node)) {
    return true;
  }
  if (format == kOpFormat_FRACTAL_ZN_RNN || format == kOpFormat_ND_RNN_BIAS) {
    return true;
  }
  auto info_builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(input_node));
  MS_EXCEPTION_IF_NULL(info_builder);
  info_builder->SetInputsFormat({format, kOpFormat_DEFAULT});
  info_builder->SetOutputsFormat({format});
  AnfAlgo::SetSelectKernelBuildInfo(info_builder->Build(), cast_node.get());
  auto cast_input_node = common::AnfAlgo::VisitKernel(common::AnfAlgo::GetInputNode(cast_node, 0), 0);
  SetWeightFormat(cast_input_node.first, {format}, cast_node, 0, true);
  return true;
}

void RefreshInputParameter(const CNodePtr &kernel_node, const AnfNodePtr &input_kernel_node,
                           const std::string &input_format, size_t input_index) {
  auto input_with_index = common::AnfAlgo::VisitKernelWithReturnType(input_kernel_node, 0);
  MS_EXCEPTION_IF_NULL(input_with_index.first);
  auto real_input_node = input_with_index.first;
  MS_EXCEPTION_IF_NULL(real_input_node);
  if (RefreshCastAndParamWeightFormat(real_input_node, input_format)) {
    return;
  }
  if (real_input_node->isa<Parameter>() && !common::AnfAlgo::IsParameterWeight(real_input_node->cast<ParameterPtr>())) {
    return;
  }

  std::vector<std::string> output_format = {input_format};
  SetWeightFormat(real_input_node, output_format, kernel_node, input_index);
}

void SetTensorDeviceInfo(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto selected_kernel_info = AnfAlgo::GetSelectKernelBuildInfo(kernel_node);
  MS_EXCEPTION_IF_NULL(selected_kernel_info);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  size_t real_input_num = 0;
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    auto input_object_type = selected_kernel_info->GetInputKernelObjectType(input_index);
    if (input_object_type == kernel::KernelObjectType::TUPLE_UNFOLD) {
      std::vector<KernelWithIndex> kernels_with_index =
        common::AnfAlgo::GetRealPrevNodesOutput(kernel_node, input_index);
      for (auto &i : kernels_with_index) {
        RefreshInputParameter(kernel_node, i.first, selected_kernel_info->GetInputFormat(real_input_num),
                              real_input_num);
        ++real_input_num;
      }
    } else {
      auto input_kernel_node = common::AnfAlgo::GetInputNode(kernel_node, input_index);
      MS_EXCEPTION_IF_NULL(input_kernel_node);
      RefreshInputParameter(kernel_node, input_kernel_node, selected_kernel_info->GetInputFormat(real_input_num),
                            real_input_num);
      ++real_input_num;
    }
  }
}

std::string TryBackoffCpu(const KernelGraphPtr &graph, const CNodePtr &node,
                          const std::pair<std::string, ExceptionType> &failure_info) {
  // The Pynative_mode and task_sink does not support the backoff ability.
  if (!AnfAlgo::IsNodeSupportKernelSelectBackoff(node, graph)) {
    return failure_info.first;
  }

  if (graph->is_graph_run_mode()) {
    return failure_info.first +
           "\nThe operator is not supported in task sink mode. You can try to set JitLevel to O0 to execute this "
           "operator.";
  }

  MS_LOG(INFO) << "Try to use backoff CPU kernel, node:" << node->fullname_with_scope();
  // Erease  kAttrDynInputSizes before cpu kernel select, since cpu may expand it according to kAttrDynInputSizes
  // and make wrong choose, for example, the TupleToTensor op
  if (common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, node)) {
    common::AnfAlgo::EraseNodeAttr(kAttrDynInputSizes, node);
  }
  auto [cpu_msg, cpu_etype] = device::cpu::SetKernelInfoWithMsg(node);
  if (cpu_msg.empty()) {
    SetTensorDeviceInfo(node);
    AnfAlgo::SetKernelSelectBackoffInfo(node, failure_info);
  } else {
    std::ostringstream oss;
    oss << "Ascend operator selection failed info: " << failure_info.first
        << "\nCPU operator selection failed type: " << cpu_etype << ". failed info: " << cpu_msg;
    return oss.str();
  }
  return "";
}

std::pair<std::string, ExceptionType> CollectNotMatchMessage(
  const CNodePtr &node, const std::vector<kernel::KernelBuildInfoPtr> &kernel_info_list,
  const transform::ErrorAclType acl_err_type) {
  std::stringstream ss;
  ExceptionType etype;
  MS_EXCEPTION_IF_NULL(node);

  switch (acl_err_type) {
    case transform::kUnknownOp: {
      ss << "The current operator needs to be supplemented with an adapter, please check in `transform` directory."
         << " node is " << node->fullname_with_scope() << trace::DumpSourceLines(node) << std::endl;
      etype = NotSupportError;
      break;
    }
    case transform::kInValidType: {
      ss << "The supported input and output data types for the current operator are: node is "
         << node->fullname_with_scope() << trace::DumpSourceLines(node) << std::endl;
      std::string name = GetCNodeFuncName(node);
      const auto &info = transform::GeAdapterManager::GetInstance().GetInfo(name, true);
      const auto &input_supported_dtypes = info->input_supported_dtypes();
      for (auto [index, dtypes] : input_supported_dtypes) {
        ss << "InputDesc [" << index << "] support {";
        for (auto dtype : dtypes) {
          std::string dtype_str = transform::ge_dtype_str_map.find(dtype) == transform::ge_dtype_str_map.end()
                                    ? ""
                                    : transform::ge_dtype_str_map[dtype];
          ss << dtype_str << ",";
        }
        ss << "}" << std::endl;
      }
      const auto &output_supported_dtypes = info->output_supported_dtypes();
      for (auto [index, dtypes] : output_supported_dtypes) {
        ss << "OutputDesc [" << index << "] support {";
        for (auto dtype : dtypes) {
          std::string dtype_str = transform::ge_dtype_str_map.find(dtype) == transform::ge_dtype_str_map.end()
                                    ? ""
                                    : transform::ge_dtype_str_map[dtype];
          ss << dtype_str << ",";
        }
        ss << "}" << std::endl;
      }
      ss << std::endl;
      ss << "But current operator's input and output data types is:" << std::endl;
      size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
      size_t output_num = AnfUtils::GetOutputTensorNum(node);
      for (size_t i = 0; i < input_num; ++i) {
        ss << "InputDesc [" << i << "] is ";
        ss << TypeIdToString(common::AnfAlgo::GetPrevNodeOutputInferDataType(node, i)) << std::endl;
      }
      for (size_t i = 0; i < output_num; ++i) {
        ss << "OutputDesc [" << i << "] is ";
        ss << TypeIdToString(common::AnfAlgo::GetOutputInferDataType(node, i)) << std::endl;
      }
      etype = TypeError;
      break;
    }
    case transform::kSpecialOp: {
      ss << "The current operator is specified not to select ACL. Please contact the relevant engineer for help."
         << "node is " << node->fullname_with_scope() << trace::DumpSourceLines(node) << std::endl;
      etype = NotSupportError;
      break;
    }
    case transform::kInvalidBuildInfo: {
      auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
      MS_EXCEPTION_IF_NULL(kernel_info);
      auto build_info = kernel_info->select_kernel_build_info();
      MS_EXCEPTION_IF_NULL(build_info);
      ss << "Invalid Kernel Build Info! Kernel type: " << kernel::KernelTypeLabel(KernelType::HOST_KERNEL)
         << ", node: " << node->fullname_with_scope() << KernelSelectDebugString(build_info, kernel_info_list)
         << trace::DumpSourceLines(node);
      etype = TypeError;
      break;
    }
    default: {
      ss << "The current [" << node->fullname_with_scope() << "] operator did not match any operator prototype library."
         << trace::DumpSourceLines(node) << std::endl;
      etype = NotSupportError;
      break;
    }
  }
  return std::make_pair(ss.str(), etype);
}

bool SetMatchKernelInfo(const CNodePtr &kernel_node, const std::vector<kernel::KernelBuildInfoPtr> &kernel_info_list,
                        const KernelType &kernel_type, transform::ErrorAclType *err_type) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (kernel_info_list.empty()) {
    return false;
  }
  GenerateKernelBuildInfo(kernel_node, kernel_type);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(build_info);
  bool find_valid = std::any_of(kernel_info_list.begin(), kernel_info_list.end(),
                                [&build_info](const kernel::KernelBuildInfoPtr &item) {
                                  MS_EXCEPTION_IF_NULL(item);
                                  return item->IsSimilarityKernelBuildInfo(*build_info);
                                });
  if (!find_valid) {
    *err_type = transform::ErrorAclType::kInvalidBuildInfo;
  }
  return find_valid;
}

static std::once_flag kAclnnEnableListInit;
static std::unordered_set<std::string> kAclnnEnableList;
bool ReadAclnnEnableEnv(const std::string &op_name) {
  static auto enable_aclnn_env = common::GetEnv("MS_ENABLE_ACLNN");
  if (enable_aclnn_env == "1") {
    return kernel::IsRegisteredAclnnOp(op_name);
  }

  static auto read_config = !enable_aclnn_env.empty() && enable_aclnn_env != "0";
  if (read_config) {
    std::call_once(kAclnnEnableListInit, []() {
      std::ifstream in_file(enable_aclnn_env);
      if (!in_file.is_open()) {
        MS_LOG(WARNING) << "MS_ENABLE_ACLNN set path:" << enable_aclnn_env << " is invalid.";
        return;
      }
      std::string line;
      while (getline(in_file, line)) {
        kAclnnEnableList.insert(line);
      }
      in_file.close();
    });

    if (kAclnnEnableList.count(op_name) != 0) {
      return kernel::IsRegisteredAclnnOp(op_name);
    }
  }

  static std::set<std::string> kAscendcKernelList = {"AllFinite"};
  if (kAscendcKernelList.count(op_name) != 0) {
    return true;
  }

  return false;
}

bool IsDynamicTuple(const CNodePtr &kernel, bool is_acl, size_t output_num) {
  MS_EXCEPTION_IF_NULL(kernel);
  if (is_acl) {
    auto cnode_output_object_type =
      kernel::TypeIdToKernelObjectType(AnfAlgo::GetAbstractObjectType(kernel->abstract()));
    return output_num == 1 && cnode_output_object_type == kernel::KernelObjectType::TUPLE;
  }

  return kernel->abstract() != nullptr && kernel->abstract()->isa<abstract::AbstractTuple>() &&
         kernel->abstract()->cast<abstract::AbstractTuplePtr>()->dynamic_len();
}
}  // namespace

bool IsEmptyTupleInput(const CNodePtr &kernel, const size_t i, const TypeId cur_type_id) {
  auto input_node = common::AnfAlgo::GetPrevNodeOutput(kernel, i).first;
  if (input_node->isa<ValueNode>()) {
    auto value_node = input_node->cast<ValueNodePtr>();
    if (cur_type_id == kTypeUnknown && value_node->value() != nullptr && value_node->value()->isa<ValueTuple>() &&
        value_node->value()->cast<ValueTuplePtr>()->size() == 0) {
      MS_LOG(DEBUG) << "Set int64 type for empty value tuple node:" << value_node->DebugString();
      return true;
    }
  }
  return false;
}

TypeId GetInputDeviceType(const AnfNodePtr &kernel_node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  TypeId type = kTypeUnknown;
  auto [input_node, idx] = common::AnfAlgo::GetPrevNodeOutput(kernel_node, input_idx);
  MS_EXCEPTION_IF_NULL(input_node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(input_node->kernel_info());
  if (kernel_info != nullptr && kernel_info->select_kernel_build_info() != nullptr) {
    type = AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, input_idx);
    if (type == kTypeUnknown) {
      MS_LOG(DEBUG) << "This node kernel build info in valid, it may be parameter or value node: "
                    << kernel_node->DebugString() << ", idx: " << input_idx
                    << ", input node: " << input_node->DebugString();
      type = common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_idx);
      auto build_info = kernel_info->GetMutableSelectKernelBuildInfo();
      MS_EXCEPTION_IF_NULL(build_info);
      build_info->SetOutputDeviceType(type, idx);
    }
  } else {
    MS_LOG(DEBUG) << "Node no build info, node name: " << kernel_node->DebugString() << ", idx: " << input_idx
                  << ", input node: " << input_node->DebugString();
    type = common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_idx);
  }
  return type;
}

void GenerateKernelBuildInfo(const CNodePtr &kernel, const KernelType &kernel_type) {
  MS_EXCEPTION_IF_NULL(kernel);
  std::vector<std::string> input_formats;
  std::vector<std::string> output_formats;
  std::vector<std::string> input_reshape_types;
  std::vector<std::string> output_reshape_types;
  auto input_num = common::AnfAlgo::GetInputTensorNum(kernel);
  auto output_num = AnfUtils::GetOutputTensorNum(kernel);
  auto output_object_type = kernel::KernelObjectType::TENSOR;

  auto process_tuple_output = [&output_num, &output_object_type, &output_formats, &output_reshape_types](
                                const CNodePtr &kernel, bool is_acl, size_t cur_output_number) {
    auto cnode_output_object_type =
      kernel::TypeIdToKernelObjectType(AnfAlgo::GetAbstractObjectType(kernel->abstract()));
    if (IsDynamicTuple(kernel, is_acl, cur_output_number)) {
      MS_LOG(INFO) << "Node " << kernel->fullname_with_scope() << " output is real tuple";
      output_object_type = cnode_output_object_type;
      output_num = 1;
      auto output_format = output_formats[kFirstItem];
      auto output_reshape_type = output_reshape_types[kFirstItem];
      output_formats.clear();
      output_reshape_types.clear();
      output_formats.emplace_back(output_format);
      output_reshape_types.emplace_back(output_reshape_type);
    } else if (cnode_output_object_type == kernel::KernelObjectType::SCALAR) {
      output_object_type = cnode_output_object_type;
    }
  };
  if (kernel_type == ACL_KERNEL) {
    transform::AclHelper::GetValidKernelBuildInfo(kernel, &input_formats, &output_formats, &input_reshape_types,
                                                  &output_reshape_types);
    // NOTE: acl default output objecttype is tensor, here are 2 special case:
    // case 1: when cnode output is tuple, and ge ops prototype output num is 1, the real output objecttype is tuple;
    // case 2: when cnode output is scalar, the real output objecttype is scalar
    // others: output objecttype is tensor
    std::string name = GetCNodeFuncName(kernel);
    const auto &info = transform::GeAdapterManager::GetInstance().GetInfo(name, true);
    auto adapter_output_num = info->GetNumStaticOutputsOfMsOpProto();
    process_tuple_output(kernel, true, adapter_output_num);
  } else if (kernel_type == OPAPI_KERNEL || kernel_type == INTERNAL_KERNEL) {
    transform::OpApiUtil::GetValidKernelBuildInfo(kernel, &input_formats, &output_formats, &input_reshape_types,
                                                  &output_reshape_types);
    process_tuple_output(kernel, false, 1);
  } else {
    auto cand_format = GetValidFormat(input_num, output_num);
    if (cand_format.empty()) {
      MS_LOG_WITH_NODE(EXCEPTION, kernel) << "The kernel: " << kernel->fullname_with_scope()
                                          << " does not have a supported dynamic shape format on the Ascend platform.";
    }
    input_formats = cand_format.at(kFirstItem).first;
    output_formats = cand_format.at(kFirstItem).second;
    input_reshape_types.assign(input_num, "");
    output_reshape_types.assign(output_num, "");
  }

  std::vector<TypeId> input_types;
  input_types.reserve(input_num);
  std::vector<TypeId> output_types;
  output_types.reserve(output_num);
  std::vector<kernel::KernelObjectType> output_object_types;
  output_object_types.reserve(output_num);
  auto input_object_types = kernel::TypeIdToKernelObjectType(AnfAlgo::GetAllInputObjectType(kernel));

  for (size_t i = 0; i < input_num; i++) {
    auto cur_input_type = GetInputDeviceType(kernel, i);
    if (kernel_type != RT_KERNEL && IsEmptyTupleInput(kernel, i, cur_input_type)) {
      cur_input_type = TypeId::kNumberTypeInt64;
    }
    (void)input_types.emplace_back(cur_input_type);
  }
  for (size_t i = 0; i < output_num; i++) {
    (void)output_types.emplace_back(common::AnfAlgo::GetOutputInferDataType(kernel, i));
    // no tuple in PyNative dynamic shape
    (void)output_object_types.emplace_back(output_object_type);
  }
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetKernelType(kernel_type);
  builder->SetInputsFormat(input_formats);
  builder->SetInputsDeviceType(input_types);
  builder->SetInputsKernelObjectType(input_object_types);
  builder->SetOutputsFormat(output_formats);
  builder->SetOutputsDeviceType(output_types);
  builder->SetOutputsKernelObjectType(output_object_types);
  builder->SetInputsReshapeType(input_reshape_types);
  builder->SetOutputsReshapeType(output_reshape_types);
  if (input_formats.size() != input_types.size() || input_formats.size() != input_object_types.size()) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel) << "The input buildInfo size kernel: " << kernel->fullname_with_scope()
                                        << "is not equal, input_formats size: " << input_formats.size()
                                        << ", input_types size: " << input_types.size()
                                        << ", input_object_types size: " << input_object_types.size();
  }
  if (output_formats.size() != output_types.size() || output_formats.size() != output_object_types.size()) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel) << "The output buildInfo size kernel: " << kernel->fullname_with_scope()
                                        << "is not equal, output_formats size: " << output_formats.size()
                                        << ", output_types size: " << output_types.size()
                                        << ", output_object_types size: " << output_object_types.size();
  }
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), kernel.get());
}

void GenerateKernelPacketBuildInfo(const CNodePtr &node) {
  MS_LOG(DEBUG) << "Begin set kernel info for kernelpacket "
                << common::AnfAlgo::GetNodeAttr<std::string>(node, kAttrKernelPacketNode) << ". The node is "
                << node->fullname_with_scope() << ", " << node->DebugString();
  auto sub_fg = common::AnfAlgo::GetNodeAttr<FuncGraphPtr>(node, kAttrFuncGraph);
  MS_EXCEPTION_IF_NULL(sub_fg);
  std::vector<std::string> input_formats;
  std::vector<TypeId> input_types;
  auto input_object_types = kernel::TypeIdToKernelObjectType(AnfAlgo::GetAllInputObjectType(node));
  auto input_num = common::AnfAlgo::GetInputTensorNum(node);
  input_formats.reserve(input_num);
  input_types.reserve(input_num);
  if (sub_fg->parameters().size() < input_num) {
    MS_LOG(INTERNAL_EXCEPTION) << "For kernel " << node->DebugString() << ", the input tensor num (" << input_num
                               << ") is greater than the parameter num of inner graph " << sub_fg->ToString();
  }
  for (size_t i = 0; i < input_num; i++) {
    auto cur_input_type = GetInputDeviceType(node, i);
    if (IsEmptyTupleInput(node, i, cur_input_type)) {
      cur_input_type = TypeId::kNumberTypeInt64;
    }
    (void)input_types.emplace_back(cur_input_type);
    (void)input_formats.emplace_back(AnfAlgo::GetPrevNodeOutputFormat(node, i));
    // set parameters' output info.
    auto param_infobuilder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    MS_EXCEPTION_IF_NULL(param_infobuilder);
    param_infobuilder->SetOutputsDeviceType({input_types.back()});
    param_infobuilder->SetOutputsFormat({input_formats.back()});
    param_infobuilder->SetOutputsKernelObjectType({input_object_types[i]});
    sub_fg->parameters()[i]->set_kernel_info(std::make_shared<device::KernelInfo>());
    AnfAlgo::SetSelectKernelBuildInfo(param_infobuilder->Build(), sub_fg->parameters()[i].get());
  }
  auto inner_build_info = AnfAlgo::GetSelectKernelBuildInfo(sub_fg->output());
  MS_EXCEPTION_IF_NULL(inner_build_info);
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetKernelType(KernelType::AKG_KERNEL);
  builder->SetInputsFormat(input_formats);
  builder->SetInputsDeviceType(input_types);
  builder->SetInputsKernelObjectType(input_object_types);
  builder->SetOutputsFormat(inner_build_info->GetAllOutputFormats());
  builder->SetOutputsDeviceType(inner_build_info->GetAllOutputDeviceTypes());
  builder->SetOutputsKernelObjectType(inner_build_info->GetAllOutputKernelObjectTypes());
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), node.get());
  MS_LOG(DEBUG) << "Finish set kernel info for kernelpacket node " << node->fullname_with_scope();
}

void HandleKernelSelectFailure(const KernelGraphPtr &graph, const CNodePtr &node,
                               const std::pair<std::string, ExceptionType> &failure_info) {
  auto msg = TryBackoffCpu(graph, node, failure_info);
  if (!msg.empty()) {
    MS_EXCEPTION(failure_info.second) << msg;
  }
}

std::vector<std::string> SplitString(const std::string &str, char delim) {
  std::stringstream ss(str);
  std::string item;
  std::vector<std::string> elems;
  while (std::getline(ss, item, delim)) {
    if (!item.empty()) {
      elems.emplace_back(item);
    }
  }
  return elems;
}

std::tuple<bool, std::string, ExceptionType> SelectKernelInfoWithMsg(const KernelGraphPtr &graph,
                                                                     const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  static std::vector<std::set<std::string>> op_selected_type(kOpTypeNumber);
  transform::ErrorAclType acl_err_type = transform::ErrorAclType::kNormalOp;
  std::tuple<bool, std::string, ExceptionType> result = std::make_tuple(true, "", NoExceptionType);
  std::string op_name = common::AnfAlgo::GetCNodeName(node);

  if (IsEnableInternalNode(node)) {
    GenerateKernelBuildInfo(node, KernelType::INTERNAL_KERNEL);
    if (op_selected_type[kInternalOpSelect].count(op_name) == 0) {
      (void)op_selected_type[kInternalOpSelect].insert(op_name);
      MS_VLOG(VL_ASCEND_KERNEL_SELECT) << op_name << " select internal kernel.";
      MS_LOG(INFO) << op_name << " select internal kernel.";
    }
    return result;
  }

  // The shape op use the cpu kernel priorly.
  static const std::set<std::string> select_host_priorly = {kShapeOpName, kTensorShapeOpName};
  if ((select_host_priorly.count(op_name) != 0) && AnfAlgo::IsNodeSupportKernelSelectBackoff(node, graph)) {
    MS_VLOG(VL_ASCEND_KERNEL_SELECT) << op_name << " select host kernel priorly.";
    return {false, op_name + " select host kernel priorly.", NotSupportError};
  }

  if (IsEnableAclnn(graph, node)) {
    GenerateKernelBuildInfo(node, KernelType::OPAPI_KERNEL);
    if (op_selected_type[kAclnnOpSelect].count(op_name) == 0) {
      (void)op_selected_type[kAclnnOpSelect].insert(op_name);
      MS_VLOG(VL_ASCEND_KERNEL_SELECT) << op_name << " select aclnn kernel.";
      MS_LOG(INFO) << op_name << " select aclnn kernel.";
    }
    return result;
  }

  // for backend inline
  if (IsOneOfPrimitiveCNode(node, {prim::kPrimCallInline, prim::kPrimSwitch, prim::kPrimPartialInline,
                                   prim::kPrimConditionSwitch, prim::kPrimConditionGather, prim::kPrimReshapeExt})) {
    GenerateKernelBuildInfo(node, KernelType::RT_KERNEL);
    return result;
  }

  auto kernel_type = transform::AclHelper::GetKernelInfoFromGe(node, &acl_err_type);
  if (kernel_type == KernelType::ACL_KERNEL) {
    GenerateKernelBuildInfo(node, kernel_type);
    if (op_selected_type[kAclOpSelect].count(op_name) == 0) {
      (void)op_selected_type[kAclOpSelect].insert(op_name);
      MS_VLOG(VL_ASCEND_KERNEL_SELECT) << op_name << " select aclop kernel.";
      MS_LOG(INFO) << op_name << " select aclop kernel.";
    }
    return result;
  }

  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list{};
  if (kernel_type == KernelType::HCCL_KERNEL) {
    kernel::HcclMetadataInfo(node, &kernel_info_list);
    GenerateKernelBuildInfo(node, kernel_type);
    if (op_selected_type[kHcclOpSelect].count(op_name) == 0) {
      (void)op_selected_type[kHcclOpSelect].insert(op_name);
      MS_VLOG(VL_ASCEND_KERNEL_SELECT) << op_name << " select hccl kernel.";
      MS_LOG(INFO) << op_name << " select hccl kernel.";
    }
    return result;
  }

  if (common::AnfAlgo::HasNodeAttr(kAttrKernelPacketNode, node)) {
    GenerateKernelPacketBuildInfo(node);
    return result;
  }

  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> host_kernel_info_list{};
  kernel::HostMetadataInfo(node, &host_kernel_info_list);
  auto match_res = SetMatchKernelInfo(node, host_kernel_info_list, KernelType::HOST_KERNEL, &acl_err_type);
  if (match_res) {
    if (op_selected_type[kHostOpSelect].count(op_name) == 0) {
      (void)op_selected_type[kHostOpSelect].insert(op_name);
      MS_VLOG(VL_ASCEND_KERNEL_SELECT) << op_name << " select host kernel.";
      MS_LOG(INFO) << op_name << " select host kernel.";
    }
    return result;
  }

  auto [msg, etype] = CollectNotMatchMessage(node, host_kernel_info_list, acl_err_type);
  return {false, msg, etype};
}

bool IsEnableAclnn(const KernelGraphPtr &kernel_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(node);

  if (IsPrimitiveCNode(node, prim::kPrimCustom)) {
    auto primitive = GetCNodePrimitive(node);
    auto op_type = GetValue<std::string>(primitive->GetAttr("reg_op_name"));
    auto op_api_func = transform::GetOpApiFunc(op_type.c_str());
    if (op_api_func != nullptr) {
      MS_LOG(INFO) << "Kernel of custom op " << node->fullname_with_scope() << "is selected AclNN.";
      return true;
    }
  }

  if (kernel_graph->is_from_single_op()) {
    return false;
  }

  static bool special_format = GetFormatMode() == "0";
  if (special_format && !kernel_graph->is_dynamic_shape()) {
    return false;
  }

  static std::unordered_map<std::string, bool> kIsEnableAclnnMap;
  std::string op_name = common::AnfAlgo::GetCNodeName(node);
  auto iter = kIsEnableAclnnMap.find(op_name);
  if (iter != kIsEnableAclnnMap.end()) {
    return iter->second;
  }
  if (kernel::IsEnabledAclnnDispatch(op_name)) {
    if (!kernel::IsRegisteredAclnnOp(op_name)) {
      MS_LOG(WARNING) << "Kernel " << node->fullname_with_scope()
                      << " is enabled dispatch in yaml, but not registered an aclnn kernelmod.";
      return false;
    }
    kIsEnableAclnnMap.insert({op_name, true});
    return true;
  }

  bool ret = ReadAclnnEnableEnv(op_name);
  kIsEnableAclnnMap.insert({op_name, ret});
  return ret;
}

bool IsEnableInternalNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto enable_internal = false;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->IsEnableInferBoost()) {
    enable_internal = true;
  }

  std::string op_name = common::AnfAlgo::GetCNodeName(node);
  if (op_name == "ReshapeExt") {
    enable_internal = true;
  }

  std::string disable_name_list = common::GetEnv("MS_DISABLE_INTERNAL_KERNELS_LIST");
  std::vector<std::string> op_name_vec = SplitString(disable_name_list, ',');
  if (std::any_of(op_name_vec.begin(), op_name_vec.end(),
                  [&op_name](const std::string &name) { return name == op_name; })) {
    enable_internal = false;
  }

  return enable_internal && kernel::IsRegisteredInternalKernel(node);
}

void SetKernelInfoBeforeCreateKernel(const std::vector<CNodePtr> &nodes) {
  for (const auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    auto build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
    if (build_info != nullptr && build_info->valid()) {
      continue;
    }

    const auto &kernel_graph = AnfAlgo::FetchKernelGraph(node.get());
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto [select_res, msg, etype] = SelectKernelInfoWithMsg(kernel_graph, node);
    if (!select_res) {
      MS_LOG(INFO) << "node is " << node->fullname_with_scope() << " should backoff";
      MS_VLOG(VL_ASCEND_KERNEL_SELECT) << "node is " << node->fullname_with_scope() << " should backoff";
      std::pair<std::string, ExceptionType> failure_info = std::make_pair(msg, etype);
      HandleKernelSelectFailure(kernel_graph, node, failure_info);
    }
  }
}

class AscendGraphKernelInfo : public GraphKernelInfo {
 public:
  AscendGraphKernelInfo() = default;
  virtual ~AscendGraphKernelInfo() = default;
  void SetKernelInfo(const CNodePtr &kernel_node, KernelType) override {
    MS_EXCEPTION_IF_NULL(kernel_node);
    const auto &kernel_graph = AnfAlgo::FetchKernelGraph(kernel_node.get());
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto [select_res, msg, etype] = SelectKernelInfoWithMsg(kernel_graph, kernel_node);
    if (select_res) {
      return;
    }
    MS_LOG(INFO) << "node is " << kernel_node->fullname_with_scope() << " should backoff";
    std::pair<std::string, ExceptionType> failure_info = std::make_pair(msg, etype);
    HandleKernelSelectFailure(kernel_graph, kernel_node, failure_info);
  }
};
REG_GRAPH_KERNEL_INFO(kAscendDevice, AscendGraphKernelInfo);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
