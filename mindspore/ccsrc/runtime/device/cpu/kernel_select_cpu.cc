/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "runtime/device/cpu/kernel_select_cpu.h"
#include <string>
#include <memory>
#include <algorithm>
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "backend/kernel_compiler/kernel_build_info.h"
#include "backend/kernel_compiler/oplib/opinfo.h"
#include "backend/kernel_compiler/oplib/oplib.h"
#include "backend/kernel_compiler/cpu/pyfunc/py_func_cpu_kernel.h"
#include "backend/kernel_compiler/cpu/custom/custom_aot_cpu_kernel.h"
#include "backend/kernel_compiler/cpu/custom/custom_julia_cpu_kernel.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace device {
namespace cpu {
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;
using mindspore::kernel::KernelBuildInfo;
namespace {
constexpr auto kParamDynamic = "dynamic";

bool IsInputNotCNode(const CNodePtr &kernel_node, size_t input_index) {
  auto input_node = AnfAlgo::VisitKernel(kernel_node->input(input_index + 1), 0).first;
  MS_EXCEPTION_IF_NULL(input_node);
  if (input_node->isa<Parameter>() || input_node->isa<ValueNode>()) {
    return true;
  }
  return false;
}

void GetOutputDtypes(const CNodePtr &kernel_node, std::vector<TypeId> *output_types) {
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    TypeId dtype = kTypeUnknown;
    dtype = AnfAlgo::GetOutputInferDataType(kernel_node, output_index);
    output_types->emplace_back(dtype);
  }
}

void GetOutputFormat(const CNodePtr &kernel_node, std::vector<std::string> *output_formats) {
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    output_formats->emplace_back(kOpFormat_DEFAULT);
  }
}

void GetInputDtypes(const CNodePtr &kernel_node, std::vector<TypeId> *input_types,
                    std::vector<size_t> *input_no_cnode_indexes) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    TypeId dtype = kTypeUnknown;
    if (IsInputNotCNode(kernel_node, input_index)) {
      input_no_cnode_indexes->emplace_back(input_index);
      dtype = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_index);
    } else {
      dtype = AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, input_index);
    }
    input_types->emplace_back(dtype);
  }
}

void GetInputFormat(const CNodePtr &kernel_node, std::vector<std::string> *input_formats) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    input_formats->emplace_back(kOpFormat_DEFAULT);
  }
}

void GetOutputFormatsAndDtypes(const CNodePtr &kernel_node, const KernelAttr &kernel_attr,
                               std::vector<std::string> *output_formats, std::vector<TypeId> *output_types) {
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    output_formats->emplace_back(kernel_attr.GetOutputAttr(output_index).second);
    auto dtype = kernel_attr.GetOutputAttr(output_index).first;
    output_types->emplace_back(dtype);
  }
}

bool InputDtypeMatch(TypeId InputAttr, TypeId input_type, bool strict) {
  if (InputAttr == input_type) {
    return true;
  }
  if (!strict && InputAttr == kNumberTypeInt32 && (input_type == kNumberTypeInt16 || input_type == kNumberTypeInt64)) {
    return true;
  }
  if (!strict && InputAttr == kNumberTypeFloat32 &&
      (input_type == kNumberTypeFloat16 || input_type == kNumberTypeFloat64)) {
    return true;
  }
  return false;
}

int GetOutputDtypeMatchedNum(const KernelAttr &kernel_attr, const std::vector<TypeId> &output_types) {
  if (kernel_attr.GetOutputSize() != output_types.size()) {
    MS_LOG(DEBUG) << "required output num:" << kernel_attr.GetInputSize()
                  << ", actual output num:" << output_types.size();
    return 0;
  }
  int data_type_matched_num = 0;
  auto output_num = output_types.size();
  for (size_t i = 0; i < output_num; ++i) {
    if (kernel_attr.GetOutputAttr(i).first != output_types[i]) {
      MS_LOG(DEBUG) << "required dtype:" << kernel_attr.GetOutputAttr(i).first
                    << ", actual output dtype:" << output_types[i];
    } else {
      data_type_matched_num++;
    }
  }
  return data_type_matched_num;
}

int GetInputDtypeFormatMatchedNum(const KernelAttr &kernel_attr, const std::vector<TypeId> &input_types,
                                  const std::vector<size_t> &input_not_cnode_indexes, bool strict) {
  if (kernel_attr.GetInputSize() != input_types.size()) {
    MS_LOG(DEBUG) << "required input num:" << kernel_attr.GetInputSize() << ", actual input num:" << input_types.size();
    return 0;
  }
  int data_type_matched_num = 0;
  auto input_num = input_types.size();
  for (size_t i = 0; i < input_num; ++i) {
    if (!InputDtypeMatch(kernel_attr.GetInputAttr(i).first, input_types[i], strict)) {
      MS_LOG(DEBUG) << "required dtype:" << kernel_attr.GetInputAttr(i).first
                    << ", actual input dtype:" << input_types[i];
    } else {
      data_type_matched_num++;
    }
  }
  return data_type_matched_num;
}

void ExpandKernelAttr(const CNodePtr &kernel_node, KernelAttr *kernel_attr) {
  MS_EXCEPTION_IF_NULL(kernel_attr);
  size_t attr_num = kernel_attr->GetInputSize();
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (attr_num == 0) {
    MS_LOG(EXCEPTION) << "Input size is empty";
    return;  // To pass the CI Check_Cppcheck
  }
  // Only support one dynamic input like Concat or
  // many dynamic input but each input has same number like DynamicStitch
  std::string format = kOpFormat_DEFAULT;
  std::vector<DataType> attr_list;
  size_t each_attr_input_num = input_num / attr_num;
  for (size_t i = 0; i < attr_num; ++i) {
    TypeId input_dtype = kernel_attr->GetInputAttr(i).first;
    for (size_t j = 0; j < each_attr_input_num; ++j) {
      (void)attr_list.emplace_back(input_dtype, format);
    }
  }
  kernel_attr->SetInputAttrList(attr_list);

  TypeId output_dtype = kernel_attr->GetOutputAttr(0).first;
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t i = 1; i < output_num; ++i) {
    kernel_attr->AddOutputAttr(output_dtype);
  }
}

void SetKernelBuildInfo(const std::vector<std::string> &input_formats, const std::vector<TypeId> &input_types,
                        const std::vector<std::string> &output_formats, const std::vector<TypeId> &output_types,
                        AnfNode *kernel_node) {
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetInputsFormat(input_formats);
  builder->SetInputsDeviceType(input_types);
  builder->SetOutputsFormat(output_formats);
  builder->SetOutputsDeviceType(output_types);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), kernel_node);
}

void KernelNotSupportException(const AnfNodePtr &kernel_node, const std::vector<TypeId> &input_types,
                               const std::vector<TypeId> &infer_output_types) {
  std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
  std::stringstream operator_info;
  operator_info << "Operator[" << kernel_name << "] ";
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num > 0) {
    operator_info << " input(";
    for (size_t i = 0; i < input_num; ++i) {
      operator_info << TypeIdLabel(input_types[i]);
      if (i != input_num - 1) {
        operator_info << ",";
      }
    }
    operator_info << ") ";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num > 0) {
    operator_info << "output(";
    for (size_t i = 0; i < output_num; ++i) {
      operator_info << TypeIdLabel(infer_output_types[i]);
      if (i != output_num - 1) {
        operator_info << ",";
      }
    }
    operator_info << ") ";
  }
  operator_info
    << "is not supported. This error means the current input type is not supported, please refer to the MindSpore "
       "doc for supported types.\n";
  MS_EXCEPTION(TypeError) << operator_info.str() << trace::DumpSourceLines(kernel_node);
}

void UpdateDynamicKernelBuildInfoAndAttrs(const CNodePtr &kernel_node) {
  const std::string &op_name = AnfAlgo::GetCNodeName(kernel_node);
  MS_LOG(INFO) << "Operator name: " << op_name;
  // Set kernel build info
  std::vector<TypeId> input_types;
  std::vector<size_t> input_not_cnode_indexes;
  GetInputDtypes(kernel_node, &input_types, &input_not_cnode_indexes);
  std::vector<TypeId> output_types;
  GetOutputDtypes(kernel_node, &output_types);
  std::vector<std::string> input_formats;
  GetInputFormat(kernel_node, &input_formats);
  std::vector<std::string> output_formats;
  GetOutputFormat(kernel_node, &output_formats);
  SetKernelBuildInfo(input_formats, input_types, output_formats, output_types, kernel_node.get());

  // Set kernel attrs
  KernelAttr attr;
  for (size_t i = 0; i < input_types.size(); i++) {
    attr.AddInputAttr(input_types[i]);
  }
  for (size_t j = 0; j < output_types.size(); j++) {
    attr.AddInputAttr(output_types[j]);
  }
  std::vector<KernelAttr> kernel_attrs =
    kernel::NativeCpuKernelModFactory::GetInstance().GetSupportedKernelAttrList(AnfAlgo::GetCNodeName(kernel_node));
  kernel_attrs.emplace_back(attr);
  kernel::NativeCpuKernelModFactory::GetInstance().UpdateKernelAttrs(op_name, kernel_attrs);
  return;
}

void AddKernelAttr(const CNodePtr &kernel_node, const KernelAttr &kernel_attr) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<KernelAttr> kernel_attrs =
    kernel::NativeCpuKernelModFactory::GetInstance().GetSupportedKernelAttrList(AnfAlgo::GetCNodeName(kernel_node));
  if (kernel_attrs.size() == 1 && kernel_attrs[0].GetInputSize() == 0 && kernel_attrs[0].GetOutputSize() == 0) {
    kernel_attrs.clear();
  }
  kernel_attrs.emplace_back(kernel_attr);
  kernel::NativeCpuKernelModFactory::GetInstance().UpdateKernelAttrs(AnfAlgo::GetCNodeName(kernel_node), kernel_attrs);
}

void UpdateCustomKernelBuildInfoAndAttrs(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  // Custom op's kernel type can only be CPU_KERNEL on CPU
  builder->SetKernelType(KernelType::CPU_KERNEL);
  builder->SetProcessor(kernel::Processor::CPU);
  // Set inputs info
  std::vector<TypeId> input_types;
  std::vector<size_t> input_not_cnode_indexes;
  GetInputDtypes(kernel_node, &input_types, &input_not_cnode_indexes);
  std::vector<std::string> input_formats;
  GetInputFormat(kernel_node, &input_formats);
  builder->SetInputsDeviceType(input_types);
  builder->SetInputsFormat(input_formats);
  // Set inputs info
  std::vector<TypeId> output_types;
  GetOutputDtypes(kernel_node, &output_types);
  std::vector<std::string> output_formats;
  GetOutputFormat(kernel_node, &output_formats);
  builder->SetOutputsDeviceType(output_types);
  builder->SetOutputsFormat(output_formats);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), kernel_node.get());
  // Update kernel attrs
  KernelAttr attr;
  if (input_types.size() != input_formats.size()) {
    MS_LOG(EXCEPTION) << "input types size " << input_types.size() << " is not equal to input formats size "
                      << input_formats.size() << " in op: " << AnfAlgo::GetCNodeName(kernel_node);
  }
  for (size_t i = 0; i < input_types.size(); ++i) {
    attr.AddInputAttr(input_types[i], input_formats[i]);
  }
  if (output_types.size() != output_formats.size()) {
    MS_LOG(EXCEPTION) << "output types size " << output_types.size() << " is not equal to output formats size "
                      << output_formats.size() << " in op: " << AnfAlgo::GetCNodeName(kernel_node);
  }
  for (size_t i = 0; i < output_types.size(); ++i) {
    attr.AddOutputAttr(output_types[i], output_formats[i]);
  }
  AddKernelAttr(kernel_node, attr);
}

KernelAttr FillNoneInKernelAttr(const CNodePtr &kernel_node, const std::vector<TypeId> &input_types,
                                const std::vector<TypeId> &output_types, const KernelAttr &kernel_attr) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  // Only process Custom op
  if (!IsPrimitiveCNode(kernel_node, prim::kPrimCustom)) {
    return kernel_attr;
  }
  auto input_num = input_types.size();
  auto output_num = output_types.size();
  if (kernel_attr.GetInputSize() != input_types.size() || kernel_attr.GetOutputSize() != output_types.size()) {
    MS_LOG(DEBUG) << "required input num:" << kernel_attr.GetInputSize() << ", actual input num:" << input_num;
    MS_LOG(DEBUG) << "required input num:" << kernel_attr.GetOutputSize() << ", actual input num:" << output_num;
    return kernel_attr;
  }
  KernelAttr result;
  bool has_none = false;
  // Fill inputs info.
  for (size_t i = 0; i < input_num; ++i) {
    auto type_format = kernel_attr.GetInputAttr(i);
    if (type_format.first == TypeId::kMetaTypeNone) {
      has_none = true;
      type_format.first = input_types[i];
    }
    if (type_format.second.empty()) {
      has_none = true;
      type_format.second = kOpFormat_DEFAULT;
    }
    result.AddInputAttr(type_format.first, type_format.second);
  }
  // Fill outputs info.
  for (size_t i = 0; i < output_num; ++i) {
    auto type_format = kernel_attr.GetOutputAttr(i);
    if (type_format.first == TypeId::kMetaTypeNone) {
      has_none = true;
      type_format.first = output_types[i];
    }
    if (type_format.second.empty()) {
      has_none = true;
      type_format.second = kOpFormat_DEFAULT;
    }
    result.AddOutputAttr(type_format.first, type_format.second);
  }
  if (has_none) {
    AddKernelAttr(kernel_node, result);
  }
  return result;
}
}  // namespace

bool IsDynamicParamKernel(const std::string &op_name) {
  const auto &op_info = kernel::OpLib::FindOp(op_name, kernel::OpImplyType::kCPU);
  if (op_info == nullptr) {
    return false;
  }

  const auto &input_io_info = op_info->inputs_ptr();
  if (input_io_info.size() != 1 || input_io_info[0]->param_type() != kParamDynamic) {
    return false;
  }

  const auto &output_io_info = op_info->outputs_ptr();
  if (output_io_info.size() != 1 || output_io_info[0]->param_type() != kParamDynamic) {
    return false;
  }

  return true;
}

bool SelectKernel(const CNodePtr &kernel_node, KernelAttr *selected_kernel_attr,
                  const std::vector<KernelAttr> &kernel_attrs, const std::vector<TypeId> &input_types,
                  const std::vector<size_t> &input_not_cnode_indexes, const std::vector<TypeId> &output_types,
                  std::pair<bool, bool> *matched, bool strict) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(selected_kernel_attr);
  MS_EXCEPTION_IF_NULL(matched);
  MS_LOG(DEBUG) << "Select kernel for op: " << AnfAlgo::GetCNodeName(kernel_node);
  for (auto kernel_attr : kernel_attrs) {
    if (kernel_attr.GetAllSame()) {
      ExpandKernelAttr(kernel_node, &kernel_attr);
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (kernel_attr.GetOutputSize() != output_num) {
      MS_LOG(DEBUG) << "Output num is not equal!";
      continue;
    }
    auto new_kernel_attr = FillNoneInKernelAttr(kernel_node, input_types, output_types, kernel_attr);
    int input_dtype_matched_num =
      GetInputDtypeFormatMatchedNum(new_kernel_attr, input_types, input_not_cnode_indexes, strict);
    int output_dtype_matched_num = GetOutputDtypeMatchedNum(new_kernel_attr, output_types);
    // All formats and data types matched
    if (input_dtype_matched_num == SizeToInt(input_types.size())) {
      *selected_kernel_attr = new_kernel_attr;
      matched->first = true;
      if (output_dtype_matched_num == SizeToInt(output_types.size())) {
        matched->second = true;
        return true;
      }
    }
  }
  return false;
}

void SetKernelInfo(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  const std::string &op_name = AnfAlgo::GetCNodeName(kernel_node);
  if (IsPrimitiveCNode(kernel_node, prim::kPrimCustom)) {
    if (!kernel::NativeCpuKernelModFactory::GetInstance().SearchRegisteredOp(op_name)) {
      auto tp = AnfAlgo::GetNodeAttr<std::string>(kernel_node, kAttrFuncType);
      if (tp == kCustomTypePyfunc) {
        kernel::NativeCpuKernelRegistrar(op_name, KernelAttr(),
                                         []() { return std::make_shared<kernel::PyFuncCpuKernelMod>(); });
      } else if (tp == kCustomTypeAOT) {
        kernel::NativeCpuKernelRegistrar(op_name, KernelAttr(),
                                         []() { return std::make_shared<kernel::CustomAOTCpuKernelMod>(); });
      } else if (tp == kCustomTypeJULIA) {
        kernel::NativeCpuKernelRegistrar(op_name, KernelAttr(),
                                         []() { return std::make_shared<kernel::CustomJULIACpuKernelMod>(); });
      } else {
        MS_LOG(EXCEPTION)
          << "Unsupported func type for Custom operator on CPU, it should be 'pyfunc' or 'aot' or 'julia', "
          << "but got [" << tp << "] for Custom operator [" << op_name << "]";
      }
    }
    // If Custom op has not set reg info, then infer info from inputs
    if (mindspore::kernel::OpLib::FindOp(op_name, kernel::OpImplyType::kCPU) == nullptr) {
      MS_LOG(WARNING) << "Not find operator information for Custom operator[" << op_name << "]. "
                      << "Infer operator information from inputs. For more details, "
                      << "please refer to 'mindspore.ops.Custom' at https://www.mindspore.cn.";
      return UpdateCustomKernelBuildInfoAndAttrs(kernel_node);
    }
  } else if (IsDynamicParamKernel(op_name)) {
    // Select for dynamic kernel(both the number and data type are undetermined).
    return UpdateDynamicKernelBuildInfoAndAttrs(kernel_node);
  }

  std::vector<std::string> input_formats;
  std::vector<TypeId> input_types;
  std::vector<size_t> input_not_cnode_indexes;
  std::vector<std::string> selected_output_formats;
  std::vector<TypeId> output_types;
  std::vector<TypeId> selected_output_types;
  MS_LOG(INFO) << "SetKernelInfo, CNode Name: " << op_name;
  auto kernel_attrs =
    kernel::NativeCpuKernelModFactory::GetInstance().GetSupportedKernelAttrList(AnfAlgo::GetCNodeName(kernel_node));
  if (kernel_attrs.empty() || (kernel_attrs[0].GetInputSize() == 0 && kernel_attrs[0].GetOutputSize() == 0)) {
    MS_LOG(DEBUG) << "Operator[" << op_name << "] will get ops attr info.";
    auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, kernel::OpImplyType::kCPU);
    if (op_info_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Not find op[" << op_name << "] in cpu. For more details, "
                        << "please refer to the list of supported cpu operations at https://www.mindspore.cn.";
    }
    kernel_attrs.clear();
    kernel::NativeCpuKernelModFactory::GetInstance().SetKernelAttrs(op_info_ptr, &kernel_attrs);
    kernel::NativeCpuKernelModFactory::GetInstance().UpdateKernelAttrs(op_name, kernel_attrs);
  }
  GetInputDtypes(kernel_node, &input_types, &input_not_cnode_indexes);
  GetOutputDtypes(kernel_node, &output_types);
  KernelAttr selected_kernel_attr;
  std::pair<bool, bool> matched = std::make_pair(false, false);
  if (!SelectKernel(kernel_node, &selected_kernel_attr, kernel_attrs, input_types, input_not_cnode_indexes,
                    output_types, &matched, true)) {
    if (op_name == "Cast") {
      KernelNotSupportException(kernel_node, input_types, output_types);
    }
    matched = std::make_pair(false, false);
    (void)SelectKernel(kernel_node, &selected_kernel_attr, kernel_attrs, input_types, input_not_cnode_indexes,
                       output_types, &matched, false);
    if (!matched.first) {
      KernelNotSupportException(kernel_node, input_types, output_types);
    }
  }

  if (matched.first || input_types.size() == input_not_cnode_indexes.size()) {
    MS_LOG(INFO) << "Input format and dtype is matched";
    GetOutputFormatsAndDtypes(kernel_node, selected_kernel_attr, &selected_output_formats, &selected_output_types);
    for (size_t index = 0; index < selected_kernel_attr.GetInputSize(); index++) {
      input_types[index] = selected_kernel_attr.GetInputAttr(index).first;
      input_formats.emplace_back(selected_kernel_attr.GetInputAttr(index).second);
    }
  }
  SetKernelBuildInfo(input_formats, input_types, selected_output_formats, selected_output_types, kernel_node.get());
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
