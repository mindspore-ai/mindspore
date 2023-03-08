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

#include "src/extendrt/utils/kernel_build_utils.h"
#include <string>
#include <memory>
#include <algorithm>
#include "kernel/common_utils.h"
#include "plugin/factory/ms_factory.h"
#include "kernel/kernel_build_info.h"
#include "kernel/oplib/opinfo.h"
#include "kernel/oplib/oplib.h"
#include "utils/trace_base.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_info.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace infer {
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;
using mindspore::kernel::KernelBuildInfo;
namespace {
constexpr auto kParamDynamic = "dynamic";
constexpr auto kInputNum = 3;
constexpr auto kNameCustomAscend = "CustomAscend";
constexpr auto kNameTranspose = "Transpose";
constexpr auto kCustomTypeAscend = "acl_build";

bool IsInputNotCNode(const CNodePtr &kernel_node, size_t input_index) {
  auto input_node = common::AnfAlgo::VisitKernel(kernel_node->input(input_index + 1), 0).first;
  MS_EXCEPTION_IF_NULL(input_node);
  if (input_node->isa<Parameter>() || input_node->isa<ValueNode>()) {
    return true;
  }
  return false;
}

void GetOutputDtypes(const CNodePtr &kernel_node, std::vector<TypeId> *output_types) {
  size_t output_num = AnfUtils::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    TypeId dtype = common::AnfAlgo::GetOutputInferDataType(kernel_node, output_index);
    output_types->emplace_back(dtype);
  }
}

void GetOutputFormat(const CNodePtr &kernel_node, std::vector<std::string> *output_formats) {
  size_t output_num = AnfUtils::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    output_formats->emplace_back(kOpFormat_DEFAULT);
  }
}

void GetInputDtypes(const CNodePtr &kernel_node, std::vector<TypeId> *input_types,
                    std::vector<size_t> *input_no_cnode_indexes) {
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    TypeId dtype = kTypeUnknown;
    if (IsInputNotCNode(kernel_node, input_index)) {
      input_no_cnode_indexes->emplace_back(input_index);
      dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_index);
    } else {
      dtype = AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, input_index);
    }
    input_types->emplace_back(dtype);
  }
}

void GetInputFormat(const CNodePtr &kernel_node, std::vector<std::string> *input_formats) {
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    input_formats->emplace_back(kOpFormat_DEFAULT);
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

int GetOutputDtypeMatchedNum(const kernel::KernelAttr &kernel_attr, const std::vector<TypeId> &output_types) {
  if (kernel_attr.GetOutputSize() != output_types.size()) {
    MS_LOG(DEBUG) << "required output num:" << kernel_attr.GetInputSize()
                  << ", actual output num:" << output_types.size();
    return 0;
  }
  int data_type_matched_num = 0;
  auto output_num = output_types.size();
  for (size_t i = 0; i < output_num; ++i) {
    if (kernel_attr.GetOutputAttr(i).dtype != output_types[i]) {
      MS_LOG(DEBUG) << "required dtype:" << kernel_attr.GetOutputAttr(i).dtype
                    << ", actual output dtype:" << output_types[i];
    } else {
      data_type_matched_num++;
    }
  }
  return data_type_matched_num;
}

int GetInputDtypeFormatMatchedNum(const kernel::KernelAttr &kernel_attr, const std::vector<TypeId> &input_types,
                                  const std::vector<size_t> &input_not_cnode_indexes, bool strict) {
  if (kernel_attr.GetInputSize() != input_types.size()) {
    MS_LOG(DEBUG) << "required input num:" << kernel_attr.GetInputSize() << ", actual input num:" << input_types.size();
    return 0;
  }
  int data_type_matched_num = 0;
  auto input_num = input_types.size();
  for (size_t i = 0; i < input_num; ++i) {
    if (!InputDtypeMatch(kernel_attr.GetInputAttr(i).dtype, input_types[i], strict)) {
      MS_LOG(DEBUG) << "required dtype:" << kernel_attr.GetInputAttr(i).dtype
                    << ", actual input dtype:" << input_types[i];
    } else {
      data_type_matched_num++;
    }
  }
  return data_type_matched_num;
}

void ExpandKernelAttr(const CNodePtr &kernel_node, kernel::KernelAttr *kernel_attr) {
  MS_EXCEPTION_IF_NULL(kernel_attr);
  size_t attr_num = kernel_attr->GetInputSize();
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
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
    TypeId input_dtype = kernel_attr->GetInputAttr(i).dtype;
    for (size_t j = 0; j < each_attr_input_num; ++j) {
      (void)attr_list.emplace_back(input_dtype, format);
    }
  }
  kernel_attr->SetInputAttrList(attr_list);

  TypeId output_dtype = kernel_attr->GetOutputAttr(0).dtype;
  size_t output_num = AnfUtils::GetOutputTensorNum(kernel_node);
  for (size_t i = 1; i < output_num; ++i) {
    (void)kernel_attr->AddOutputAttr(output_dtype);
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

void UpdateDynamicKernelBuildInfo(const CNodePtr &kernel_node) {
  const std::string &op_name = common::AnfAlgo::GetCNodeName(kernel_node);
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
}

bool CheckKernelInfo(const std::shared_ptr<KernelBuildInfo> &alternative_kernel_info,
                     const std::shared_ptr<KernelBuildInfo> &selected_kernel_info) {
  MS_EXCEPTION_IF_NULL(selected_kernel_info);
  MS_EXCEPTION_IF_NULL(alternative_kernel_info);
  size_t selected_input_num = selected_kernel_info->GetInputNum();
  size_t alternative_input_num = alternative_kernel_info->GetInputNum();
  if (selected_input_num != alternative_input_num) {
    return false;
  }
  for (size_t i = 0; i < selected_input_num; i++) {
    auto format = alternative_kernel_info->GetInputFormat(i);
    if (selected_kernel_info->GetInputFormat(i) != format && (!format.empty())) {
      return false;
    }
    auto type = alternative_kernel_info->GetInputDeviceType(i);
    if (selected_kernel_info->GetInputDeviceType(i) != type && (type != TypeId::kMetaTypeNone)) {
      return false;
    }
  }

  size_t selected_output_num = selected_kernel_info->GetOutputNum();
  size_t alternative_output_num = alternative_kernel_info->GetOutputNum();
  if (selected_output_num != alternative_output_num) {
    return false;
  }
  for (size_t i = 0; i < selected_output_num; i++) {
    auto format = alternative_kernel_info->GetOutputFormat(i);
    if (selected_kernel_info->GetOutputFormat(i) != format && (!format.empty())) {
      return false;
    }
    auto type = alternative_kernel_info->GetOutputDeviceType(i);
    if (selected_kernel_info->GetOutputDeviceType(i) != type && (type != TypeId::kMetaTypeNone)) {
      return false;
    }
  }
  return true;
}

void UpdateCustomKernelBuildInfo(const CNodePtr &kernel_node, bool is_akg_op) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  const std::string &op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  std::shared_ptr<mindspore::kernel::OpInfo> kernel_attr = nullptr;
  if (is_akg_op) {
#ifndef USE_LLVM
    MS_LOG(EXCEPTION) << "When calling AKG-CPU operator, found LLVM 12.0.1 not installed, please check: "
                         "https://www.mindspore.cn/install for installing LLVM on MindSpore.";
#else
    builder->SetKernelType(KernelType::AKG_KERNEL);
#endif
    kernel_attr = mindspore::kernel::OpLib::FindOp(op_name, kernel::OpImplyType::kImplyAKG);
    if (kernel_attr == nullptr) {
      MS_LOG(WARNING) << "Not find operator information for Custom operator[" << op_name << "]. "
                      << "Infer operator information from inputs. For more details, "
                      << "please refer to 'mindspore.ops.Custom' at https://www.mindspore.cn.";
    }
  } else {
    builder->SetKernelType(KernelType::CPU_KERNEL);
  }
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
  if (op_name == kNameCustomAscend) {
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), kernel_node.get());
  }
  // check reg info if kernel_attr is not null
  if (kernel_attr != nullptr) {
    std::vector<std::shared_ptr<KernelBuildInfo>> kernel_info_list;
    if (!ParseMetadata(kernel_node, kernel_attr, kernel::Processor::CPU, &kernel_info_list)) {
      MS_LOG(EXCEPTION) << "Parsed metadata of op[" << op_name << "] failed.";
    }
    if (kernel_info_list.empty()) {
      MS_LOG(EXCEPTION) << "Not find valid metadata of op[" << op_name << "].";
    }
    bool match = std::any_of(kernel_info_list.begin(), kernel_info_list.end(),
                             [&](const std::shared_ptr<KernelBuildInfo> &alternative_kernel_info) {
                               return CheckKernelInfo(alternative_kernel_info, builder->Build());
                             });
    if (!match) {
      MS_LOG(ERROR) << "Not find op[" << op_name << "] which both match data type and format in akg";
    }
  }
}

kernel::KernelAttr FillNoneInKernelAttr(const CNodePtr &kernel_node, const std::vector<TypeId> &input_types,
                                        const std::vector<TypeId> &output_types,
                                        const kernel::KernelAttr &kernel_attr) {
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
  kernel::KernelAttr result;
  // Fill inputs info.
  for (size_t i = 0; i < input_num; ++i) {
    auto type_format = kernel_attr.GetInputAttr(i);
    if (type_format.dtype == TypeId::kMetaTypeNone) {
      type_format.dtype = input_types[i];
    }
    if (type_format.format.empty()) {
      type_format.format = kOpFormat_DEFAULT;
    }
    (void)result.AddInputAttr(type_format.dtype, type_format.format);
  }
  // Fill outputs info.
  for (size_t i = 0; i < output_num; ++i) {
    auto type_format = kernel_attr.GetOutputAttr(i);
    if (type_format.dtype == TypeId::kMetaTypeNone) {
      type_format.dtype = output_types[i];
    }
    if (type_format.format.empty()) {
      type_format.format = kOpFormat_DEFAULT;
    }
    (void)result.AddOutputAttr(type_format.dtype, type_format.format);
  }
  return result;
}
}  // namespace

bool IsDynamicParamKernel(const std::string &op_name) {
  const auto &op_info = kernel::OpLib::FindOp(op_name, kernel::OpImplyType::kImplyCPU);
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

bool SelectKernel(const CNodePtr &kernel_node, kernel::KernelAttr *selected_kernel_attr,
                  const std::vector<kernel::KernelAttr> &kernel_attrs, const std::vector<TypeId> &input_types,
                  const std::vector<size_t> &input_not_cnode_indexes, const std::vector<TypeId> &output_types,
                  std::pair<bool, bool> *matched, bool strict) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(selected_kernel_attr);
  MS_EXCEPTION_IF_NULL(matched);
  MS_LOG(DEBUG) << "Select kernel for op: " << common::AnfAlgo::GetCNodeName(kernel_node);
  for (auto kernel_attr : kernel_attrs) {
    if (kernel_attr.GetAllSame()) {
      ExpandKernelAttr(kernel_node, &kernel_attr);
    }
    size_t output_num = AnfUtils::GetOutputTensorNum(kernel_node);
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

kernel::KernelAttr BuildKernelFromInput(const std::vector<TypeId> &inputs, const std::vector<TypeId> &outputs,
                                        const kernel::KernelAttr &origin_attr) {
  kernel::KernelAttr attr = origin_attr;
  for (auto in_dtype : inputs) {
    (void)attr.AddInputAttr(in_dtype);
  }
  for (auto out_dtype : outputs) {
    (void)attr.AddOutputAttr(out_dtype);
  }
  (void)attr.AddSkipCheckAttr(true);
  return attr;
}

std::pair<std::string, ExceptionType> SetKernelInfoWithMsg(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  const std::string &op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  if (IsPrimitiveCNode(kernel_node, prim::kPrimCustom)) {
    auto tp = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, kAttrFuncType);
    if (IsOneOfCustomAkgType(tp)) {
      UpdateCustomKernelBuildInfo(kernel_node, true);
      return {};
    }
    if (tp == kCustomTypeAscend) {
      UpdateCustomKernelBuildInfo(kernel_node, false);
      return {};
    }
    // If Custom op has not set reg info, then infer info from inputs
    if (mindspore::kernel::OpLib::FindOp(op_name, kernel::OpImplyType::kImplyCPU) == nullptr) {
      MS_LOG(WARNING) << "Not find operator information for Custom operator[" << op_name << "]. "
                      << "Infer operator information from inputs. For more details, "
                      << "please refer to 'mindspore.ops.Custom' at https://www.mindspore.cn.";
      UpdateCustomKernelBuildInfo(kernel_node, false);
      return {};
    }
  } else if (IsDynamicParamKernel(op_name)) {
    // Select for dynamic kernel(both the number and data type are undetermined).
    UpdateDynamicKernelBuildInfo(kernel_node);
    return {};
  } else if (IsAKGSparseOP(kernel_node)) {
    UpdateCustomKernelBuildInfo(kernel_node, true);
    return {};
  }

  std::vector<std::string> input_formats;
  std::vector<TypeId> input_types;
  std::vector<size_t> input_not_cnode_indexes;
  std::vector<std::string> selected_output_formats;
  std::vector<TypeId> selected_output_types;
  MS_LOG(INFO) << "SetKernelInfo, CNode Name: " << op_name;
  GetInputDtypes(kernel_node, &input_types, &input_not_cnode_indexes);
  GetInputFormat(kernel_node, &input_formats);
  GetOutputDtypes(kernel_node, &selected_output_types);
  GetOutputFormat(kernel_node, &selected_output_formats);

  SetKernelBuildInfo(input_formats, input_types, selected_output_formats, selected_output_types, kernel_node.get());
  return {};
}

void SetKernelInfo(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto [msg, etype] = SetKernelInfoWithMsg(kernel_node);
  if (msg.empty()) return;
  MS_EXCEPTION(etype) << msg;
}

void CopyInputWeights(const CNodePtr &kernel_node, const std::vector<kernel::KernelTensorPtr> &inputs) {
  std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name == kNameCustomAscend || kernel_name == kNameTranspose) {
    auto node_input_size = kernel_node->inputs().size();
    if (node_input_size < kInputNum) {
      MS_LOG(ERROR) << "Input num of custom ascend kernel should larger than " << (kInputNum - 1) << ", real num is "
                    << node_input_size;
      return;
    }
    if (node_input_size != inputs.size() + 1) {
      MS_LOG(ERROR) << "Input num of custom ascend kernel [" << node_input_size << "]"
                    << " is not equal to kernel tensor size[" << (inputs.size() + 1) << "].";
      return;
    }
    auto weight_input = kernel_node->input(node_input_size - 1);
    if (!weight_input->isa<Parameter>()) {
      MS_LOG(ERROR) << "Om input is not parameter.";
      return;
    }
    ParameterPtr weight_param = weight_input->cast<ParameterPtr>();
    if (weight_param == nullptr || !weight_param->has_default()) {
      MS_LOG(ERROR) << "Om param is invalid, val= " << weight_param;
      return;
    }
    auto tensor = std::static_pointer_cast<tensor::Tensor>(weight_param->default_param());
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Tensor is nullptr.";
      return;
    }
    if (tensor->data_c() == nullptr || tensor->Size() == 0) {
      MS_LOG(ERROR) << "Tensor data is invalid.";
      return;
    }
    auto new_addr = malloc(tensor->Size());
    if (new_addr == nullptr) {
      MS_LOG(ERROR) << "Malloc failed, size= " << tensor->Size();
      return;
    }
    memcpy(new_addr, tensor->data_c(), tensor->Size());
    kernel::AddressPtr addr_ptr = std::make_shared<kernel::Address>(new_addr, tensor->Size());
    inputs[inputs.size() - 1]->SetData(addr_ptr);
  }
}
}  // namespace infer
}  // namespace mindspore
