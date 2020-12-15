/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "utils/trace_base.h"
namespace mindspore {
namespace device {
namespace cpu {
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;
using mindspore::kernel::KernelBuildInfo;
namespace {
bool IsInputNotCNode(const CNodePtr &kernel_node, size_t input_index) {
  auto input_node = AnfAlgo::VisitKernel(kernel_node->input(input_index + 1), 0).first;
  MS_EXCEPTION_IF_NULL(input_node);
  if (input_node->isa<Parameter>() || input_node->isa<ValueNode>()) {
    return true;
  }
  return false;
}

void UpdatePrevNotCNodeFormatDtype(const KernelAttr &kernel_attr, const std::vector<size_t> &input_not_cnode_indexes,
                                   const CNodePtr kernel_node) {
  for (auto &input_index : input_not_cnode_indexes) {
    auto input_node = AnfAlgo::VisitKernel(kernel_node->input(input_index + 1), 0).first;
    MS_EXCEPTION_IF_NULL(input_node);
    std::vector<TypeId> output_types;
    output_types.emplace_back(kernel_attr.GetInputAttr(input_index).first);
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    MS_EXCEPTION_IF_NULL(builder);
    builder->SetOutputsFormat({kOpFormat_DEFAULT});
    builder->SetOutputsDeviceType(output_types);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), input_node.get());
  }
}

void GetOutputInferFormatsAndDtypes(const CNodePtr &kernel_node, std::vector<std::string> *output_formats,
                                    std::vector<TypeId> *output_types) {
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    TypeId dtype = kTypeUnknown;
    dtype = AnfAlgo::GetOutputInferDataType(kernel_node, output_index);
    output_formats->emplace_back(kOpFormat_DEFAULT);
    output_types->emplace_back(dtype);
  }
}

void GetInputFormatsAndDtypes(const CNodePtr &kernel_node, std::vector<std::string> *input_formats,
                              std::vector<TypeId> *input_types, std::vector<size_t> *input_no_cnode_indexes) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    TypeId dtype = kTypeUnknown;
    if (IsInputNotCNode(kernel_node, input_index)) {
      input_no_cnode_indexes->emplace_back(input_index);
      dtype = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_index);
    } else {
      dtype = AnfAlgo::GetPrevNodeOutputDeviceDataType(kernel_node, input_index);
    }
    input_formats->emplace_back(kOpFormat_DEFAULT);
    input_types->emplace_back(dtype);
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

std::pair<int, int> GetOutputDtypeFormatMatchedNum(const KernelAttr &kernel_attr,
                                                   const std::vector<std::string> &output_formats,
                                                   const std::vector<TypeId> &output_types) {
  if (kernel_attr.GetOutputSize() != output_types.size()) {
    MS_LOG(DEBUG) << "required output num:" << kernel_attr.GetInputSize()
                  << ", actual output num:" << output_types.size();
    return std::make_pair(0, 0);
  }
  int data_type_matched_num = 0;
  int format_matched_num = 0;
  auto output_num = output_types.size();
  for (size_t i = 0; i < output_num; ++i) {
    if (kernel_attr.GetOutputAttr(i).first != output_types[i]) {
      MS_LOG(DEBUG) << "required dtype:" << kernel_attr.GetOutputAttr(i).first
                    << ", actual output dtype:" << output_types[i];
    } else {
      data_type_matched_num++;
    }

    if (kernel_attr.GetOutputAttr(i).second != output_formats[i]) {
      MS_LOG(DEBUG) << "required format:" << kernel_attr.GetOutputAttr(i).second
                    << ", actual output format:" << output_formats[i];
    } else {
      format_matched_num++;
    }
  }
  return std::make_pair(data_type_matched_num, format_matched_num);
}

std::pair<int, int> GetInputDtypeFormatMatchedNum(const KernelAttr &kernel_attr,
                                                  const std::vector<std::string> &input_formats,
                                                  const std::vector<TypeId> &input_types,
                                                  const std::vector<size_t> &input_not_cnode_indexes, bool strict) {
  if (kernel_attr.GetInputSize() != input_types.size()) {
    MS_LOG(DEBUG) << "required input num:" << kernel_attr.GetInputSize() << ", actual input num:" << input_types.size();
    return std::make_pair(0, 0);
  }
  int data_type_matched_num = 0;
  int format_matched_num = 0;
  auto input_num = input_types.size();
  for (size_t i = 0; i < input_num; ++i) {
    bool is_not_cnode_idx = std::any_of(input_not_cnode_indexes.begin(), input_not_cnode_indexes.end(),
                                        [i](size_t index) { return index == i; });
    bool have_cnode_input = (input_types.size() != input_not_cnode_indexes.size());
    if (have_cnode_input && is_not_cnode_idx) {
      data_type_matched_num++;
      format_matched_num++;
      continue;
    }
    if (is_not_cnode_idx) {
      if (!InputDtypeMatch(kernel_attr.GetInputAttr(i).first, input_types[i], strict)) {
        MS_LOG(DEBUG) << "required dtype:" << kernel_attr.GetInputAttr(i).first
                      << ", actual input dtype:" << input_types[i];
      } else {
        data_type_matched_num++;
      }
      format_matched_num++;
      continue;
    }
    if (kernel_attr.GetInputAttr(i).first != input_types[i]) {
      MS_LOG(DEBUG) << "required dtype:" << kernel_attr.GetInputAttr(i).first
                    << ", actual input dtype:" << input_types[i];
    } else {
      data_type_matched_num++;
    }

    if (kernel_attr.GetInputAttr(i).second != input_formats[i]) {
      MS_LOG(DEBUG) << "required format:" << kernel_attr.GetInputAttr(i).second
                    << ", actual input format:" << input_formats[i];
    } else {
      format_matched_num++;
    }
  }
  return std::make_pair(data_type_matched_num, format_matched_num);
}

void ExpandKernelAttr(const CNodePtr &kernel_node, KernelAttr *kernel_attr) {
  MS_EXCEPTION_IF_NULL(kernel_attr);
  TypeId input_dtype = kernel_attr->GetInputAttr(0).first;
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t i = 1; i < input_num; ++i) {
    kernel_attr->AddInputAttr(input_dtype);
  }

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
  operator_info << "is not support.";
  MS_EXCEPTION(TypeError) << operator_info.str() << " Trace: " << trace::DumpSourceLines(kernel_node);
}
}  // namespace
bool SelectKernel(const CNodePtr &kernel_node, KernelAttr *selected_kernel_attr,
                  const std::vector<KernelAttr> &kernel_attrs, const std::vector<std::string> &input_formats,
                  const std::vector<TypeId> &input_types, const std::vector<size_t> &input_not_cnode_indexes,
                  const std::vector<std::string> &infer_output_formats, const std::vector<TypeId> &infer_output_types,
                  std::pair<bool, bool> *matched, bool strict) {
  int max_type_matched_num = -1;
  int max_format_matched_num = -1;
  for (auto kernel_attr : kernel_attrs) {
    if (kernel_attr.GetAllSame()) {
      ExpandKernelAttr(kernel_node, &kernel_attr);
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (kernel_attr.GetOutputSize() != output_num) {
      MS_LOG(DEBUG) << "Output num is not equal!";
      continue;
    }
    std::pair<int, int> input_type_format_matched_num =
      GetInputDtypeFormatMatchedNum(kernel_attr, input_formats, input_types, input_not_cnode_indexes, strict);
    std::pair<int, int> output_type_format_matched_num =
      GetOutputDtypeFormatMatchedNum(kernel_attr, infer_output_formats, infer_output_types);
    // Data type first
    if (input_type_format_matched_num.first > max_type_matched_num) {
      max_type_matched_num = input_type_format_matched_num.first;
      max_format_matched_num = input_type_format_matched_num.second;
      *selected_kernel_attr = kernel_attr;
    } else if (input_type_format_matched_num.first == max_type_matched_num &&
               input_type_format_matched_num.second > max_format_matched_num) {
      max_format_matched_num = input_type_format_matched_num.second;
      *selected_kernel_attr = kernel_attr;
    } else if (input_type_format_matched_num.first == max_type_matched_num &&
               input_type_format_matched_num.second == max_format_matched_num) {
      if (output_type_format_matched_num.first == SizeToInt(infer_output_types.size()) &&
          output_type_format_matched_num.second == SizeToInt(infer_output_types.size())) {
        *selected_kernel_attr = kernel_attr;
      }
    }
    // All formats and data types matched
    if (input_type_format_matched_num.first == SizeToInt(input_types.size()) &&
        input_type_format_matched_num.second == SizeToInt(input_types.size())) {
      matched->first = true;
      if (output_type_format_matched_num.first == SizeToInt(infer_output_types.size()) &&
          output_type_format_matched_num.second == SizeToInt(infer_output_types.size())) {
        matched->second = true;
        return true;
      }
    }
  }
  return false;
}
void SetKernelInfo(const CNodePtr &kernel_node) {
  std::vector<std::string> input_formats;
  std::vector<TypeId> input_types;
  std::vector<size_t> input_not_cnode_indexes;
  std::vector<std::string> output_formats;
  std::vector<TypeId> output_types;
  std::vector<std::string> infer_output_formats;
  std::vector<TypeId> infer_output_types;
  MS_LOG(INFO) << "SetKernelInfo, CNode Name: " << AnfAlgo::GetCNodeName(kernel_node);
  auto kernel_attrs =
    kernel::CPUKernelFactory::GetInstance().GetSupportedKernelAttrList(AnfAlgo::GetCNodeName(kernel_node));
  if (kernel_attrs.empty()) {
    MS_LOG(EXCEPTION) << "Operator[" << AnfAlgo::GetCNodeName(kernel_node)
                      << "] is not support. Trace: " << trace::DumpSourceLines(kernel_node);
  }
  GetInputFormatsAndDtypes(kernel_node, &input_formats, &input_types, &input_not_cnode_indexes);
  GetOutputInferFormatsAndDtypes(kernel_node, &infer_output_formats, &infer_output_types);
  KernelAttr selected_kernel_attr;
  std::pair<bool, bool> matched = std::make_pair(false, false);
  if (!SelectKernel(kernel_node, &selected_kernel_attr, kernel_attrs, input_formats, input_types,
                    input_not_cnode_indexes, infer_output_formats, infer_output_types, &matched, true)) {
    if (AnfAlgo::GetCNodeName(kernel_node) == "Cast") {
      KernelNotSupportException(kernel_node, input_types, infer_output_types);
    }
    matched = std::make_pair(false, false);
    SelectKernel(kernel_node, &selected_kernel_attr, kernel_attrs, input_formats, input_types, input_not_cnode_indexes,
                 infer_output_formats, infer_output_types, &matched, false);
    if (!matched.first) {
      KernelNotSupportException(kernel_node, input_types, infer_output_types);
    }
  }

  if (selected_kernel_attr.GetInputSize() > 0 &&
      (matched.first || input_types.size() == input_not_cnode_indexes.size())) {
    MS_LOG(INFO) << "Input format and dtype is matched";
    GetOutputFormatsAndDtypes(kernel_node, selected_kernel_attr, &output_formats, &output_types);
    UpdatePrevNotCNodeFormatDtype(selected_kernel_attr, input_not_cnode_indexes, kernel_node);
    for (auto &input_index : input_not_cnode_indexes) {
      input_types[input_index] = selected_kernel_attr.GetInputAttr(input_index).first;
    }
  }
  SetKernelBuildInfo(input_formats, input_types, output_formats, output_types, kernel_node.get());
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
