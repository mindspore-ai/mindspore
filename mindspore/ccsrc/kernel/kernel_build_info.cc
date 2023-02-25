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

#include "kernel/kernel_build_info.h"

#include <algorithm>
#include <unordered_map>
#include "utils/log_adapter.h"
#include "include/common/debug/anf_dump_utils.h"

namespace mindspore {
namespace kernel {
std::string KernelObjectTypeLabel(const KernelObjectType &obj_type) {
  std::unordered_map<KernelObjectType, std::string> trans_map{{KernelObjectType::TUPLE, "Tuple"},
                                                              {KernelObjectType::SCALAR, "Scalar"},
                                                              {KernelObjectType::TENSOR, "Tensor"},
                                                              {KernelObjectType::UNKNOWN_TYPE, "Unknown"},
                                                              {KernelObjectType::TUPLE_UNFOLD, "TupleUnfold"}};
  if (trans_map.find(obj_type) == trans_map.end()) {
    return "Unknown";
  }
  return trans_map[obj_type];
}

std::string KernelTypeLabel(const KernelType &kernel_type) {
  std::unordered_map<KernelType, std::string> trans_map{{KernelType::UNKNOWN_KERNEL_TYPE, "UNKNOWN_KERNEL_TYPE"},
                                                        {KernelType::AKG_KERNEL, "AKG_KERNEL"},
                                                        {KernelType::AICPU_KERNEL, "AICPU_KERNEL"},
                                                        {KernelType::RT_KERNEL, "RT_KERNEL"},
                                                        {KernelType::HCCL_KERNEL, "HCCL_KERNEL"},
                                                        {KernelType::TBE_KERNEL, "TBE_KERNEL"},
                                                        {KernelType::HOST_KERNEL, "HOST_KERNEL"},
                                                        {KernelType::CPU_KERNEL, "CPU_KERNEL"},
                                                        {KernelType::GPU_KERNEL, "GPU_KERNEL"},
                                                        {KernelType::BISHENG_KERNEL, "BISHENG_KERNEL"},
                                                        {KernelType::ACL_KERNEL, "ACL_KERNEL"}};
  if (trans_map.find(kernel_type) == trans_map.end()) {
    return "UNKNOWN_KERNEL_TYPE";
  }
  return trans_map[kernel_type];
}

std::string OpTypeLabel(const OpType &op_type) {
  std::unordered_map<OpType, std::string> trans_map{
    {OpType::UNKNOWN_OP_TYPE, "UNKNOWN_OP_TYPE"}, {OpType::DYNAMIC, "DYNAMIC"}, {OpType::SKIP, "SKIP"}};
  if (trans_map.find(op_type) == trans_map.end()) {
    return "UNKNOWN_OP_TYPE";
  }
  return trans_map[op_type];
}

std::string KernelBuildInfo::GetInputFormat(size_t input_index) const {
  if (input_index >= inputs_format_.size()) {
    MS_LOG(ERROR) << "The index [" << input_index << "] is exceed the number of input node";
    return kInvalidFormat;
  }
  return inputs_format_[input_index];
}

std::string KernelBuildInfo::GetOutputFormat(size_t output_index) const {
  if (output_index >= outputs_format_.size()) {
    MS_LOG(ERROR) << "The index [" << output_index << "] is exceed the number of output";
    return kInvalidFormat;
  }
  return outputs_format_[output_index];
}

TypeId KernelBuildInfo::GetInputDeviceType(size_t input_index) const {
  if (input_index >= inputs_device_type_.size()) {
    MS_LOG(ERROR) << "The index [" << input_index << "] is exceed the number of input";
    return TypeId::kNumberTypeEnd;
  }
  return inputs_device_type_[input_index];
}

TypeId KernelBuildInfo::GetOutputDeviceType(size_t output_index) const {
  if (output_index >= outputs_device_type_.size()) {
    MS_LOG(ERROR) << "The index [" << output_index << "] is exceed the number of output";
    return TypeId::kNumberTypeEnd;
  }
  return outputs_device_type_[output_index];
}

KernelObjectType KernelBuildInfo::GetInputKernelObjectType(size_t input_index) const {
  if (inputs_kernel_object_type_.empty()) {
    return KernelObjectType::UNKNOWN_TYPE;
  }
  if (input_index >= inputs_kernel_object_type_.size()) {
    bool has_tuple_unfold =
      std::any_of(inputs_kernel_object_type_.begin(), inputs_kernel_object_type_.end(),
                  [](const KernelObjectType &obj_type) { return obj_type == KernelObjectType::TUPLE_UNFOLD; });
    // tuple unfold may correspond to many formats or dtypes
    if (!has_tuple_unfold) {
      MS_LOG(ERROR) << "The input index [" << input_index
                    << "] is exceed the number of input:" << inputs_kernel_object_type_.size();
    }
    return KernelObjectType::UNKNOWN_TYPE;
  }
  return inputs_kernel_object_type_[input_index];
}

KernelObjectType KernelBuildInfo::GetOutputKernelObjectType(size_t output_index) const {
  if (outputs_kernel_object_type_.empty()) {
    return KernelObjectType::UNKNOWN_TYPE;
  }

  // tuple unfold may correspond to many formats or dtypes
  bool has_tuple_unfold =
    std::any_of(outputs_kernel_object_type_.begin(), outputs_kernel_object_type_.end(),
                [](const KernelObjectType &obj_type) { return obj_type == KernelObjectType::TUPLE_UNFOLD; });
  if (has_tuple_unfold) {
    return KernelObjectType::UNKNOWN_TYPE;
  }

  if (output_index >= outputs_kernel_object_type_.size()) {
    MS_LOG(ERROR) << "The output index [" << output_index
                  << "] is exceed the number of output:" << outputs_kernel_object_type_.size();
    return KernelObjectType::UNKNOWN_TYPE;
  }
  return outputs_kernel_object_type_[output_index];
}

const std::vector<KernelObjectType> &KernelBuildInfo::GetAllOutputElementsKernelObjectTypes() const {
  return output_elements_kernel_object_type_;
}

const std::string &KernelBuildInfo::GetOriginDataFormat() const { return origin_data_format_; }

const std::vector<std::string> &KernelBuildInfo::GetAllInputFormats() const { return inputs_format_; }

const std::vector<std::string> &KernelBuildInfo::GetAllOutputFormats() const { return outputs_format_; }

const std::vector<TypeId> &KernelBuildInfo::GetAllInputDeviceTypes() const { return inputs_device_type_; }

const std::vector<TypeId> &KernelBuildInfo::GetAllOutputDeviceTypes() const { return outputs_device_type_; }

const std::vector<KernelObjectType> &KernelBuildInfo::GetAllOutputKernelObjectTypes() const {
  return outputs_kernel_object_type_;
}

const std::vector<KernelObjectType> &KernelBuildInfo::GetAllInputKernelObjectTypes() const {
  return inputs_kernel_object_type_;
}

void KernelBuildInfo::SetOpType(const OpType &op_type) { op_type_ = op_type; }

void KernelBuildInfo::SetOutputsKernelObjectType(const std::vector<KernelObjectType> &outputs_kernel_object_type) {
  outputs_kernel_object_type_ = outputs_kernel_object_type;
}

void KernelBuildInfo::SetInputsKernelObjectType(const std::vector<KernelObjectType> &inputs_kernel_object_type) {
  inputs_kernel_object_type_ = inputs_kernel_object_type;
}

void KernelBuildInfo::SetOutputElementsKernelObjectType(
  const std::vector<KernelObjectType> &output_elements_kernel_object_type) {
  output_elements_kernel_object_type_ = output_elements_kernel_object_type;
}

void KernelBuildInfo::SetInputsFormat(const std::vector<std::string> &inputs_format) { inputs_format_ = inputs_format; }

void KernelBuildInfo::SetInputsDeviceType(const std::vector<TypeId> &inputs_device_type) {
  inputs_device_type_ = inputs_device_type;
}

void KernelBuildInfo::SetOutputFormat(const std::string &format, size_t index) {
  if (index >= outputs_format_.size()) {
    MS_LOG(EXCEPTION) << "The index [" << index << "] is exceed the number of output";
  }
  outputs_format_[index] = format;
}

void KernelBuildInfo::SetOutputsFormat(const std::vector<std::string> &outputs_format) {
  outputs_format_ = outputs_format;
}

void KernelBuildInfo::SetOutputDeviceType(const TypeId &output_device_type, size_t index) {
  if (index >= outputs_device_type_.size()) {
    MS_LOG(EXCEPTION) << "The index [" << index << "] is exceed the number of output";
  }
  outputs_device_type_[index] = output_device_type;
}

void KernelBuildInfo::SetOutputsDeviceType(const std::vector<TypeId> &outputs_device_type) {
  outputs_device_type_ = outputs_device_type;
}

size_t KernelBuildInfo::GetInputNum() const { return inputs_format_.size(); }

size_t KernelBuildInfo::GetOutputNum() const { return outputs_format_.size(); }

size_t KernelBuildInfo::GetOutputNumWithoutMonad() const {
  const auto count = std::count_if(outputs_device_type_.begin(), outputs_device_type_.end(),
                                   [](TypeId type) { return type != TypeId::kObjectTypeUMonad; });
  return static_cast<size_t>(count);
}

std::string KernelBuildInfo::GetInputReshapeType(size_t input_index) const {
  if (input_reshape_type_.empty()) {
    return "";
  }
  if (input_index >= input_reshape_type_.size()) {
    MS_LOG(EXCEPTION) << "The index [" << input_index << "] is exceed the number of input node size "
                      << input_reshape_type_.size();
  }
  return input_reshape_type_[input_index];
}

std::string KernelBuildInfo::GetOutputReshapeType(size_t output_index) const {
  if (output_reshape_type_.empty()) {
    return "";
  }
  if (output_index >= output_reshape_type_.size()) {
    MS_LOG(EXCEPTION) << "The index [" << output_index << "] is exceed the number of output node size "
                      << output_reshape_type_.size();
  }
  return output_reshape_type_[output_index];
}

std::string KernelBuildInfo::ToString() const {
  std::ostringstream output_buffer;
  output_buffer << "(";
  for (size_t index = 0; index < GetInputNum(); ++index) {
    if (index != 0) {
      output_buffer << ", ";
    }
    output_buffer << "<" << TypeIdLabel(GetInputDeviceType(index)) << "x" << GetInputFormat(index) << ">";
  }
  output_buffer << ", object_type: [";
  auto input_object_types = GetAllInputKernelObjectTypes();
  for (size_t index = 0; index < input_object_types.size(); ++index) {
    if (index != 0) {
      output_buffer << ",";
    }
    output_buffer << KernelObjectTypeLabel(input_object_types[index]);
  }

  output_buffer << "]) -> (";
  for (size_t index = 0; index < GetOutputNum(); ++index) {
    if (index != 0) {
      output_buffer << ",";
    }
    output_buffer << "<" << TypeIdLabel(GetOutputDeviceType(index)) << "x" << GetOutputFormat(index) << ">";
  }
  output_buffer << ", object_type: [";
  auto output_object_types = GetAllOutputKernelObjectTypes();
  for (size_t index = 0; index < output_object_types.size(); ++index) {
    if (index != 0) {
      output_buffer << ", ";
    }
    output_buffer << KernelObjectTypeLabel(output_object_types[index]);
  }
  output_buffer << "], kernel_type: " << KernelTypeLabel(kernel_type());
  output_buffer << ", op_type: " << OpTypeLabel(op_type());
  output_buffer << ")";
  return output_buffer.str();
}

bool KernelBuildInfo::IsSimilarityKernelBuildInfo(const KernelBuildInfo &other) const {
  if (inputs_format_ != other.inputs_format_ || outputs_format_ != other.outputs_format_) {
    if (op_pattern_ != kFormatAgnosticPattern) {
      return false;
    } else {
      MS_LOG(INFO) << "This kernel build info:" << this->ToString()
                   << ", other kernel build info: " << other.ToString();
    }
  }
  return !(inputs_device_type_ != other.inputs_device_type_ || outputs_device_type_ != other.outputs_device_type_);
}

bool KernelBuildInfo::operator==(const KernelBuildInfo &other) const {
  if (kernel_type_ != other.kernel_type_ || processor_ != other.processor_) {
    return false;
  }
  return IsSimilarityKernelBuildInfo(other);
}

bool KernelBuildInfo::IsInputDefaultPadding() const { return input_reshape_type_.empty(); }

bool KernelBuildInfo::IsOutputDefaultPadding() const { return output_reshape_type_.empty(); }

bool KernelBuildInfo::operator!=(const KernelBuildInfo &other) const { return !((*this) == other); }

void KernelBuildInfo::KernelBuildInfoBuilder::SetKernelType(const KernelType &kernel_type) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->kernel_type_ = kernel_type;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetOpType(const OpType &op_type) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->op_type_ = op_type;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetOriginDataFormat(const std::string &origin_data_format) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->origin_data_format_ = origin_data_format;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetInputsFormat(const std::vector<std::string> &inputs_format) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->inputs_format_ = inputs_format;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetOutputsFormat(const std::vector<std::string> &outputs_format) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->outputs_format_ = outputs_format;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetInputsDeviceType(const std::vector<TypeId> &inputs_device_type) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->inputs_device_type_ = inputs_device_type;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetOutputsDeviceType(const std::vector<TypeId> &outputs_device_type) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->outputs_device_type_ = outputs_device_type;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetFusionType(const std::string &fusion_type) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->fusion_type_ = fusion_type;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetCoreType(const std::string &core_type) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->core_type_ = core_type;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetOutputDataDesc(const std::vector<nlohmann::json> &data_desc) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->output_data_desc_ = data_desc;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetProcessor(Processor processor) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->processor_ = processor;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetInputsKernelObjectType(
  const std::vector<KernelObjectType> &inputs_kernel_object_type) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->inputs_kernel_object_type_ = inputs_kernel_object_type;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetOutputsKernelObjectType(
  const std::vector<KernelObjectType> &outputs_kernel_object_type) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->outputs_kernel_object_type_ = outputs_kernel_object_type;
}

std::shared_ptr<KernelBuildInfo> KernelBuildInfo::KernelBuildInfoBuilder::Build() { return kernel_build_info_; }

void KernelBuildInfo::KernelBuildInfoBuilder::SetInputsReshapeType(const std::vector<std::string> &input_reshape_type) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->input_reshape_type_ = input_reshape_type;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetOutputsReshapeType(
  const std::vector<std::string> &output_reshape_type) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->output_reshape_type_ = output_reshape_type;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetOpPattern(OpPattern pattern) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->op_pattern_ = pattern;
}
void KernelBuildInfo::KernelBuildInfoBuilder::SetInputFormat(const std::string &format, size_t index) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  auto index_limit = kernel_build_info_->inputs_format_.size();
  if (index >= index_limit) {
    MS_LOG(EXCEPTION) << "Index of input format out of range! The value should be less than: " << index_limit
                      << ", but got: " << index;
  }
  kernel_build_info_->inputs_format_[index] = format;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetOutputFormat(const std::string &format, size_t index) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  auto index_limit = kernel_build_info_->outputs_format_.size();
  if (index >= index_limit) {
    MS_LOG(EXCEPTION) << "Index of output format out of range! The value should be less than: " << index_limit
                      << ", but got: " << index;
  }
  kernel_build_info_->outputs_format_[index] = format;
}
void KernelBuildInfo::KernelBuildInfoBuilder::SetInputReshapeType(const std::string &input_reshape_type, size_t index) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  auto index_limit = kernel_build_info_->input_reshape_type_.size();
  if (index >= index_limit) {
    MS_LOG(EXCEPTION) << "Index of input_reshape_type out of range! The value should be less than: " << index_limit
                      << ", but got: " << index;
  }
  (void)std::copy(input_reshape_type.begin(), input_reshape_type.end(),
                  std::back_inserter(kernel_build_info_->input_reshape_type_[index]));
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetOutputReshapeType(const std::string &output_reshape_type,
                                                                   size_t index) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  auto index_limit = kernel_build_info_->output_reshape_type_.size();
  if (index >= index_limit) {
    MS_LOG(EXCEPTION) << "Index of output_reshape_type out of range! The value should be less than: " << index_limit
                      << ", but got: " << index;
  }
  (void)std::copy(output_reshape_type.begin(), output_reshape_type.end(),
                  std::back_inserter(kernel_build_info_->output_reshape_type_[index]));
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetOutputDeviceType(const TypeId &output_device_type, size_t index) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  auto index_limit = kernel_build_info_->outputs_device_type_.size();
  if (index >= index_limit) {
    MS_LOG(EXCEPTION) << "Index of output_device_type out of range! The value should be less than: " << index_limit
                      << ", but got: " << index;
  }
  kernel_build_info_->outputs_device_type_[index] = output_device_type;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetInputDeviceType(const TypeId &input_device_type, size_t index) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  auto index_limit = kernel_build_info_->inputs_device_type_.size();
  if (index >= index_limit) {
    MS_LOG(EXCEPTION) << "Index of input_device_type out of range! The value should be less than: " << index_limit
                      << ", but got: " << index;
  }
  kernel_build_info_->inputs_device_type_[index] = input_device_type;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetValid(bool valid) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->valid_ = valid;
}
}  // namespace kernel
}  // namespace mindspore
