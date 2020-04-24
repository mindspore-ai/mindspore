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
#include "utils/log_adapter.h"
#include "debug/anf_ir_dump.h"
namespace mindspore {
namespace kernel {
std::string KernelBuildInfo::GetInputFormat(size_t input_index) const {
  if (input_index >= inputs_format_.size()) {
    MS_LOG(EXCEPTION) << "The index [" << input_index << "] is exceed the number of input node";
  }
  return inputs_format_[input_index];
}

std::string KernelBuildInfo::GetOutputFormat(size_t output_index) const {
  if (output_index >= outputs_format_.size()) {
    MS_LOG(EXCEPTION) << "The index [" << output_index << "] is exceed the number of input node";
  }
  return outputs_format_[output_index];
}

TypeId KernelBuildInfo::GetInputDeviceType(size_t input_index) const {
  if (input_index >= inputs_device_type_.size()) {
    MS_LOG(EXCEPTION) << "The index [" << input_index << "] is exceed the number of input node";
  }
  return inputs_device_type_[input_index];
}

TypeId KernelBuildInfo::GetOutputDeviceType(size_t output_index) const {
  if (output_index >= outputs_device_type_.size()) {
    MS_LOG(EXCEPTION) << "The index [" << output_index << "] is exceed the number of input node";
  }
  return outputs_device_type_[output_index];
}

std::vector<std::string> KernelBuildInfo::GetAllInputFormats() const { return inputs_format_; }

std::vector<std::string> KernelBuildInfo::GetAllOutputFormats() const { return outputs_format_; }

std::vector<TypeId> KernelBuildInfo::GetAllInputDeviceTypes() const { return inputs_device_type_; }

std::vector<TypeId> KernelBuildInfo::GetAllOutputDeviceTypes() const { return outputs_device_type_; }

size_t KernelBuildInfo::GetInputNum() const { return inputs_format_.size(); }

size_t KernelBuildInfo::GetOutputNum() const { return outputs_format_.size(); }

std::vector<Axis> KernelBuildInfo::GetInputReshapeType(size_t input_index) const {
  if (input_index >= input_reshape_type_.size()) {
    MS_LOG(EXCEPTION) << "The index [" << input_index << "] is exceed the number of input node size "
                      << input_reshape_type_.size();
  }
  return input_reshape_type_[input_index];
}

std::vector<Axis> KernelBuildInfo::GetOutputReshapeType(size_t output_index) const {
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
    output_buffer << "<" << ToShortString(GetInputDeviceType(index)) << "x" << GetInputFormat(index) << ">";
  }
  output_buffer << ") -> (";
  for (size_t index = 0; index < GetOutputNum(); ++index) {
    if (index != 0) {
      output_buffer << ", ";
    }
    output_buffer << "<" << ToShortString(GetOutputDeviceType(index)) << "x" << GetOutputFormat(index) << ">";
  }
  output_buffer << ")";
  return output_buffer.str();
}

bool KernelBuildInfo::operator==(const KernelBuildInfo &other) const {
  if (kernel_type_ != other.kernel_type_ || fusion_type_ != other.fusion_type_ || processor_ != other.processor_) {
    return false;
  }
  if (inputs_format_ != other.inputs_format_ || outputs_format_ != other.outputs_format_) {
    return false;
  }
  return !(inputs_device_type_ != other.inputs_device_type_ || outputs_device_type_ != other.outputs_device_type_);
}

bool KernelBuildInfo::IsInputDefaultPadding() const { return output_reshape_type_.empty(); }

bool KernelBuildInfo::IsOutputDefaultPadding() const { return input_reshape_type_.empty(); }

void KernelBuildInfo::KernelBuildInfoBuilder::SetKernelType(const KernelType &kernel_type) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->kernel_type_ = kernel_type;
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

void KernelBuildInfo::KernelBuildInfoBuilder::SetFusionType(FusionType fusion_type) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->fusion_type_ = fusion_type;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetProcessor(Processor processor) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->processor_ = processor;
}

std::shared_ptr<KernelBuildInfo> KernelBuildInfo::KernelBuildInfoBuilder::Build() { return kernel_build_info_; }

void KernelBuildInfo::KernelBuildInfoBuilder::SetInputReshapeType(
  const std::vector<std::vector<Axis>> &input_reshape_type) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->input_reshape_type_ = input_reshape_type;
}

void KernelBuildInfo::KernelBuildInfoBuilder::SetOutputReshapeType(
  const std::vector<std::vector<Axis>> &output_reshape_type) {
  MS_EXCEPTION_IF_NULL(kernel_build_info_);
  kernel_build_info_->output_reshape_type_ = output_reshape_type;
}
}  // namespace kernel
}  // namespace mindspore
