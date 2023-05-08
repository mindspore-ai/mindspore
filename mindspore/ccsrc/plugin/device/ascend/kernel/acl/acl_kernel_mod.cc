/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/acl/acl_kernel_mod.h"
#include <functional>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "transform/acl_ir/acl_helper.h"

namespace mindspore {
namespace kernel {
namespace {
std::string MsTensorDescString(const TensorParams &param) {
  std::stringstream ss;
  ss << "[TensorDesc] ";
  ss << "DataType = " << TypeIdToString(param.data_type);
  ss << ", Origin Format = " << param.ori_format;
  ss << ", Origin Shape = " << VectorToString(param.ori_shape);
  ss << ", Device Format = " << param.dev_format;
  ss << ", Device Shape = " << VectorToString(param.dev_shape);
  return ss.str();
}
}  // namespace

void AclKernelMod::GetInputInfo(const std::vector<KernelTensorPtr> &inputs) {
  if (input_device_formats_.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Acl kernel's input size is not equal with format's size:" << input_device_formats_.size()
                      << " - input's size:" << inputs.size();
  }

  std::string format = transform::AclHelper::GetFormatFromAttr(primitive_ptr_);

  size_t idx = 0;
  for (auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    TensorParams params;
    auto device_type = input_device_types_[idx];
    params.data_type = device_type;
    auto shape = input->GetShapeVector();
    if (!IsValidShape(shape)) {
      // early stop if any input shape contains -1/-2, which means input shape is dynamic
      MS_LOG(EXCEPTION) << "In Resize function, input shape must be valid!";
    }
    if (format.length() != 0) {
      params.ori_format = format;
    } else {
      params.ori_format = transform::AclHelper::ConvertOriginShapeAndFormat(kernel_name_, idx, true, &shape);
    }

    params.ori_shape = shape;
    params.dev_format = input_device_formats_[idx];
    auto groups = transform::AclHelper::GetFracZGroupFromAttr(primitive_ptr_);
    params.dev_shape = trans::TransShapeToDevice(shape, params.dev_format, device_type, groups);
    (void)input_params_.emplace_back(params);
    ++idx;
  }
}

int AclKernelMod::GetOutputInfo(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &outputs) {
  int ret = KRET_OK;
  if (output_device_formats_.size() != outputs.size()) {
    MS_LOG(EXCEPTION) << "Acl kernel's output size is not equal with format's size:" << output_device_formats_.size()
                      << " - output's size:" << outputs.size();
  }

  std::string format = transform::AclHelper::GetFormatFromAttr(primitive_ptr_);

  size_t idx = 0;
  for (auto &output : outputs) {
    MS_EXCEPTION_IF_NULL(output);
    TensorParams params;
    auto device_type = output_device_types_[idx];
    params.data_type = device_type;
    size_t type_size = GetTypeByte(TypeIdToType(params.data_type));
    size_t tensor_size = 0;

    auto shape = output->GetShapeVector();
    if (format.length() != 0) {
      params.ori_format = format;
    } else {
      params.ori_format = transform::AclHelper::ConvertOriginShapeAndFormat(kernel_name_, idx, false, &shape);
    }
    auto device_shape = shape;
    auto groups = transform::AclHelper::GetFracZGroupFromAttr(primitive_ptr_);
    if (!IsValidShape(shape)) {
      auto max_shape = output->GetMaxShape();
      if (max_shape.empty()) {
        auto primitive = base_operator->GetPrim();
        MS_ERROR_IF_NULL(primitive);
        MS_LOG(EXCEPTION) << "For " << primitive->name()
                          << ", the max_shape should not be empty when input shape is known.";
      } else {
        tensor_size = SizeOf(max_shape) * type_size;
        shape = max_shape;
        device_shape = trans::TransShapeToDevice(shape, output_device_formats_[idx], device_type, groups);
        ret = KRET_UNKNOWN_OUT_SHAPE;
      }
    } else {
      device_shape = trans::TransShapeToDevice(shape, output_device_formats_[idx], device_type, groups);
      tensor_size = device_shape.empty()
                      ? type_size
                      : std::accumulate(device_shape.begin(), device_shape.end(), type_size, std::multiplies<size_t>());
      tensor_size = std::max(tensor_size, type_size);
    }
    params.dev_shape = device_shape;
    params.dev_format = output_device_formats_[idx];
    params.ori_shape = shape;
    (void)output_params_.emplace_back(params);
    (void)output_size_list_.emplace_back(tensor_size);
    ++idx;
  }
  return ret;
}

int AclKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                         const std::vector<KernelTensorPtr> &outputs,
                         const std::map<uint32_t, tensor::TensorPtr> &inputs_on_host) {
  int ret = KRET_OK;
  primitive_ptr_ = base_operator->GetPrim();
  MS_ERROR_IF_NULL(primitive_ptr_);
  kernel_name_ = primitive_ptr_->name();

  this->inputs_ = inputs;
  this->outputs_ = outputs;

  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
  input_params_.clear();
  output_params_.clear();
  ms_attr_str_.clear();

  GetInputInfo(inputs);
  ret = GetOutputInfo(base_operator, outputs);

  inputs_on_host_ = inputs_on_host;
  return ret;
}

std::string AclKernelMod::DebugString() const {
  std::stringstream ss;
  ss << "[MsLaunchInfo]OpType:" << kernel_name_ << std::endl;
  for (size_t i = 0; i < input_params_.size(); ++i) {
    auto param = input_params_[i];
    ss << "InputDesc[" << i << "]:";
    ss << MsTensorDescString(param) << std::endl;
  }
  for (size_t i = 0; i < ms_attr_str_.size(); ++i) {
    ss << "Attr[" << i << "] " << ms_attr_str_[i] << std::endl;
  }
  for (size_t i = 0; i < output_params_.size(); ++i) {
    auto param = output_params_[i];
    ss << "OutputDesc[" << i << "]:";
    ss << MsTensorDescString(param) << std::endl;
  }
  return ss.str();
}

bool AclKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }

  if (converter_ == nullptr) {
    converter_ = std::make_shared<transform::AclConverter>();
  }
  converter_->Reset();
  converter_->SetIsNeedRetrieveOutputShape(IsNeedRetrieveOutputShape());
  converter_->ConvertToAclOpType(kernel_name_);
  converter_->ResizeAclOpInputs(primitive_ptr_);
  converter_->ConvertAttrToAclInput(primitive_ptr_->attrs(), kernel_name_, &inputs_on_host_);
  converter_->ConvertToAclInput(primitive_ptr_, inputs_on_host_, inputs, input_params_);
  converter_->ConvertToAclOutput(kernel_name_, outputs, output_params_);
  converter_->ConvertToAclAttr(primitive_ptr_->attrs(), kernel_name_, &ms_attr_str_);
  if (!inputs_on_host_.empty()) {
    converter_->ConvertInputToAclAttr(inputs_on_host_, kernel_name_);
  }
  // cppcheck-suppress unreadVariable
  auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
  MS_LOG(INFO) << this->DebugString();
  MS_LOG(INFO) << converter_->DebugString();
  converter_->Run(stream_ptr);
  return true;
}

void AclKernelMod::SetDeviceInfo(const std::vector<std::string> &input_device_formats,
                                 const std::vector<std::string> &output_device_formats,
                                 const std::vector<TypeId> &input_device_types,
                                 const std::vector<TypeId> &output_device_types) {
  if (input_device_formats.size() != input_device_types.size()) {
    MS_LOG(EXCEPTION) << "Acl kernel's input size is not equal with format's size:" << input_device_formats.size()
                      << " and type's size:" << input_device_types.size();
  }
  if (output_device_formats.size() != output_device_types.size()) {
    MS_LOG(EXCEPTION) << "Acl kernel's output size is not equal with format's size:" << output_device_formats.size()
                      << " and type's size:" << output_device_types.size();
  }
  input_device_formats_ = input_device_formats;
  output_device_formats_ = output_device_formats;
  input_device_types_ = input_device_types;
  output_device_types_ = output_device_types;
}

std::vector<TaskInfoPtr> AclKernelMod::GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                               const std::vector<AddressPtr> &, uint32_t) {
  MS_LOG(EXCEPTION) << "Acl kernels do not support task sink mode.";
  return {};
}

void AclKernelMod::SyncData() {
  std::vector<std::vector<int64_t>> output_shape = converter_->SyncData();
  for (size_t i = 0; i < output_shape.size(); ++i) {
    outputs_[i]->SetShapeVector(output_shape[i]);
  }
}

bool AclKernelMod::IsNeedRetrieveOutputShape() { return transform::AclHelper::IsNeedRetrieveOutputShape(kernel_name_); }
}  // namespace kernel
}  // namespace mindspore
