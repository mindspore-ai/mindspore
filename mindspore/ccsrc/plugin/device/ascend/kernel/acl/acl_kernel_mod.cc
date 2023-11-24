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
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include "ir/tensor.h"
#include "runtime/stream.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "transform/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "mindspore/core/ops/op_utils.h"
#include "mindspore/ccsrc/include/transform/graph_ir/utils.h"

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

bool AclKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  converter_ = std::make_shared<transform::AclConverter>();
  return true;
}

void AclKernelMod::PackageInput(const size_t idx, const std::string &format, ShapeVector *shape) {
  MS_EXCEPTION_IF_NULL(shape);
  auto &params = input_params_[idx];
  if (!format.empty()) {
    params.ori_format = format;
    transform::AclHelper::PaddingOriShape(kernel_name_, idx, format, shape);
  } else {
    params.ori_format = transform::AclHelper::ConvertOriginShapeAndFormat(kernel_name_, idx, params.dev_format, shape);
  }

  params.ori_shape = *shape;
  if (!params.is_default) {
    auto groups = transform::AclHelper::GetFracZGroupFromAttr(primitive_);
    params.dev_shape = trans::TransShapeToDevice(*shape, params.dev_format, params.data_type, groups);
  } else {
    params.dev_shape = *shape;
  }
}

void AclKernelMod::PackageOutput(const size_t idx, const ShapeVector &shape) {
  auto &params = output_params_[idx];
  size_t type_size = params.type_size;
  size_t tensor_size = 0;
  params.ori_format = shape.size() == kDim4 ? kOpFormat_NCHW : kOpFormat_DEFAULT;
  ShapeVector dev_shape;
  if (!params.is_default) {
    auto groups = transform::AclHelper::GetFracZGroupFromAttr(primitive_);
    dev_shape = trans::TransShapeToDevice(shape, params.dev_format, params.data_type, groups);
  } else {
    dev_shape = shape;
  }
  tensor_size = dev_shape.empty()
                  ? type_size
                  : std::accumulate(dev_shape.begin(), dev_shape.end(), type_size, std::multiplies<size_t>());
  tensor_size = std::max(tensor_size, type_size);
  params.ori_shape = shape;
  params.dev_shape = dev_shape;
  output_size_list_[idx] = tensor_size;
}

std::string AclKernelMod::GetFormatFromInput(const std::vector<KernelTensor *> &inputs) {
  return converter_->GetFormatFromInputAttrMap(inputs, kernel_name_);
}

void AclKernelMod::GetInputInfo(const std::vector<KernelTensor *> &inputs) {
  if (input_params_.size() != inputs.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Acl kernel's input size is not equal with acl param's size:" << input_params_.size()
                               << " - input's size:" << inputs.size();
  }

  std::string format = transform::AclHelper::GetFormatFromAttr(primitive_);
  if (format.empty()) {
    format = converter_->GetFormatFromInputAttrMap(inputs, kernel_name_);
  }

  for (size_t i = 0; i < input_params_.size(); i++) {
    auto &input = inputs[i];
    MS_EXCEPTION_IF_NULL(input);
    auto shape = input->GetShapeVector();
    if (!IsValidShape(shape)) {
      // early stop if any input shape contains -1/-2, which means input shape is dynamic
      MS_LOG(INTERNAL_EXCEPTION) << "For " << kernel_name_ << ", Resize failed because of invalid shape: " << shape;
    }
    PackageInput(i, format, &shape);
  }
}

int AclKernelMod::GetOutputInfo(const std::vector<KernelTensor *> &outputs) {
  int ret = KRET_OK;
  if (output_params_.size() != outputs.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Acl kernel's output size is not equal with output param's size:"
                               << output_params_.size() << " - output's size:" << outputs.size();
  }

  for (size_t i = 0; i < output_params_.size(); i++) {
    auto &output = outputs[i];
    MS_EXCEPTION_IF_NULL(output);
    auto shape = output->GetShapeVector();
    if (!IsValidShape(shape)) {
      shape = output->GetMaxShape();
      if (shape.empty()) {
        MS_LOG(EXCEPTION) << "For " << kernel_name_
                          << ", the max_shape should not be empty when input shape is unknown.";
      }
      ret = KRET_UNKNOWN_OUT_SHAPE;
    }
    PackageOutput(i, shape);
  }
  return ret;
}

void AclKernelMod::CreateAclConverter() {
  MS_EXCEPTION_IF_NULL(converter_);
  converter_->Reset();
  converter_->ConvertToAclOpType(kernel_name_);
  converter_->ResizeAclOpInputs(primitive_);
  converter_->ConvertAttrToAclInput(primitive_->attrs(), kernel_name_, &input_to_host_array_);
  if (inputs_ != nullptr) {
    converter_->ConvertInputToAclAttr(*inputs_, kernel_name_);
  }
  if (transform::AclHelper::IsPrintDebugString()) {
    ms_attr_str_.clear();
    converter_->ConvertToAclAttr(primitive_->attrs(), kernel_name_, &ms_attr_str_);
  } else {
    converter_->ConvertToAclAttr(primitive_->attrs(), kernel_name_, nullptr);
  }
  converter_->ProcessRunnerSpecialInfo(kernel_name_, output_params_);
}

int AclKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  GetInputInfo(inputs);
  int ret = GetOutputInfo(outputs);
  inputs_ = &inputs;
  CreateAclConverter();

  return ret;
}

std::string AclKernelMod::DebugString() const {
  if (!transform::AclHelper::IsPrintDebugString()) {
    return "";
  }
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

void AclKernelMod::SetValueDependArgs(const std::set<int64_t> &indices) {
  auto info = transform::GeAdapterManager::GetInstance().GetInfo(kernel_name_, true);
  MS_EXCEPTION_IF_NULL(info);

  value_depend_args_.clear();
  for (auto item : indices) {
    if (info->input_attr_map().count(item) == 0) {
      value_depend_args_.emplace(item);
    }
  }
}

bool AclKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }
  inputs_ = &inputs;

  // Process value depend arguments, value depend arguments reside both on device and host (which were synchronized at
  // type and shape inference stage). Inside the ACL internal, it may also need to sync there arguments to host for
  // operator validation (e.g. ReduceSum), so put it on host will be more efficiencient to reduce the count of sync from
  // device to host.
  MS_EXCEPTION_IF_NULL(converter_);
  converter_->ConvertValueDependToHostInput(kernel_name_, inputs, input_params_, value_depend_args_,
                                            &input_to_host_array_);
  converter_->ConvertToAclInput(primitive_, input_to_host_array_, inputs, input_params_);
  converter_->ConvertToAclOutput(kernel_name_, outputs, output_params_);
  converter_->SetRunnerSpecialInfo();
  // cppcheck-suppress unreadVariable
  auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
  MS_LOG(DEBUG) << this->DebugString();
  MS_LOG(DEBUG) << converter_->DebugString();
  // release gil before run
  GilReleaseWithCheck release_gil;
  try {
    converter_->Run(stream_ptr);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Kernel launch failed, msg: " << e.what();
    return false;
  }
  return true;
}

void AclKernelMod::SetDeviceInfo(const std::vector<std::string> &input_device_formats,
                                 const std::vector<std::string> &output_device_formats,
                                 const std::vector<TypeId> &input_device_types,
                                 const std::vector<TypeId> &output_device_types) {
  if (input_device_formats.size() != input_device_types.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Acl kernel's input size is not equal with format's size:"
                               << input_device_formats.size() << " and type's size:" << input_device_types.size();
  }
  if (output_device_formats.size() != output_device_types.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Acl kernel's output size is not equal with format's size:"
                               << output_device_formats.size() << " and type's size:" << output_device_types.size();
  }

  if (primitive_ == nullptr && op_ != nullptr) {
    primitive_ = op_->GetPrim();
  }
  auto in_def_flag =
    primitive_ == nullptr ? true : transform::AclHelper::GetDefaultFormatFlagFromAttr(primitive_, true);
  input_params_.resize(input_device_formats.size());
  for (size_t i = 0; i < input_device_formats.size(); i++) {
    input_params_[i].data_type = input_device_types[i];
    input_params_[i].dev_format = input_device_formats[i];
    input_params_[i].is_default =
      in_def_flag && transform::AclHelper::CheckDefaultSupportFormat(input_device_formats[i]);
    input_params_[i].type_size = GetTypeByte(TypeIdToType(input_params_[i].data_type));
  }
  input_size_list_.resize(input_device_formats.size(), 0);

  auto out_def_flag =
    primitive_ == nullptr ? true : transform::AclHelper::GetDefaultFormatFlagFromAttr(primitive_, false);
  output_params_.resize(output_device_formats.size());
  for (size_t i = 0; i < output_device_formats.size(); i++) {
    output_params_[i].data_type = output_device_types[i];
    output_params_[i].dev_format = output_device_formats[i];
    output_params_[i].is_default =
      out_def_flag && transform::AclHelper::CheckDefaultSupportFormat(output_device_formats[i]);
    output_params_[i].type_size = GetTypeByte(TypeIdToType(output_params_[i].data_type));
  }
  output_size_list_.resize(output_device_formats.size(), 0);
}

void AclKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> & /*inputs*/,
                                            const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(converter_);
  std::vector<std::vector<int64_t>> output_shape = converter_->SyncData();
  if (outputs.size() != output_shape.size()) {
    MS_LOG(EXCEPTION) << "Size of outputs is " << outputs.size() << ", which is not equal to size of shape vector "
                      << output_shape.size();
  }
  for (size_t i = 0; i < output_shape.size(); ++i) {
    outputs[i]->SetShapeVector(output_shape[i]);
  }
}
}  // namespace kernel
}  // namespace mindspore
