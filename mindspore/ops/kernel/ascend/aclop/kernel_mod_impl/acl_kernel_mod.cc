/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclop/kernel_mod_impl/acl_kernel_mod.h"
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_util.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/acl_adapter_info.h"
#include "abstract/ops/primitive_infer_map.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "include/backend/common/ms_device_shape_transfer.h"

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

void SetParamsDataTypeIfComplexInput(const PrimitivePtr &prim, std::vector<TensorParams> *input_params,
                                     std::vector<TensorParams> *output_params) {
  MS_EXCEPTION_IF_NULL(input_params);
  MS_EXCEPTION_IF_NULL(output_params);
  if (input_params->empty() || output_params->empty() || (*input_params)[0].data_type != TypeId::kNumberTypeComplex64) {
    return;
  }

  auto info = device::ascend::GeAdapterManager::GetInstance().GetInfo(prim->name(), true);
  MS_EXCEPTION_IF_NULL(info);
  if (!device::ascend::AclAdapterManager::GetInstance().CheckAclAdapter(info->op_type())) {
    return;
  }
  auto acl_info = device::ascend::AclAdapterManager::GetInstance().GetOpInfo(info->op_type());
  if (!acl_info.is_complex_parallel_concerned()) {
    return;
  }

  for (size_t i = 0; i < input_params->size(); i++) {
    if ((*input_params)[i].data_type == TypeId::kNumberTypeComplex64) {
      (*input_params)[i].data_type = TypeId::kNumberTypeInt64;
    }
  }

  for (size_t i = 0; i < output_params->size(); i++) {
    if ((*output_params)[i].data_type == TypeId::kNumberTypeComplex64) {
      (*output_params)[i].data_type = TypeId::kNumberTypeInt64;
    }
  }
}

bool AclKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  converter_ = std::make_shared<device::ascend::AclConverter>();
  converter_->ConvertToAclOpType(kernel_name_);
  if (device::ascend::AclHelper::IsPrintDebugString()) {
    ms_attr_str_.clear();
    converter_->ConvertToAclAttr(primitive_->attrs(), kernel_name_, &ms_attr_str_);
  } else {
    converter_->ConvertToAclAttr(primitive_->attrs(), kernel_name_, nullptr);
  }
  converter_->ProcessRunnerSpecialInfo(kernel_name_, output_params_, is_dynamic_);
  return true;
}

void AclKernelMod::PackageInput(const size_t idx, const std::string &format, ShapeVector *shape) {
  MS_EXCEPTION_IF_NULL(shape);
  auto &params = input_params_[idx];
  if (!format.empty()) {
    params.ori_format = format;
    device::ascend::AclHelper::PaddingOriShape(kernel_name_, idx, format, shape);
  } else {
    params.ori_format =
      device::ascend::AclHelper::ConvertOriginShapeAndFormat(kernel_name_, idx, params.dev_format, shape);
  }

  params.ori_shape = *shape;
  if (!params.is_default) {
    auto groups = device::ascend::AclHelper::GetFracZGroupFromAttr(primitive_);
    params.dev_shape = trans::TransShapeToDevice(*shape, params.dev_format, params.data_type, groups);
  } else {
    params.dev_shape = *shape;
  }

  params.dev_format = params.dev_format == kOpFormat_DEFAULT ? params.ori_format : params.dev_format;
}

void AclKernelMod::PackageOutput(const size_t idx, const ShapeVector &shape) {
  auto &params = output_params_[idx];
  size_t type_size = params.type_size;
  size_t tensor_size = 0;
  params.ori_format = shape.size() == kDim4 ? kOpFormat_NCHW : kOpFormat_DEFAULT;
  ShapeVector dev_shape;
  if (!params.is_default) {
    auto groups = device::ascend::AclHelper::GetFracZGroupFromAttr(primitive_);
    dev_shape = trans::TransShapeToDevice(shape, params.dev_format, params.data_type, groups);
  } else {
    dev_shape = shape;
  }
  tensor_size = dev_shape.empty()
                  ? type_size
                  : std::accumulate(dev_shape.begin(), dev_shape.end(), type_size, std::multiplies<size_t>());
  params.ori_shape = shape;
  params.dev_shape = dev_shape;
  output_size_list_[idx] = tensor_size;
}

std::string AclKernelMod::GetFormatFromInput(const std::vector<KernelTensor *> &inputs) {
  return converter_->GetFormatFromInputAttrMap(inputs, kernel_name_);
}

void AclKernelMod::GetInputInfo(const std::vector<KernelTensor *> &inputs) {
  if (input_params_.size() != inputs.size()) {
    MS_LOG(INFO) << "Acl kernel's input size is not equal with acl param's size:" << input_params_.size()
                 << " - input's size:" << inputs.size();
  }

  std::string format = device::ascend::AclHelper::GetFormatFromAttr(primitive_);
  if (format.empty()) {
    format = converter_->GetFormatFromInputAttrMap(inputs, kernel_name_);
  }

  for (size_t i = 0; i < input_params_.size(); i++) {
    if (i >= inputs.size()) {
      continue;
    }
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
      MS_LOG(EXCEPTION) << "Shape of output " << i << " is invalid, of which value is " << shape;
    }
    PackageOutput(i, shape);
  }
  return ret;
}

void AclKernelMod::RefreshAclConverter(const std::vector<KernelTensor *> &inputs) {
  MS_EXCEPTION_IF_NULL(converter_);
  converter_->Reset();
  converter_->ConvertMsIdxToGeIdx(primitive_, inputs);
  converter_->ConvertAttrToAclInput(primitive_->attrs(), kernel_name_, &input_params_);
  converter_->ConvertInputToAclAttr(inputs, kernel_name_);
}

int AclKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  RefreshAclConverter(inputs);
  GetInputInfo(inputs);
  int ret = GetOutputInfo(outputs);
  return ret;
}

std::string AclKernelMod::DebugString() const {
  if (!device::ascend::AclHelper::IsPrintDebugString()) {
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
  auto info = device::ascend::GeAdapterManager::GetInstance().GetInfo(kernel_name_, true);
  MS_EXCEPTION_IF_NULL(info);

  value_depend_args_.clear();
  for (auto ms_proto_idx : indices) {
    if (info->input_attr_map().count(ms_proto_idx) == 0) {
      value_depend_args_.emplace(ms_proto_idx);
    }
  }
}

bool AclKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }

  // Process value depend arguments, value depend arguments reside both on device and host (which were synchronized at
  // type and shape inference stage). Inside the ACL internal, it may also need to sync there arguments to host for
  // operator validation (e.g. ReduceSum), so put it on host will be more efficiencient to reduce the count of sync from
  // device to host.
  MS_EXCEPTION_IF_NULL(converter_);
  auto is_need_skip_execute = converter_->IsNeedSkipExecute(kernel_name_, inputs, outputs, stream_ptr);
  if (is_need_skip_execute) {
    return true;
  }
  SetParamsDataTypeIfComplexInput(primitive_, &input_params_, &output_params_);
  converter_->ConvertValueDependToHostInput(kernel_name_, inputs, input_params_, value_depend_args_);
  converter_->ConvertToAclInput(primitive_, inputs, input_params_);
  converter_->ConvertToAclOutput(primitive_, outputs, output_params_);
  converter_->SetRunnerSpecialInfo();
  // cppcheck-suppress unreadVariable
  auto lock = device::ascend::AclUtil::LockRuntime(stream_ptr);
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

std::vector<size_t> AclKernelMod::GetLaunchIgnoredInputAddressIdx() const {
  static const std::map<std::string, std::vector<size_t>> launch_ignored_input_addr_idx = {
    {kReshapeOpName, {kIndex1}},
    {"Conv2DBackpropInput", {2}},
    {"ReduceSum", {1}},
    {"ResizeNearestNeighborV2Grad", {1}},
    {"StandardNormal", {0}},
    {"Slice", {1, 2}},
    {"Ones", {0, 1}},
    {"Zeros", {0, 1}},
    {"StridedSlice", {1, 2, 3}},
    {"StridedSliceGrad", {1, 2, 3, 4}},
    {"Print", {0}},
    {"UniformInt", {0}}};
  if (launch_ignored_input_addr_idx.count(kernel_name_) > 0) {
    return launch_ignored_input_addr_idx.at(kernel_name_);
  } else {
    return {};
  }
}

std::vector<size_t> AclKernelMod::GetUseLessOutputIdx() const {
  static const std::map<std::string, std::vector<size_t>> output_use_less_idx = {
    {"Print", {kIndex0}},        {"TensorDump", {kIndex0}},        {"TensorSummary", {kIndex0}},
    {"ImageSummary", {kIndex0}}, {"ScalarSummary", {kIndex0}},     {"HistogramSummary", {kIndex0}},
    {"Send", {kIndex0}},         {"BatchNorm", {kIndex1, kIndex2}}};
  if (output_use_less_idx.count(kernel_name_) > 0) {
    return output_use_less_idx.at(kernel_name_);
  } else {
    return {};
  }
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

  auto in_def_flag =
    primitive_ == nullptr ? true : device::ascend::AclHelper::GetDefaultFormatFlagFromAttr(primitive_, true);
  input_params_.resize(input_device_formats.size());
  for (size_t i = 0; i < input_device_formats.size(); i++) {
    input_params_[i].data_type = input_device_types[i];
    input_params_[i].dev_format = input_device_formats[i];
    input_params_[i].is_default =
      in_def_flag && device::ascend::AclHelper::CheckDefaultSupportFormat(input_device_formats[i]);
    input_params_[i].type_size = GetTypeByte(TypeIdToType(input_params_[i].data_type));
  }
  input_size_list_.resize(input_device_formats.size(), 0);

  auto out_def_flag =
    primitive_ == nullptr ? true : device::ascend::AclHelper::GetDefaultFormatFlagFromAttr(primitive_, false);
  output_params_.resize(output_device_formats.size());
  for (size_t i = 0; i < output_device_formats.size(); i++) {
    output_params_[i].data_type = output_device_types[i];
    output_params_[i].dev_format = output_device_formats[i];
    output_params_[i].is_default =
      out_def_flag && device::ascend::AclHelper::CheckDefaultSupportFormat(output_device_formats[i]);
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
    size_t dtype_byte = GetTypeByte(TypeIdToType(outputs[i]->dtype_id()));
    size_t update_size = LongToSize(
      std::accumulate(output_shape[i].begin(), output_shape[i].end(), dtype_byte, std::multiplies<int64_t>()));
    outputs[i]->set_size(update_size);
  }
}
}  // namespace kernel
}  // namespace mindspore
