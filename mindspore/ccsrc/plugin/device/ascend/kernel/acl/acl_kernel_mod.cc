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

tensor::TensorPtr GetDependValueTensor(const AddressPtr &address, const TypeId type, const ShapeVector &shape,
                                       void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (address != nullptr && address->addr != nullptr) {
    auto tensor = std::make_shared<tensor::Tensor>(type, shape);
    MS_EXCEPTION_IF_NULL(tensor);
    auto status = aclrtMemcpyAsync(tensor->data_c(), tensor->Size(), address->addr, address->size,
                                   ACL_MEMCPY_DEVICE_TO_HOST, stream_ptr);
    if (status != ACL_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "aclrtMemcpyAsync depend tensor failed! tensor size is " << tensor->Size()
                        << " and address size is " << address->size;
    }
    auto sync_status = aclrtSynchronizeStreamWithTimeout(stream_ptr, -1);
    if (sync_status != ACL_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "aclrtSynchronizeStreamWithTimeout depend tensor failed! tensor size is " << tensor->Size()
                        << " and address size is " << address->size;
    }
    return tensor;
  }
  return nullptr;
}
}  // namespace

void AclKernelMod::PackageInput(const size_t idx, const std::string &format, ShapeVector *shape) {
  MS_EXCEPTION_IF_NULL(shape);
  auto &params = input_params_[idx];
  if (!format.empty()) {
    params.ori_format = format;
  } else {
    params.ori_format = transform::AclHelper::ConvertOriginShapeAndFormat(kernel_name_, idx, params.dev_format, shape);
  }

  params.ori_shape = *shape;
  if (!params.is_default) {
    auto groups = transform::AclHelper::GetFracZGroupFromAttr(primitive_ptr_);
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
    auto groups = transform::AclHelper::GetFracZGroupFromAttr(primitive_ptr_);
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

void AclKernelMod::SetPrimitive(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  primitive_ptr_ = primitive;
  kernel_name_ = primitive_ptr_->name();
}

void AclKernelMod::GetInputInfo(const std::vector<KernelTensorPtr> &inputs) {
  if (input_params_.size() != inputs.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Acl kernel's input size is not equal with acl param's size:" << input_params_.size()
                               << " - input's size:" << inputs.size();
  }

  std::string format = transform::AclHelper::GetFormatFromAttr(primitive_ptr_);

  for (size_t i = 0; i < input_params_.size(); i++) {
    auto &input = inputs[i];
    MS_EXCEPTION_IF_NULL(input);
    auto shape = input->GetShapeVector();
    if (!IsValidShape(shape)) {
      // early stop if any input shape contains -1/-2, which means input shape is dynamic
      MS_LOG(INTERNAL_EXCEPTION) << "In Resize function, input shape must be valid!";
    }
    PackageInput(i, format, &shape);
  }
}

int AclKernelMod::GetOutputInfo(const std::vector<KernelTensorPtr> &outputs) {
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
        MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the max_shape should not be empty when input shape is known.";
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
  converter_->ResizeAclOpInputs(primitive_ptr_);
  converter_->ConvertAttrToAclInput(primitive_ptr_->attrs(), kernel_name_, &input_to_host_array_);
  if (!input_to_host_array_.empty()) {
    converter_->ConvertInputToAclAttr(input_to_host_array_, kernel_name_);
  }
  if (transform::AclHelper::IsPrintDebugString()) {
    ms_attr_str_.clear();
    converter_->ConvertToAclAttr(primitive_ptr_->attrs(), kernel_name_, &ms_attr_str_);
  } else {
    converter_->ConvertToAclAttr(primitive_ptr_->attrs(), kernel_name_, nullptr);
  }
  converter_->ProcessRunnerSpecialInfo(kernel_name_, output_params_);
}

int AclKernelMod::Resize(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs,
                         const std::map<uint32_t, tensor::TensorPtr> &inputs_on_host) {
  int ret = KRET_OK;
  primitive_ptr_ = op_->GetPrim();
  MS_ERROR_IF_NULL(primitive_ptr_);
  kernel_name_ = primitive_ptr_->name();

  this->inputs_ = inputs;
  this->outputs_ = outputs;

  GetInputInfo(inputs);
  ret = GetOutputInfo(outputs);
  input_to_host_array_.build(inputs_on_host);
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

bool AclKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }

  if (need_convert_host_tensor_) {
    auto anf_node = anf_node_.lock();
    MS_EXCEPTION_IF_NULL(anf_node);
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    std::set<int64_t> depend_list = abstract::GetValueDependArgIndices(cnode);
    input_to_host_array_.clear();
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (depend_list.find(i) != depend_list.end()) {
        auto input_param = input_params_.at(i);
        auto depended_value = GetDependValueTensor(inputs[i], input_param.data_type, input_param.ori_shape, stream_ptr);
        if (depended_value == nullptr) {
          continue;
        }
        input_to_host_array_.emplace(i, depended_value);
      }
    }
    need_convert_host_tensor_ = false;
    // need recreate converter when first launch in static shape.
    CreateAclConverter();
  }

  MS_EXCEPTION_IF_NULL(converter_);
  converter_->ConvertToAclInput(primitive_ptr_, input_to_host_array_, inputs, input_params_);
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

  if (primitive_ptr_ == nullptr && op_ != nullptr) {
    primitive_ptr_ = op_->GetPrim();
  }
  auto in_def_flag =
    primitive_ptr_ == nullptr ? true : transform::AclHelper::GetDefaultFormatFlagFromAttr(primitive_ptr_, true);
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
    primitive_ptr_ == nullptr ? true : transform::AclHelper::GetDefaultFormatFlagFromAttr(primitive_ptr_, false);
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

std::vector<TaskInfoPtr> AclKernelMod::GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                               const std::vector<AddressPtr> &, uint32_t) {
  MS_LOG(EXCEPTION) << "Acl kernels do not support task sink mode.";
  return {};
}

void AclKernelMod::SyncOutputShape() {
  MS_EXCEPTION_IF_NULL(converter_);
  std::vector<std::vector<int64_t>> output_shape = converter_->SyncData();
  for (size_t i = 0; i < output_shape.size(); ++i) {
    outputs_[i]->SetShapeVector(output_shape[i]);
  }
}

bool AclKernelMod::IsNeedRetrieveOutputShape() {
  MS_EXCEPTION_IF_NULL(converter_);
  return converter_->is_need_retrieve_output_shape();
}
}  // namespace kernel
}  // namespace mindspore
