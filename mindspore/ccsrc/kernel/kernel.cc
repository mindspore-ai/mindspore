/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "kernel/kernel.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <set>

#include "kernel/format_utils.h"
#include "kernel/common_utils.h"
#include "utils/ms_context.h"
#include "include/backend/device_synchronizer_utils.h"

namespace mindspore {
namespace kernel {
constexpr int64_t kInvalidShape = -2;

namespace {
using ShapeTransposeFunc = std::function<void(const ShapeVector *, ShapeVector *)>;

void TransposeDefaultShape(const ShapeVector *host_shape_vector, ShapeVector *device_shape_vector) {
  MS_EXCEPTION_IF_NULL(host_shape_vector);
  MS_EXCEPTION_IF_NULL(device_shape_vector);
  *device_shape_vector = *host_shape_vector;
}

void TransposeNCHWShape(const ShapeVector *host_shape_vector, ShapeVector *device_shape_vector) {
  MS_EXCEPTION_IF_NULL(host_shape_vector);
  MS_EXCEPTION_IF_NULL(device_shape_vector);
  if (host_shape_vector->size() != kDim4) {
    MS_LOG(EXCEPTION) << "The host shape dims should be 4, but got: " << host_shape_vector->size();
  }
  *device_shape_vector = *host_shape_vector;
}

void TransposeNHWCShape(const ShapeVector *host_shape_vector, ShapeVector *device_shape_vector) {
  MS_EXCEPTION_IF_NULL(host_shape_vector);
  MS_EXCEPTION_IF_NULL(device_shape_vector);

  if (host_shape_vector->size() != kDim4) {
    MS_LOG(EXCEPTION) << "The host shape dims should be 4, but got: " << host_shape_vector->size();
  }
  device_shape_vector->resize(kDim4);

  device_shape_vector->at(kIndex0) = host_shape_vector->at(kIndex0);
  device_shape_vector->at(kIndex1) = host_shape_vector->at(kIndex2);
  device_shape_vector->at(kIndex2) = host_shape_vector->at(kIndex3);
  device_shape_vector->at(kIndex3) = host_shape_vector->at(kIndex1);
}
}  // namespace

KernelHostInfo::KernelHostInfo(const KernelHostInfo &other) {
  shape_vector_after_format_trasform_ = other.shape_vector_after_format_trasform_;
  type_id_ = other.type_id_;
  kernel_tensor_value_ = other.kernel_tensor_value_;
}

KernelTensor::KernelTensor() { address_common_ = std::make_shared<AddressCommon>(); }

KernelTensor::KernelTensor(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value) {
  host_info_ = std::make_unique<KernelHostInfo>();
  address_common_ = std::make_shared<AddressCommon>();

  if (type) {
    SetType(type);
  }
  if (shape) {
    // Note: for performance, the function `SetShape` uses host_info_->type_id_, so need to SetType first.
    SetShape(shape);
  }
  if (value) {
    SetValue(value);
  }
}

KernelTensor::KernelTensor(void *device_ptr, size_t size, Format format, TypeId dtype_id, const ShapeVector &host_shape,
                           const string &device_name, uint32_t device_id, const UserDataPtr &user_data)
    : host_shape_(host_shape),
      user_data_(user_data),
      address_common_(
        std::make_shared<AddressCommon>(device_ptr, size, host_shape, format, dtype_id, device_name, device_id)) {
  if (dtype_id == kTypeUnknown) {
    SetType(TypeIdToType(dtype_id));
  } else {
    SetType(std::make_shared<TensorType>(TypeIdToType(dtype_id)));
  }
}

KernelTensor::KernelTensor(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value,
                           void *device_ptr, size_t size, const std::string &format, TypeId dtype_id,
                           const ShapeVector &host_shape, const string &device_name, uint32_t device_id,
                           const UserDataPtr &user_data)
    : KernelTensor(shape, type, value) {
  address_common_->pointer_ref_count_->set_ptr(device_ptr);
  address_common_->size_ = size;
  address_common_->format_ = GetFormatFromStrToEnum(format);
  address_common_->dtype_id_ = dtype_id;
  address_common_->device_name_ = device_name;
  address_common_->device_id_ = device_id;
  host_shape_ = host_shape;
  user_data_ = user_data;
}

KernelTensor::KernelTensor(const AddressCommonPtr &address_common, const abstract::BaseShapePtr &shape,
                           const TypePtr &type, const ValuePtr &value, const ShapeVector &host_shape,
                           const UserDataPtr &user_data)
    : KernelTensor(shape, type, value) {
  address_common_ = address_common;
  host_shape_ = host_shape;
  user_data_ = user_data;
}

KernelTensor::KernelTensor(const KernelTensor &other) {
  // Copy host info.
  shape_ = other.shape_ != nullptr ? other.shape_->Clone() : abstract::kNoShape;
  type_ = other.shape_ != nullptr ? other.type_->Clone() : kTypeAny;
  value_ = other.value_;

  if (other.host_info_) {
    host_info_ = std::make_unique<KernelHostInfo>(*other.host_info_);
    host_info_->kernel_tensor_value_ = other.host_info_->kernel_tensor_value_ != nullptr
                                         ? std::make_shared<KernelTensorValue>(*other.host_info_->kernel_tensor_value_)
                                         : nullptr;
  }

  // Copy device info.
  task_id_on_stream_ = other.task_id_on_stream_;
  address_common_ = std::make_shared<AddressCommon>(*other.address_common_);
  device_synchronizer_ = other.device_synchronizer_;
  host_shape_ = other.host_shape_;
  user_data_ = other.user_data_;
}

inline void KernelTensor::CheckHostInfoValid() {
  if (MS_UNLIKELY(!host_info_)) {
    host_info_ = std::make_unique<KernelHostInfo>();
  }
}
namespace {
ShapeVector GetShapeVectorByBaseShape(const abstract::BaseShapePtr &base_shape) {
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::NoShape>()) {
    return {};
  } else if (base_shape->isa<abstract::Shape>()) {
    return base_shape->cast<abstract::ShapePtr>()->shape();
  } else if (base_shape->isa<abstract::DynamicSequenceShape>()) {
    return {-1};
  } else if (base_shape->isa<abstract::SequenceShape>()) {
    const auto &sequence_shape = base_shape->cast<abstract::SequenceShapePtr>();
    MS_EXCEPTION_IF_NULL(sequence_shape);
    if (sequence_shape->size() == 0) {
      return {0};
    }
    ShapeVector shape_vector = {SizeToLong(sequence_shape->size())};
    const auto &sub_shape_vector = GetShapeVectorByBaseShape(sequence_shape->shape()[0]);
    shape_vector.insert(shape_vector.end(), sub_shape_vector.begin(), sub_shape_vector.end());
    return shape_vector;
  }
  MS_LOG(EXCEPTION) << "Invalid shape:" << base_shape->ToString();
}
}  // namespace

void KernelTensor::SetHostInfo(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value) {
  CheckHostInfoValid();
  if (type) {
    SetType(type);
  }
  if (shape) {
    SetShape(shape);
  }
  if (value) {
    SetValue(value);
  }
}

void KernelTensor::SetShape(const abstract::BaseShapePtr &shape) {
  MS_EXCEPTION_IF_NULL(shape);
  shape_ = shape;
  CheckHostInfoValid();

  // Note: for performance, the function `SetShape` uses host_info_->type_id_, so need to SetType first.
  switch (host_info_->type_id_) {
    case kObjectTypeMapTensorType:
    case kObjectTypeTensorType: {
      // The shape type check will affect the performance. The following check will be deleted after the framework is
      // stable.
      if (shape_->isa<abstract::NoShape>()) {
        address_common_->shape_vector_ = {};
      } else {
        if (!shape_->isa<abstract::TensorShape>()) {
          MS_LOG(EXCEPTION) << "Expected TensorShape for SetShape, but got: " << shape_->type_name() << ", "
                            << shape_->ToString();
        }
        address_common_->shape_vector_ = shape_->GetShapeVector();
      }

      break;
    }

    case kObjectTypeList:
    case kObjectTypeTuple: {
      if (shape->isa<abstract::DynamicSequenceShape>()) {
        address_common_->shape_vector_ = {-1};
        break;
      }
      const auto &seq_shape = shape_->cast<abstract::SequenceShapePtr>();
      if (seq_shape == nullptr) {
        MS_LOG(EXCEPTION) << "Expected SequenceShape for SetShape, but got: " << shape_->type_name() << ", "
                          << shape_->ToString();
      }
      address_common_->shape_vector_.clear();
      address_common_->shape_vector_.push_back(seq_shape->size());
      const auto &shapes = seq_shape->shape();
      if (shapes.empty()) {
        break;
      }
      const auto &element_shape = shapes[0];
      MS_EXCEPTION_IF_NULL(element_shape);
      if (element_shape->isa<abstract::TensorShape>()) {
        const ShapeVector &element_shape_vector = element_shape->GetShapeVector();
        address_common_->shape_vector_.insert(address_common_->shape_vector_.end(), element_shape_vector.begin(),
                                              element_shape_vector.end());
      } else if (element_shape->isa<abstract::SequenceShape>()) {
        const ShapeVector &element_shape_vector = GetShapeVectorByBaseShape(element_shape);
        address_common_->shape_vector_.insert(address_common_->shape_vector_.end(), element_shape_vector.begin(),
                                              element_shape_vector.end());
      }

      break;
    }

    case kTypeUnknown: {
      MS_LOG(EXCEPTION) << "Can not set shape for unknown type, please set correct type for kernel tensor first.";
    }

    default:
      MS_EXCEPTION_IF_NULL(type_);
      MS_LOG(DEBUG) << "Need not set shape for: " << type_->ToString();
  }

  // Update size_ after shape changed.
  // Note: calculate memory size should be executed after 'SetType' and 'SetShape'.
  CalculateMemSize();
}

void KernelTensor::CalculateMemSize() {
  MS_EXCEPTION_IF_NULL(host_info_);
  if (host_info_->type_id_ == kObjectTypeTensorType || host_info_->type_id_ == kObjectTypeTuple ||
      host_info_->type_id_ == kObjectTypeList) {
    // If address_common_->shape_vector_ is a dynamic shape, device_info_->size_ will be 0.
    size_t element_num = SizeOf(address_common_->shape_vector_);
    address_common_->size_ = element_num * UnitSizeInBytes(address_common_->dtype_id_);
  } else if (host_info_->type_id_ == kObjectTypeNumber) {
    address_common_->size_ = UnitSizeInBytes(address_common_->dtype_id_);
  }
}

void KernelTensor::SetShapeVector(const ShapeVector &shape_vector) {
  CheckHostInfoValid();
  if (host_info_->type_id_ == kObjectTypeTensorType || host_info_->type_id_ == kObjectTypeMapTensorType) {
    address_common_->shape_vector_ = shape_vector;
    MS_EXCEPTION_IF_NULL(shape_);
    shape_->SetShapeVector(address_common_->shape_vector_);

    MS_LOG(DEBUG) << "Set shape vector: " << shape_vector
                  << ", the format: " << GetFormatFromEnumToStr(address_common_->format_);
    return;
  }

  if (host_info_->type_id_ == kObjectTypeNumber) {
    if (!shape_vector.empty()) {
      MS_LOG(EXCEPTION) << "For Number Type, shape should be empty, but got " << shape_vector;
    }
    return;
  }

  MS_LOG(EXCEPTION) << "Only support Scalar/Tensor/MapTensor type to set shape vector currently, but got type: "
                    << TypeIdLabel(host_info_->type_id_);
}

void KernelTensor::SetShapeVector(ShapeVector &&shape_vector) {
  CheckHostInfoValid();
  if (host_info_->type_id_ == kObjectTypeTensorType || host_info_->type_id_ == kObjectTypeMapTensorType) {
    address_common_->shape_vector_ = std::move(shape_vector);
    MS_EXCEPTION_IF_NULL(shape_);
    shape_->SetShapeVector(address_common_->shape_vector_);

    MS_LOG(DEBUG) << "Set shape vector: " << shape_vector
                  << ", the format: " << GetFormatFromEnumToStr(address_common_->format_);
    return;
  }

  if (host_info_->type_id_ == kObjectTypeNumber) {
    if (!shape_vector.empty()) {
      MS_LOG(EXCEPTION) << "For Number Type, shape should be empty, but got " << shape_vector;
    }
    return;
  }

  MS_LOG(EXCEPTION) << "Only support Scalar/Tensor/MapTensor type to set shape vector currently, but got type: "
                    << TypeIdLabel(host_info_->type_id_);
}

const ShapeVector &KernelTensor::TransposeToDeviceShape() const {
  MS_EXCEPTION_IF_NULL(host_info_);
  if (host_info_->type_id_ != kObjectTypeTensorType) {
    MS_LOG(EXCEPTION) << "Only TensorType could transpose device shape, but got: " << TypeIdLabel(host_info_->type_id_);
  }

  static const mindspore::HashMap<mindspore::Format, ShapeTransposeFunc> shape_trans_funcs = {
    {Format::DEFAULT_FORMAT, TransposeDefaultShape},
    {Format::NCHW, TransposeNCHWShape},
    {Format::NHWC, TransposeNHWCShape}};

  auto iter = shape_trans_funcs.find(address_common_->format_);
  if (iter == shape_trans_funcs.end()) {
    MS_LOG(EXCEPTION) << "Can not find shape transpose function for format: "
                      << GetFormatFromEnumToStr(address_common_->format_);
  }

  // The shape of the device corresponding to 'address_common_->shape_vector_'. For example, if format is NHWC, the
  // shape of the device and host may be different.
  iter->second(&address_common_->shape_vector_, &host_info_->shape_vector_after_format_trasform_);
  return host_info_->shape_vector_after_format_trasform_;
}

bool KernelTensor::NeedTransposeToDeviceShape() const noexcept {
  static std::set<mindspore::Format> black_list{Format::DEFAULT_FORMAT, Format::NCHW, Format::ND, Format::NCDHW};
  auto it = black_list.find(address_common_->format_);
  return it == black_list.end();
}

const ShapeVector &KernelTensor::GetDeviceShapeVector() const {
  MS_EXCEPTION_IF_NULL(host_info_);
  if (NeedTransposeToDeviceShape()) {
    std::lock_guard<std::mutex> lock(host_info_->shape_transform_mutex_);
    return TransposeToDeviceShape();
  }
  return address_common_->shape_vector_;
}

void KernelTensor::SetType(const TypePtr &type) {
  MS_EXCEPTION_IF_NULL(type);
  CheckHostInfoValid();
  type_ = type;
  host_info_->type_id_ = type_->object_type();
  if (host_info_->type_id_ == kTypeUnknown) {
    host_info_->type_id_ = type_->type_id();
    MS_EXCEPTION_IF_CHECK_FAIL((host_info_->type_id_ != kTypeUnknown),
                               "Got a unknown type id, type info: " + type_->ToString());
  }

  switch (host_info_->type_id_) {
    case kObjectTypeTensorType: {
      auto tensor_type_ptr = type_->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(tensor_type_ptr);
      auto element_type = tensor_type_ptr->element();
      if (element_type) {
        address_common_->dtype_id_ = element_type->type_id();
      }
    } break;

    case kObjectTypeTuple: {
      auto tuple_type = type_->cast<TuplePtr>();
      MS_EXCEPTION_IF_NULL(tuple_type);
      TypePtr element_type = nullptr;
      if (tuple_type->dynamic_len()) {
        element_type = tuple_type->dynamic_element_type();
        if (element_type == nullptr) {
          return;
        }
      } else {
        const TypePtrList &element_types = tuple_type->elements();
        if (element_types.empty()) {
          return;
        }
        element_type = element_types[0];
      }
      SetSequenceDType(element_type);
    } break;

    case kObjectTypeList: {
      auto list_type = type_->cast<ListPtr>();
      MS_EXCEPTION_IF_NULL(list_type);
      TypePtr element_type = nullptr;
      if (list_type->dynamic_len()) {
        element_type = list_type->dynamic_element_type();
        if (element_type == nullptr) {
          return;
        }
      } else {
        const TypePtrList &element_types = list_type->elements();
        if (element_types.empty()) {
          return;
        }
        element_type = element_types[0];
      }
      SetSequenceDType(element_type);
    } break;

    default:
      address_common_->dtype_id_ = type->type_id();
      MS_LOG(DEBUG) << "Set dtype for: " << type->ToString();
  }
}

void KernelTensor::SetSequenceDType(const TypePtr &element_type) {
  MS_EXCEPTION_IF_NULL(element_type);
  if (element_type->object_type() == kObjectTypeTensorType) {
    // Tensor type element.
    auto tensor_type_ptr = element_type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type_ptr);
    auto tensor_element_type = tensor_type_ptr->element();
    if (tensor_element_type) {
      address_common_->dtype_id_ = tensor_element_type->type_id();
    }
  } else if (element_type->object_type() == kObjectTypeNumber) {
    // Scalar type element.
    address_common_->dtype_id_ = element_type->type_id();
  } else if (element_type->object_type() == kObjectTypeString) {
    // String type element.
    address_common_->dtype_id_ = element_type->type_id();
  } else if (element_type->object_type() == kObjectTypeTuple) {
    // Sequence type element.
    auto tuple_type = element_type->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_type);
    if (tuple_type->dynamic_len()) {
      if (tuple_type->dynamic_element_type() == nullptr) {
        return;
      }
      SetSequenceDType(tuple_type->dynamic_element_type());
      return;
    }
    const TypePtrList &element_types = tuple_type->elements();
    if (element_types.empty() || element_types[0] == nullptr) {
      return;
    }
    SetSequenceDType(element_types[0]);
    return;
  } else if (element_type->object_type() == kObjectTypeList) {
    // Sequence type element.
    auto list_type = element_type->cast<ListPtr>();
    MS_EXCEPTION_IF_NULL(list_type);
    if (list_type->dynamic_len()) {
      if (list_type->dynamic_element_type() == nullptr) {
        return;
      }
      SetSequenceDType(list_type->dynamic_element_type());
      return;
    }
    const TypePtrList &element_types = list_type->elements();
    if (element_types.empty() || element_types[0] == nullptr) {
      return;
    }
    SetSequenceDType(element_types[0]);
    return;
  } else {
    MS_LOG(EXCEPTION) << "Unsupported element type[" << element_type->ToString()
                      << "] to set element data type for KernelTensor.";
  }
}

std::string KernelTensor::GetStringFormat() const { return GetFormatFromEnumToStr(address_common_->format_); }

void KernelTensor::SetStringFormat(const std::string &format) {
  address_common_->format_ = GetFormatFromStrToEnum(format);
}

ValuePtr KernelTensor::GetValue() const {
  MS_EXCEPTION_IF_NULL(host_info_);
  std::lock_guard<std::mutex> lock(host_info_->value_mutex_);

  // There is a origin value in KernelTensor(maybe come from a ValueNode).
  if (address_common_->dtype_id_ == kMetaTypeNone) {
    return kNone;
  } else if (value_ && !value_->isa<ValueAny>()) {
    if (host_info_->kernel_tensor_value_ == nullptr) {
      host_info_->kernel_tensor_value_ = ConvertValueToKernelTensorValue(value_);
      return host_info_->kernel_tensor_value_ ? host_info_->kernel_tensor_value_ : value_;
    }
    return host_info_->kernel_tensor_value_;
  }

  // Sync value data from device.
  if (!SyncDataFromDeviceToHost()) {
    MS_LOG(EXCEPTION) << "Sync data from device to host side failed";
  }
  return host_info_->kernel_tensor_value_;
}

const void *KernelTensor::GetValuePtr() {
  CheckHostInfoValid();
  std::lock_guard<std::mutex> lock(host_info_->value_mutex_);

  // There is a origin value in KernelTensor(maybe come from a ValueNode).
  if (address_common_->dtype_id_ == kMetaTypeNone) {
    return nullptr;
  } else if (value_ && !value_->isa<ValueAny>()) {
    if (host_info_->kernel_tensor_value_ == nullptr) {
      host_info_->kernel_tensor_value_ = ConvertValueToKernelTensorValue(value_);
    }
    MS_EXCEPTION_IF_NULL(host_info_->kernel_tensor_value_);
    return host_info_->kernel_tensor_value_->GetDataPtr();
  }

  // Sync value data from device.
  if (!SyncDataFromDeviceToHost()) {
    MS_LOG(EXCEPTION) << "Sync data from device to host side failed";
  }
  return host_info_->kernel_tensor_value_->GetDataPtr();
}

bool KernelTensor::SyncDataFromDeviceToHost() const {
  // Note: must release lock when wait async resize or launch kernel finish, because the kernels' resize and launch
  // tasks which are waited maybe use this kernel's GetValue and try lock this mutex to avoid deadlock.
  host_info_->value_mutex_.unlock();
  WaitAsyncResizeAndLaunchFinish();
  host_info_->value_mutex_.lock();

  void *device_ptr = this->device_ptr();
  if (device_ptr == nullptr) {
    MS_LOG(ERROR) << "Not malloc device memory yet, sync data from device to host side failed, size: "
                  << address_common_->size_;
    return false;
  }

  MS_EXCEPTION_IF_NULL(host_info_);
  // For performance, the CPU back-end does not need to copy the device to host, and directly uses the
  // device pointer in the kernel Tensor.
  if (address_common_->device_name_ == kCPUDevice) {
    if (!host_info_->kernel_tensor_value_) {
      host_info_->kernel_tensor_value_ = std::make_shared<KernelTensorValue>(device_ptr, address_common_->size_, type_);
    } else {
      host_info_->kernel_tensor_value_->SetDataPtr(device_ptr);
      host_info_->kernel_tensor_value_->Resize(address_common_->size_);
    }
    return true;
  }

  if (!host_info_->kernel_tensor_value_) {
    host_info_->kernel_tensor_value_ = std::make_shared<KernelTensorValue>(address_common_->size_, type_);
  } else {
    host_info_->kernel_tensor_value_->Resize(address_common_->size_);
  }

  if (address_common_->size_ == 0) {
    return true;
  }

  void *host_ptr = host_info_->kernel_tensor_value_->GetMutableDataPtr();
  MS_EXCEPTION_IF_NULL(host_ptr);

  MS_EXCEPTION_IF_NULL(device_synchronizer_);
  if (!device_synchronizer_->SyncDeviceToHost(
        host_ptr, device_ptr, address_common_->size_, address_common_->device_name_, address_common_->device_id_,
        address_common_->format_, address_common_->shape_vector_, address_common_->stream_id_, user_data_)) {
    MS_LOG(EXCEPTION) << "Sync data from device to host side failed";
  }
  return true;
}

bool KernelTensor::IsDynamicShape() const {
  const auto &shape = this->GetShapeVector();
  return std::any_of(shape.cbegin(), shape.cend(), [](auto i) { return i < 0; });
}

ShapeVector KernelTensor::GetMaxShape() const {
  MS_EXCEPTION_IF_NULL(host_info_);
  if (host_info_->type_id_ != kObjectTypeTensorType) {
    return {};
  }
  if (shape_ == nullptr || !shape_->isa<abstract::Shape>()) {
    return {};
  }

  return shape_->cast<abstract::ShapePtr>()->max_shape();
}

int KernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KRET_OK;
  workspace_size_list_.clear();
  output_size_list_.clear();

  for (size_t idx = 0; idx < outputs.size(); idx++) {
    auto &output = outputs[idx];
    size_t tensor_size = 0;
    MS_EXCEPTION_IF_NULL(output);
    size_t type_size = UnitSizeInBytes(output->dtype_id());
    if (type_size == 0) {
      MS_LOG(WARNING) << "The type size is 0, type: " << TypeIdToType(output->dtype_id())->ToString();
    }

    const auto &shape = output->GetShapeVector();
    if (!IsValidShape(shape)) {
      MS_LOG(WARNING) << "Invalid shape:" << mindspore::ToString(shape) << ", kernel name:" << kernel_name();
      // Note:
      // If output shape is unknown, the op is a compute-depended op, and the output_size_list_ can be set by default
      // size: type_size.
      tensor_size = type_size;
      ret = KRET_UNKNOWN_OUT_SHAPE;
    } else {
      if (shape.empty()) {
        tensor_size = type_size;
      } else {
        auto cur_out_shape_num = SizeOf(shape);
        tensor_size = cur_out_shape_num * type_size;
        if (type_size != 0 && tensor_size / type_size != cur_out_shape_num) {
          MS_EXCEPTION(ValueError) << "For " << kernel_name_ << ", the shape of outputs[" << output_size_list_.size()
                                   << "]: " << shape
                                   << " is too big, mindspore cannot apply for such a large amount of memory.";
        }
      }
    }
    (void)output_size_list_.emplace_back(tensor_size);
  }
  return static_cast<int>(ret);
}

std::vector<std::vector<int64_t>> GetShapes(const std::vector<KernelTensor *> &tensors) {
  std::vector<std::vector<int64_t>> shapes(tensors.size());
  for (size_t idx = 0; idx < shapes.size(); idx++) {
    shapes[idx] = tensors[idx]->GetShapeVector();
  }
  return shapes;
}

void ConvertLaunchInfoToAddr(const KernelLaunchInfo &launch_info, KernelLaunchAddr *mem_info) {
  (mem_info->inputs_).clear();
  (mem_info->outputs_).clear();
  (mem_info->workspaces_).clear();
  std::transform((launch_info.inputs_).begin(), (launch_info.inputs_).end(), std::back_inserter(mem_info->inputs_),
                 [](const auto &input) { return std::make_shared<Address>(input->device_ptr(), input->size()); });
  std::transform(
    (launch_info.workspaces_).begin(), (launch_info.workspaces_).end(), std::back_inserter(mem_info->workspaces_),
    [](const auto &workspace) { return std::make_shared<Address>(workspace->device_ptr(), workspace->size()); });
  std::transform((launch_info.outputs_).begin(), (launch_info.outputs_).end(), std::back_inserter(mem_info->outputs_),
                 [](const auto &output) { return std::make_shared<Address>(output->device_ptr(), output->size()); });
}

int ConvertReductionForAclnn(Reduction reduction) {
  std::unordered_map<Reduction, int64_t> reduction_map = {
    {Reduction::REDUCTION_SUM, 2}, {Reduction::MEAN, 1}, {Reduction::NONE, 0}};
  auto iter = reduction_map.find(reduction);
  if (iter == reduction_map.end()) {
    MS_LOG(EXCEPTION) << "For ConvertReductionForAclnn, the value of reduction is invalid.";
  }
  return iter->second;
}
}  // namespace kernel
}  // namespace mindspore
