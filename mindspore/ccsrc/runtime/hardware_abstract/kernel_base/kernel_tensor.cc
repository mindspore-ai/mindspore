/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"
#include <utility>
#include <set>
#include <string>
#include <memory>
#include "ir/format_utils.h"
#include "include/utils/utils.h"
#include "include/utils/callback.h"
#include "ops_utils/op_constants.h"
#include "utils/ms_context.h"
#include "utils/convert_utils.h"
#include "ir/dtype/tensor_type.h"

namespace mindspore {
namespace kernel {
namespace {
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

KernelHostInfo::KernelHostInfo(const KernelHostInfo &other) {
  shape_vector_after_format_trasform_ = other.shape_vector_after_format_trasform_;
  type_id_ = other.type_id_;
  kernel_tensor_value_ = other.kernel_tensor_value_;
}

KernelTensor::KernelTensor() {
  device_address_ = std::make_shared<DeviceAddress>();
  ref_cnt_ = std::make_shared<RefCount>();
}

KernelTensor::KernelTensor(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value) {
  host_info_ = std::make_unique<KernelHostInfo>();
  device_address_ = std::make_shared<DeviceAddress>();
  ref_cnt_ = std::make_shared<RefCount>();

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

KernelTensor::KernelTensor(const DeviceAddressPtr &device_address, const TensorStorageInfoPtr &storage_info,
                           TypeId dtype_id, const ShapeVector &host_shape, const UserDataPtr &user_data) {
  MS_EXCEPTION_IF_NULL(device_address);
  device_address_ = device_address;
  shape_vector_ = host_shape;
  tensor_storage_info_ = storage_info;
  user_data_ = user_data;
  if (dtype_id == kTypeUnknown) {
    SetType(TypeIdToType(dtype_id));
  } else {
    SetType(std::make_shared<TensorType>(TypeIdToType(dtype_id)));
  }
  ref_cnt_ = std::make_shared<RefCount>();
}

KernelTensor::KernelTensor(const DeviceAddressPtr &device_address, const TensorStorageInfoPtr &storage_info,
                           const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value,
                           void *device_ptr, size_t size, const std::string &format, TypeId dtype_id,
                           const ShapeVector &host_shape, const string &device_name, const UserDataPtr &user_data)
    : KernelTensor(shape, type, value) {
  MS_EXCEPTION_IF_NULL(device_address);
  tensor_storage_info_ = storage_info;
  device_address_ = device_address;
  device_address_->device_pointer()->set_ptr(device_ptr);
  device_address_->SetSize(size);
  device_address_->SetDeviceType(device::GetDeviceTypeByName(device_name));
  if (IsDynamic(host_shape)) {
    shape_vector_ = host_shape;
  }
  set_format(kernel::GetFormatFromStrToEnum(format));
  dtype_id_ = dtype_id;
  user_data_ = user_data;
  ref_cnt_ = std::make_shared<RefCount>();
}

KernelTensor::KernelTensor(const DeviceAddressPtr &device_address, const TensorStorageInfoPtr &storage_info,
                           const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(device_address);
  device_address_ = device_address;
  tensor_storage_info_ = storage_info;

  host_info_ = std::make_unique<KernelHostInfo>();
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
  ref_cnt_ = std::make_shared<RefCount>();
}

KernelTensor::KernelTensor(const KernelTensor &other) {
  // Copy host info.
  shape_ = other.shape_ != nullptr ? other.shape_->Clone() : abstract::kNoShape;
  type_ = other.type_ != nullptr ? other.type_->Clone() : kTypeAny;
  value_ = other.value_;
  user_data_ = other.user_data_;
  need_sync_user_data_ = other.need_sync_user_data_;
  dtype_id_ = other.dtype_id_;
  shape_vector_ = other.shape_vector_;
  format_ = other.format_;
  if (other.host_info_) {
    host_info_ = std::make_unique<KernelHostInfo>(*other.host_info_);
    host_info_->kernel_tensor_value_ = other.host_info_->kernel_tensor_value_ != nullptr
                                         ? std::make_shared<KernelTensorValue>(*other.host_info_->kernel_tensor_value_)
                                         : nullptr;
  }

  // Copy device info.
  task_id_on_stream_ = other.task_id_on_stream_;
  MS_EXCEPTION_IF_NULL(other.device_address_);
  device_address_ = other.device_address_->CloneDeviceAddress();
  ref_cnt_ = std::make_shared<RefCount>();
  tensor_storage_info_ = other.tensor_storage_info();
}

inline void KernelTensor::CheckHostInfoValid() {
  if (MS_UNLIKELY(!host_info_)) {
    host_info_ = std::make_unique<KernelHostInfo>();
  }
}

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

std::string KernelTensor::ToString() const {
  std::stringstream ofs;
  ofs << this << " shape:" << (GetShape() == nullptr ? "null" : GetShape()->ToString())
      << " type:" << (GetType() == nullptr ? "null" : GetType()->ToString())
      << " value:" << (value_ == nullptr ? "null" : value_->ToString());
  ofs << " flag:" << flag_ << " user data:" << (user_data_ != nullptr)
      << " need sync user data:" << need_sync_user_data_ << " managed by somas:" << managed_by_somas_;
  ofs << "ref count:" << ref_cnt_->ToString() << " dtype:" << TypeIdLabel(dtype_id_) << " shape vector:{";
  std::for_each(shape_vector_.begin(), shape_vector_.end(), [&ofs](int64_t axis) { ofs << axis << " "; });
  ofs << "} format:" << format_;
  if (device_address_ != nullptr) {
    return ofs.str() + " device address:" + device_address_->ToString();
  }
  return ofs.str() + "device address:0";
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
        shape_vector_ = {};
      } else {
        if (!shape_->isa<abstract::TensorShape>()) {
          MS_LOG(EXCEPTION) << "Expected TensorShape for SetShape, but got: " << shape_->type_name() << ", "
                            << shape_->ToString();
        }
        shape_vector_ = shape_->GetShapeVector();
      }

      break;
    }

    case kObjectTypeList:
    case kObjectTypeTuple: {
      if (shape->isa<abstract::DynamicSequenceShape>()) {
        shape_vector_ = {-1};
        break;
      }
      const auto &seq_shape = shape_->cast<abstract::SequenceShapePtr>();
      if (seq_shape == nullptr) {
        MS_LOG(EXCEPTION) << "Expected SequenceShape for SetShape, but got: " << shape_->type_name() << ", "
                          << shape_->ToString();
      }
      shape_vector_ = {SizeToLong(seq_shape->size())};
      const auto &shapes = seq_shape->shape();
      if (shapes.empty()) {
        break;
      }
      const auto &element_shape = shapes[0];
      MS_EXCEPTION_IF_NULL(element_shape);
      auto shape_vector = GetShapeVector();
      if (element_shape->isa<abstract::TensorShape>()) {
        const ShapeVector &element_shape_vector = element_shape->GetShapeVector();
        shape_vector.insert(shape_vector.end(), element_shape_vector.begin(), element_shape_vector.end());
      } else if (element_shape->isa<abstract::SequenceShape>()) {
        const ShapeVector &element_shape_vector = GetShapeVectorByBaseShape(element_shape);
        shape_vector.insert(shape_vector.end(), element_shape_vector.begin(), element_shape_vector.end());
      }
      shape_vector_ = shape_vector;

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
    // If device_address_->GetShapeVector() is a dynamic shape, device_info_->size_ will be 0.
    size_t element_num = SizeOf(GetShapeVector());
    device_address_->SetSize(element_num * UnitSizeInBytes(dtype_id()));
  } else if (host_info_->type_id_ == kObjectTypeNumber) {
    device_address_->SetSize(UnitSizeInBytes(dtype_id()));
  }
}

void KernelTensor::SetShapeVector(const ShapeVector &shape_vector) {
  CheckHostInfoValid();
  if (host_info_->type_id_ == kObjectTypeTensorType || host_info_->type_id_ == kObjectTypeMapTensorType) {
    shape_vector_ = shape_vector;
    MS_EXCEPTION_IF_NULL(shape_);
    shape_->SetShapeVector(shape_vector);

    MS_LOG(DEBUG) << "Set shape vector: " << shape_vector << ", the format: " << format();
    return;
  }

  if (host_info_->type_id_ == kObjectTypeNumber) {
    if (!shape_vector.empty()) {
      MS_LOG(EXCEPTION) << "For Number Type, shape should be empty, but got " << shape_vector;
    }
    return;
  }

  if (host_info_->type_id_ == kObjectTypeString) {
    if (!shape_vector.empty()) {
      MS_LOG(EXCEPTION) << "For String Type, shape should be empty, but got " << shape_vector;
    }
    return;
  }
  if (host_info_->type_id_ == kObjectTypeTuple || host_info_->type_id_ == kObjectTypeList) {
    shape_vector_ = shape_vector;
    return;
  }

  if (host_info_->type_id_ == kMetaTypeNone) {
    return;
  }
  MS_LOG(EXCEPTION) << "Only support Scalar/Tensor/MapTensor type to set shape vector currently, but got type: "
                    << TypeIdLabel(host_info_->type_id_);
}

void KernelTensor::SetShapeVector(ShapeVector &&shape_vector) {
  CheckHostInfoValid();
  if (host_info_->type_id_ == kObjectTypeTensorType || host_info_->type_id_ == kObjectTypeMapTensorType) {
    shape_vector_ = shape_vector;
    MS_EXCEPTION_IF_NULL(shape_);
    shape_->SetShapeVector(GetShapeVector());

    MS_LOG(DEBUG) << "Set shape vector: " << shape_vector << ", the format: " << format();
    return;
  }

  if (host_info_->type_id_ == kObjectTypeNumber) {
    if (!shape_vector.empty()) {
      MS_LOG(EXCEPTION) << "For String Type, shape should be empty, but got " << shape_vector;
    }
    return;
  }

  if (host_info_->type_id_ == kObjectTypeString) {
    if (!shape_vector.empty()) {
      MS_LOG(EXCEPTION) << "For Number Type, shape should be empty, but got " << shape_vector;
    }
    return;
  }

  MS_LOG(EXCEPTION) << "Only support Scalar/Tensor/MapTensor type to set shape vector currently, but got type: "
                    << TypeIdLabel(host_info_->type_id_);
}

using ShapeTransposeFunc = std::function<void(const ShapeVector *, ShapeVector *)>;
const ShapeVector &KernelTensor::TransposeToDeviceShape() const {
  MS_EXCEPTION_IF_NULL(host_info_);
  if (host_info_->type_id_ != kObjectTypeTensorType) {
    MS_LOG(EXCEPTION) << "Only TensorType could transpose device shape, but got: " << TypeIdLabel(host_info_->type_id_);
  }

  static const mindspore::HashMap<mindspore::Format, ShapeTransposeFunc> shape_trans_funcs = {
    {Format::DEFAULT_FORMAT, TransposeDefaultShape},
    {Format::NCHW, TransposeNCHWShape},
    {Format::NHWC, TransposeNHWCShape}};

  auto iter = shape_trans_funcs.find(format());
  if (iter == shape_trans_funcs.end()) {
    MS_LOG(EXCEPTION) << "Can not find shape transpose function for format: " << format();
  }

  // The shape of the device corresponding to 'device_address_->GetShapeVector()'. For example, if format is NHWC, the
  // shape of the device and host may be different.
  iter->second(&GetShapeVector(), &host_info_->shape_vector_after_format_trasform_);
  return host_info_->shape_vector_after_format_trasform_;
}

bool KernelTensor::NeedTransposeToDeviceShape() const noexcept {
  static std::set<mindspore::Format> black_list{Format::DEFAULT_FORMAT, Format::NCHW, Format::ND, Format::NCDHW};
  auto it = black_list.find(format());
  return it == black_list.end();
}

const ShapeVector &KernelTensor::GetDeviceShapeVector() const {
  MS_EXCEPTION_IF_NULL(host_info_);
  if (NeedTransposeToDeviceShape()) {
    std::lock_guard<std::mutex> lock(host_info_->shape_transform_mutex_);
    return TransposeToDeviceShape();
  }
  return GetShapeVector();
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
        set_dtype_id(element_type->type_id());
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
      set_dtype_id(type->type_id());
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
      set_dtype_id(tensor_element_type->type_id());
    }
  } else if (element_type->object_type() == kObjectTypeNumber) {
    // Scalar type element.
    set_dtype_id(element_type->type_id());
  } else if (element_type->object_type() == kObjectTypeString) {
    // String type element.
    set_dtype_id(element_type->type_id());
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

std::string KernelTensor::GetStringFormat() const { return kernel::GetFormatFromEnumToStr(format()); }

void KernelTensor::SetStringFormat(const std::string &format) { set_format(kernel::GetFormatFromStrToEnum(format)); }

ValuePtr KernelTensor::GetValue() const {
  MS_EXCEPTION_IF_NULL(host_info_);
  std::lock_guard<std::mutex> lock(host_info_->value_mutex_);

  // There is a origin value in KernelTensor(maybe come from a ValueNode).
  if (dtype_id() == kMetaTypeNone) {
    return kNone;
  }
  if (!SetKernelTensorValue()) {
    MS_LOG(EXCEPTION) << "Failed to get value from kernel tensor:" << this->ToString() << ", this pointer: " << this;
  }
  return host_info_->kernel_tensor_value_ != nullptr ? host_info_->kernel_tensor_value_ : value_;
}

const void *KernelTensor::GetValuePtr() {
  CheckHostInfoValid();
  std::lock_guard<std::mutex> lock(host_info_->value_mutex_);

  // There is a origin value in KernelTensor(maybe come from a ValueNode).
  if (dtype_id() == kMetaTypeNone) {
    return nullptr;
  }
  if (!SetKernelTensorValue()) {
    MS_LOG(EXCEPTION) << "Failed to get value from kernel tensor:" << this->ToString() << ", this pointer: " << this;
  }
  MS_EXCEPTION_IF_NULL(host_info_->kernel_tensor_value_);
  return host_info_->kernel_tensor_value_->GetDataPtr();
}

bool KernelTensor::SyncDataFromDeviceToHost() const {
  // Note: must release lock when wait async resize or launch kernel finish, because the kernels' resize and launch
  // tasks which are waited maybe use this kernel's GetValue and try lock this mutex to avoid deadlock.
  host_info_->value_mutex_.unlock();
  constexpr char kWaitAsyncResizeAndLaunchFinishCallback[] = "WaitAsyncResizeAndLaunchFinish";
  static const auto wait_resize_and_launch_finish =
    callback::CommonCallback::GetInstance().GetCallback<void>(kWaitAsyncResizeAndLaunchFinishCallback);
  if (wait_resize_and_launch_finish) {
    wait_resize_and_launch_finish();
  }
  host_info_->value_mutex_.lock();

  void *device_ptr = this->device_ptr();
  if (device_ptr == nullptr) {
    MS_LOG(INFO) << "Not malloc device memory yet, sync data from device to host side failed, size: "
                 << device_address_->size();
    return false;
  }

  MS_EXCEPTION_IF_NULL(host_info_);
  // For performance, the CPU back-end does not need to copy the device to host, and directly uses the
  // device pointer in the kernel Tensor.
  if (device_address_->GetDeviceType() == device::DeviceType::kCPU) {
    if (!host_info_->kernel_tensor_value_) {
      host_info_->kernel_tensor_value_ =
        std::make_shared<KernelTensorValue>(device_ptr, device_address_->size(), type_);
    } else {
      host_info_->kernel_tensor_value_->SetDataPtr(device_ptr);
      host_info_->kernel_tensor_value_->Resize(device_address_->size());
    }
    return true;
  }

  if (!host_info_->kernel_tensor_value_) {
    host_info_->kernel_tensor_value_ = std::make_shared<KernelTensorValue>(device_address_->size(), type_);
  } else {
    host_info_->kernel_tensor_value_->Resize(device_address_->size());
  }

  if (device_address_->size() == 0) {
    return true;
  }

  void *host_ptr = host_info_->kernel_tensor_value_->GetMutableDataPtr();
  MS_EXCEPTION_IF_NULL(host_ptr);

  MS_EXCEPTION_IF_NULL(device_address_);
  const auto &tensor_storage_info = this->tensor_storage_info();
  if (tensor_storage_info != nullptr && (SizeOf(tensor_storage_info->shape) != SizeOf(tensor_storage_info->ori_shape) ||
                                         !tensor_storage_info->is_contiguous)) {
    MS_LOG(EXCEPTION) << "Not support get value from non-contiguous input:" << ToString();
  }
  if (!CopyToHost(device_address_->GetDeviceType(), host_ptr, device_ptr, device_address_->size(),
                  device_address_->stream_id())) {
    MS_LOG(EXCEPTION) << "Sync data from device to host side failed, device type:" << device_address_->GetDeviceType();
  }
  return true;
}

bool KernelTensor::SetKernelTensorValue() const {
  // The tensor is const value
  if (value_ != nullptr && !value_->isa<ValueAny>()) {
    if (host_info_->kernel_tensor_value_ == nullptr) {
      host_info_->kernel_tensor_value_ = ConvertValueToKernelTensorValue(value_);
    }
    return true;
  }

  // The tensor is variable value that is set in user_data.
  if (user_data() != nullptr) {
    auto var_host_value = user_data()->get<std::pair<ValuePtr, bool>>("variable_host_value");
    if (var_host_value != nullptr) {
      if (var_host_value->second) {
        MS_LOG(DEBUG) << "Set kernel_tensor_value from host value in user data: " << var_host_value->first->ToString();
        host_info_->kernel_tensor_value_ = ConvertValueToKernelTensorValue(var_host_value->first);
        var_host_value->second = false;
      }
      return true;
    }
    // Set user data for PyExecute infer.
    if (user_data()->has(kGetValueByUserDataHandler)) {
      const auto &handler = user_data()->get<ValuePtr (*)(const UserDataPtr &)>(kGetValueByUserDataHandler);
      if (handler != nullptr) {
        auto value = (*handler)(user_data());
        if (value != nullptr) {
          host_info_->kernel_tensor_value_ = ConvertValueToKernelTensorValue(value);
          return true;
        }
      }
    }
  }

  // Sync value data from device.
  if (!SyncDataFromDeviceToHost()) {
    MS_LOG(INFO) << "Sync data from device to host side failed";
    return false;
  }
  return true;
}

void KernelTensor::set_device_ptr(void *ptr) { device_address_->device_pointer()->set_ptr(ptr); }
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

const DeviceAddressPtr &KernelTensor::device_address() const { return device_address_; }
void KernelTensor::set_device_address(const DeviceAddressPtr &device_address) { device_address_ = device_address; }

ContinuousKernelTensorsPtr KernelTensor::continuous_kernel_tensors() const { return continuous_kernel_tensors_; }

void KernelTensor::set_continuous_kernel_tensors(const ContinuousKernelTensorsPtr &continuous_kernel_tensors) {
  continuous_kernel_tensors_ = continuous_kernel_tensors;
}

mindspore::Format KernelTensor::format() const { return format_; }
void KernelTensor::set_format(mindspore::Format format) { format_ = format; }

size_t KernelTensor::flag() const { return flag_; }

void KernelTensor::set_flag(size_t flag) { flag_ = flag; }

void KernelTensor::UpdateFlag(size_t flag) { SET_FLAG(flag_, flag); }

void KernelTensor::ClearFlag(size_t flag) { CLEAR_FLAG(flag_, flag); }

bool KernelTensor::IsNotNeedAlloc() const {
  return device_address_->IsPtrValid() || TEST_FLAG(flag(), device::kDeviceAddressFlagNotUsed);
}

bool KernelTensor::IsNotNeedAllocWOLock() const {
  return (device_ptr() != nullptr) || TEST_FLAG(flag(), device::kDeviceAddressFlagNotUsed);
}

// Return the valid device ptr.
void *KernelTensor::GetValidPtr(size_t) {
  if (user_data_ == nullptr || (!need_sync_user_data_)) {
    return device_ptr();
  }
  std::lock_guard<std::mutex> lock(ptr_mutex_);
  if (!need_sync_user_data_) {
    return device_ptr();
  }
  auto sync_handler = user_data()->get<SyncUserDataHandler>(kSyncUserDataHandler);
  if (sync_handler == nullptr) {
    MS_LOG(WARNING) << "For device address:" << this << ", the sync user data handler is null.";
    return device_ptr();
  }
  (*sync_handler)(this);
  need_sync_user_data_ = false;
  return device_ptr();
}

bool KernelTensor::is_ptr_persisted() const { return ref_cnt_->is_ptr_persisted_; }

void KernelTensor::set_is_ptr_persisted(bool is_ptr_persisted) { ref_cnt_->is_ptr_persisted_ = is_ptr_persisted; }

void KernelTensor::IncreaseNewRefCount(const std::string &op_name, size_t i) {
  IncreaseNewRefCount(i);
  MS_LOG(DEBUG) << "Op:" << op_name << " increase new ref count for device address:" << ToString();
}
size_t KernelTensor::DecreaseNewRefCount(const std::string &op_name) {
  size_t ref_count = DecreaseNewRefCount();
  MS_LOG(DEBUG) << "Op:" << op_name << " decrease new ref count for device address:" << ToString();
  return ref_count;
}

// The related interface of static reference count operation.
void KernelTensor::set_original_ref_count(size_t original_ref_count) {
  ref_cnt_->original_ref_count_ = original_ref_count;
}
size_t KernelTensor::original_ref_count() const { return ref_cnt_->original_ref_count_; }
void KernelTensor::set_ref_count(size_t ref_count) { ref_cnt_->ref_count_ = ref_count; }
size_t KernelTensor::ref_count() const { return ref_cnt_->ref_count_.load(); }
void KernelTensor::IncreaseOriginalRefCount() {
  if (ref_cnt_->original_ref_count_ < SIZE_MAX) {
    ref_cnt_->original_ref_count_++;
  }
}
void KernelTensor::DecreaseOriginalRefCount() {
  if ((ref_cnt_->original_ref_count_ < SIZE_MAX) && (ref_cnt_->original_ref_count_ > 0)) {
    ref_cnt_->original_ref_count_--;
  }
}

void KernelTensor::IncreaseRefCount(size_t increase_cnt) {
  if (ref_count() < SIZE_MAX && (SIZE_MAX - ref_count()) > increase_cnt) {
    ref_cnt_->ref_count_ += increase_cnt;
    return;
  }
  MS_LOG(EXCEPTION) << "The reference count is:" << ref_count() << ", and can't add: " << increase_cnt << " more.";
}
size_t KernelTensor::DecreaseRefCount() { return --ref_cnt_->ref_count_; }
void KernelTensor::ResetRefCount() { ref_cnt_->ref_count_ = ref_cnt_->original_ref_count_; }

// The related interface of dynamic reference count operation.
void KernelTensor::set_dynamic_ref_count(int32_t dynamic_ref_count) {
  ref_cnt_->dynamic_ref_count_ = dynamic_ref_count;
}
int32_t KernelTensor::dynamic_ref_count() const { return ref_cnt_->dynamic_ref_count_; }

void KernelTensor::IncreaseDynamicRefCount(const std::string &op_object, int32_t increase_cnt) {
  if (ref_cnt_->dynamic_ref_count_ < INT32_MAX && (INT32_MAX - ref_cnt_->dynamic_ref_count_) > increase_cnt) {
    auto ret = ref_cnt_->dynamic_ref_count_.fetch_add(increase_cnt) + increase_cnt;
    MS_LOG(DEBUG) << op_object << " increases dynamic ref count to:" << ret << " for ptr:" << device_ptr();
    return;
  }
  MS_LOG(EXCEPTION) << "The dynamic reference count is:" << ref_cnt_->dynamic_ref_count_
                    << ", and can't add: " << increase_cnt << " more.";
}
void KernelTensor::IncreaseDynamicRefCount(const std::string &op_object) {
  if (ref_cnt_->dynamic_ref_count_ < INT32_MAX) {
    auto ret = ++ref_cnt_->dynamic_ref_count_;
    MS_LOG(DEBUG) << op_object << " increases dynamic ref count to:" << ret << " for ptr:" << device_ptr();
  }
}
int32_t KernelTensor::DecreaseDynamicRefCount(const std::string &op_object) {
  if (ref_cnt_->dynamic_ref_count_ <= 0) {
    MS_LOG(EXCEPTION) << "The dynamic reference count is invalid value:" << ref_cnt_->dynamic_ref_count_;
  }
  auto ret = --ref_cnt_->dynamic_ref_count_;
  MS_LOG(DEBUG) << op_object << " The dynamic ref count decreases to:" << ret << " for ptr:" << device_ptr();
  return ret;
}

// New ref count interface.
void KernelTensor::IncreaseNewRefCount(size_t i) {
  if (ref_cnt_->new_ref_count_ < SIZE_MAX) {
    ref_cnt_->new_ref_count_ += i;
  }
}
size_t KernelTensor::DecreaseNewRefCount() {
  if (ref_cnt_->new_ref_count_ == 0) {
    MS_LOG(EXCEPTION) << "Failed to decrease ref count:" << this;
  }
  if (ref_cnt_->new_ref_count_ == SIZE_MAX) {
    return SIZE_MAX;
  }
  return --ref_cnt_->new_ref_count_;
}
void KernelTensor::set_new_ref_count(size_t new_ref_count) { ref_cnt_->new_ref_count_ = new_ref_count; }
size_t KernelTensor::new_ref_count() const { return ref_cnt_->new_ref_count_.load(); }

void KernelTensor::set_pointer_ref_count(KernelTensor *const other) {
  if (other->device_address() == nullptr) {
    MS_LOG(WARNING) << "Kernel tensor: " << this << " has no device address.";
  }
  MS_LOG(DEBUG) << "Kernel tensor: " << this->ToString()
                << ", set pointer ref count from kernel tensor: " << other->ToString();
  auto other_device_address = other->device_address();
  device_address_->set_device_pointer(other_device_address->device_pointer());
  ref_cnt_ = other->ref_cnt_;
}
}  // namespace kernel

bool SyncCopy(kernel::KernelTensor *const dst_kernel_tensor, kernel::KernelTensor *const src_kernel_tensor,
              size_t stream_id) {
  MS_EXCEPTION_IF_NULL(dst_kernel_tensor);
  MS_EXCEPTION_IF_NULL(src_kernel_tensor);
  MS_EXCEPTION_IF_NULL(dst_kernel_tensor->device_address());
  MS_EXCEPTION_IF_NULL(src_kernel_tensor->device_address());
  DeviceAddressExtPtr src_ext =
    std::make_shared<DeviceAddressExt>(src_kernel_tensor->format(), src_kernel_tensor->dtype_id(),
                                       src_kernel_tensor->GetShapeVector(), src_kernel_tensor->tensor_storage_info());
  DeviceAddressExtPtr dst_ext =
    std::make_shared<DeviceAddressExt>(dst_kernel_tensor->format(), dst_kernel_tensor->dtype_id(),
                                       dst_kernel_tensor->GetShapeVector(), dst_kernel_tensor->tensor_storage_info());
  MS_LOG(DEBUG) << "Sync copy from kernel tensor:" << src_kernel_tensor->ToString()
                << " src ext:" << src_ext->ToString() << " to:" << dst_kernel_tensor->ToString()
                << " dst ext:" << dst_ext->ToString() << " stream id:" << stream_id;
  return SyncCopy(dst_kernel_tensor->device_address(), src_kernel_tensor->device_address(), stream_id, src_ext,
                  dst_ext);
}

bool AsyncCopy(kernel::KernelTensor *const dst_kernel_tensor, kernel::KernelTensor *const src_kernel_tensor,
               size_t stream_id, bool keep_src) {
  MS_EXCEPTION_IF_NULL(dst_kernel_tensor);
  MS_EXCEPTION_IF_NULL(src_kernel_tensor);
  MS_EXCEPTION_IF_NULL(dst_kernel_tensor->device_address());
  MS_EXCEPTION_IF_NULL(src_kernel_tensor->device_address());
  DeviceAddressExtPtr src_ext =
    std::make_shared<DeviceAddressExt>(src_kernel_tensor->format(), src_kernel_tensor->dtype_id(),
                                       src_kernel_tensor->GetShapeVector(), src_kernel_tensor->tensor_storage_info());
  DeviceAddressExtPtr dst_ext =
    std::make_shared<DeviceAddressExt>(dst_kernel_tensor->format(), dst_kernel_tensor->dtype_id(),
                                       dst_kernel_tensor->GetShapeVector(), dst_kernel_tensor->tensor_storage_info());
  MS_LOG(DEBUG) << "Async copy from kernel tensor:" << src_kernel_tensor->ToString()
                << " src ext:" << src_ext->ToString() << " to:" << dst_kernel_tensor->ToString()
                << " dst ext:" << dst_ext->ToString() << " stream id:" << stream_id;
  return AsyncCopy(dst_kernel_tensor->device_address(), src_kernel_tensor->device_address(), stream_id, keep_src,
                   src_ext, dst_ext);
}

bool SyncCopy(kernel::KernelTensor *const dst_kernel_tensor, tensor::Tensor *const src_tensor, size_t stream_id) {
  MS_EXCEPTION_IF_NULL(dst_kernel_tensor);
  MS_EXCEPTION_IF_NULL(src_tensor);
  MS_EXCEPTION_IF_NULL(dst_kernel_tensor->device_address());
  MS_EXCEPTION_IF_NULL(src_tensor->device_address());
  DeviceAddressExtPtr src_ext = std::make_shared<DeviceAddressExt>(src_tensor->format(), src_tensor->data_type(),
                                                                   src_tensor->shape(), src_tensor->storage_info());
  DeviceAddressExtPtr dst_ext =
    std::make_shared<DeviceAddressExt>(dst_kernel_tensor->format(), dst_kernel_tensor->dtype_id(),
                                       dst_kernel_tensor->GetShapeVector(), dst_kernel_tensor->tensor_storage_info());
  MS_LOG(DEBUG) << "Sync copy from tensor:" << src_tensor->ToString() << " addr:" << src_tensor
                << " src ext:" << src_ext->ToString() << " to kernel tensor:" << dst_kernel_tensor->ToString()
                << " dst ext:" << dst_ext->ToString() << " stream id:" << stream_id;
  return SyncCopy(dst_kernel_tensor->device_address(), src_tensor->device_address(), stream_id, src_ext, dst_ext);
}

bool AsyncCopy(kernel::KernelTensor *const dst_kernel_tensor, tensor::Tensor *const src_tensor, size_t stream_id,
               bool keep_src) {
  MS_EXCEPTION_IF_NULL(dst_kernel_tensor);
  MS_EXCEPTION_IF_NULL(src_tensor);
  MS_EXCEPTION_IF_NULL(dst_kernel_tensor->device_address());
  MS_EXCEPTION_IF_NULL(src_tensor->device_address());
  DeviceAddressExtPtr src_ext = std::make_shared<DeviceAddressExt>(src_tensor->format(), src_tensor->data_type(),
                                                                   src_tensor->shape(), src_tensor->storage_info());
  DeviceAddressExtPtr dst_ext =
    std::make_shared<DeviceAddressExt>(dst_kernel_tensor->format(), dst_kernel_tensor->dtype_id(),
                                       dst_kernel_tensor->GetShapeVector(), dst_kernel_tensor->tensor_storage_info());
  MS_LOG(DEBUG) << "Async copy from tensor:" << src_tensor->ToString() << " addr:" << src_tensor
                << " src ext:" << src_ext->ToString() << " to kernel tensor:" << dst_kernel_tensor->ToString()
                << " dst ext:" << dst_ext->ToString() << " stream id:" << stream_id;
  return AsyncCopy(dst_kernel_tensor->device_address(), src_tensor->device_address(), stream_id, keep_src, src_ext,
                   dst_ext);
}

bool SyncCopy(const tensor::TensorPtr &dst_tensor, kernel::KernelTensor *const src_kernel_tensor, size_t stream_id) {
  MS_EXCEPTION_IF_NULL(dst_tensor);
  MS_EXCEPTION_IF_NULL(src_kernel_tensor);
  MS_EXCEPTION_IF_NULL(dst_tensor->device_address());
  MS_EXCEPTION_IF_NULL(src_kernel_tensor->device_address());
  DeviceAddressExtPtr src_ext =
    std::make_shared<DeviceAddressExt>(src_kernel_tensor->format(), src_kernel_tensor->dtype_id(),
                                       src_kernel_tensor->GetShapeVector(), src_kernel_tensor->tensor_storage_info());
  DeviceAddressExtPtr dst_ext = std::make_shared<DeviceAddressExt>(dst_tensor->format(), dst_tensor->data_type(),
                                                                   dst_tensor->shape(), dst_tensor->storage_info());
  MS_LOG(DEBUG) << "Sync copy from kernel tensor:" << src_kernel_tensor->ToString()
                << " src ext:" << src_ext->ToString() << " to tensor:" << dst_tensor->ToString()
                << " addr:" << dst_tensor.get() << " dst ext:" << dst_ext->ToString() << " stream id:" << stream_id;
  return SyncCopy(dst_tensor->device_address(), src_kernel_tensor->device_address(), stream_id, src_ext, dst_ext);
}

bool AsyncCopy(const tensor::TensorPtr &dst_tensor, kernel::KernelTensor *const src_kernel_tensor, size_t stream_id,
               bool keep_src) {
  MS_EXCEPTION_IF_NULL(dst_tensor);
  MS_EXCEPTION_IF_NULL(src_kernel_tensor);
  MS_EXCEPTION_IF_NULL(dst_tensor->device_address());
  MS_EXCEPTION_IF_NULL(src_kernel_tensor->device_address());
  DeviceAddressExtPtr src_ext =
    std::make_shared<DeviceAddressExt>(src_kernel_tensor->format(), src_kernel_tensor->dtype_id(),
                                       src_kernel_tensor->GetShapeVector(), src_kernel_tensor->tensor_storage_info());
  DeviceAddressExtPtr dst_ext = std::make_shared<DeviceAddressExt>(dst_tensor->format(), dst_tensor->data_type(),
                                                                   dst_tensor->shape(), dst_tensor->storage_info());
  MS_LOG(DEBUG) << "Async copy from kernel tensor:" << src_kernel_tensor->ToString()
                << " src ext:" << src_ext->ToString() << " to tensor:" << dst_tensor->ToString()
                << " addr:" << dst_tensor.get() << " dst ext:" << dst_ext->ToString() << " stream id:" << stream_id;
  return AsyncCopy(dst_tensor->device_address(), src_kernel_tensor->device_address(), stream_id, keep_src, src_ext,
                   dst_ext);
}
}  // namespace mindspore
