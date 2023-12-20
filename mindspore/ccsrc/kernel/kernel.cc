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

KernelTensor::KernelTensor() { ptr_ref_cnt_ = std::make_shared<PointerRefCount>(); }

KernelTensor::KernelTensor(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value) {
  if (type) {
    SetType(type);
  }
  if (shape) {
    // Note: for performance, the function `SetShape` uses type_id_, so need to SetType first.
    SetShape(shape);
  }
  if (value) {
    SetValue(value);
  }

  // Update size_ at constructing KernelTensor.
  // Note: calculate memory size should be executed after 'SetType' and 'SetShape'.
  CalculateMemSize();

  ptr_ref_cnt_ = std::make_shared<PointerRefCount>();
}

KernelTensor::KernelTensor(void *device_ptr, size_t size, const std::string &format, TypeId dtype_id,
                           const ShapeVector &host_shape, const string &device_name, uint32_t device_id,
                           const UserDataPtr &user_data)
    : host_shape_(host_shape),
      dtype_id_(dtype_id),
      format_(GetFormatFromStrToEnum(format)),
      ptr_ref_cnt_(std::make_shared<PointerRefCount>(device_ptr)),
      size_(size),
      device_name_(device_name),
      device_id_(device_id),
      user_data_(user_data) {}

KernelTensor::KernelTensor(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value,
                           void *device_ptr, size_t size, const std::string &format, TypeId dtype_id,
                           const ShapeVector &host_shape, const string &device_name, uint32_t device_id,
                           const UserDataPtr &user_data)
    : KernelTensor(shape, type, value) {
  host_shape_ = host_shape;
  dtype_id_ = dtype_id;
  format_ = GetFormatFromStrToEnum(format);
  ptr_ref_cnt_->set_ptr(device_ptr);
  size_ = size;
  device_name_ = device_name;
  device_id_ = device_id;
  user_data_ = user_data;
}

KernelTensor::KernelTensor(const KernelTensor &other) {
  shape_ = other.shape_ != nullptr ? other.shape_->Clone() : abstract::kNoShape;
  type_ = other.shape_ != nullptr ? other.type_->Clone() : kTypeAny;
  value_ = other.value_;
  shape_vector_ = other.shape_vector_;
  device_shape_vector_ = other.device_shape_vector_;
  host_shape_ = other.host_shape_;
  type_id_ = other.type_id_;
  dtype_ = other.dtype_ != nullptr ? other.dtype_->Clone() : kTypeAny;
  dtype_id_ = other.dtype_id_;
  element_size_in_bytes_ = other.element_size_in_bytes_;
  kernel_tensor_value_ =
    other.kernel_tensor_value_ != nullptr ? std::make_shared<KernelTensorValue>(*other.kernel_tensor_value_) : nullptr;
  format_ = other.format_;
  padding_type_ = other.padding_type_;
  // Only copy device pointer, reference count and deleter should be managed by self from initial state.
  ptr_ref_cnt_ = other.ptr_ref_cnt_ != nullptr ? std::make_shared<PointerRefCount>(other.ptr_ref_cnt_->ptr())
                                               : std::make_shared<PointerRefCount>();
  size_ = other.size_;
  device_name_ = other.device_name_;
  device_id_ = other.device_id_;
  stream_id_ = other.stream_id_;
  user_data_ = other.user_data_;
  device_synchronizer_ = other.device_synchronizer_;
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

void KernelTensor::SetShape(const abstract::BaseShapePtr &shape) {
  MS_EXCEPTION_IF_NULL(shape);
  shape_ = shape;
  // Note: for performance, the function `SetShape` uses type_id_, so need to SetType first.
  if (type_id_ == kObjectTypeTensorType || type_id_ == kObjectTypeMapTensorType) {
    // The shape type check will affect the performance. The following check will be deleted after the framework is
    // stable.
    if (!shape_->isa<abstract::TensorShape>()) {
      MS_LOG(EXCEPTION) << "Expected TensorShape for SetShape, but got: " << shape_->type_name() << ", "
                        << shape_->ToString();
    }
    shape_vector_ = shape_->GetShapeVector();
  } else if (type_id_ == kObjectTypeTuple || type_id_ == kObjectTypeList) {
    if (shape->isa<abstract::DynamicSequenceShape>()) {
      shape_vector_ = {-1};
      return;
    }
    const auto &seq_shape = shape_->cast<abstract::SequenceShapePtr>();
    if (seq_shape == nullptr) {
      MS_LOG(EXCEPTION) << "Expected SequenceShape for SetShape, but got: " << shape_->type_name() << ", "
                        << shape_->ToString();
    }
    shape_vector_.clear();
    shape_vector_.push_back(seq_shape->size());
    const auto &shapes = seq_shape->shape();
    if (shapes.empty()) {
      return;
    }
    const auto &element_shape = shapes[0];
    MS_EXCEPTION_IF_NULL(element_shape);
    if (element_shape->isa<abstract::TensorShape>()) {
      const ShapeVector &element_shape_vector = element_shape->GetShapeVector();
      shape_vector_.insert(shape_vector_.end(), element_shape_vector.begin(), element_shape_vector.end());
    } else if (element_shape->isa<abstract::SequenceShape>()) {
      const ShapeVector &element_shape_vector = GetShapeVectorByBaseShape(element_shape);
      shape_vector_.insert(shape_vector_.end(), element_shape_vector.begin(), element_shape_vector.end());
    }
  }
}

void KernelTensor::CalculateMemSize() {
  if (type_id_ == kObjectTypeNumber) {
    size_ = element_size_in_bytes_;
  } else {
    // If shape_vector_ is a dynamic shape, size_ will be 0.
    size_t element_num = SizeOf(shape_vector_);
    size_ = element_num * element_size_in_bytes_;
  }
}

void KernelTensor::SetShapeVector(const ShapeVector &shape_vector) {
  if (type_id_ == kObjectTypeTensorType || type_id_ == kObjectTypeMapTensorType) {
    shape_vector_ = shape_vector;
    shape_->SetShapeVector(shape_vector_);

    MS_LOG(DEBUG) << "Set shape vector: " << shape_vector << ", the format: " << GetFormatFromEnumToStr(format_);
    return;
  }

  if (type_id_ == kObjectTypeNumber) {
    if (!shape_vector.empty()) {
      MS_LOG(EXCEPTION) << "For Number Type, shape should be empty, but got " << shape_vector;
    }
    return;
  }

  MS_LOG(EXCEPTION) << "Only support Scalar/Tensor/MapTensor type to set shape vector currently, but got type: "
                    << TypeIdLabel(type_id_)
                    << ", please use KernelTensor::SetShape(const abstract::BaseShapePtr &shape) instead.";
}

void KernelTensor::SetShapeVector(ShapeVector &&shape_vector) {
  if (type_id_ == kObjectTypeTensorType || type_id_ == kObjectTypeMapTensorType) {
    shape_vector_ = std::move(shape_vector);
    shape_->SetShapeVector(shape_vector_);

    MS_LOG(DEBUG) << "Set shape vector: " << shape_vector << ", the format: " << GetFormatFromEnumToStr(format_);
    return;
  }

  if (type_id_ == kObjectTypeNumber) {
    if (!shape_vector.empty()) {
      MS_LOG(EXCEPTION) << "For Number Type, shape should be empty, but got " << shape_vector;
    }
    return;
  }

  MS_LOG(EXCEPTION) << "Only support Scalar/Tensor/MapTensor type to set shape vector currently, but got type: "
                    << TypeIdLabel(type_id_)
                    << ", please use KernelTensor::SetShape(const abstract::BaseShapePtr &shape) instead.";
}

void KernelTensor::TransposeToDeviceShape() const {
  if (type_id_ != kObjectTypeTensorType) {
    MS_LOG(EXCEPTION) << "Only TensorType could transpose device shape, but got: " << TypeIdLabel(type_id_);
  }

  static const mindspore::HashMap<mindspore::Format, ShapeTransposeFunc> shape_trans_funcs = {
    {Format::DEFAULT_FORMAT, TransposeDefaultShape},
    {Format::NCHW, TransposeNCHWShape},
    {Format::NHWC, TransposeNHWCShape}};

  auto iter = shape_trans_funcs.find(format_);
  if (iter == shape_trans_funcs.end()) {
    MS_LOG(EXCEPTION) << "Can not find shape transpose function for format: " << GetFormatFromEnumToStr(format_);
  }

  iter->second(&shape_vector_, &device_shape_vector_);
}

bool KernelTensor::NeedTransposeToDeviceShape() const noexcept {
  static std::set<mindspore::Format> black_list{Format::DEFAULT_FORMAT, Format::NCHW, Format::ND, Format::NCDHW};
  auto it = black_list.find(format_);
  return it == black_list.end();
}

const ShapeVector &KernelTensor::GetDeviceShapeVector() const {
  if (NeedTransposeToDeviceShape()) {
    std::lock_guard<std::mutex> lock(device_shape_mutex_);
    TransposeToDeviceShape();
    return device_shape_vector_;
  }
  return shape_vector_;
}

void KernelTensor::SetType(const TypePtr &type) {
  MS_EXCEPTION_IF_NULL(type);
  type_ = type;
  type_id_ = type_->object_type();

  switch (type_id_) {
    case kObjectTypeTensorType: {
      auto tensor_type_ptr = type_->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(tensor_type_ptr);
      auto element_type = tensor_type_ptr->element();
      if (element_type) {
        dtype_ = element_type;
        dtype_id_ = dtype_->type_id();
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

    case kObjectTypeNumber:
    case kObjectTypeString:
    case kObjectTypeMapTensorType:
    case kObjectTypeUMonad:
    case kObjectTypeIOMonad: {
      dtype_ = type;
      dtype_id_ = dtype_->type_id();
    } break;

    default:
      MS_LOG(DEBUG) << "Need not set dtype for: " << type->ToString();
  }

  element_size_in_bytes_ = GetTypeByte(dtype_);
}

void KernelTensor::SetSequenceDType(const TypePtr &element_type) {
  MS_EXCEPTION_IF_NULL(element_type);
  if (element_type->object_type() == kObjectTypeTensorType) {
    // Tensor type element.
    auto tensor_type_ptr = element_type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type_ptr);
    auto tensor_element_type = tensor_type_ptr->element();
    if (tensor_element_type) {
      dtype_ = tensor_element_type;
      dtype_id_ = dtype_->type_id();
    }
  } else if (element_type->object_type() == kObjectTypeNumber) {
    // Scalar type element.
    dtype_ = element_type;
    dtype_id_ = dtype_->type_id();
  } else if (element_type->object_type() == kObjectTypeString) {
    // String type element.
    dtype_ = element_type;
    dtype_id_ = dtype_->type_id();
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

std::string KernelTensor::GetStringFormat() const { return GetFormatFromEnumToStr(format_); }

void KernelTensor::SetStringFormat(const std::string &format) { format_ = GetFormatFromStrToEnum(format); }

const std::string &KernelTensor::padding_type() const { return padding_type_; }

void KernelTensor::set_padding_type(const std::string &padding_type) { padding_type_ = padding_type; }

bool KernelTensor::SyncDataFromDeviceToHost() const {
  void *device_ptr = this->device_ptr();
  if (device_ptr == nullptr) {
    MS_LOG(ERROR) << "Not malloc device memory yet, sync data from device to host side failed, size: " << size_;
    return false;
  }

  // For performance, the CPU back-end does not need to copy the device to host, and directly uses the
  // device pointer in the kernel Tensor.
  if (device_name_ == kCPUDevice) {
    if (!kernel_tensor_value_) {
      kernel_tensor_value_ = std::make_shared<KernelTensorValue>(device_ptr, size_, type_);
    } else {
      kernel_tensor_value_->SetDataPtr(device_ptr);
      kernel_tensor_value_->Resize(size_);
    }
    return true;
  }

  if (!kernel_tensor_value_) {
    kernel_tensor_value_ = std::make_shared<KernelTensorValue>(size_, type_);
  } else {
    kernel_tensor_value_->Resize(size_);
  }

  if (size_ == 0) {
    return true;
  }

  void *host_ptr = kernel_tensor_value_->GetMutableDataPtr();
  MS_EXCEPTION_IF_NULL(host_ptr);

  MS_EXCEPTION_IF_NULL(device_synchronizer_);
  if (!device_synchronizer_->SyncDeviceToHost(host_ptr, device_ptr, size_, device_name_, device_id_, format_,
                                              shape_vector_, stream_id_, user_data_)) {
    MS_LOG(EXCEPTION) << "Sync data from device to host side failed";
  }
  return true;
}

string KernelTensor::GetAbstractName() const {
  const TensorInfo &info = std::get<TensorInfo>(meta_);
  if (info.base_ == nullptr) {
    return "null(no abstract base)";
  }
  return info.base_->ToString();
}

bool KernelTensor::IsDynamicShape() const {
  const auto &shape = this->GetShapeVector();
  return std::any_of(shape.cbegin(), shape.cend(), [](auto i) { return i < 0; });
}

size_t KernelTensor::GetSizeInBytes() const {
  auto unit_size = GetTypeByte(TypeIdToType(GetDtype()));
  auto shapes = this->GetShapeVector();
  if (shapes.size() == 0) {
    return unit_size;
  }

  auto cur_size = unit_size;
  for (const auto val : shapes) {
    if (val < 0) {
      MS_LOG_EXCEPTION << "Invalid shape value " << val << " for calculating size. Abstract name: " << GetAbstractName()
                       << ". Please contact MindSpore support.";
    }
    if (val == 0) {
      MS_LOG_WARNING << "One dim of the shape is 0. Abstract name: " << GetAbstractName() << ".";
    }
    cur_size *= val;
  }

  return cur_size;
}

TypeId GetSeqElementsDtype(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (!abs->isa<abstract::AbstractSequence>()) {
    return TypeId::kTypeUnknown;
  }
  TypePtr type_ptr;
  auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abs);
  if (seq_abs->dynamic_len()) {
    if (seq_abs->dynamic_len_element_abs() == nullptr) {
      return TypeId::kTypeUnknown;
    }
    type_ptr = seq_abs->dynamic_len_element_abs()->BuildType();
  } else {
    if (seq_abs->elements().empty() || seq_abs->elements()[0] == nullptr) {
      return TypeId::kTypeUnknown;
    }
    type_ptr = seq_abs->elements()[0]->BuildType();
  }
  MS_EXCEPTION_IF_NULL(type_ptr);
  if (type_ptr->isa<TensorType>()) {
    auto tensor_ptr = type_ptr->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    auto elem = tensor_ptr->element();
    if (elem == nullptr) {
      return TypeId::kTypeUnknown;
    }
    return elem->type_id();
  }
  return type_ptr->type_id();
}

TypeId KernelTensor::GetDtype() const {
  if (meta_type_ == kObjectTypeNumber) {
    // Scalar
    const ScalarInfo &info = std::get<ScalarInfo>(meta_);
    return info.base_->BuildType()->type_id();
  } else if (meta_type_ == kObjectTypeTuple) {
    // Tuple
    const TupleInfo &info = std::get<TupleInfo>(meta_);
    return GetSeqElementsDtype(info.base_);
  } else if (meta_type_ == kObjectTypeList) {
    // List
    const ListInfo &info = std::get<ListInfo>(meta_);
    return GetSeqElementsDtype(info.base_);
  } else if (meta_type_ == kMetaTypeNone) {
    return TypeId::kMetaTypeNone;
  } else {
    // Tensor
    const TensorInfo &info = std::get<TensorInfo>(meta_);
    if (info.base_ == nullptr) {
      return TypeId::kTypeUnknown;
    }
    auto type_ptr = info.base_->BuildType();
    if (type_ptr == nullptr || !type_ptr->isa<TensorType>()) {
      return TypeId::kTypeUnknown;
    }
    auto tensor_ptr = type_ptr->cast<TensorTypePtr>();
    auto elem = tensor_ptr->element();
    if (elem == nullptr) {
      return TypeId::kTypeUnknown;
    }
    return elem->type_id();
  }
  return kTypeUnknown;
}

ShapeVector GetSequenceFlattenShape(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (!abs->isa<abstract::AbstractSequence>()) {
    return {};
  }
  auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abs);
  if (seq_abs->dynamic_len()) {
    return {-1};
  }
  if (seq_abs->elements().empty() || seq_abs->elements()[0] == nullptr) {
    MS_LOG(INFO) << "Empty sequence abstract:" << seq_abs->ToString();
    return {0};
  }
  auto type_ptr = seq_abs->elements()[0]->BuildType();
  MS_EXCEPTION_IF_NULL(type_ptr);
  if (!type_ptr->isa<TensorType>()) {
    return {static_cast<int64_t>(seq_abs->elements().size())};
  }
  // for tuple of tensor, the tensors shape must be same
  ShapeVector flatten_shp;
  (void)flatten_shp.emplace_back(seq_abs->elements().size());
  if (seq_abs->elements().empty()) {
    return flatten_shp;
  }
  auto tensor_shp_ptr = seq_abs->elements()[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(tensor_shp_ptr);
  MS_LOG(DEBUG) << "tensor shape:" << tensor_shp_ptr->ToString() << " for abstract:" << abs->ToString();
  MS_EXCEPTION_IF_NULL(tensor_shp_ptr->cast<abstract::ShapePtr>());
  auto shape = tensor_shp_ptr->cast<abstract::ShapePtr>()->shape();
  (void)flatten_shp.insert(flatten_shp.end(), shape.begin(), shape.end());
  return flatten_shp;
}

ShapeVector KernelTensor::GetMaxShape() const {
  if (meta_type_ != kObjectTypeTensorType) {
    return {};
  }
  auto base_shape_ptr = GetBaseShape();
  if (base_shape_ptr == nullptr || !base_shape_ptr->isa<abstract::Shape>()) {
    return {};
  }

  return base_shape_ptr->cast<abstract::ShapePtr>()->max_shape();
}

std::vector<TypeId> KernelTensor::GetListOrTupleDtype() const {
  const TensorInfo &info = std::get<TensorInfo>(meta_);
  if (info.base_ == nullptr) {
    return {TypeId::kTypeUnknown};
  }

  auto type_ptr = info.base_->BuildType();
  if (type_ptr == nullptr || !type_ptr->isa<List>() || !type_ptr->isa<Tuple>()) {
    return {TypeId::kTypeUnknown};
  }

  std::vector<TypeId> types;
  if (type_ptr->isa<List>()) {
    auto tuple_ptr = type_ptr->cast<TuplePtr>();
    auto elements = tuple_ptr->elements();
    (void)std::transform(elements.begin(), elements.end(), std::back_inserter(types),
                         [](const TypePtr &t) { return t->type_id(); });
  } else if (type_ptr->isa<Tuple>()) {
    auto tuple_ptr = type_ptr->cast<TuplePtr>();
    auto elements = tuple_ptr->elements();
    (void)std::transform(elements.begin(), elements.end(), std::back_inserter(types),
                         [](const TypePtr &t) { return t->type_id(); });
  } else if (type_ptr->isa<List>()) {
    auto list_ptr = type_ptr->cast<ListPtr>();
    auto elements = list_ptr->elements();
    (void)std::transform(elements.begin(), elements.end(), std::back_inserter(types),
                         [](const TypePtr &t) { return t->type_id(); });
  } else {
    types.push_back(TypeId::kTypeUnknown);
  }

  return types;
}

ShapeArray KernelTensor::GetListOrTupleShapeVector() const {
  auto base_shape_ptr = GetBaseShape();
  // ListShape or TupleShape is inherited from SequenceShape.
  if (base_shape_ptr == nullptr || !base_shape_ptr->isa<abstract::SequenceShape>()) {
    return {};
  }
  auto sequence_shape_ptr = base_shape_ptr->cast<abstract::SequenceShapePtr>();
  auto base_shape_list = sequence_shape_ptr->shape();
  std::vector<std::vector<int64_t>> shape_vector_list;
  for (auto base_shape : base_shape_list) {
    if (base_shape == nullptr || !base_shape->isa<abstract::Shape>()) {
      return {};
    }
    auto tmp_shape = base_shape->cast<abstract::ShapePtr>()->shape();
    shape_vector_list.push_back(tmp_shape);
  }

  return shape_vector_list;
}

void KernelTensor::SetDtype(const TypePtr &dtype) {
  TensorInfo &info = std::get<TensorInfo>(meta_);
  if (info.base_ == nullptr) {
    return;
  }
  info.base_->set_type(dtype);
}

abstract::BaseShapePtr KernelTensor::GetBaseShape() const {
  if (meta_type_ != kObjectTypeTensorType) {
    return nullptr;
  }
  const TensorInfo &info = std::get<TensorInfo>(meta_);
  if (info.base_ == nullptr) {
    return nullptr;
  }
  return info.base_->BuildShape();
}

void KernelTensor::SetBaseShape(const abstract::BaseShapePtr &base_shape) {
  TensorInfo &info = std::get<TensorInfo>(meta_);
  if (info.base_ == nullptr) {
    return;
  }
  info.base_->set_shape(base_shape);
}

int KernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KRET_OK;
  workspace_size_list_.clear();
  output_size_list_.clear();

  for (size_t idx = 0; idx < outputs.size(); idx++) {
    auto &output = outputs[idx];
    size_t tensor_size = 0;
    MS_EXCEPTION_IF_NULL(output);
    size_t type_size = GetTypeByte(output->dtype());
    if (type_size == 0) {
      MS_LOG(WARNING) << "The type size is 0, type: " << output->dtype()->ToString();
    }
    const auto &shape = output->GetShapeVector();
    if (!IsValidShape(shape)) {
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

// ===========================Old interface===========================
std::vector<std::vector<int64_t>> GetShapes(const std::vector<KernelTensorPtr> &tensors) {
  std::vector<std::vector<int64_t>> shapes(tensors.size());
  for (size_t idx = 0; idx < shapes.size(); idx++) {
    shapes[idx] = tensors[idx]->GetShapeVector();
  }
  return shapes;
}

// ===========================New interface===========================
std::vector<std::vector<int64_t>> GetShapes(const std::vector<KernelTensor *> &tensors) {
  std::vector<std::vector<int64_t>> shapes(tensors.size());
  for (size_t idx = 0; idx < shapes.size(); idx++) {
    shapes[idx] = tensors[idx]->GetShapeVector();
  }
  return shapes;
}
}  // namespace kernel
}  // namespace mindspore
