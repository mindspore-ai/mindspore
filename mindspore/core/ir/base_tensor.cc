/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ir/base_tensor.h"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <functional>
#include <memory>
#include <utility>
#include <map>
#include <vector>
#include "mindapi/base/type_id.h"
#include "abstract/utils.h"
#include "abstract/abstract_value.h"
#include "base/complex_storage.h"
#include "utils/log_adapter.h"
#include "mindspore/ccsrc/include/common/utils/convert_utils.h"
#include "utils/shape_utils.h"
#include "utils/temp_file_manager.h"

namespace mindspore {
namespace tensor {
static std::string MakeId() {
  // Use atomic to make id generator thread safe.
  static std::atomic<uint64_t> last_id{1};
  return "T" + std::to_string(last_id.fetch_add(1, std::memory_order_relaxed));
}

static TypeId TypeIdOf(const TypePtr &data_type, TypeId defaultTypeId) {
  return data_type ? data_type->type_id() : defaultTypeId;
}

std::string ShapeToString(const ShapeVector &shape) {
  std::string str = "[";
  const size_t count = shape.size();
  for (size_t i = 0; i < count; ++i) {
    if (i > 0) {
      str.append(", ");
    }
    str.append(std::to_string(shape[i]));
  }
  return str.append("]");
}

// Tensor chunk data.
template <typename T>
class TensorChunkData : public TensorDataImpl<T> {
 public:
  explicit TensorChunkData(size_t size) : TensorDataImpl<T>(ShapeVector{static_cast<int64_t>(size)}) {}

  ~TensorChunkData() override = default;

  bool has_sub_data() const override { return true; }
};

// Tensor compression data.
template <typename T>
class CompressionTensorData : public TensorDataImpl<T> {
 public:
  explicit CompressionTensorData(size_t size) : TensorDataImpl<T>(ShapeVector{static_cast<int64_t>(size)}) {}

  ~CompressionTensorData() override = default;
};

BaseTensor::BaseTensor(const BaseTensor &tensor)
    : MetaTensor(tensor),
      is_forward_output_(tensor.is_forward_output_),
      need_pipeline_sync_(tensor.need_pipeline_sync_),
      id_(tensor.id_),
      device_sync_(tensor.device_sync_),
      sync_status_(tensor.sync_status_),
      auto_grad_meta_data_(tensor.auto_grad_meta_data_),
      data_(tensor.data_),
      base_shape_ptr_(tensor.base_shape_ptr_),
      contiguous_callback_(tensor.contiguous_callback_) {
  user_data_ = tensor.user_data_;
}

BaseTensor::BaseTensor(const BaseTensor &tensor, TypeId data_type)
    : MetaTensor(data_type, tensor.shape_),
      is_forward_output_(tensor.is_forward_output_),
      need_pipeline_sync_(tensor.need_pipeline_sync_),
      id_(tensor.data_type_ != data_type ? MakeId() : tensor.id_),
      device_sync_(tensor.device_sync_),
      sync_status_(tensor.sync_status_),
      auto_grad_meta_data_(tensor.auto_grad_meta_data_),
      data_(MakeTensorData(data_type, tensor.shape_, tensor.data_->data(), tensor.data_type_)),
      base_shape_ptr_(tensor.base_shape_ptr_),
      contiguous_callback_(tensor.contiguous_callback_) {
  user_data_ = tensor.user_data_;
}

BaseTensor &BaseTensor::operator=(const BaseTensor &tensor) {
  if (this == &tensor) {
    return *this;
  }
  is_forward_output_ = tensor.is_forward_output_;
  data_ = tensor.data_;
  id_ = tensor.id_;
  sync_status_ = tensor.sync_status_;
  device_sync_ = tensor.device_sync_;
  need_pipeline_sync_ = tensor.need_pipeline_sync_;
  lazy_callback_ = tensor.lazy_callback_;
  contiguous_callback_ = tensor.contiguous_callback_;
  user_data_ = tensor.user_data_;
  base_shape_ptr_ = tensor.base_shape_ptr_;
  auto_grad_meta_data_ = tensor.auto_grad_meta_data_;
  return *this;
}

BaseTensor::BaseTensor(TypeId data_type, const ShapeVector &shape, TensorDataPtr data)
    : MetaTensor(data_type, shape), id_(MakeId()), data_(std::move(data)) {}

BaseTensor::BaseTensor(TypeId data_type, const ShapeVector &shape)
    : BaseTensor(data_type, shape, MakeTensorData(data_type, shape)) {}

BaseTensor::BaseTensor(TypeId data_type, const ShapeVector &shape, void *data, size_t data_len)
    : BaseTensor(data_type, shape, MakeTensorData(data_type, shape, data, data_len)) {}

BaseTensor::BaseTensor(TypeId data_type, const ShapeVector &shape, void *data, TypeId src_data_type)
    : BaseTensor(data_type, shape, MakeTensorData(data_type, shape, data, src_data_type)) {}

BaseTensor::BaseTensor(const std::vector<int64_t> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt64), {static_cast<int>(input.size())}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, shape_, input.data(), input.size())) {}

BaseTensor::BaseTensor(const std::vector<int32_t> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt32), {static_cast<int>(input.size())}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, shape_, input.data(), input.size())) {}

BaseTensor::BaseTensor(const std::vector<double> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {static_cast<int>(input.size())}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, shape_, input.data(), input.size())) {}

BaseTensor::BaseTensor(const std::vector<float> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {static_cast<int>(input.size())}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, shape_, input.data(), input.size())) {}

BaseTensor::BaseTensor(int64_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt64), {}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)) {}

BaseTensor::BaseTensor(int32_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt32), {}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)) {}

BaseTensor::BaseTensor(int16_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt16), {}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)) {}

BaseTensor::BaseTensor(int8_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt8), {}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)) {}

BaseTensor::BaseTensor(double input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)) {}

BaseTensor::BaseTensor(float input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)) {}

BaseTensor::BaseTensor(float16 input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat16), {}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)) {}
#ifndef KERNEL_EXECUTOR_ANDROID
BaseTensor::BaseTensor(bfloat16 input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeBFloat16), {}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)) {}
#endif
BaseTensor::BaseTensor(uint64_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeUInt64), {}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)) {}

BaseTensor::BaseTensor(uint32_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeUInt32), {}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)) {}

BaseTensor::BaseTensor(uint16_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeUInt16), {}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)) {}

BaseTensor::BaseTensor(uint8_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeUInt8), {}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)) {}

BaseTensor::BaseTensor(bool input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeBool), {}),
      id_(MakeId()),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)) {}

BaseTensor::BaseTensor(TypeId data_type, size_t data_size)
    : BaseTensor(data_type, ShapeVector{static_cast<int64_t>(data_size)},
                 MakeTensorData<TensorChunkData>(data_type, data_size)) {}

BaseTensor::BaseTensor(TypeId origin_data_type, const ShapeVector &shape, size_t compression_data_size,
                       TensorCompressionType compression_type)
    : BaseTensor(origin_data_type, shape,
                 MakeTensorData<CompressionTensorData>(kNumberTypeInt8, compression_data_size)) {}

bool BaseTensor::operator==(const BaseTensor &tensor) const {
  return (&tensor == this || (MetaTensor::operator==(tensor) && data_ == tensor.data_));
}

bool BaseTensor::ValueEqual(const BaseTensor &tensor) const {
  if (is_parameter_ != tensor.is_parameter_) {
    return false;
  }
  if (is_parameter_ && param_info_->name() != tensor.param_info_->name()) {
    return false;
  }
  return (&tensor == this || (MetaTensor::operator==(tensor) && data_->equals(*tensor.data_)));
}

void BaseTensor::ExecuteLazyTask() const {
  if (lazy_callback_ != nullptr && need_pipeline_sync_) {
    lazy_callback_();
  }

  if (contiguous_callback_ != nullptr && storage_info() != nullptr) {
    device_sync_ = contiguous_callback_(device_address());
    device_sync_->set_original_ref_count(SIZE_MAX);
    device_sync_->ResetRefCount();
  }
}

DeviceSyncPtr BaseTensor::device_address() const { return device_sync_; }

const TensorStorageInfoPtr BaseTensor::storage_info() const {
  if (device_sync_ == nullptr) {
    return nullptr;
  }

  return device_sync_->GetTensorStorageInfo();
}

bool BaseTensor::is_contiguous() const {
  const auto &storage = storage_info();
  return storage == nullptr || storage->is_contiguous;
}

std::vector<int64_t> BaseTensor::stride() const {
  const auto &storage = storage_info();
  if (storage != nullptr) {
    return storage->strides;
  }

  if (shape_.empty()) {
    return {};
  }
  std::vector<int64_t> ret(shape_.size(), 1);
  int64_t stride = 1;
  for (size_t i = shape_.size() - 1; i > 0; --i) {
    stride *= shape_[i];
    ret[i - 1] = stride;
  }
  return ret;
}

const int64_t BaseTensor::storage_offset() const {
  const auto &storage = storage_info();
  return storage == nullptr ? 0 : SizeToLong(storage->storage_offset);
}

void BaseTensor::set_device_address(const DeviceSyncPtr &device_sync, bool need_update_ref_count) {
  device_sync_ = device_sync;
  // To support the old and new runtime coexistence, the output of old runtime may be the input of new runtime, so the
  // device address cannot be released through ref count and set max ref count in this scenario.
  if (need_update_ref_count && (device_sync_ != nullptr)) {
    device_sync_->set_original_ref_count(SIZE_MAX);
    device_sync_->ResetRefCount();
  }
}

BaseTensor &BaseTensor::AssignValue(const BaseTensor &tensor) {
  if (this != &tensor) {
    ExecuteLazyTask();
    contiguous_callback_ = tensor.contiguous_callback_;
    MetaTensor::operator=(tensor);
    device_sync_ = tensor.device_address();
    need_pipeline_sync_ = tensor.need_pipeline_sync_;
    is_forward_output_ = tensor.is_forward_output_;
    sync_status_ = tensor.sync_status_;
    MS_EXCEPTION_IF_NULL(data_);
    if (data_->is_sub_data()) {
      // If tensor data is sub data, we should keep data
      // memory address unchange and copy data to it.
      CopyTensorData(data_, tensor.data_);
    } else {
      data_ = tensor.data_;
    }
    if (!is_parameter_) {
      id_ = tensor.id_;
      auto_grad_meta_data_ = tensor.auto_grad_meta_data_;
    }
  }
  return *this;
}

abstract::AbstractBasePtr BaseTensor::ToAbstract() {
  auto tens = shared_from_base<BaseTensor>();
  auto dtype = tens->Dtype();
  if (!IsSubType(dtype, kNumber) && !IsSubType(dtype, kString) && !IsSubType(dtype, kTensorType)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber or kString or kTensor but got: " << dtype->ToString() << ".";
  }
  abstract::AbstractTensorPtr abs_tensor = nullptr;
  if (base_shape_ptr_ == nullptr) {
    auto tensor_shape = tens->shape();
    abs_tensor = std::make_shared<abstract::AbstractTensor>(dtype, tensor_shape);
  } else {
    abs_tensor = std::make_shared<abstract::AbstractTensor>(dtype, base_shape_ptr_);
  }
  // if is parameter always no value.
  if (is_parameter_) {
    auto param_name = param_info_->name();
    auto ref_key = std::make_shared<RefKey>(param_name);
    abs_tensor = std::make_shared<abstract::AbstractRefTensor>(abs_tensor, ref_key);
  } else {
    abs_tensor->set_value(shared_from_base<BaseTensor>());
  }
  return abs_tensor;
}

abstract::AbstractBasePtr BaseTensor::GetAbstractCache() {
  auto abs = abstract_.lock();
  if (abs != nullptr) {
    MS_LOG(DEBUG) << "Get cached abstract " << abs->ToString() << " real tensor shape is " << shape_;
    return abs;
  }
  return ToAbstract();
}

std::string BaseTensor::GetShapeAndDataTypeInfo() const {
  std::ostringstream buf;
  buf << "Tensor shape:[" << shape() << "]" << this->Dtype()->ToString();
  return buf.str();
}

std::string BaseTensor::ToStringInternal(size_t limit_size) const {
  std::ostringstream buf;
  auto dtype = Dtype();
  MS_EXCEPTION_IF_NULL(dtype);
  buf << "Tensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString() << ", value=";
  if (limit_size == 0 || DataSize() < limit_size) {
    // Only print data for small tensor.
    buf << ((data().ndim() > 1) ? "\n" : "") << data().ToString(data_type_, shape_, false);
  } else {
    buf << "[...]";
  }
  if (is_parameter_) {
    buf << ", name=" << param_info_->name();
  }
  buf << ")";
  return buf.str();
}

std::string BaseTensor::ToString() const {
  constexpr size_t small_tensor_size = 30;
  return ToStringInternal(small_tensor_size);
}

std::string BaseTensor::ToStringNoLimit() const { return ToStringInternal(0); }

std::string BaseTensor::ToStringRepr() const {
  std::ostringstream buf;
  auto dtype = Dtype();
  MS_EXCEPTION_IF_NULL(dtype);
  buf << "Tensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString()
      << ", value=" << ((data().ndim() > 1) ? '\n' : ' ') << data().ToString(data_type_, shape_, true) << ')';
  return buf.str();
}

void BaseTensor::data_sync(bool need_wait) const {
  if (need_wait) {
    device_sync_ = device_address();
    ExecuteLazyTask();
  }
  if (device_sync_ == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(data_);
  if (data_->is_sub_data()) {
    return;
  }

  std::vector<size_t> shape_tmp;
  (void)std::transform(shape().begin(), shape().end(), std::back_inserter(shape_tmp), LongToSize);
  auto size = abstract::ShapeSize(shape_tmp) * abstract::TypeIdSize(data_type());
  auto address = device_sync_;
  if (size != 0 && !address->SyncDeviceToHost(shape(), size, data_type(), data_c())) {
    MS_LOG(INTERNAL_EXCEPTION) << "SyncDeviceToHost failed.";
  }
  if (!data_->file_path().empty()) {
    device_sync_ = nullptr;
  }
  sync_status_ = kNeedSyncHostToDevice;
}

TypeId BaseTensor::set_data_type(TypeId data_type) {
  if (data_type != data_type_) {
    MS_EXCEPTION_IF_NULL(data_);
    data_ = MakeTensorData(data_type, shape_, data_->data(), data_type_);
    return MetaTensor::set_data_type(data_type);
  }
  return data_type;
}

size_t BaseTensor::set_shape(const ShapeVector &shape) {
  abstract_.reset();
  if (DataSize() != SizeOf(shape)) {
    data_ = MakeTensorData(data_type_, shape);
  }
  return MetaTensor::set_shape(shape);
}
}  // namespace tensor
}  // namespace mindspore
