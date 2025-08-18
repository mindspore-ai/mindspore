/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "ir/tensor.h"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <functional>
#include <memory>
#include <utility>
#include <algorithm>
#include <map>
#include <vector>
#include <string>

#include "mindapi/base/type_id.h"
#include "abstract/abstract_value.h"
#include "base/complex_storage.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "utils/system/env.h"
#include "utils/temp_file_manager.h"
#include "utils/ms_context.h"
#include "device_address/device_address_maker.h"
#include "ir/tensor_new.h"
#include "ir/dtype/tensor_type.h"
#include "utils/stream_guard.h"
#include "base/float16.h"
#include "ir/dtype/type_id.h"
#include "ir/format_utils.h"
namespace mindspore {
namespace tensor {
static uint64_t MakeId() {
  // Use atomic to make id generator thread safe.
  static std::atomic<uint64_t> last_id{1};
  return last_id.fetch_add(1, std::memory_order_relaxed);
}

std::unique_ptr<DeviceInfo> CopyDeviceInfo(const std::unique_ptr<DeviceInfo> &device_info) {
  return device_info == nullptr ? nullptr : std::make_unique<DeviceInfo>(device_info);
}

Tensor::Tensor(const Tensor &tensor)
    : MetaTensor(tensor),
      contiguous_callback_(tensor.contiguous_callback_),
      id_(tensor.id_),
      tensor_name_(tensor.tensor_name_),
      version_(tensor.version_),
      format_(tensor.format_),
      device_sync_(tensor.device_sync_),
      storage_info_(tensor.storage_info_),
      auto_grad_meta_data_(tensor.auto_grad_meta_data_),
      base_shape_ptr_(tensor.base_shape_ptr_),
      cache_tensor_ptr_(tensor.cache_tensor_ptr_),
      hashmap_tensor_ptr_(tensor.hashmap_tensor_ptr_),
      device_info_(CopyDeviceInfo(tensor.device_info_)),
      pin_mem_register_(tensor.pin_mem_register_),
      sync_status_(tensor.sync_status_),
      compression_type_(tensor.compression_type_),
      need_pipeline_sync_(tensor.need_pipeline_sync_),
      init_flag_(tensor.init_flag_),
      cache_enable_(tensor.cache_enable_),
      copy_done_flag_(tensor.copy_done_flag_) {
  user_data_ = tensor.user_data_;
}

Tensor::Tensor(const Tensor &tensor, TypeId data_type)
    : MetaTensor(data_type, tensor.shape_),
      contiguous_callback_(tensor.contiguous_callback_),
      id_(tensor.data_type_ != data_type ? MakeId() : tensor.id_),
      tensor_name_(tensor.tensor_name_),
      version_(tensor.version_),
      format_(tensor.format_),
      device_sync_(MakeDeviceAddress(data_type, tensor.shape_,
                                     MakeTensorData(data_type, tensor.shape_, tensor.data_c(), tensor.data_type_))),
      auto_grad_meta_data_(tensor.auto_grad_meta_data_),
      base_shape_ptr_(tensor.base_shape_ptr_),
      cache_tensor_ptr_(tensor.cache_tensor_ptr_),
      hashmap_tensor_ptr_(tensor.hashmap_tensor_ptr_),
      device_info_(CopyDeviceInfo(tensor.device_info_)),
      pin_mem_register_(tensor.pin_mem_register_),
      sync_status_(tensor.sync_status_),
      compression_type_(tensor.compression_type_),
      need_pipeline_sync_(tensor.need_pipeline_sync_),
      init_flag_(tensor.init_flag_),
      cache_enable_(tensor.cache_enable_),
      copy_done_flag_(tensor.copy_done_flag_) {
  MS_LOG(WARNING) << "Changing tensor data type is unsafe!";
}

Tensor &Tensor::operator=(const Tensor &tensor) {
  if (this == &tensor) {
    return *this;
  }
  id_ = tensor.id_;
  sync_status_ = tensor.sync_status_;
  version_ = tensor.version_;
  format_ = tensor.format_;
  device_sync_ = tensor.device_sync_;
  storage_info_ = tensor.storage_info_;
  need_pipeline_sync_ = tensor.need_pipeline_sync_;
  lazy_callback_ = tensor.lazy_callback_;
  contiguous_callback_ = tensor.contiguous_callback_;
  user_data_ = tensor.user_data_;
  base_shape_ptr_ = tensor.base_shape_ptr_;
  auto_grad_meta_data_ = tensor.auto_grad_meta_data_;
  init_flag_ = tensor.init_flag_;
  cache_enable_ = tensor.cache_enable_;
  cache_tensor_ptr_ = tensor.cache_tensor_ptr_;
  hashmap_tensor_ptr_ = tensor.hashmap_tensor_ptr_;
  pin_mem_register_ = tensor.pin_mem_register_;
  compression_type_ = tensor.compression_type_;
  tensor_name_ = tensor.tensor_name_;
  cast_dtype_ = tensor.cast_dtype_;
  graph_output_ = tensor.graph_output_;
  quant_params_ = tensor.quant_params_;
  updated_by_device_ = tensor.updated_by_device_;
  device_info_ = CopyDeviceInfo(tensor.device_info_);
  copy_done_flag_ = tensor.copy_done_flag_;
  return *this;
}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, DeviceAddressPtr device_address)
    : MetaTensor(data_type, shape), id_(MakeId()), device_sync_(std::move(device_address)) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape)
    : Tensor(data_type, shape, MakeDeviceAddress(data_type, shape)) {}

Tensor::Tensor(TypeId origin_data_type, const ShapeVector &shape, size_t compression_data_size,
               TensorCompressionType compression_type)
    : Tensor(
        origin_data_type, shape,
        MakeDeviceAddress(kNumberTypeInt8, ShapeVector{static_cast<int64_t>(compression_data_size)},
                          MakeTensorData(kNumberTypeInt8, ShapeVector{static_cast<int64_t>(compression_data_size)}))) {
  compression_type_ = compression_type;
}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, bool ref_mem, void *data)
    : Tensor(data_type, shape, MakeDeviceAddress(data_type, shape, MakeTensorData(data_type, shape, ref_mem, data))) {}

Tensor::~Tensor() {
  try {
    UnPinMemory();
    pin_mem_register_ = nullptr;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Exception when destruct tensor. Error info " << e.what();
  }
}

bool Tensor::operator==(const Tensor &tensor) const {
  return (&tensor == this || (MetaTensor::operator==(tensor) && device_sync_ == tensor.device_sync_));
}

// assign value to this tensor
Tensor &Tensor::AssignValue(const Tensor &tensor) {
  if (this != &tensor) {
    ExecuteLazyTask();
    contiguous_callback_ = tensor.contiguous_callback_;
    MetaTensor::operator=(tensor);
    device_sync_ = tensor.device_address();
    set_storage_info(tensor.storage_info());
    set_format(tensor.format());
    need_pipeline_sync_ = tensor.need_pipeline_sync_;
    sync_status_ = tensor.sync_status_;
    version_ = tensor.version_;
    if (this->auto_grad_meta_data() != nullptr && this->auto_grad_meta_data()->input_type() == InputType::kInput) {
      MS_LOG(EXCEPTION)
        << "Can not modify tensor id of input tensor from network by assign value, this may caused by slice op, "
           "please check your code to avoid this error!";
    }
    if (!is_parameter_) {
      id_ = tensor.id_;
      auto_grad_meta_data_ = tensor.auto_grad_meta_data_;
    }

    device_info_ = CopyDeviceInfo(tensor.device_info_);

    // Need execute callback when update host value of Tensor.
    ExecuteUpdateValueCallback();
  }
  return *this;
}

void Tensor::shallow_copy_from(const Tensor &other) {
  if (this != &other) {
    ExecuteLazyTask();
    MetaTensor::operator=(other);
    contiguous_callback_ = other.contiguous_callback_;
    device_sync_ = other.device_address();
    storage_info_ = other.storage_info();
    device_info_ = CopyDeviceInfo(other.device_info_);
    need_pipeline_sync_ = other.need_pipeline_sync();
  }
}

abstract::AbstractBasePtr Tensor::ToAbstract() {
  auto tens = shared_from_base<Tensor>();
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
    abs_tensor->set_value(shared_from_base<Tensor>());
  }
  return abs_tensor;
}

bool TensorEqual(const Tensor &self, const Tensor &other) {
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto self_ptr = static_cast<const uint8_t *>(self_cpu->data_c());
  auto other_ptr = static_cast<const uint8_t *>(other_cpu->data_c());
  if (self_ptr == nullptr || other_ptr == nullptr) {
    return false;
  }
  if (self_ptr == other_ptr) {
    return true;
  }
  return self.DataNDim() == other.DataNDim() && self.DataNBytes() == other.DataNBytes() &&
         std::equal(self_ptr, self_ptr + self.DataNBytes(), other_ptr);
}

bool Tensor::ValueEqual(const Tensor &tensor) const {
  if (is_parameter_ != tensor.is_parameter_) {
    return false;
  }
  if (is_parameter_ && param_info_->name() != tensor.param_info_->name()) {
    return false;
  }
  return (&tensor == this || (MetaTensor::operator==(tensor) && TensorEqual(*this, tensor)));
}

TypeId Tensor::set_data_type(TypeId data_type) {
  if (data_type != data_type_) {
    if (device_sync_ == nullptr) {
      // For Parameter with initializer.
      // The Parameter is not initialized yet.
      id_ = MakeId();
      return MetaTensor::set_data_type(data_type);
    }

    if (device_sync_->GetDeviceType() != device::DeviceType::kCPU) {
      auto cpu_tensor = cpu();
      device_sync_ = cpu_tensor->device_address();
      format_ = cpu_tensor->format();
      storage_info_ = nullptr;
    }
    auto new_dtype_address = MakeDeviceAddress(data_type, shape_, true);
    MS_EXCEPTION_IF_NULL(new_dtype_address);
    DeviceAddressExtPtr src_ext =
      std::make_shared<DeviceAddressExt>(this->format(), this->data_type(), this->shape(), this->storage_info());
    DeviceAddressExtPtr dst_ext = std::make_shared<DeviceAddressExt>(Format::DEFAULT_FORMAT, data_type, shape_);
    if (!SyncCopy(new_dtype_address, device_sync_, device_sync_->stream_id(), src_ext, dst_ext)) {
      MS_LOG(EXCEPTION) << "Sync copy failed";
    }
    device_sync_ = new_dtype_address;
    format_ = dst_ext->format_;
    storage_info_ = nullptr;
    id_ = MakeId();
    return MetaTensor::set_data_type(data_type);
  }
  return data_type;
}

size_t Tensor::set_shape(const ShapeVector &shape) {
  bool is_shape_unknown = std::any_of(shape_.begin(), shape_.end(), [](int64_t value) { return value < 0; });
  auto cur_data_size = DataSize();
  auto incoming_size = SizeOf(shape);
  if (!is_shape_unknown && cur_data_size < incoming_size) {
    // For dynamic shape scene.
    MS_LOG(INFO) << "It's not recommended to set " << ToString() << " shape to " << shape;
    if (device_sync_ != nullptr) {
      auto incoming_bytes = incoming_size * DataItemSize();
      if (incoming_bytes > device_sync_->GetSize()) {
        MS_LOG(WARNING) << "Cannot set " << ToString() << " shape to " << shape << ". The data size is "
                        << device_sync_->GetSize();
      }
    }
  }
  MS_LOG(DEBUG) << "Change shape of Tensor " << ToString() << " to " << shape;
  return MetaTensor::set_shape(shape);
}

std::string Tensor::GetShapeAndDataTypeInfo() const {
  std::ostringstream buf;
  buf << "Tensor shape:[" << shape() << "]" << this->Dtype()->ToString();
  return buf.str();
}

std::string Tensor::ToStringInternal(size_t limit_size) const {
  std::ostringstream buf;
  auto dtype = Dtype();
  MS_EXCEPTION_IF_NULL(dtype);
  buf << "Tensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString() << ", value=";
  if (limit_size == 0 || DataSize() < limit_size) {
    // Only print data for small tensor.
    buf << ((DataDim() > 1) ? "\n" : "") << DataToString(false);
  } else {
    buf << "[...]";
  }
  if (is_parameter_) {
    buf << ", name=" << param_info_->name();
  }
  buf << ")";
  return buf.str();
}

std::string Tensor::ToString() const {
  constexpr size_t small_tensor_size = 30;
  return ToStringInternal(small_tensor_size);
}

std::string Tensor::ToStringNoLimit() const { return ToStringInternal(0); }

std::string Tensor::ToStringRepr() const {
  std::ostringstream buf;
  auto dtype = Dtype();
  MS_EXCEPTION_IF_NULL(dtype);
  buf << "Tensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString()
      << ", value=" << ((DataNDim() > 1) ? '\n' : ' ') << DataToString(true) << ')';
  return buf.str();
}

const DeviceAddressPtr &Tensor::device_address() const { return device_sync_; }

void Tensor::set_device_address(const DeviceAddressPtr &device_sync, bool need_update_ref_count) {
  device_sync_ = device_sync;
}

void Tensor::set_(DeviceAddressPtr &&device_sync, const TensorStorageInfoPtr &storage_info, const ShapeVector &shape) {
  set_device_address(device_sync);
  set_storage_info(storage_info);
  set_shape(shape);
}

TensorStorageInfoPtr Tensor::storage_info() const { return storage_info_; }

void Tensor::set_storage_info(const TensorStorageInfoPtr &storage_info) { storage_info_ = storage_info; }

TensorStorageInfoPtr Tensor::implicit_copy_storage_info() const { return implicit_copy_storage_info_; }

void Tensor::set_implicit_copy_storage_info(const TensorStorageInfoPtr &storage_info) {
  implicit_copy_storage_info_ = storage_info;
}

bool Tensor::is_contiguous() const {
  const auto &storage = storage_info();
  return storage == nullptr || storage->is_contiguous;
}

bool Tensor::NeedContiguous() const {
  const auto &storage = storage_info();
  if (storage == nullptr) {
    return false;
  }
  return !storage->is_contiguous || storage->storage_offset != 0;
}

std::vector<int64_t> Tensor::stride() const {
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

size_t Tensor::storage_offset() const {
  const auto &storage = storage_info();
  return storage == nullptr ? 0 : storage->storage_offset;
}

void Tensor::ExecuteLazyTask() const {
  if (lazy_callback_ != nullptr && need_pipeline_sync_) {
    lazy_callback_();
  }
}

DeviceAddressPtr Tensor::CallContiguousCallback() const {
  DeviceAddressPtr contiguous_device_address = nullptr;
  if (contiguous_callback_ != nullptr && storage_info() != nullptr) {
    auto self_tensor = std::make_shared<Tensor>(*this);
    contiguous_device_address = contiguous_callback_(self_tensor);
  }
  return contiguous_device_address;
}

void *Tensor::data_c() const {
  if (device_sync_ == nullptr) {
    MS_LOG(EXCEPTION) << "Cannot access uninitialized tensor data";
  }
  if (device_sync_->GetDeviceType() != device::DeviceType::kCPU) {
    MS_LOG(EXCEPTION) << "Can't access data on " << device::GetDeviceNameByType(device_sync_->GetDeviceType());
  }

  // Load data from file
  if (!offload_file_.empty()) {
    auto fs = mindspore::system::Env::GetFileSystem();
    MS_EXCEPTION_IF_NULL(fs);
    if (fs->FileExist(offload_file_)) {
      auto file = fs->CreateWriteFile(offload_file_, "r+");
      if (device_sync_->GetMutablePtr() == nullptr) {
        device_sync_ = MakeDeviceAddress(data_type_, shape_, true);
      }
      MS_EXCEPTION_IF_NULL(file);
      bool success = file->PRead(device_sync_->GetMutablePtr(), DataNBytes(), 0);
      if (!success) {
        MS_LOG(WARNING) << "Tensor load data from file: " << offload_file_ << " failed!";
      }
      if (!file->Close()) {
        MS_LOG(WARNING) << "Close tensor file: " << offload_file_ << " failed!";
      }
    } else {
      MS_LOG(WARNING) << "Invalid tensor file path: " << offload_file_;
    }
  }

  return device_sync_->GetMutablePtr();
}

TensorPtr Tensor::cpu() const {
  ExecuteLazyTask();
  DeviceAddressPtr device_address;
  TensorStorageInfoPtr storage_info;
  auto contiguous_address = CallContiguousCallback();
  if (contiguous_address != nullptr) {
    device_address = contiguous_address;
    storage_info = nullptr;
  } else {
    device_address = device_sync_;
    storage_info = storage_info_;
  }

  if (device_address == nullptr) {
    MS_LOG(WARNING) << "Can't do cpu() for uninitialized tensor " << ToString();
    auto ret = std::make_shared<Tensor>(data_type_, shape_, MakeDeviceAddress(data_type_, shape_, true));
    ret->set_need_pipeline_sync(true);
    return ret;
  }
  if (device_address->GetDeviceType() == device::DeviceType::kCPU) {
    auto ret = std::make_shared<Tensor>(data_type_, shape_, device_address);
    ret->set_format(format_);
    ret->set_storage_info(storage_info);
    ret->set_need_pipeline_sync(true);
    return ret;
  }

  auto dst = MakeDeviceAddress(data_type_, shape_, true);
  MS_EXCEPTION_IF_NULL(dst);
  DeviceAddressExtPtr src_ext =
    std::make_shared<DeviceAddressExt>(this->format(), data_type_, shape_, storage_info);
  DeviceAddressExtPtr dst_ext = std::make_shared<DeviceAddressExt>(Format::DEFAULT_FORMAT, data_type_, shape_);
  if (!SyncCopy(dst, device_address, CurrentStream::id(), src_ext, dst_ext)) {
    MS_LOG(EXCEPTION) << "SyncCopy failed for " << ToString();
  }
  auto ret = std::make_shared<Tensor>(data_type_, shape_, dst);
  ret->set_storage_info(nullptr);
  ret->set_need_pipeline_sync(true);

  return ret;
}

std::string Tensor::DataToString(bool use_comma) const {
  if (device_sync_ == nullptr) {
    return "<uninitialized>";
  }
  if (device_sync_->GetDeviceType() != device::DeviceType::kCPU) {
    return "<" + device::GetDeviceNameByType(device_sync_->GetDeviceType()) + ">";
  }
  if (device_sync_->has_data()) {
    const auto &data = device_sync_->data();
    return data->ToString(data_type_, shape_, use_comma);
  }
  return GetTensorDataString(data_type_, shape_, device_sync_->GetMutablePtr(), DataSize(), DataDim(), use_comma);
}

const void *Tensor::unsafe_data() const {
  if (device_sync_ == nullptr) {
    return nullptr;
  }
  return device_sync_->GetMutablePtr();
}

void Tensor::ExecuteUpdateValueCallback() const {
  if (update_value_callback_ != nullptr) {
    update_value_callback_(this);
  }
}

void Tensor::SetDeviceInfo(const std::string &format, const TypePtr &data_type, const std::string &host_format) {
  DeviceInfo info(format, data_type, host_format);
  set_device_info(info);
}

void Tensor::data_sync_directly(const DeviceAddress *const device_sync, bool need_wait) const {}

bool Tensor::Offload(const std::string &file_path) {
  if (file_path.empty()) {
    return false;
  }

  auto fs = mindspore::system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  auto file = fs->CreateWriteFile(file_path);
  MS_EXCEPTION_IF_NULL(file);
  TempFileManager::GetInstance().Register(file_path);
  bool success = file->PWrite(data_c(), DataNBytes(), 0);
  if (!file->Close()) {
    MS_LOG(WARNING) << "Close tensor file: " << file_path << " failed!";
  }
  if (!success) {
    MS_LOG(WARNING) << "Tensor write data to file: " << file_path << " failed!";
    return false;
  }

  if (file_path == GetOffloadFilePath()) {
    offload_file_.clear();
  }

  // Make CPU device address and not init the data in device address.
  device_sync_ = MakeDeviceAddress(data_type_, shape_, false);
  offload_file_ = file_path;
  return true;
}

const std::string &Tensor::GetOffloadFilePath() const { return offload_file_; }

bool Tensor::CheckStub() {
#if defined(WITH_BACKEND)
  return false;
#else
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string backend_name = context_ptr->backend_policy();
  if (backend_name == "vm") {
    return false;
  }
  return true;
#endif
}

void Tensor::PinMemory(PinnedMemRegister *pin_mem_register) {
  if (pin_mem_register == nullptr) {
    return;
  }
  pin_mem_register_ = pin_mem_register;
  pin_mem_register_->RegisterPinnedMem(data_c(), Size());
}

void Tensor::UnPinMemory() {
  if (pin_mem_register_ == nullptr) {
    return;
  }
  pin_mem_register_->UnRegisterPinnedMem(data_c());
}

const ShapeVector &Tensor::shape_c() const { return shape(); }

Format Tensor::format() const { return format_; }

void Tensor::set_format(const Format &format) { format_ = format; }

ssize_t Tensor::DataItemSize() const {
  if (device_sync_ != nullptr && device_sync_->has_data()) {
    return device_sync_->data()->itemsize();
  }
  return static_cast<ssize_t>(abstract::TypeIdSize(data_type_));
}

bool Tensor::operator==(const Value &other) const {
  if (other.isa<Tensor>()) {
    auto &other_ = static_cast<const Tensor &>(other);
    return *this == other_;
  }
  return false;
}

bool Tensor::requires_grad() { return grad_impl()->requires_grad(shared_from_base<Tensor>()); }

void Tensor::set_requires_grad(bool requires_grad) {
  return grad_impl()->set_requires_grad(shared_from_base<Tensor>(), requires_grad);
}

bool Tensor::retains_grad() { return grad_impl()->retains_grad(shared_from_base<Tensor>()); }

void Tensor::retain_grad() { return grad_impl()->retain_grad(shared_from_base<Tensor>()); }

TensorPtr Tensor::grad() { return grad_impl()->grad(shared_from_base<Tensor>()); }

void Tensor::set_grad(const TensorPtr &grad) { grad_impl()->set_grad(shared_from_base<Tensor>(), grad); }

bool Tensor::is_leaf() { return grad_impl()->is_leaf(shared_from_base<Tensor>()); }

size_t Tensor::output_index() { return grad_impl()->output_index(shared_from_base<Tensor>()); }

BackwardNodePtr Tensor::grad_node() { return grad_impl()->grad_node(shared_from_base<Tensor>()); }

void Tensor::InitializeGradImpl(GradHookInterfacePtr grad_impl) {
  if (grad_impl_ != nullptr) {
    MS_LOG(EXCEPTION) << "Grad hook can only initialize once!";
  }
  grad_impl_ = std::move(grad_impl);
}

const GradHookInterfacePtr &Tensor::grad_impl() {
  MS_EXCEPTION_IF_NULL(grad_impl_);
  return grad_impl_;
}

CSRTensor::CSRTensor(const TensorPtr indptr, const TensorPtr indices, const TensorPtr values, const ShapeVector &shape)
    : MetaSparseTensor(values->data_type(), shape), indptr_(indptr), indices_(indices), values_(values) {}

std::string CSRTensor::ToString() const {
  std::ostringstream buf;
  MS_EXCEPTION_IF_NULL(values_);
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(indptr_);
  auto dtype = values_->Dtype();
  auto values_cpu = values_->cpu();
  auto indices_cpu = indices_->cpu();
  auto indptr_cpu = indptr_->cpu();
  buf << "CSRTensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString() << ", indptr=";
  buf << indptr_cpu->ToString() << ", indices=" << indices_cpu->ToString() << ", values=";
  buf << values_cpu->ToString() << ")";
  return buf.str();
}

abstract::AbstractBasePtr CSRTensor::ToAbstract() {
  auto dtype = values_->Dtype();
  if (!IsSubType(dtype, kNumber) && !IsSubType(dtype, kString) && !IsSubType(dtype, kTensorType)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber or kString or kTensor but got: " << dtype->ToString() << ".";
  }

  auto indptr = indptr_->ToAbstract()->cast<abstract::AbstractTensorPtr>();
  auto indices = indices_->ToAbstract()->cast<abstract::AbstractTensorPtr>();
  auto values = values_->ToAbstract()->cast<abstract::AbstractTensorPtr>();
  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape_.begin(), shape_.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  auto shape = std::make_shared<abstract::AbstractTuple>(abstract_shape);
  AbstractBasePtrList element_list{indptr, indices, values, shape};

  return std::make_shared<abstract::AbstractCSRTensor>(element_list);
}

const size_t CSRTensor::GetSizeAt(size_t index) const {
  if (index == kIndptrIdx) {
    MS_EXCEPTION_IF_NULL(indptr_);
    return indptr_->DataNBytes();
  } else if (index == kIndicesIdx) {
    MS_EXCEPTION_IF_NULL(indices_);
    return indices_->DataNBytes();
  } else if (index == kValuesIdx) {
    MS_EXCEPTION_IF_NULL(values_);
    return values_->DataNBytes();
  } else if (index >= kIndicesIdx && index < kShapeIdx + shape().size()) {
    return sizeof(int64_t);
  }
  MS_LOG(EXCEPTION) << "Invalid index: " << index << " for CSRTensor: " << ToString();
}

TensorPtr CSRTensor::GetTensorAt(size_t index) const {
  if (index == kIndptrIdx) {
    MS_EXCEPTION_IF_NULL(indptr_);
    return indptr_;
  } else if (index == kIndicesIdx) {
    MS_EXCEPTION_IF_NULL(indices_);
    return indices_;
  } else if (index == kValuesIdx) {
    MS_EXCEPTION_IF_NULL(values_);
    return values_;
  } else if (index >= kShapeIdx && index < kShapeIdx + shape().size()) {
    return from_scalar(shape_[index - kShapeIdx], TypeIdToType(kNumberTypeInt64));
  }
  MS_LOG(EXCEPTION) << "Invalid index: " << index << " for CSRTensor: " << ToString();
}

bool CSRTensor::operator==(const Value &other) const {
  if (other.isa<CSRTensor>()) {
    auto &other_ = static_cast<const CSRTensor &>(other);
    return *this == other_;
  }
  return false;
}

COOTensor::COOTensor(const TensorPtr indices, const TensorPtr values, const ShapeVector &shape)
    : MetaSparseTensor(values->data_type(), shape), indices_(indices), values_(values) {}

TensorPtr COOTensor::GetTensorAt(size_t index) const {
  if (index == kIndicesIdx) {
    MS_EXCEPTION_IF_NULL(indices_);
    return indices_;
  } else if (index == kValuesIdx) {
    MS_EXCEPTION_IF_NULL(values_);
    return values_;
  } else if (index >= kShapeIdx && index < kShapeIdx + shape().size()) {
    return tensor::from_scalar(shape_[index - kShapeIdx], TypeIdToType(kNumberTypeInt64));
  }
  MS_LOG(EXCEPTION) << "Invalid index: " << index << " for COOTensor: " << ToString();
}

std::string COOTensor::ToString() const {
  std::ostringstream buf;
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(values_);
  auto indices_cpu = indices_->cpu();
  auto values_cpu = values_->cpu();
  auto dtype = values_->Dtype();
  buf << "COOTensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString()
      << ", indices=" << indices_cpu->ToString() << ", values=" << values_cpu->ToString() << ")";
  return buf.str();
}

abstract::AbstractBasePtr COOTensor::ToAbstract() {
  MS_EXCEPTION_IF_NULL(values_);
  auto dtype = values_->Dtype();
  if (!IsSubType(dtype, kNumber) && !IsSubType(dtype, kString) && !IsSubType(dtype, kTensorType)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber or kString or kTensor but got: " << dtype->ToString() << ".";
  }
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(indices_->ToAbstract());
  MS_EXCEPTION_IF_NULL(values_->ToAbstract());
  auto indices = indices_->ToAbstract()->cast<abstract::AbstractTensorPtr>();
  auto values = values_->ToAbstract()->cast<abstract::AbstractTensorPtr>();
  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape_.begin(), shape_.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  auto shape = std::make_shared<abstract::AbstractTuple>(abstract_shape);
  AbstractBasePtrList element_list{indices, values, shape};

  return std::make_shared<abstract::AbstractCOOTensor>(element_list);
}

bool COOTensor::operator==(const Value &other) const {
  if (other.isa<COOTensor>()) {
    auto &other_ = static_cast<const COOTensor &>(other);
    return *this == other_;
  }
  return false;
}

RowTensor::RowTensor(const TensorPtr indices, const TensorPtr values, const ShapeVector &shape)
    : MetaSparseTensor(values->data_type(), shape), indices_(indices), values_(values) {}

std::string RowTensor::ToString() const {
  std::ostringstream buf;
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(values_);
  auto dtype = values_->Dtype();
  buf << "RowTensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString()
      << ", indices=" << indices_->ToString() << ", values=" << values_->ToString() << ")";
  return buf.str();
}

abstract::AbstractBasePtr RowTensor::ToAbstract() {
  auto dtype = values_->Dtype();
  if (!IsSubType(dtype, kNumber) && !IsSubType(dtype, kString) && !IsSubType(dtype, kTensorType)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber or kString or kTensor but got: " << dtype->ToString() << ".";
  }
  auto abs_sparse_tensor = std::make_shared<abstract::AbstractRowTensor>(dtype, shape_);
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(indices_->ToAbstract());
  MS_EXCEPTION_IF_NULL(values_->ToAbstract());
  abs_sparse_tensor->set_indices(indices_->ToAbstract()->cast<abstract::AbstractTensorPtr>());
  abs_sparse_tensor->set_values(values_->ToAbstract()->cast<abstract::AbstractTensorPtr>());

  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape_.begin(), shape_.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  abs_sparse_tensor->set_dense_shape(std::make_shared<abstract::AbstractTuple>(abstract_shape));

  return abs_sparse_tensor;
}

bool RowTensor::operator==(const Value &other) const {
  if (other.isa<RowTensor>()) {
    auto &other_ = static_cast<const RowTensor &>(other);
    return *this == other_;
  }
  return false;
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
}  // namespace tensor
namespace {
DeviceAddressExtPtr MakeDeviceAddressExt(const tensor::TensorPtr &tensor) {
  return std::make_shared<DeviceAddressExt>(tensor->format(), tensor->data_type(), tensor->shape(),
                                            tensor->storage_info());
}
}  // namespace

bool SyncCopy(const tensor::TensorPtr &dst, const tensor::TensorPtr &src, size_t stream_id) {
  auto dst_ext = MakeDeviceAddressExt(dst);
  auto src_ext = MakeDeviceAddressExt(src);
  return SyncCopy(dst->device_address(), src->device_address(), stream_id, src_ext, dst_ext);
}

bool AsyncCopy(const tensor::TensorPtr &dst, const tensor::TensorPtr &src, size_t stream_id, bool keep_src) {
  auto dst_ext = MakeDeviceAddressExt(dst);
  auto src_ext = MakeDeviceAddressExt(src);
  return AsyncCopy(dst->device_address(), src->device_address(), stream_id, keep_src, src_ext, dst_ext);
}
}  // namespace mindspore
