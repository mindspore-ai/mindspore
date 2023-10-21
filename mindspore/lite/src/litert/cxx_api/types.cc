/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "include/api/types.h"
#include <cstring>
#include <limits>
#include <numeric>
#include "include/api/status.h"
#include "include/api/dual_abi_helper.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"
#include "src/litert/cxx_api/tensor_utils.h"
#include "src/common/log_adapter.h"
#ifdef ENABLE_CLOUD_INFERENCE
#include <fstream>
#include "utils/file_utils.h"
#include "ir/dtype.h"
#include "utils/convert_utils_base.h"
#include "extendrt/kernel/ascend/plugin/ascend_allocator_plugin.h"
#endif

namespace mindspore {
class Buffer::Impl {
 public:
  Impl() : data_() {}
  ~Impl() = default;
  Impl(const void *data, size_t data_len) {
    if (data != nullptr) {
      (void)SetData(data, data_len);
    } else {
      ResizeData(data_len);
    }
  }

  const void *Data() const { return data_.data(); }
  void *MutableData() { return data_.data(); }
  size_t DataSize() const { return data_.size(); }

  void ResizeData(size_t data_len) { data_.resize(data_len); }

  bool SetData(const void *data, size_t data_len) {
    ResizeData(data_len);
    if (DataSize() != data_len) {
      MS_LOG(ERROR) << "Set data failed, tensor current data size " << DataSize() << " not match data len " << data_len;
      return false;
    }

    if (data == nullptr) {
      return data_len == 0;
    }

    if (MutableData() == nullptr) {
      MS_LOG(ERROR) << "Set data failed, data len " << data_len;
      return false;
    }

    (void)memcpy(MutableData(), data, data_len);
    return true;
  }

 protected:
  std::vector<uint8_t> data_;
};

MSTensor::MSTensor() {
  auto impl = std::make_shared<LiteTensorImpl>(new (std::nothrow) lite::Tensor());
  if (impl != nullptr) {
    impl->set_from_session(false);
  } else {
    MS_LOG(ERROR) << "Failed to allocate tensor impl.";
  }
  impl_ = impl;
}
MSTensor::MSTensor(std::nullptr_t) : impl_(nullptr) {}
MSTensor::MSTensor(const std::shared_ptr<Impl> &impl) : impl_(impl) {}
MSTensor::MSTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                   const void *data, size_t data_len)
    : impl_(LiteTensorImpl::CreateTensorImplByDeepCopy(CharToString(name), type, shape, data, data_len)) {}
MSTensor::~MSTensor() = default;

bool MSTensor::operator==(std::nullptr_t) const { return impl_ == nullptr; }

bool MSTensor::operator!=(std::nullptr_t) const { return impl_ != nullptr; }

bool MSTensor::operator==(const MSTensor &tensor) const {
  if (impl_ == nullptr) {
    return false;
  }
  auto lite_impl = std::static_pointer_cast<LiteTensorImpl>(impl_);
  auto lite_tensor_impl = std::static_pointer_cast<LiteTensorImpl>(tensor.impl_);
  if (lite_tensor_impl == nullptr) {
    MS_LOG(ERROR) << "Cast lite tensor impl ptr failed.";
    return false;
  }

  return lite_impl->lite_tensor() == lite_tensor_impl->lite_tensor();
}

bool MSTensor::operator!=(const MSTensor &tensor) const { return !operator==(tensor); }

MSTensor *MSTensor::CreateTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                                 const void *data, size_t data_len, const std::vector<char> &device,
                                 int device_id) noexcept {
  MS_LOG(INFO) << "device id: " << device_id << ", device type: " << device;
  auto device_type = CharToString(device);
  if (!device_type.empty() && device_type == "ascend") {
#ifdef ENABLE_CLOUD_INFERENCE
    kernel::AscendAllocatorPlugin::GetInstance().Register();
    // check device id
    device_id = device_id == -1 ? kernel::AscendAllocatorPlugin::GetInstance().GetCurrentDeviceId() : device_id;
    // check device data size
    size_t element_size = CalTensorDataSize(shape, type);
    MS_CHECK_FALSE_MSG(data_len != 0 && element_size != data_len, nullptr, "data len not equal element size.");
    // malloc device data
    void *device_data = kernel::AscendAllocatorPlugin::GetInstance().Malloc(element_size, device_id);
    MS_CHECK_TRUE_MSG(device_data != nullptr, nullptr, "malloc device data failed.");
    // create tensor
    auto impl = LiteTensorImpl::CreateTensorImpl(CharToString(name), type, shape, nullptr, 0);
    if (impl == nullptr) {
      kernel::AscendAllocatorPlugin::GetInstance().Free(device_data, device_id);
      MS_LOG(ERROR) << "Allocate tensor impl failed.";
      return nullptr;
    }
    if (data != nullptr) {
      // init device data by host data buf
      auto status = kernel::AscendAllocatorPlugin::GetInstance().CopyHostDataToDevice(const_cast<void *>(data),
                                                                                      device_data, element_size);
      if (status != kSuccess) {
        kernel::AscendAllocatorPlugin::GetInstance().Free(device_data, device_id);
        MS_LOG(ERROR) << "copy host data to device data failed.";
        return nullptr;
      }
    }

    // init impl
    impl->SetDeviceData(device_data);
    impl->SetDeviceId(device_id);
    impl->SetDevice(device_type);

    auto ms_tensor = new (std::nothrow) MSTensor(impl);
    if (ms_tensor == nullptr) {
      kernel::AscendAllocatorPlugin::GetInstance().Free(device_data, device_id);
      MS_LOG(ERROR) << "Allocate MSTensor failed.";
      return nullptr;
    }
    impl->set_own_data(true);
    return ms_tensor;
#endif
    MS_LOG(ERROR) << "Unsupported Feature.";
    return nullptr;
  }
  if (data_len > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "data_len is error.";
    return nullptr;
  }
  if (data_len > 0 && data == nullptr) {
    MS_LOG(ERROR) << "Null data ptr of tensor.";
    return nullptr;
  }
  if (data_len == 0 && data != nullptr) {
    MS_LOG(ERROR) << "Data len doesn't match the data buffer size.";
    return nullptr;
  }

  void *new_data = nullptr;
  if (data != nullptr) {
    new_data = malloc(data_len);
    if (new_data == nullptr) {
      MS_LOG(ERROR) << "Allocate data failed.";
      return nullptr;
    }
    (void)memcpy(new_data, data, data_len);
  }
  auto impl = LiteTensorImpl::CreateTensorImpl(CharToString(name), type, shape, new_data, data_len);
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Allocate tensor impl failed.";
    free(new_data);
    return nullptr;
  }

  auto ms_tensor = new (std::nothrow) MSTensor(impl);
  if (ms_tensor == nullptr) {
    MS_LOG(ERROR) << "Allocate MSTensor failed.";
    free(new_data);
    return nullptr;
  }
  impl->set_own_data(true);
  return ms_tensor;
}

MSTensor *MSTensor::CreateTensor(const std::vector<char> &name, const MSTensor &tensor, const std::vector<char> &device,
                                 int device_id) noexcept {
#ifdef ENABLE_CLOUD_INFERENCE
  kernel::AscendAllocatorPlugin::GetInstance().Register();
  auto dst_device_type = CharToString(device);
  if (!dst_device_type.empty() && dst_device_type != "ascend") {
    MS_LOG(ERROR) << "only support create ascend device tensor.";
    return nullptr;
  }

  auto src_device_type = tensor.GetDevice();
  if (!src_device_type.empty() && src_device_type != "ascend") {
    MS_LOG(ERROR) << "only tensor tensor is ascend device tensor.";
    return nullptr;
  }
  if (src_device_type.empty() && static_cast<MSTensor>(tensor).GetDeviceData() != nullptr) {
    MS_LOG(ERROR) << "tensor tensor is device tensor, but device data is nullptr.";
    return nullptr;
  }

  if (src_device_type.empty() && dst_device_type.empty()) {
    MS_LOG(INFO) << "copy host tensor to host tensor.";
    if (tensor.Data() != nullptr) {
      return CreateTensor(tensor.Name(), tensor.DataType(), tensor.Shape(), static_cast<MSTensor>(tensor).MutableData(),
                          tensor.DataSize());
    } else {
      return CreateTensor(tensor.Name(), tensor.DataType(), tensor.Shape(), nullptr, 0);
    }
  } else if (src_device_type == "ascend" && dst_device_type == "ascend") {
    MS_LOG(INFO) << "copy device tensor to device tensor.";
    auto new_tensor =
      CreateTensor(tensor.Name(), tensor.DataType(), tensor.Shape(), nullptr, tensor.DataSize(), "ascend", device_id);
    auto status = kernel::AscendAllocatorPlugin::GetInstance().CopyDeviceDataToDevice(
      static_cast<MSTensor>(tensor).GetDeviceData(), new_tensor->GetDeviceData(), new_tensor->DataSize(),
      tensor.DataSize(), tensor.GetDeviceId(), new_tensor->GetDeviceId());
    if (status != kSuccess) {
      return nullptr;
    }
    return new_tensor;
  } else if (src_device_type.empty() && dst_device_type == "ascend") {
    MS_LOG(INFO) << "copy host tensor to device tensor.";
    return CreateTensor(tensor.Name(), tensor.DataType(), tensor.Shape(), static_cast<MSTensor>(tensor).MutableData(),
                        tensor.DataSize(), dst_device_type, device_id);
  } else if (src_device_type == "ascend" && dst_device_type.empty()) {
    MS_LOG(INFO) << "copy device tensor to host tensor.";
    auto host_form_device = malloc(tensor.DataSize());
    MS_CHECK_FALSE_MSG(host_form_device == nullptr, nullptr, "malloc host buf failed.");
    auto status = kernel::AscendAllocatorPlugin::GetInstance().CopyDeviceDataToHost(
      static_cast<MSTensor>(tensor).GetDeviceData(), host_form_device, tensor.DataSize(), tensor.GetDeviceId());
    if (status != kSuccess) {
      free(host_form_device);
      return nullptr;
    }
    auto new_tensor =
      CreateTensor(tensor.Name(), tensor.DataType(), tensor.Shape(), host_form_device, tensor.DataSize());
    free(host_form_device);
    host_form_device = nullptr;
    return new_tensor;
  } else {
    MS_LOG(ERROR) << "device type is wrong.";
    return nullptr;
  }
#endif
  MS_LOG(ERROR) << "Unsupported Feature.";
  return nullptr;
}

MSTensor *MSTensor::CreateRefTensor(const std::vector<char> &name, enum DataType type,
                                    const std::vector<int64_t> &shape, const void *data, size_t data_len,
                                    bool own_data) noexcept {
  auto impl = LiteTensorImpl::CreateTensorImpl(CharToString(name), type, shape, data, data_len);
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Allocate tensor impl failed.";
    return nullptr;
  }
  impl->set_own_data(own_data);
  auto ms_tensor = new (std::nothrow) MSTensor(impl);
  if (ms_tensor == nullptr) {
    MS_LOG(ERROR) << "Allocate tensor impl failed.";
    return nullptr;
  }
  return ms_tensor;
}

MSTensor MSTensor::CreateDeviceTensor(const std::vector<char> &name, enum DataType type,
                                      const std::vector<int64_t> &shape, void *data, size_t data_len) noexcept {
#ifdef ENABLE_CLOUD_INFERENCE
  auto impl = LiteTensorImpl::CreateTensorImpl(CharToString(name), type, shape, nullptr, 0);
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Allocate tensor impl failed.";
    return MSTensor(nullptr);
  }
  if (data_len < impl->DataSize()) {
    MS_LOG(ERROR) << "The size " << data_len << " of data cannot be less that the memory size required by the shape "
                  << shape << " and data type " << TypeIdToString(static_cast<enum TypeId>(type));
    return MSTensor(nullptr);
  }
  impl->SetDeviceData(data);
  return MSTensor(impl);
#else
  MS_LOG(ERROR) << "Unsupported Feature.";
  return MSTensor(nullptr);
#endif
}

MSTensor *MSTensor::CreateTensorFromFile(const std::vector<char> &file, enum DataType type,
                                         const std::vector<int64_t> &shape) noexcept {
#ifdef ENABLE_CLOUD_INFERENCE
  try {
    std::string file_str = CharToString(file);

    auto realpath = mindspore::FileUtils::GetRealPath(file_str.c_str());
    if (!realpath.has_value()) {
      MS_LOG(ERROR) << "Get real path failed, path=" << file_str;
      return nullptr;
    }

    // Read image file
    auto file_path = realpath.value();
    if (file_path.empty()) {
      MS_LOG(ERROR) << "Can not find any input file.";
      return nullptr;
    }

    std::ifstream ifs(file_path, std::ios::in | std::ios::binary);
    if (!ifs.good()) {
      MS_LOG(ERROR) << "File: " + file_path + " does not exist.";
      return nullptr;
    }
    if (!ifs.is_open()) {
      MS_LOG(ERROR) << "File: " + file_path + " open failed.";
      return nullptr;
    }

    auto &io_seekg1 = ifs.seekg(0, std::ios::end);
    if (!io_seekg1.good() || io_seekg1.fail() || io_seekg1.bad()) {
      ifs.close();
      MS_LOG(ERROR) << "Failed to seekg file: " + file_path;
      return nullptr;
    }

    size_t size = static_cast<size_t>(ifs.tellg());
    std::vector<int64_t> tensor_shape;
    tensor_shape = shape.empty() ? std::vector<int64_t>{static_cast<int64_t>(size)} : shape;
    MSTensor *ret = new (std::nothrow) MSTensor(file_path, type, tensor_shape, nullptr, size);
    if (ret == nullptr) {
      ifs.close();
      MS_LOG(ERROR) << "Malloc memory failed.";
      return nullptr;
    }
    auto &io_seekg2 = ifs.seekg(0, std::ios::beg);
    if (!io_seekg2.good() || io_seekg2.fail() || io_seekg2.bad()) {
      ifs.close();
      MS_LOG(ERROR) << "Failed to seekg file: " + file_path;
      return nullptr;
    }

    std::map<enum DataType, size_t> TypeByte = {
      {DataType::kTypeUnknown, 0},       {DataType::kObjectTypeString, 0},  {DataType::kNumberTypeBool, 1},
      {DataType::kNumberTypeInt8, 1},    {DataType::kNumberTypeInt16, 2},   {DataType::kNumberTypeInt32, 4},
      {DataType::kNumberTypeInt64, 8},   {DataType::kNumberTypeUInt8, 1},   {DataType::kNumberTypeUInt16, 2},
      {DataType::kNumberTypeUInt32, 4},  {DataType::kNumberTypeUInt64, 8},  {DataType::kNumberTypeFloat16, 2},
      {DataType::kNumberTypeFloat32, 4}, {DataType::kNumberTypeFloat64, 8},
    };

    if (LongToSize(ret->ElementNum()) * TypeByte[type] != size) {
      ifs.close();
      MS_LOG(ERROR) << "Tensor data size: " << LongToSize(ret->ElementNum()) * TypeByte[type]
                    << " not match input data length: " << size;
      return nullptr;
    }

    auto &io_read = ifs.read(reinterpret_cast<char *>(ret->MutableData()), static_cast<std::streamsize>(size));
    if (!io_read.good() || io_read.fail() || io_read.bad()) {
      ifs.close();
      MS_LOG(ERROR) << "Failed to read file: " + file_path;
      return nullptr;
    }
    ifs.close();

    return ret;
  } catch (...) {
    MS_LOG(ERROR) << "Unknown error occurred.";
    return nullptr;
  }
#else
  MS_LOG(ERROR) << "Unsupported Feature.";
  return nullptr;
#endif
}

MSTensor *MSTensor::CharStringsToTensor(const std::vector<char> &name, const std::vector<std::vector<char>> &inputs) {
#ifndef STRING_KERNEL_CLIP
  auto impl = LiteTensorImpl::StringsToTensorImpl(CharToString(name), VectorCharToString(inputs));
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Allocate tensor impl failed.";
    return nullptr;
  }
  auto ms_tensor = new (std::nothrow) MSTensor(impl);
  if (ms_tensor == nullptr) {
    MS_LOG(ERROR) << "Allocate tensor impl failed.";
    return nullptr;
  }
  return ms_tensor;
#else
  MS_LOG(ERROR) << unsupport_string_tensor_log;
  return nullptr;
#endif
}

std::vector<std::vector<char>> MSTensor::TensorToStringChars(const MSTensor &tensor) {
#ifndef STRING_KERNEL_CLIP
  if (tensor.impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor.";
    return {{}};
  }
  auto lite_impl = std::static_pointer_cast<LiteTensorImpl>(tensor.impl_);
  return VectorStringToChar(LiteTensorImpl::TensorImplToStrings(lite_impl));
#else
  std::vector<std::vector<char>> empty;
  MS_LOG(ERROR) << unsupport_string_tensor_log;
  return empty;
#endif
}

MSTensor *MSTensor::Clone() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor.";
    return nullptr;
  }
  auto data_len = this->DataSize();
  if (data_len > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "Illegal data size of tensor.";
    return nullptr;
  }
  if (data_len > 0 && impl_->Data() == nullptr) {
    MS_LOG(ERROR) << "Null data ptr of tensor.";
    return nullptr;
  }
  if (data_len == 0 && impl_->Data() != nullptr) {
    MS_LOG(ERROR) << "Data len doesn't match the data buffer size.";
    return nullptr;
  }

  void *new_data = nullptr;
  if (impl_->Data() != nullptr) {
    new_data = malloc(data_len);
    if (new_data == nullptr) {
      MS_LOG(ERROR) << "Allocate data failed.";
      return nullptr;
    }
    (void)memcpy(new_data, impl_->MutableData(), data_len);
  }

  auto impl = LiteTensorImpl::CreateTensorImpl(this->Name(), this->DataType(), this->Shape(), new_data, data_len);
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Allocate tensor impl failed.";
    if (new_data != nullptr) {
      free(new_data);
    }
    return nullptr;
  }

  auto ms_tensor = new (std::nothrow) MSTensor(impl);
  if (ms_tensor == nullptr) {
    MS_LOG(ERROR) << "Allocate MSTensor failed.";
    if (new_data != nullptr) {
      free(new_data);
    }
    return nullptr;
  }
  impl->set_own_data(true);
  return ms_tensor;
}

std::vector<char> MSTensor::CharName() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return std::vector<char>();
  }
  return StringToChar(impl_->Name());
}

int64_t MSTensor::ElementNum() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return -1;
  }
  return std::static_pointer_cast<MutableTensorImpl>(impl_)->ElementNum();
}

enum DataType MSTensor::DataType() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return DataType::kTypeUnknown;
  }
  return impl_->DataType();
}

const std::vector<int64_t> &MSTensor::Shape() const {
  static const std::vector<int64_t> empty{};
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return empty;
  }
  return impl_->Shape();
}

std::shared_ptr<const void> MSTensor::Data() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return nullptr;
  }
  return impl_->Data();
}

void *MSTensor::MutableData() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return nullptr;
  }
  return impl_->MutableData();
}

void MSTensor::SetDeviceData(void *data) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return;
  }
  std::static_pointer_cast<MutableTensorImpl>(impl_)->SetDeviceData(data);
}

void *MSTensor::GetDeviceData() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return nullptr;
  }
  return std::static_pointer_cast<MutableTensorImpl>(impl_)->GetDeviceData();
}

bool MSTensor::IsConst() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return false;
  }
  return std::static_pointer_cast<MutableTensorImpl>(impl_)->IsConst();
}

size_t MSTensor::DataSize() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return 0;
  }
  return impl_->DataSize();
}

std::string MSTensor::GetDevice() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return "";
  }
  return std::static_pointer_cast<MutableTensorImpl>(impl_)->GetDevice();
}

int MSTensor::GetDeviceId() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return -1;
  }
  return std::static_pointer_cast<MutableTensorImpl>(impl_)->GetDeviceId();
}

bool MSTensor::IsDevice() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return false;
  }
  return impl_->IsDevice();
}

void MSTensor::DestroyTensorPtr(MSTensor *tensor) noexcept {
  if (tensor != nullptr) {
    delete tensor;
  }
}

void MSTensor::SetShape(const std::vector<int64_t> &shape) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return;
  }

  std::static_pointer_cast<MutableTensorImpl>(impl_)->SetShape(shape);
}

void MSTensor::SetDataType(enum DataType data_type) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return;
  }

  std::static_pointer_cast<MutableTensorImpl>(impl_)->SetDataType(data_type);
}

void MSTensor::SetTensorName(const std::vector<char> &name) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return;
  }
  std::static_pointer_cast<MutableTensorImpl>(impl_)->SetName(CharToString(name));
}

void MSTensor::SetAllocator(std::shared_ptr<Allocator> allocator) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return;
  }

  return std::static_pointer_cast<MutableTensorImpl>(impl_)->SetAllocator(allocator);
}

std::shared_ptr<Allocator> MSTensor::allocator() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return nullptr;
  }

  return std::static_pointer_cast<MutableTensorImpl>(impl_)->GetAllocator();
}

void MSTensor::SetFormat(mindspore::Format format) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return;
  }

  return std::static_pointer_cast<MutableTensorImpl>(impl_)->SetFormat(format);
}

mindspore::Format MSTensor::format() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return mindspore::Format::NHWC;
  }

  return std::static_pointer_cast<MutableTensorImpl>(impl_)->Format();
}

void MSTensor::SetData(void *data, bool own_data) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return;
  }

  return std::static_pointer_cast<MutableTensorImpl>(impl_)->SetData(data, own_data);
}

std::vector<QuantParam> MSTensor::QuantParams() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return std::vector<QuantParam>{};
  }

  return std::static_pointer_cast<MutableTensorImpl>(impl_)->GetQuantParams();
}

void MSTensor::SetQuantParams(std::vector<QuantParam> quant_params) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return;
  }

  return std::static_pointer_cast<MutableTensorImpl>(impl_)->SetQuantParams(quant_params);
}

Buffer::Buffer() : impl_(std::make_shared<Impl>()) {}
Buffer::Buffer(const void *data, size_t data_len) : impl_(std::make_shared<Impl>(data, data_len)) {}
Buffer::~Buffer() = default;

Buffer Buffer::Clone() const {
  Buffer ret;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "impl is nullptr.";
    return ret;
  }
  ret.impl_ = std::make_shared<Impl>(*impl_);
  return ret;
}

const void *Buffer::Data() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "impl is nullptr.";
    return nullptr;
  }
  return impl_->Data();
}

void *Buffer::MutableData() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "impl is nullptr.";
    return nullptr;
  }
  return impl_->MutableData();
}

size_t Buffer::DataSize() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "impl is nullptr.";
    return 0;
  }
  return impl_->DataSize();
}

bool Buffer::ResizeData(size_t data_len) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "impl is nullptr.";
    return false;
  }
  impl_->ResizeData(data_len);
  return true;
}

bool Buffer::SetData(const void *data, size_t data_len) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "impl is nullptr.";
    return false;
  }
  return impl_->SetData(data, data_len);
}

std::vector<char> CharVersion() {
  std::string version = VERSION_STR;
  return StringToChar("MindSpore Lite " + version);
}
}  // namespace mindspore
