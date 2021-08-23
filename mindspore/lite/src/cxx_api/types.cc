/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <string.h>
#include <limits.h>
#include <numeric>
#include "include/api/status.h"
#include "include/api/dual_abi_helper.h"
#include "src/cxx_api/tensor/tensor_impl.h"
#include "src/common/log_adapter.h"
#include "include/version.h"

namespace mindspore {
class Buffer::Impl {
 public:
  Impl() : data_() { MS_LOG(ERROR) << "Unsupported feature."; }
  ~Impl() = default;
  Impl(const void *data, size_t data_len) { MS_LOG(ERROR) << "Unsupported feature."; }

  const void *Data() const {
    MS_LOG(ERROR) << "Unsupported feature.";
    return nullptr;
  }
  void *MutableData() {
    MS_LOG(ERROR) << "Unsupported feature.";
    return nullptr;
  }
  size_t DataSize() const {
    MS_LOG(ERROR) << "Unsupported feature.";
    return 0;
  }

  bool ResizeData(size_t data_len) {
    MS_LOG(ERROR) << "Unsupported feature.";
    return false;
  }

  bool SetData(const void *data, size_t data_len) {
    MS_LOG(ERROR) << "Unsupported feature.";
    return false;
  }

 protected:
  std::vector<uint8_t> data_;
};

MSTensor::MSTensor() : impl_(std::make_shared<Impl>()) {}
MSTensor::MSTensor(std::nullptr_t) : impl_(nullptr) {}
MSTensor::MSTensor(const std::shared_ptr<Impl> &impl) : impl_(impl) {}
MSTensor::MSTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                   const void *data, size_t data_len)
    : impl_(Impl::CreateTensorImpl(CharToString(name), type, shape, data, data_len)) {}
MSTensor::~MSTensor() = default;

bool MSTensor::operator==(std::nullptr_t) const { return impl_ == nullptr; }

bool MSTensor::operator!=(std::nullptr_t) const { return impl_ != nullptr; }

bool MSTensor::operator==(const MSTensor &tensor) const { return impl_->lite_tensor() == tensor.impl_->lite_tensor(); }

MSTensor *MSTensor::CreateTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                                 const void *data, size_t data_len) noexcept {
  if (data_len < 0 || data_len > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "data_len is error.";
    return nullptr;
  }
  if (data_len > 0 && data == nullptr) {
    MS_LOG(ERROR) << "Mull data ptr of tensor.";
    return nullptr;
  }
  auto impl = Impl::CreateTensorImpl(CharToString(name), type, shape, nullptr, data_len);
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Allocate tensor impl failed.";
    return nullptr;
  }
  impl->set_own_data(true);

  auto ms_tensor = new (std::nothrow) MSTensor(impl);
  if (ms_tensor == nullptr) {
    MS_LOG(ERROR) << "Allocate tensor impl failed.";
    return nullptr;
  }

  if (data != nullptr) {
    if (ms_tensor->MutableData() == nullptr) {
      MS_LOG(ERROR) << "Allocate data failed.";
      delete ms_tensor;
      return nullptr;
    }
    ::memcpy(ms_tensor->MutableData(), data, data_len);
  }
  return ms_tensor;
}

MSTensor *MSTensor::CreateRefTensor(const std::vector<char> &name, enum DataType type,
                                    const std::vector<int64_t> &shape, const void *data, size_t data_len) noexcept {
  auto impl = Impl::CreateTensorImpl(CharToString(name), type, shape, data, data_len);
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
}

MSTensor *MSTensor::CreateDevTensor(const std::vector<char> &name, enum DataType type,
                                    const std::vector<int64_t> &shape, const void *data, size_t data_len) noexcept {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return nullptr;
}

MSTensor *MSTensor::CharStringsToTensor(const std::vector<char> &name, const std::vector<std::vector<char>> &inputs) {
  auto impl = Impl::StringsToTensorImpl(CharToString(name), VectorCharToString(inputs));
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
}

std::vector<std::vector<char>> MSTensor::TensorToStringChars(const MSTensor &tensor) {
  if (tensor.impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor.";
    std::vector<std::vector<char>> empty;
    return empty;
  }
  return VectorStringToChar(Impl::TensorImplToStrings(tensor.impl_));
}

MSTensor *MSTensor::Clone() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor.";
    return nullptr;
  }
  auto data_len = this->DataSize();
  if (data_len <= 0 || data_len > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "Illegal data size of tensor.";
    return nullptr;
  }
  auto impl = Impl::CreateTensorImpl(this->Name(), this->DataType(), this->Shape(), nullptr, data_len);
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Allocate tensor impl failed.";
    return nullptr;
  }
  impl->set_own_data(true);

  auto ms_tensor = new (std::nothrow) MSTensor(impl);
  if (ms_tensor == nullptr) {
    MS_LOG(ERROR) << "Allocate tensor impl failed.";
    return nullptr;
  }

  if (impl_->Data() != nullptr) {
    if (ms_tensor->MutableData() == nullptr) {
      MS_LOG(ERROR) << "Allocate data failed.";
      delete ms_tensor;
      return nullptr;
    }
    ::memcpy(ms_tensor->MutableData(), impl_->MutableData(), data_len);
  }
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
  return impl_->ElementNum();
}

enum DataType MSTensor::DataType() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return DataType::kTypeUnknown;
  }
  return impl_->DataType();
}

const std::vector<int64_t> &MSTensor::Shape() const {
  static std::vector<int64_t> empty;
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

size_t MSTensor::DataSize() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return 0;
  }
  return impl_->DataSize();
}

bool MSTensor::IsDevice() const {
  MS_LOG(ERROR) << "Unsupported feature.";
  return false;
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
  impl_->SetShape(shape);
}

void MSTensor::SetDataType(enum DataType data_type) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return;
  }
  impl_->SetDataType(data_type);
}

void MSTensor::SetTensorName(const std::string &name) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return;
  }
  impl_->SetName(name);
}

void MSTensor::SetAllocator(std::shared_ptr<Allocator> allocator) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return;
  }
  return impl_->SetAllocator(allocator);
}

std::shared_ptr<Allocator> MSTensor::allocator() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return nullptr;
  }
  return impl_->allocator();
}

void MSTensor::SetFormat(mindspore::Format format) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return;
  }
  return impl_->SetFormat(format);
}

mindspore::Format MSTensor::format() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return mindspore::Format::NHWC;
  }
  return impl_->format();
}

void MSTensor::SetData(void *data) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return;
  }
  return impl_->SetData(data);
}

std::vector<QuantParam> MSTensor::QuantParams() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return std::vector<QuantParam>{};
  }
  return impl_->QuantParams();
}

void MSTensor::SetQuantParams(std::vector<QuantParam> quant_params) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor implement.";
    return;
  }
  return impl_->SetQuantParams(quant_params);
}

Buffer::Buffer() : impl_(nullptr) { MS_LOG(ERROR) << "Unsupported feature."; }
Buffer::Buffer(const void *data, size_t data_len) : impl_(nullptr) { MS_LOG(ERROR) << "Unsupported feature."; }
Buffer::~Buffer() = default;

Buffer Buffer::Clone() const {
  MS_LOG(ERROR) << "Unsupported feature.";
  return Buffer();
}

const void *Buffer::Data() const {
  MS_LOG(ERROR) << "Unsupported feature.";
  return nullptr;
}

void *Buffer::MutableData() {
  MS_LOG(ERROR) << "Unsupported feature.";
  return nullptr;
}

size_t Buffer::DataSize() const {
  MS_LOG(ERROR) << "Unsupported feature.";
  return 0;
}

bool Buffer::ResizeData(size_t data_len) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return false;
}

bool Buffer::SetData(const void *data, size_t data_len) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return false;
}

std::vector<char> CharVersion() { return StringToChar(lite::Version()); }
}  // namespace mindspore
