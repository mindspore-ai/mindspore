/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/cxx_api/tensor/tensor_impl.h"
#include "src/tensor.h"
#include "src/common/log_adapter.h"

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
MSTensor::MSTensor(const std::string &name, enum DataType type, const std::vector<int64_t> &shape, const void *data,
                   size_t data_len)
    : impl_(std::make_shared<Impl>(name, type, shape, data, data_len)) {}
MSTensor::~MSTensor() = default;

bool MSTensor::operator==(std::nullptr_t) const { return impl_ == nullptr; }

MSTensor MSTensor::CreateTensor(const std::string &name, enum DataType type, const std::vector<int64_t> &shape,
                                const void *data, size_t data_len) noexcept {
  auto impl = std::make_shared<Impl>(name, type, shape, data, data_len);
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Allocate tensor impl failed.";
    return MSTensor(nullptr);
  }
  return MSTensor(impl);
}

MSTensor MSTensor::CreateRefTensor(const std::string &name, enum DataType type, const std::vector<int64_t> &shape,
                                   const void *data, size_t data_len) noexcept {
  auto tensor = CreateTensor(name, type, shape, data, data_len);
  if (tensor == nullptr) {
    return MSTensor(nullptr);
  }
  tensor.impl_->set_need_copy(false);
  return tensor;
}

MSTensor MSTensor::Clone() const {
  MSTensor ret;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor inpmlement.";
    ret.impl_ = nullptr;
    return ret;
  }
  ret.impl_ = impl_->Clone();
  return ret;
}

const std::string &MSTensor::Name() const {
  static std::string empty = "";
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor inpmlement.";
    return empty;
  }
  return impl_->Name();
}

int64_t MSTensor::ElementNum() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor inpmlement.";
    return -1;
  }
  return impl_->ElementNum();
}

enum DataType MSTensor::DataType() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor inpmlement.";
    return DataType::kTypeUnknown;
  }
  return impl_->DataType();
}

const std::vector<int64_t> &MSTensor::Shape() const {
  static std::vector<int64_t> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor inpmlement.";
    return empty;
  }
  return impl_->Shape();
}

std::shared_ptr<const void> MSTensor::Data() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor inpmlement.";
    return nullptr;
  }
  return impl_->Data();
}

void *MSTensor::MutableData() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor inpmlement.";
    return nullptr;
  }
  return impl_->MutableData();
}

size_t MSTensor::DataSize() const {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor inpmlement.";
    return 0;
  }
  return impl_->DataSize();
}

bool MSTensor::IsDevice() const {
  MS_LOG(ERROR) << "Unsupported feature.";
  return false;
}

Buffer::Buffer() : impl_(std::make_shared<Impl>()) { MS_LOG(ERROR) << "Unsupported feature."; }
Buffer::Buffer(const void *data, size_t data_len) : impl_(std::make_shared<Impl>(data, data_len)) {
  MS_LOG(ERROR) << "Unsupported feature.";
}
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
}  // namespace mindspore
