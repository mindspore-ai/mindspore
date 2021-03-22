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
#include <numeric>
#include "securec/include/securec.h"
#include "utils/utils.h"
#include "mindspore/core/ir/api_tensor_impl.h"

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

  bool ResizeData(size_t data_len) {
    data_.resize(data_len);
    return true;
  }

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

    auto ret = memcpy_s(MutableData(), DataSize(), data, data_len);
    if (ret != 0) {
      MS_LOG(ERROR) << "Set data memcpy_s failed, ret = " << ret;
      return false;
    }
    return true;
  }

 protected:
  std::vector<uint8_t> data_;
};

class TensorDefaultImpl : public MSTensor::Impl {
 public:
  TensorDefaultImpl() : buffer_(), name_(), type_(DataType::kTypeUnknown), shape_() {}
  ~TensorDefaultImpl() override = default;
  TensorDefaultImpl(const std::string &name, enum DataType type, const std::vector<int64_t> &shape, const void *data,
                    size_t data_len)
      : buffer_(data, data_len), name_(name), type_(type), shape_(shape) {}

  const std::string &Name() const override { return name_; }
  enum DataType DataType() const override { return type_; }
  const std::vector<int64_t> &Shape() const override { return shape_; }

  std::shared_ptr<const void> Data() const override {
    return std::shared_ptr<const void>(buffer_.Data(), [](const void *) {});
  }

  void *MutableData() override { return buffer_.MutableData(); }
  size_t DataSize() const override { return buffer_.DataSize(); }

  bool IsDevice() const override { return false; }

  std::shared_ptr<Impl> Clone() const override {
    return std::make_shared<TensorDefaultImpl>(name_, type_, shape_, buffer_.Data(), buffer_.DataSize());
  }

 private:
  Buffer buffer_;
  std::string name_;
  enum DataType type_;
  std::vector<int64_t> shape_;
};

class TensorReferenceImpl : public MSTensor::Impl {
 public:
  TensorReferenceImpl() : data_(nullptr), data_size_(0), name_(), type_(DataType::kTypeUnknown), shape_() {}
  ~TensorReferenceImpl() override = default;
  TensorReferenceImpl(const std::string &name, enum DataType type, const std::vector<int64_t> &shape, const void *data,
                      size_t data_len)
      : data_(data), data_size_(data_len), name_(name), type_(type), shape_(shape) {}

  const std::string &Name() const override { return name_; }
  enum DataType DataType() const override { return type_; }
  const std::vector<int64_t> &Shape() const override { return shape_; }

  std::shared_ptr<const void> Data() const override {
    return std::shared_ptr<const void>(data_, [](const void *) {});
  }

  void *MutableData() override { return const_cast<void *>(data_); }
  size_t DataSize() const override { return data_size_; }

  bool IsDevice() const override { return false; }

  std::shared_ptr<Impl> Clone() const override {
    return std::make_shared<TensorReferenceImpl>(name_, type_, shape_, data_, data_size_);
  }

 protected:
  const void *data_;
  size_t data_size_;
  std::string name_;
  enum DataType type_;
  std::vector<int64_t> shape_;
};

MSTensor *MSTensor::CreateTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                                 const void *data, size_t data_len) noexcept {
  std::string name_str = CharToString(name);
  try {
    std::shared_ptr<Impl> impl = std::make_shared<TensorDefaultImpl>(name_str, type, shape, data, data_len);
    MSTensor *ret = new MSTensor(impl);
    return ret;
  } catch (const std::bad_alloc &) {
    MS_LOG(ERROR) << "Malloc memory failed.";
    return nullptr;
  } catch (...) {
    MS_LOG(ERROR) << "Unknown error occurred.";
    return nullptr;
  }
}

MSTensor *MSTensor::CreateRefTensor(const std::vector<char> &name, enum DataType type,
                                    const std::vector<int64_t> &shape, const void *data, size_t data_len) noexcept {
  std::string name_str = CharToString(name);
  try {
    std::shared_ptr<Impl> impl = std::make_shared<TensorReferenceImpl>(name_str, type, shape, data, data_len);
    MSTensor *ret = new MSTensor(impl);
    return ret;
  } catch (const std::bad_alloc &) {
    MS_LOG(ERROR) << "Malloc memory failed.";
    return nullptr;
  } catch (...) {
    MS_LOG(ERROR) << "Unknown error occurred.";
    return nullptr;
  }
}

MSTensor *MSTensor::CharStringsToTensor(const std::vector<char> &name, const std::vector<std::vector<char>> &str) {
  // num(4 bytes) + offset1(4 bytes) + offset2(4 bytes) + ... + data1(str1.len) + data2(str2.len) + ...
  // str1.len() = offset2 - offset1
  // data1.begin() = start + offset1
  size_t mem_size = 0;
  mem_size += sizeof(int32_t);  // for num
  for (const auto &s : str) {
    mem_size += sizeof(int32_t);  // for offset
    mem_size += s.size();         // for data
  }

  auto tensor = CreateTensor(name, DataType::kObjectTypeString, {static_cast<int64_t>(mem_size)}, nullptr, mem_size);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Create tensor failed.";
    return nullptr;
  }

  int32_t *data = reinterpret_cast<int32_t *>(tensor->MutableData());
  if (data == nullptr) {
    MS_LOG(ERROR) << "Create tensor failed.";
    DestroyTensorPtr(tensor);
    return nullptr;
  }
  uint8_t *cur_data = reinterpret_cast<uint8_t *>(data + 1 + str.size());
  *reinterpret_cast<int32_t *>(data) = str.size();
  for (size_t i = 0; i < str.size(); ++i) {
    int32_t offset = (cur_data - reinterpret_cast<uint8_t *>(data));
    data[i + 1] = offset;
    if (str[i].empty()) {
      continue;
    }
    auto ret = memcpy_s(reinterpret_cast<void *>(cur_data), str[i].size(), str[i].data(), str[i].size());
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s failed, ret = " << ret;
      DestroyTensorPtr(tensor);
      return nullptr;
    }
    cur_data += str[i].size();
  }

  return tensor;
}

std::vector<std::vector<char>> MSTensor::TensorToStringChars(const MSTensor &tensor) {
  if (tensor == nullptr || tensor.DataType() != DataType::kObjectTypeString || tensor.DataSize() < 4) {
    MS_LOG(ERROR) << "Invalid tensor.";
    return {};
  }

  std::vector<std::vector<char>> strings;
  auto host_data = tensor.Data();
  const int32_t *data = reinterpret_cast<const int32_t *>(host_data.get());
  int32_t str_num = data[0];
  if (str_num == 0) {
    return {};
  }
  if (str_num < 0) {
    MS_LOG(ERROR) << "str num " << str_num << " cannot be negative.";
    return {};
  }

  if (tensor.DataSize() < (str_num + 1) * sizeof(int32_t)) {
    MS_LOG(ERROR) << "Invalid tensor data size " << tensor.DataSize() << ", need " << (str_num + 1) * sizeof(int32_t)
                  << " at least for " << str_num << " strings.";
    return {};
  }
  for (size_t i = 0; i < static_cast<size_t>(str_num); ++i) {
    strings.push_back({});
    auto &str = strings[i];
    int32_t str_len;
    int32_t offset = data[i + 1];
    if (i + 1 != static_cast<size_t>(str_num)) {
      str_len = data[i + 1 + 1] - offset;
    } else {
      str_len = tensor.DataSize() - offset;
    }

    if (str_len == 0) {
      continue;
    }

    if (str_len < 0) {
      MS_LOG(ERROR) << "str " << i << " len " << str_len << " cannot be negative.";
      return {};
    }

    str.resize(str_len);
    const uint8_t *cur_data = reinterpret_cast<const uint8_t *>(data) + offset;
    auto ret = memcpy_s(reinterpret_cast<void *>(str.data()), str.size(), cur_data, str_len);
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s failed, ret = " << ret;
      return {};
    }
  }

  return strings;
}

void MSTensor::DestroyTensorPtr(MSTensor *tensor) noexcept {
  if (tensor != nullptr) {
    delete tensor;
  }
}

MSTensor::MSTensor() : impl_(std::make_shared<TensorDefaultImpl>()) {}
MSTensor::MSTensor(std::nullptr_t) : impl_(nullptr) {}
MSTensor::MSTensor(const std::shared_ptr<Impl> &impl) : impl_(impl) { MS_EXCEPTION_IF_NULL(impl); }
MSTensor::MSTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                   const void *data, size_t data_len)
    : impl_(std::make_shared<TensorDefaultImpl>(CharToString(name), type, shape, data, data_len)) {}
MSTensor::~MSTensor() = default;

bool MSTensor::operator==(std::nullptr_t) const { return impl_ == nullptr; }

bool MSTensor::operator!=(std::nullptr_t) const { return impl_ != nullptr; }

MSTensor *MSTensor::Clone() const {
  MS_EXCEPTION_IF_NULL(impl_);
  try {
    MSTensor *ret = new MSTensor();
    ret->impl_ = impl_->Clone();
    return ret;
  } catch (const std::bad_alloc &) {
    MS_LOG(ERROR) << "Malloc memory failed.";
    return nullptr;
  } catch (...) {
    MS_LOG(ERROR) << "Unknown error occurred.";
    return nullptr;
  }
}

std::vector<char> MSTensor::CharName() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return StringToChar(impl_->Name());
}

enum DataType MSTensor::DataType() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->DataType();
}

const std::vector<int64_t> &MSTensor::Shape() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Shape();
}

int64_t MSTensor::ElementNum() const {
  MS_EXCEPTION_IF_NULL(impl_);
  const auto &shape = impl_->Shape();
  if (shape.empty()) {
    // element number of scalar is 1
    return 1;
  }

  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
}

std::shared_ptr<const void> MSTensor::Data() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Data();
}

void *MSTensor::MutableData() {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->MutableData();
}

size_t MSTensor::DataSize() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->DataSize();
}

bool MSTensor::IsDevice() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->IsDevice();
}

Buffer::Buffer() : impl_(std::make_shared<Impl>()) {}
Buffer::Buffer(const void *data, size_t data_len) : impl_(std::make_shared<Impl>(data, data_len)) {}
Buffer::~Buffer() = default;

Buffer Buffer::Clone() const {
  MS_EXCEPTION_IF_NULL(impl_);
  Buffer ret;
  ret.impl_ = std::make_shared<Impl>(*impl_);
  return ret;
}

const void *Buffer::Data() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Data();
}

void *Buffer::MutableData() {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->MutableData();
}

size_t Buffer::DataSize() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->DataSize();
}

bool Buffer::ResizeData(size_t data_len) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->ResizeData(data_len);
}

bool Buffer::SetData(const void *data, size_t data_len) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->SetData(data, data_len);
}
}  // namespace mindspore
