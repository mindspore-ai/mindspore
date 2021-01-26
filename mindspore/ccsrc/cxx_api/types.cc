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

MSTensor MSTensor::CreateTensor(const std::string &name, enum DataType type, const std::vector<int64_t> &shape,
                                const void *data, size_t data_len) noexcept {
  try {
    std::shared_ptr<Impl> impl = std::make_shared<TensorDefaultImpl>(name, type, shape, data, data_len);
    return MSTensor(impl);
  } catch (const std::bad_alloc &) {
    MS_LOG(ERROR) << "Malloc memory failed.";
    return MSTensor(nullptr);
  } catch (...) {
    MS_LOG(ERROR) << "Unknown error occurred.";
    return MSTensor(nullptr);
  }
}

MSTensor MSTensor::CreateRefTensor(const std::string &name, enum DataType type, const std::vector<int64_t> &shape,
                                   const void *data, size_t data_len) noexcept {
  try {
    std::shared_ptr<Impl> impl = std::make_shared<TensorReferenceImpl>(name, type, shape, data, data_len);
    return MSTensor(impl);
  } catch (const std::bad_alloc &) {
    MS_LOG(ERROR) << "Malloc memory failed.";
    return MSTensor(nullptr);
  } catch (...) {
    MS_LOG(ERROR) << "Unknown error occurred.";
    return MSTensor(nullptr);
  }
}

MSTensor::MSTensor() : impl_(std::make_shared<TensorDefaultImpl>()) {}
MSTensor::MSTensor(std::nullptr_t) : impl_(nullptr) {}
MSTensor::MSTensor(const std::shared_ptr<Impl> &impl) : impl_(impl) { MS_EXCEPTION_IF_NULL(impl); }
MSTensor::MSTensor(const std::string &name, enum DataType type, const std::vector<int64_t> &shape, const void *data,
                   size_t data_len)
    : impl_(std::make_shared<TensorDefaultImpl>(name, type, shape, data, data_len)) {}
MSTensor::~MSTensor() = default;

bool MSTensor::operator==(std::nullptr_t) const { return impl_ == nullptr; }

MSTensor MSTensor::Clone() const {
  MS_EXCEPTION_IF_NULL(impl_);
  MSTensor ret;
  ret.impl_ = impl_->Clone();
  return ret;
}

const std::string &MSTensor::Name() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Name();
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
