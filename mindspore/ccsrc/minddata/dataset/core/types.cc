/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "include/dataset/ms_tensor.h"
#include "include/securec.h"
#include "utils/convert_utils_base.h"
#include "utils/file_utils.h"

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
    if (ret != EOK) {
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

  enum DataType DataType() const override { return type_; }

  const std::vector<int64_t> &Shape() const override { return shape_; }

  std::shared_ptr<const void> Data() const override {
    return std::shared_ptr<const void>(buffer_.Data(), [](const void *) {});
  }

  void *MutableData() override { return buffer_.MutableData(); }
  size_t DataSize() const override { return buffer_.DataSize(); }

 private:
  Buffer buffer_;
  std::string name_;
  enum DataType type_;
  std::vector<int64_t> shape_;
};

MSTensor::MSTensor() : impl_(std::make_shared<TensorDefaultImpl>()) {}

MSTensor::MSTensor(const std::shared_ptr<Impl> &impl) : impl_(impl) { MS_EXCEPTION_IF_NULL(impl); }

MSTensor::MSTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                   const void *data, size_t data_len)
    : impl_(std::make_shared<TensorDefaultImpl>(CharToString(name), type, shape, data, data_len)) {}

MSTensor::~MSTensor() = default;

enum DataType MSTensor::DataType() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->DataType();
}

const std::vector<int64_t> &MSTensor::Shape() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Shape();
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

Buffer::Buffer() : impl_(std::make_shared<Impl>()) {}

Buffer::Buffer(const void *data, size_t data_len) : impl_(std::make_shared<Impl>(data, data_len)) {}

Buffer::~Buffer() = default;

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
}  // namespace mindspore
