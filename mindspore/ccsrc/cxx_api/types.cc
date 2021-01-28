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

namespace mindspore::api {
const char *kDeviceTypeAscend310 = "Ascend310";
const char *kDeviceTypeAscend910 = "Ascend910";
const char *kDeviceTypeGpu = "GPU";

class DataImpl {
 public:
  DataImpl() : data_() {}
  ~DataImpl() = default;
  DataImpl(const void *data, size_t data_len) { SetData(data, data_len); }

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

class Buffer::Impl : public DataImpl {
 public:
  Impl() : DataImpl() {}
  ~Impl() = default;
  Impl(const void *data, size_t data_len) : DataImpl(data, data_len) {}
};

class Tensor::Impl : public DataImpl {
 public:
  Impl() : DataImpl(), name_(), type_(DataType::kMsUnknown), shape_() {}
  ~Impl() = default;
  Impl(const std::string &name, api::DataType type, const std::vector<int64_t> &shape, const void *data,
       size_t data_len)
      : DataImpl(data, data_len), name_(name), type_(type), shape_(shape) {}

  const std::string &Name() const { return name_; }
  void SetName(const std::string &name) { name_ = name; }

  api::DataType DataType() const { return type_; }
  void SetDataType(api::DataType type) { type_ = type; }

  void SetShape(const std::vector<int64_t> &shape) { shape_ = shape; }
  const std::vector<int64_t> &Shape() const { return shape_; }

  int64_t ElementNum() const {
    std::vector<int64_t> shapex = Shape();
    return std::accumulate(shapex.begin(), shapex.end(), 1LL, std::multiplies<int64_t>());
  }

  static int GetTypeSize(api::DataType type) {
    static const std::map<api::DataType, size_t> type_size_map = {
      {kMsBool, sizeof(bool)},       {kMsFloat64, sizeof(double)},   {kMsInt8, sizeof(int8_t)},
      {kMsUint8, sizeof(uint8_t)},   {kMsInt16, sizeof(int16_t)},    {kMsUint16, sizeof(uint16_t)},
      {kMsInt32, sizeof(int32_t)},   {kMsUint32, sizeof(uint32_t)},  {kMsInt64, sizeof(int64_t)},
      {kMsUint64, sizeof(uint64_t)}, {kMsFloat16, sizeof(uint16_t)}, {kMsFloat32, sizeof(float)},
    };
    auto it = type_size_map.find(type);
    if (it != type_size_map.end()) {
      return it->second;
    }

    MS_LOG(WARNING) << "Cannot find data type " << type;
    return 0;
  }

 private:
  std::string name_;
  api::DataType type_;
  std::vector<int64_t> shape_;
};

Tensor::Tensor() : impl_(std::make_shared<Impl>()) {}
Tensor::Tensor(const std::string &name, api::DataType type, const std::vector<int64_t> &shape, const void *data,
               size_t data_len)
    : impl_(std::make_shared<Impl>(name, type, shape, data, data_len)) {}
Tensor::~Tensor() = default;

Tensor Tensor::Clone() const {
  MS_EXCEPTION_IF_NULL(impl_);
  Tensor ret;
  ret.impl_ = std::make_shared<Impl>(*impl_);
  return ret;
}

const std::string &Tensor::Name() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Name();
}

void Tensor::SetName(const std::string &name) {
  MS_EXCEPTION_IF_NULL(impl_);
  impl_->SetName(name);
}

DataType Tensor::DataType() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->DataType();
}

void Tensor::SetDataType(api::DataType type) {
  MS_EXCEPTION_IF_NULL(impl_);
  impl_->SetDataType(type);
}

const std::vector<int64_t> &Tensor::Shape() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Shape();
}

void Tensor::SetShape(const std::vector<int64_t> &shape) {
  MS_EXCEPTION_IF_NULL(impl_);
  impl_->SetShape(shape);
}

const void *Tensor::Data() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Data();
}

void *Tensor::MutableData() {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->MutableData();
}

size_t Tensor::DataSize() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->DataSize();
}

bool Tensor::ResizeData(size_t data_len) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->ResizeData(data_len);
}

bool Tensor::SetData(const void *data, size_t data_len) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->SetData(data, data_len);
}

int64_t Tensor::ElementNum() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->ElementNum();
}

int Tensor::GetTypeSize(api::DataType type) { return Impl::GetTypeSize(type); }

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
}  // namespace mindspore::api
