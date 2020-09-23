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

#ifndef MINDSPORE_INCLUDE_INFER_TENSOR_H_
#define MINDSPORE_INCLUDE_INFER_TENSOR_H_

#include <utility>
#include <vector>
#include <memory>
#include <numeric>
#include <map>
#include <functional>

#include "securec/include/securec.h"
#include "include/infer_log.h"

namespace mindspore {
#define MS_API __attribute__((visibility("default")))
namespace inference {
enum DataType {
  kMSI_Unknown = 0,
  kMSI_Bool = 1,
  kMSI_Int8 = 2,
  kMSI_Int16 = 3,
  kMSI_Int32 = 4,
  kMSI_Int64 = 5,
  kMSI_Uint8 = 6,
  kMSI_Uint16 = 7,
  kMSI_Uint32 = 8,
  kMSI_Uint64 = 9,
  kMSI_Float16 = 10,
  kMSI_Float32 = 11,
  kMSI_Float64 = 12,
};

class InferTensorBase {
 public:
  InferTensorBase() = default;
  virtual ~InferTensorBase() = default;

  virtual DataType data_type() const = 0;
  virtual void set_data_type(DataType type) = 0;
  virtual std::vector<int64_t> shape() const = 0;
  virtual void set_shape(const std::vector<int64_t> &shape) = 0;
  virtual const void *data() const = 0;
  virtual size_t data_size() const = 0;
  virtual bool resize_data(size_t data_len) = 0;
  virtual void *mutable_data() = 0;

  bool set_data(const void *data, size_t data_len) {
    resize_data(data_len);
    if (mutable_data() == nullptr) {
      MSI_LOG_ERROR << "set data failed, data len " << data_len;
      return false;
    }
    if (data_size() != data_len) {
      MSI_LOG_ERROR << "set data failed, tensor current data size " << data_size() << " not match data len "
                    << data_len;
      return false;
    }
    if (data_len == 0) {
      return true;
    }
    auto ret = memcpy_s(mutable_data(), data_size(), data, data_len);
    if (ret != 0) {
      MSI_LOG_ERROR << "Set data memcpy_s failed";
      return false;
    }
    return true;
  }

  int64_t ElementNum() const {
    std::vector<int64_t> shapex = shape();
    return std::accumulate(shapex.begin(), shapex.end(), 1LL, std::multiplies<int64_t>());
  }

  int GetTypeSize(DataType type) const {
    const std::map<DataType, size_t> type_size_map{
      {kMSI_Bool, sizeof(bool)},       {kMSI_Float64, sizeof(double)},   {kMSI_Int8, sizeof(int8_t)},
      {kMSI_Uint8, sizeof(uint8_t)},   {kMSI_Int16, sizeof(int16_t)},    {kMSI_Uint16, sizeof(uint16_t)},
      {kMSI_Int32, sizeof(int32_t)},   {kMSI_Uint32, sizeof(uint32_t)},  {kMSI_Int64, sizeof(int64_t)},
      {kMSI_Uint64, sizeof(uint64_t)}, {kMSI_Float16, sizeof(uint16_t)}, {kMSI_Float32, sizeof(float)},
    };
    auto it = type_size_map.find(type);
    if (it != type_size_map.end()) {
      return it->second;
    }
    return 0;
  }
};

class InferTensor : public InferTensorBase {
 public:
  DataType type_;
  std::vector<int64_t> shape_;
  std::vector<uint8_t> data_;

 public:
  InferTensor() = default;
  ~InferTensor() = default;
  InferTensor(DataType type, std::vector<int64_t> shape, const void *data, size_t data_len) {
    set_data_type(type);
    set_shape(shape);
    set_data(data, data_len);
  }

  void set_data_type(DataType type) override { type_ = type; }
  DataType data_type() const override { return type_; }

  void set_shape(const std::vector<int64_t> &shape) override { shape_ = shape; }
  std::vector<int64_t> shape() const override { return shape_; }

  const void *data() const override { return data_.data(); }
  size_t data_size() const override { return data_.size(); }

  bool resize_data(size_t data_len) override {
    data_.resize(data_len);
    return true;
  }
  void *mutable_data() override { return data_.data(); }
};

class InferImagesBase {
 public:
  InferImagesBase() = default;
  virtual ~InferImagesBase() = default;
  virtual size_t batch_size() const = 0;
  virtual bool get(size_t index, const void *&pic_buffer, uint32_t &pic_size) const = 0;
  virtual size_t input_index() const = 0;  // the index of images as input in model
};

class RequestBase {
 public:
  RequestBase() = default;
  virtual ~RequestBase() = default;
  virtual size_t size() const = 0;
  virtual const InferTensorBase *operator[](size_t index) const = 0;
};

class ImagesRequestBase {
 public:
  ImagesRequestBase() = default;
  virtual ~ImagesRequestBase() = default;
  virtual size_t size() const = 0;
  virtual const InferImagesBase *operator[](size_t index) const = 0;
};

class ReplyBase {
 public:
  ReplyBase() = default;
  virtual ~ReplyBase() = default;
  virtual size_t size() const = 0;
  virtual InferTensorBase *operator[](size_t index) = 0;
  virtual const InferTensorBase *operator[](size_t index) const = 0;
  virtual InferTensorBase *add() = 0;
  virtual void clear() = 0;
};

class VectorInferTensorWrapReply : public ReplyBase {
 public:
  explicit VectorInferTensorWrapReply(std::vector<InferTensor> &tensor_list) : tensor_list_(tensor_list) {}
  ~VectorInferTensorWrapReply() = default;

  size_t size() const { return tensor_list_.size(); }
  InferTensorBase *operator[](size_t index) {
    if (index >= tensor_list_.size()) {
      MSI_LOG_ERROR << "visit invalid index " << index << " total size " << tensor_list_.size();
      return nullptr;
    }
    return &(tensor_list_[index]);
  }
  const InferTensorBase *operator[](size_t index) const {
    if (index >= tensor_list_.size()) {
      MSI_LOG_ERROR << "visit invalid index " << index << " total size " << tensor_list_.size();
      return nullptr;
    }
    return &(tensor_list_[index]);
  }
  InferTensorBase *add() {
    tensor_list_.push_back(InferTensor());
    return &(tensor_list_.back());
  }
  void clear() { tensor_list_.clear(); }
  std::vector<InferTensor> &tensor_list_;
};

class VectorInferTensorWrapRequest : public RequestBase {
 public:
  explicit VectorInferTensorWrapRequest(const std::vector<InferTensor> &tensor_list) : tensor_list_(tensor_list) {}
  ~VectorInferTensorWrapRequest() = default;

  size_t size() const { return tensor_list_.size(); }
  const InferTensorBase *operator[](size_t index) const {
    if (index >= tensor_list_.size()) {
      MSI_LOG_ERROR << "visit invalid index " << index << " total size " << tensor_list_.size();
      return nullptr;
    }
    return &(tensor_list_[index]);
  }
  const std::vector<InferTensor> &tensor_list_;
};
}  // namespace inference
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_INFER_TENSOR_H_
