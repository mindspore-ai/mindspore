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

#ifndef MINDSPORE_LITE_SRC_CXX_API_TENSOR_TENSOR_IMPL_H
#define MINDSPORE_LITE_SRC_CXX_API_TENSOR_TENSOR_IMPL_H

#include <cstddef>
#include <numeric>
#include <memory>
#include <algorithm>
#include <string>
#include <vector>
#include <functional>
#include "include/api/types.h"
#include "include/api/status.h"
#include "include/lite_utils.h"
#include "include/ms_tensor.h"
#include "src/tensor.h"
#include "src/common/log_adapter.h"

namespace mindspore {
using mindspore::lite::RET_OK;

class MSTensor::Impl {
 public:
  Impl() {}

  virtual ~Impl() {
    if (lite_tensor_ == nullptr) {
      return;
    }
    if (!from_session_) {
      if (!own_data_) {
        lite_tensor_->set_data(nullptr);
      }
      delete lite_tensor_;
      lite_tensor_ = nullptr;
    }
  }

  explicit Impl(tensor::MSTensor *tensor) : lite_tensor_(tensor), from_session_(true) {
    if (tensor != nullptr) {
      tensor_name_ = tensor->tensor_name();
    }
  }

  static std::shared_ptr<Impl> CreateTensorImpl(const std::string &name, enum DataType type,
                                                const std::vector<int64_t> &shape, const void *data, size_t data_len);

  static std::shared_ptr<Impl> StringsToTensorImpl(const std::string &name, const std::vector<std::string> &str);

  static std::vector<std::string> TensorImplToStrings(const std::shared_ptr<Impl> &impl) {
    std::vector<std::string> empty;
    auto lite_tensor = impl->lite_tensor();
    if (lite_tensor == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor impl.";
      return empty;
    }
    return lite::MSTensorToStrings(lite_tensor);
  }

  virtual const std::string &Name() const {
    static std::string empty = "";
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return empty;
    }
    return tensor_name_;
  }

  virtual enum DataType DataType() const {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return DataType::kTypeUnknown;
    }
    return static_cast<enum DataType>(lite_tensor_->data_type());
  }

  int64_t ElementNum() const {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return -1;
    }
    return static_cast<int64_t>(lite_tensor_->ElementsNum());
  }

  virtual const std::vector<int64_t> &Shape() {
    static std::vector<int64_t> empty;
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return empty;
    }
    auto shape = lite_tensor_->shape();
    shape_.resize(shape.size());
    std::transform(shape.begin(), shape.end(), shape_.begin(), [](int c) { return static_cast<int64_t>(c); });
    return shape_;
  }

  virtual std::shared_ptr<const void> Data() const {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return nullptr;
    }

    if (DataSize() == 0) {
      MS_LOG(ERROR) << "Invalid data size.";
      return nullptr;
    }

    return std::shared_ptr<const void>(lite_tensor_->data(), [](const void *) {});
  }

  virtual void *MutableData() {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return nullptr;
    }
    return lite_tensor_->MutableData();
  }
  virtual size_t DataSize() const {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return 0;
    }
    return lite_tensor_->Size();
  }

  virtual bool IsDevice() const { return false; }

  tensor::MSTensor *lite_tensor() { return lite_tensor_; }

  Status set_lite_tensor(tensor::MSTensor *tensor) {
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Tensor to set is null.";
      return kLiteNullptr;
    }
    lite_tensor_ = tensor;
    return kSuccess;
  }

  void set_own_data(bool own_data) { own_data_ = own_data; }

  void set_from_session(bool from_session) { from_session_ = from_session; }

 private:
  tensor::MSTensor *lite_tensor_ = nullptr;
  std::string tensor_name_ = "";
  std::vector<int64_t> shape_ = {};
  bool own_data_ = false;
  bool from_session_ = false;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_CXX_API_TENSOR_TENSOR_IMPL_H
