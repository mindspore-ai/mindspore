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
#include <cstddef>
#include <numeric>
#include <memory>
#include <algorithm>
#include <string>
#include <vector>
#include <functional>
#include "include/api/types.h"
#include "include/api/status.h"
#include "include/ms_tensor.h"
#include "src/tensor.h"
#include "src/common/log_adapter.h"

namespace mindspore {
class MSTensor::Impl {
 public:
  Impl() {}
  virtual ~Impl() = default;
  explicit Impl(tensor::MSTensor *tensor) : lite_tensor_(tensor) {
    if (tensor != nullptr) {
      tensor_name_ = tensor->tensor_name();
    }
  }

  bool operator==(std::nullptr_t) const { return lite_tensor_ == nullptr; }

  Impl(const std::string &name, enum DataType type, const std::vector<int64_t> &shape, const void *data,
       size_t data_len);

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

    return std::shared_ptr<const void>(lite_tensor_->MutableData(), [](const void *) {});
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

  virtual std::shared_ptr<Impl> Clone() const {
    MS_LOG(ERROR) << "Unsupported feature.";
    return nullptr;
  }

  tensor::MSTensor *lite_tensor() { return lite_tensor_; }

  Status set_lite_tensor(tensor::MSTensor *tensor) {
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Tensor to set is null.";
      return kLiteNullptr;
    }
    lite_tensor_ = tensor;
    return kSuccess;
  }

  void set_need_copy(bool need_copy) { need_copy_ = need_copy; }

  bool need_copy() { return need_copy_; }

 private:
  tensor::MSTensor *lite_tensor_;
  std::string tensor_name_;
  std::vector<int64_t> shape_;
  bool need_copy_ = true;
};

}  // namespace mindspore
