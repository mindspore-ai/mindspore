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

#ifndef MINDSPORE_LITE_SRC_CXX_API_TENSOR_TENSOR_IMPL_H_
#define MINDSPORE_LITE_SRC_CXX_API_TENSOR_TENSOR_IMPL_H_

#include <cstddef>
#include <numeric>
#include <memory>
#include <algorithm>
#include <string>
#include <vector>
#include <functional>
#include "include/api/types.h"
#include "include/api/status.h"
#include "include/errorcode.h"
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

  static std::shared_ptr<Impl> MS_API CreateTensorImpl(const std::string &name, enum DataType type,
                                                       const std::vector<int64_t> &shape, const void *data,
                                                       size_t data_len);

#ifndef STRING_KERNEL_CLIP
  static std::shared_ptr<Impl> MS_API StringsToTensorImpl(const std::string &name, const std::vector<std::string> &str);

  static std::vector<std::string> MS_API TensorImplToStrings(const std::shared_ptr<Impl> &impl);
#endif

  virtual const std::string &Name() const {
    static std::string empty = "";
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return empty;
    }
    return tensor_name_;
  }

  void SetName(const std::string &name) {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return;
    }
    lite_tensor_->set_tensor_name(name);
    tensor_name_ = name;
  }

  virtual enum DataType DataType() const {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return DataType::kTypeUnknown;
    }
    return static_cast<enum DataType>(lite_tensor_->data_type());
  }

  void SetDataType(enum DataType data_type) {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return;
    }
    lite_tensor_->set_data_type(static_cast<enum TypeId>(data_type));
  }

  int64_t ElementNum() const {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return -1;
    }
    return static_cast<int64_t>(lite_tensor_->ElementsNum());
  }

  virtual const std::vector<int64_t> &Shape() const {
    static std::vector<int64_t> empty;
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return empty;
    }
    auto shape = lite_tensor_->shape();
    lite_shape_.resize(shape.size());
    std::transform(shape.begin(), shape.end(), lite_shape_.begin(), [](int c) { return static_cast<int64_t>(c); });
    return lite_shape_;
  }

  virtual std::shared_ptr<Impl> Clone() const { return nullptr; }

  void SetShape(const std::vector<int64_t> &shape) {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return;
    }
    std::vector<int> tensor_shape;
    tensor_shape.resize(shape.size());
    std::transform(shape.begin(), shape.end(), tensor_shape.begin(), [](int64_t c) { return static_cast<int>(c); });
    lite_tensor_->set_shape(tensor_shape);
  }

  std::shared_ptr<Allocator> allocator() const {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return nullptr;
    }
    return lite_tensor_->allocator();
  }

  void SetAllocator(std::shared_ptr<Allocator> allocator) {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return;
    }
    lite_tensor_->set_allocator(allocator);
  }

  mindspore::Format format() {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return mindspore::Format::NHWC;
    }
    return lite_tensor_->format();
  }

  void SetFormat(mindspore::Format format) {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return;
    }
    lite_tensor_->set_format(format);
  }

  virtual std::shared_ptr<const void> Data() const {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
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
  virtual bool IsConst() const {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return false;
    }
    return lite_tensor_->IsConst();
  }

  virtual size_t DataSize() const {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return 0;
    }
    return lite_tensor_->Size();
  }

  void SetData(void *data) {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return;
    }
    lite_tensor_->set_data(data);
  }

  virtual std::vector<QuantParam> QuantParams() const {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return std::vector<QuantParam>{};
    }
    auto lite_quant_params = lite_tensor_->quant_params();
    std::vector<QuantParam> quant_params;
    for (size_t i = 0; i < lite_quant_params.size(); i++) {
      QuantParam param{};
      param.bit_num = lite_quant_params[i].bitNum;
      param.scale = lite_quant_params[i].scale;
      param.zero_point = lite_quant_params[i].zeroPoint;
      quant_params.push_back(param);
    }
    return quant_params;
  }

  void SetQuantParams(std::vector<QuantParam> quant_params) {
    if (lite_tensor_ == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor.";
      return;
    }
    std::vector<lite::LiteQuantParam> lite_quant_params;
    for (size_t i = 0; i < quant_params.size(); i++) {
      lite::LiteQuantParam lite_param{};
      lite_param.bitNum = quant_params[i].bit_num;
      lite_param.scale = quant_params[i].scale;
      lite_param.zeroPoint = quant_params[i].zero_point;
      lite_quant_params.push_back(lite_param);
    }
    lite_tensor_->set_quant_params(lite_quant_params);
  }

  virtual bool IsDevice() const { return false; }

  tensor::MSTensor *lite_tensor() const { return lite_tensor_; }

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
  mutable std::vector<int64_t> lite_shape_;
  bool own_data_ = false;
  bool from_session_ = false;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_CXX_API_TENSOR_TENSOR_IMPL_H_
