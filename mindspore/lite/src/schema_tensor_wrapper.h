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

#ifndef MINDSPORE_LITE_SRC_SCHEMA_TENSOR_WRAPPER_H_
#define MINDSPORE_LITE_SRC_SCHEMA_TENSOR_WRAPPER_H_

#include "src/common/version_manager.h"
#include "schema/model_generated.h"

namespace mindspore {
namespace lite {
class SchemaTensorWrapper {
 public:
  class TensorData {
   public:
    explicit TensorData(size_t tensor_id) : tensor_id_(tensor_id) {}
    virtual ~TensorData() {
      if (if_own_data_) {
        free(data_);
        data_ = nullptr;
      }
    }
    bool Init(const schema::Tensor &tensor, SCHEMA_VERSION schema_version);
    const size_t tensor_id_;
    size_t length_ = 0;
    void *data_ = nullptr;
    bool if_own_data_ = true;
  };

  explicit SchemaTensorWrapper(schema::Tensor *handler, TensorData *data = nullptr) : handler_(handler), data_(data) {}
  virtual ~SchemaTensorWrapper() {
    delete this->data_;
    this->data_ = nullptr;
  }
  const schema::Tensor *handler() const { return this->handler_; }
  const TensorData *data() const { return this->data_; }

 private:
  schema::Tensor *handler_ = nullptr;
  TensorData *data_ = nullptr;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_SCHEMA_TENSOR_WRAPPER_H_
