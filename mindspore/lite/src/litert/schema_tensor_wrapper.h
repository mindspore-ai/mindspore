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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_SCHEMA_TENSOR_WRAPPER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_SCHEMA_TENSOR_WRAPPER_H_

#include <utility>
#include <string>
#include "src/common/version_manager.h"
#include "schema/model_generated.h"
#ifdef ENABLE_LITE_HELPER
#include "src/common/helper/infer_helpers.h"
#endif

namespace mindspore {
namespace lite {
class SchemaTensorWrapper {
 public:
  SchemaTensorWrapper() = default;
  virtual ~SchemaTensorWrapper() {
    if (if_own_data_) {
      free(data_);
      data_ = nullptr;
    }
    this->data_ = nullptr;
  }

#ifdef ENABLE_LITE_HELPER
  bool Init(const schema::Tensor &tensor, SCHEMA_VERSION schema_version, const std::string &base_path,
            mindspore::infer::helper::InferHelpers *infer_helpers = nullptr);
#else
  bool Init(const schema::Tensor &tensor, SCHEMA_VERSION schema_version, const std::string &base_path);
#endif

  const schema::Tensor *handler() const { return this->handler_; }

  const void *data() const { return this->data_; }

  size_t length() const { return this->length_; }

  std::pair<bool, void *> ReleaseData() {
    std::pair<bool, void *> ret = std::make_pair(this->if_own_data_, this->data_);
    this->if_own_data_ = false;
    return ret;
  }

 private:
  const schema::Tensor *handler_ = nullptr;
  size_t length_ = 0;
  void *data_ = nullptr;
  bool if_own_data_ = true;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_SCHEMA_TENSOR_WRAPPER_H_
