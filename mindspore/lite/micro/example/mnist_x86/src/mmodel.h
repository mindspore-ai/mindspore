
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
#ifndef MINDSPORE_LITE_LIBRARY_SOURCE_MODEL_H_
#define MINDSPORE_LITE_LIBRARY_SOURCE_MODEL_H_

#include "include/model.h"
#include "session.h"
#include <new>
#include <string.h>

namespace mindspore::lite {
class MModel : public Model {
 public:
  void Free() override {
    if (this->buf != nullptr) {
      free(this->buf);
      this->buf = nullptr;
      this->buf_size_ = 0;
    }
  }

  void Destroy() override { Free(); }

  ~MModel() override { Destroy(); }

  void set_buf_size(size_t size) { buf_size_ = size; }
  size_t buf_size() const { return buf_size_; }

 private:
  size_t buf_size_{0};
};

Model *Model::Import(const char *model_buf, size_t size) {
  MS_NULLPTR_IF_NULL(model_buf);
  if (size == 0) {
    return nullptr;
  }
  MModel *model = new (std::nothrow) MModel();
  MS_NULLPTR_IF_NULL(model);
  model->buf = reinterpret_cast<char *>(malloc(size));
  if (model->buf == nullptr) {
    delete model;
    return nullptr;
  }
  memcpy(model->buf, model_buf, size);
  model->set_buf_size(size);
  return model;
}
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_LIBRARY_SOURCE_MODEL_H_
