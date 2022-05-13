/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "extendrt/mindir_loader/model_loader.h"

namespace mindspore::infer {
constexpr size_t kMaxModelBufferSize = static_cast<size_t>(1024) * 1024 * 1024 * 2;

int ModelLoader::InitModelBuffer(AbstractBaseModel *model, const char *model_buf, size_t size, bool take_buf) {
  if (model_buf == nullptr || size == 0) {
    MS_LOG(ERROR) << "Input model buffer is nullptr.";
    return mindspore::lite::RET_INPUT_PARAM_INVALID;
  }
  MS_ASSERT(model != nullptr);
  if (take_buf) {
    model->buf = const_cast<char *>(model_buf);
  } else {
    if (size > kMaxModelBufferSize) {
      MS_LOG(ERROR) << "Input model buffer size invalid, require (0, 2GB].";
      return mindspore::lite::RET_ERROR;
    }
    model->buf = new char[size];
    if (model->buf == nullptr) {
      MS_LOG(ERROR) << "new inner model buf fail!";
      return mindspore::lite::RET_NULL_PTR;
    }
    memcpy(model->buf, model_buf, size);
  }
  model->buf_size_ = size;
  return mindspore::lite::RET_OK;
}

ModelLoaderRegistry::ModelLoaderRegistry() {}

ModelLoaderRegistry::~ModelLoaderRegistry() {}

ModelLoaderRegistry *ModelLoaderRegistry::GetInstance() {
  static ModelLoaderRegistry instance;
  return &instance;
}
}  // namespace mindspore::infer
