
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

#ifndef MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_SESSION_H_
#define MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_SESSION_H_

#include "include/errorcode.h"
#include "include/lite_session.h"

#include "tensor.h"

namespace mindspore {
namespace lite {

#define MS_ERROR_IF_NULL(ptr)            \
  do {                                   \
    if ((ptr) == nullptr) {              \
      return mindspore::lite::RET_ERROR; \
    }                                    \
  } while (0)

#define MS_NULLPTR_IF_NULL(ptr) \
  do {                          \
    if ((ptr) == nullptr) {     \
      return nullptr;           \
    }                           \
  } while (0)

#define MS_NULLPTR_IF_ERROR(ptr)            \
  do {                                      \
    if ((ptr) != mindspore::lite::RET_OK) { \
      return nullptr;                       \
    }                                       \
  } while (0)

class LiteSession : public session::LiteSession {
 public:
  LiteSession() = default;

  ~LiteSession() override;

  void BindThread(bool if_bind) override {}

  int CompileGraph(lite::Model *model) override;

  Vector<tensor::MSTensor *> GetInputs() const override;

  mindspore::tensor::MSTensor *GetInputsByTensorName(const String &tensor_name) const override { return nullptr; }

  int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr) override;

  Vector<tensor::MSTensor *> GetOutputsByNodeName(const String &node_name) const override;

  Vector<String> GetOutputTensorNames() const override;

  mindspore::tensor::MSTensor *GetOutputByTensorName(const String &tensor_name) const override;

  int Resize(const Vector<tensor::MSTensor *> &inputs, const Vector<Vector<int>> &dims) override { return RET_ERROR; }

  int InitRuntimeBuffer();

 private:
  Vector<MTensor *> inputs_;
  Vector<MTensor *> outputs_;
  void *runtime_buffer_;
};

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_SESSION_H_
