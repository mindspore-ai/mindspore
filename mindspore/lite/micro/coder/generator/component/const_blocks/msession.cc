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

#include "coder/generator/component/const_blocks/msession.h"

namespace mindspore::lite::micro {

const char *session_header = R"RAW(
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
)RAW";

const char *session_source = R"RAW(
int LiteSession::RunGraph(const KernelCallBack &before, const KernelCallBack &after) {
  const void *inputs_data[inputs_.size()];
  for (size_t i = 0; i < inputs_.size(); ++i) {
    inputs_data[i] = inputs_[i]->MutableData();
  }
  SetInputs(inputs_data, inputs_.size());

  Inference();

  void *outputs_data[outputs_.size()];
  for (size_t i = 0; i < outputs_.size(); ++i) {
    outputs_data[i] = outputs_[i]->MutableData();
  }
  CopyOutputsData(outputs_data, outputs_.size());

  return RET_OK;
}

LiteSession::~LiteSession() {
  FreeResource();
  if (runtime_buffer_ != nullptr) {
    free(runtime_buffer_);
    runtime_buffer_ = nullptr;
  }
  for (auto &input : inputs_) {
    if (input == nullptr) {
      continue;
    }
    delete input;
    input = nullptr;
  }
  for (auto &output : outputs_) {
    if (output == nullptr) {
      continue;
    }
    delete output;
    output = nullptr;
  }
}

int LiteSession::InitRuntimeBuffer() {
  int buffer_size = GetBufferSize();
  runtime_buffer_ = malloc(buffer_size);
  if (runtime_buffer_ == nullptr) {
    return RET_ERROR;
  }
  int ret = SetBuffer(runtime_buffer_);
  if (ret != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

Vector<tensor::MSTensor *> LiteSession::GetInputs() const {
  Vector<tensor::MSTensor *> inputs;
  for (const auto &input : inputs_) {
    inputs.push_back(input);
  }
  return inputs;
}

Vector<tensor::MSTensor *> LiteSession::GetOutputsByNodeName(const String &node_name) const {
  Vector<tensor::MSTensor *> outputs;
  return outputs;
}

Vector<String> LiteSession::GetOutputTensorNames() const {
  Vector<String> output_names;
  for (const auto &output : outputs_) {
    output_names.push_back(output->tensor_name());
  }
  return output_names;
}

mindspore::tensor::MSTensor *LiteSession::GetOutputByTensorName(const String &tensor_name) const {
  for (const auto &output : outputs_) {
    if (output->tensor_name() == tensor_name) {
      return output;
    }
  }
  return nullptr;
}
}  // namespace lite
)RAW";

}  // namespace mindspore::lite::micro
