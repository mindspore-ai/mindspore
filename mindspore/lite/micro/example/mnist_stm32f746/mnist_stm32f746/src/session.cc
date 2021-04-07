
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

#include "session.h"
#include "model.h"
#include "net.h"
#include <new>

namespace mindspore {
namespace lite {
int LiteSession::CompileGraph(lite::Model *model) {
  inputs_.resize(1);
  Vector<int> in_shape_0;
  in_shape_0.resize(4);
  in_shape_0[0] = 1;
  in_shape_0[1] = 28;
  in_shape_0[2] = 28;
  in_shape_0[3] = 1;
  inputs_[0] = new (std::nothrow) MTensor(String("graph_input-0"), kNumberTypeInt8, in_shape_0);
  MS_ERROR_IF_NULL(inputs_[0]);
  outputs_.resize(1);
  Vector<int> out_shape_0;
  out_shape_0.resize(2);
  out_shape_0[0] = 1;
  out_shape_0[1] = 10;
  outputs_[0] = new (std::nothrow) MTensor(String("int8toft32_Softmax-7_post0/output-0"), kNumberTypeFloat32, out_shape_0);
  MS_ERROR_IF_NULL(outputs_[0]);
  return RET_OK;
}


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
session::LiteSession *session::LiteSession::CreateSession(const lite::Context *context) {
  auto *session = new (std::nothrow) lite::LiteSession();
  MS_NULLPTR_IF_NULL(session);
  int ret = session->InitRuntimeBuffer();
  MS_NULLPTR_IF_ERROR(ret);
  return session;
}

session::LiteSession *session::LiteSession::CreateSession(const char *model_buf, size_t size,
                                                          const lite::Context *context) {
  session::LiteSession *session = CreateSession(context);
  MS_NULLPTR_IF_NULL(session);
  lite::Model *model = lite::Model::Import(model_buf, size);
  int ret = session->CompileGraph(model);
  MS_NULLPTR_IF_ERROR(ret);
  delete model;
  return session;
}
}  // namespace mindspore

