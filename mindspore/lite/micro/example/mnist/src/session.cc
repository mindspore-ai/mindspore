
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
#include "net.h"

namespace mindspore {
namespace lite {
int LiteSession::CompileGraph(lite::Model *model) {
  inputs_.resize(1);
  inputs_[0] = new (std::nothrow) MTensor("graph_input-0", kNumberTypeFloat32, {1, 28, 28, 1, });
  MS_ERROR_IF_NULL(inputs_[0]);
  outputs_.resize(1);
  outputs_[0] = new (std::nothrow) MTensor("Softmax-7", kNumberTypeFloat32, {1, 10, });
  MS_ERROR_IF_NULL(outputs_[0]);
  for (const auto &output: outputs_) {
    output_tensor_map_[output->tensor_name()] = output;
  }
  return RET_OK;
}


int LiteSession::RunGraph(const KernelCallBack &before, const KernelCallBack &after) {
  const void *inputs_data[inputs_.size()];
  for (size_t i = 0; i < inputs_.size(); ++i) {
    inputs_data[i] = inputs_[i]->MutableData();
  }
  net_SetInputs(inputs_data, inputs_.size());

  net_Inference();

  void *outputs_data[outputs_.size()];
  for (size_t i = 0; i < outputs_.size(); ++i) {
    outputs_data[i] = outputs_[i]->MutableData();
  }
  CopyOutputsData(outputs_data, outputs_.size());

  return RET_OK;
}

LiteSession::~LiteSession() {
  net_FreeResource();
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
  for (auto &item : output_tensor_map_) {
    auto output = item.second;
    if (output == nullptr) {
      continue;
    }
    delete output;
    output = nullptr;
  }
}

int LiteSession::InitRuntimeBuffer() {
  int buffer_size = net_GetBufferSize();
  runtime_buffer_ = malloc(buffer_size);
  if (runtime_buffer_ == nullptr) {
    return RET_ERROR;
  }
  int ret = net_SetBuffer(runtime_buffer_);
  if (ret != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

std::vector<tensor::MSTensor *> LiteSession::GetInputs() const {
  std::vector<tensor::MSTensor *> inputs;
  inputs.insert(inputs.begin(), inputs_.begin(), inputs_.end());
  return inputs;
}

std::vector<tensor::MSTensor *> LiteSession::GetOutputsByNodeName(const std::string &node_name) const {
  auto iter = output_node_map_.find(node_name);
  if (iter == output_node_map_.end()) {
    std::vector<tensor::MSTensor *> empty;
    return empty;
  }
  return iter->second;
}

std::unordered_map<std::string, mindspore::tensor::MSTensor *> LiteSession::GetOutputs() const {
  return output_tensor_map_;
}

std::vector<std::string> LiteSession::GetOutputTensorNames() const {
  std::vector<std::string> output_names;
  for (const auto &item : output_node_map_) {
    for (const auto &output : item.second) {
      output_names.emplace_back(output->tensor_name());
    }
  }
  return output_names;
}

mindspore::tensor::MSTensor *LiteSession::GetOutputByTensorName(const std::string &tensor_name) const {
  auto item = output_tensor_map_.find(tensor_name);
  if (item == output_tensor_map_.end()) {
    return nullptr;
  }
  return item->second;
}

int LiteSession::Resize(const std::vector<tensor::MSTensor *> &inputs, const std::vector<std::vector<int>> &dims) {
  return RET_OK;
}

}  // namespace lite

session::LiteSession *session::LiteSession::CreateSession(const lite::Context *context) {
  auto *session = new (std::nothrow) lite::LiteSession();
  if (session == nullptr) {
    return nullptr;
  }
  session->InitRuntimeBuffer();
  return session;
}

session::LiteSession *session::LiteSession::CreateSession(const char *net_buf, size_t size,
                                                          const lite::Context *context) {
  session::LiteSession *session = CreateSession(context);
  if (session == nullptr) {
    return nullptr;
  }
  int ret = session->CompileGraph(nullptr);
  if (ret != lite::RET_OK) {
    return nullptr;
  }
  net_Init(const_cast<char *>(net_buf), size);
  return session;
}
}  // namespace mindspore

