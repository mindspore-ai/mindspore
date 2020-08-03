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
#ifndef MINDSPORE_CCSRC_SESSION_SESSION_H
#define MINDSPORE_CCSRC_SESSION_SESSION_H

#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <memory>
#include <map>

#include "backend/session/session_basic.h"
#include "ir/anf.h"
#include "include/inference.h"

#ifdef ENABLE_D
#include "runtime/context.h"
#endif

namespace mindspore {
namespace inference {
class MSInferSession : public InferSession {
 public:
  MSInferSession();
  ~MSInferSession();

  Status InitEnv(const std::string &device_type, uint32_t device_id) override;
  Status FinalizeEnv() override;
  Status LoadModelFromFile(const std::string &file_name, uint32_t &model_id) override;
  Status UnloadModel(uint32_t model_id) override;
  Status ExecuteModel(uint32_t model_id, const RequestBase &inputs, ReplyBase &outputs) override;

 private:
  std::shared_ptr<session::SessionBasic> session_impl_ = nullptr;
  std::vector<uint32_t> graph_id_;
  std::string device_type_;
  int32_t device_id_;
#ifdef ENABLE_D
  rtContext_t context_ = nullptr;
#endif

  std::shared_ptr<FuncGraph> LoadModel(const char *model_buf, size_t size, const std::string &device);
  std::shared_ptr<std::vector<char>> ReadFile(const std::string &file);
  static void RegAllOp();
  string AjustTargetName(const std::string &device);
  Status CompileGraph(std::shared_ptr<FuncGraph> funcGraphPtr, uint32_t &model_id);
  Status CheckModelInputs(uint32_t graph_id, const std::vector<tensor::TensorPtr> &inputs) const;
  std::vector<tensor::TensorPtr> RunGraph(uint32_t graph_id, const std::vector<tensor::TensorPtr> &inputs);
};
}  // namespace inference
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_SESSION_SESSION_BASIC_H
