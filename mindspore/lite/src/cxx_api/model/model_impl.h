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

#ifndef MINDSPORE_LITE_SRC_CXX_API_MODEL_MODEL_IMPL_H
#define MINDSPORE_LITE_SRC_CXX_API_MODEL_MODEL_IMPL_H

#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <unordered_map>
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/cell.h"
#include "include/lite_session.h"

namespace mindspore {
class ModelImpl {
 public:
  ModelImpl() : graph_(nullptr), session_(nullptr), context_(nullptr) {}
  ~ModelImpl() = default;

  Status Build();
  Status Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims);

  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs);

  std::vector<MSTensor> GetInputs();
  std::vector<MSTensor> GetOutputs();
  MSTensor GetInputByTensorName(const std::string &name);
  std::vector<std::string> GetOutputTensorNames();
  MSTensor GetOutputByTensorName(const std::string &name);
  std::vector<MSTensor> GetOutputsByNodeName(const std::string &name);

  static bool CheckModelSupport(const std::string &device_type, ModelType model_type);

 private:
  friend class Model;
  std::shared_ptr<Graph> graph_;
  std::shared_ptr<session::LiteSession> session_;
  std::shared_ptr<Context> context_;
  void SetGraph(const std::shared_ptr<Graph> &graph) { graph_ = graph; }
  void SetContext(const std::shared_ptr<Context> &context) { context_ = context; }
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_CXX_API_MODEL_MODEL_IMPL_H
