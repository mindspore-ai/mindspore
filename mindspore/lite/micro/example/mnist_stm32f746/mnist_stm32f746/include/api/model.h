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
#ifndef MINDSPORE_INCLUDE_API_MODEL_H
#define MINDSPORE_INCLUDE_API_MODEL_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <utility>
#include "include/api/status.h"
#include "include/api/types.h"
#include "include/api/graph.h"
#include "include/api/context.h"
#include "include/api/cell.h"
#include "include/api/dual_abi_helper.h"

namespace mindspore {
class ModelImpl;

class MS_API Model {
 public:
  Model();
  ~Model();
  Model(const Model &) = delete;
  void operator=(const Model &) = delete;

  Status Build(GraphCell graph, const std::shared_ptr<Context> &model_context = nullptr);
  Status Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims);

  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs);

  std::vector<MSTensor> GetInputs();
  inline MSTensor GetInputByTensorName(const std::string &tensor_name);

  std::vector<MSTensor> GetOutputs();
  inline std::vector<std::string> GetOutputTensorNames();
  inline MSTensor GetOutputByTensorName(const std::string &tensor_name);
  inline std::vector<MSTensor> GetOutputsByNodeName(const std::string &tensor_name);

  static bool CheckModelSupport(enum DeviceType device_type, ModelType model_type);

 private:
  // api without std::string
  MSTensor GetInputByTensorName(const std::vector<char> &tensor_name);
  std::vector<std::vector<char>> GetOutputTensorNamesChar();
  MSTensor GetOutputByTensorName(const std::vector<char> &tensor_name);
  std::vector<MSTensor> GetOutputsByNodeName(const std::vector<char> &node_name);

  std::shared_ptr<ModelImpl> impl_;
};

MSTensor Model::GetInputByTensorName(const std::string &tensor_name) {
  return GetInputByTensorName(StringToChar(tensor_name));
}

std::vector<std::string> Model::GetOutputTensorNames() { return VectorCharToString(GetOutputTensorNamesChar()); }

MSTensor Model::GetOutputByTensorName(const std::string &tensor_name) {
  return GetOutputByTensorName(StringToChar(tensor_name));
}

std::vector<MSTensor> Model::GetOutputsByNodeName(const std::string &tensor_name) {
  return GetOutputsByNodeName(StringToChar(tensor_name));
}
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_MODEL_H
