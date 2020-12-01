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
#include "include/api/status.h"
#include "include/api/types.h"
#include "include/api/graph.h"
#include "include/api/cell.h"

namespace mindspore {
namespace api {
class ModelImpl;
// todo: minddata c++ interface
class DataSet {};

class MS_API Model {
 public:
  explicit Model(const std::vector<Output> &network);
  explicit Model(const GraphCell &graph);
  ~Model();
  Model(const Model &) = delete;
  void operator=(const Model &) = delete;

  Status Build(const std::map<std::string, std::string> &options);

  Status Train(const DataSet &dataset, bool data_sink, std::map<std::string, Buffer> *outputs);
  Status Eval(const DataSet &dataset, bool data_sink, std::map<std::string, Buffer> *outputs);
  Status Predict(const std::vector<Buffer> &inputs, std::vector<Buffer> *outputs);

  Status GetInputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                       std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) const;
  Status GetOutputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                        std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) const;

  static bool CheckModelSupport(const std::string &device_type, ModelType model_type);

 private:
  std::shared_ptr<ModelImpl> impl_;
};
}  // namespace api
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_MODEL_H
