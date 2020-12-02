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

namespace mindspore {
namespace api {
class ModelImpl;
// todo: minddata c++ interface
class DataSet {};
class NetWork {};

class MS_API Model {
 public:
  Model(const std::string &device_type, uint32_t device_id);
  Model(NetWork network, const std::string &device_type, uint32_t device_id);
  ~Model();
  Model(const Model &) = delete;
  void operator=(const Model &) = delete;

  Status LoadModel(const Buffer &model_data, ModelType type, const std::map<std::string, std::string> &options);
  Status LoadModel(const std::string &file_name, ModelType type, const std::map<std::string, std::string> &options);
  Status UnloadModel();

  Status Train(const DataSet &dataset, std::map<std::string, Buffer> *outputs);
  Status Eval(const DataSet &dataset, std::map<std::string, Buffer> *outputs);
  Status Predict(const std::map<std::string, Buffer> &inputs, std::map<std::string, Buffer> *outputs);
  Status Predict(const std::vector<Buffer> &inputs, std::map<std::string, Buffer> *outputs);

  Status GetInputsInfo(std::vector<Tensor> *tensor_list) const;
  Status GetOutputsInfo(std::vector<Tensor> *tensor_list) const;

  static bool CheckModelSupport(const std::string& device_type, ModelType model_type);

 private:
  std::shared_ptr<ModelImpl> impl_;
};

extern MS_API const char* kDeviceTypeAscendCL;
extern MS_API const char* kDeviceTypeAscendMS;
}  // namespace api
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_MODEL_H
