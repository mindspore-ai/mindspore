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
#ifndef MINDSPORE_CCSRC_CXX_API_MODEL_MODEL_IMPL_H
#define MINDSPORE_CCSRC_CXX_API_MODEL_MODEL_IMPL_H
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/api/model.h"
#include "utils/utils.h"

namespace mindspore::api {
class ModelImpl {
 public:
  ModelImpl() = default;
  virtual ~ModelImpl() = default;

  virtual Status LoadModel(const Buffer &model_data, ModelType type,
                           const std::map<std::string, std::string> &options) = 0;
  virtual Status LoadModel(const std::string &file_name, ModelType type,
                           const std::map<std::string, std::string> &options) = 0;
  virtual Status UnloadModel() = 0;

  virtual Status Train(const DataSet &dataset, std::map<std::string, Buffer> *outputs) = 0;
  virtual Status Eval(const DataSet &dataset, std::map<std::string, Buffer> *outputs) = 0;
  virtual Status Predict(const std::map<std::string, Buffer> &inputs, std::map<std::string, Buffer> *outputs) = 0;

  virtual Status GetInputsInfo(std::vector<Tensor> *tensor_list) const = 0;
  virtual Status GetOutputsInfo(std::vector<Tensor> *tensor_list) const = 0;
};

using ModelCreator = std::function<std::shared_ptr<ModelImpl>(uint32_t device_id)>;
class ModelFactory {
 public:
  ModelFactory(const ModelFactory &) = delete;
  void operator=(const ModelFactory &) = delete;

  static ModelFactory &Instance() {
    static ModelFactory instance;
    return instance;
  }

  void Register(const std::string &device_name, ModelCreator &&model_creator) {
    if (model_creators_.find(device_name) == model_creators_.end()) {
      (void)model_creators_.emplace(device_name, model_creator);
    }
  }

  std::shared_ptr<ModelImpl> Create(const std::string &device_name, uint32_t device_id) {
    auto iter = model_creators_.find(device_name);
    if (model_creators_.end() != iter) {
      MS_EXCEPTION_IF_NULL(iter->second);
      return (iter->second)(device_id);
    }
    return nullptr;
  }

  bool CheckModelSupport(const std::string &device_type, ModelType /*model_type*/) {
    return std::any_of(
      model_creators_.begin(), model_creators_.end(),
      [&device_type](const std::pair<std::string, ModelCreator> &item) { return item.first == device_type; });
  }

 private:
  ModelFactory() = default;
  ~ModelFactory() = default;
  std::map<std::string, ModelCreator> model_creators_;
};

class ModelRegistrar {
 public:
  ModelRegistrar(const std::string &device_name, ModelCreator model_creator) {
    ModelFactory::Instance().Register(device_name, std::move(model_creator));
  }
  ~ModelRegistrar() = default;
};

#define API_REG_MODEL(DEVICE_NAME, MODEL_CLASS)                              \
  static const ModelRegistrar g_api_model_registrar__##DEVICE_NAME##_##_reg( \
    kDeviceType##DEVICE_NAME, [](uint32_t device_id) { return std::make_shared<MODEL_CLASS>(device_id); });

}  // namespace mindspore::api

#endif  // MINDSPORE_CCSRC_CXX_API_MODEL_MODEL_IMPL_H
