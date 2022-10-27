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
#ifndef MINDSPORE_CCSRC_CXX_API_FACTORY_H
#define MINDSPORE_CCSRC_CXX_API_FACTORY_H
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/api/context.h"
#include "include/common/utils/utils.h"
#include "cxx_api/graph/graph_impl.h"
#include "cxx_api/model/model_impl.h"

namespace mindspore {
constexpr auto Ascend310 = "Ascend310";
constexpr auto Ascend910 = "Ascend910";
constexpr auto kMS = "MS";

static inline LogStream &operator<<(LogStream &stream, DeviceType device_type) {
  std::map<DeviceType, std::string> type_str_map = {
    {kAscend, "Ascend"}, {kAscend910, "Ascend910"}, {kAscend310, "Ascend310"}, {kGPU, "GPU"}, {kCPU, "CPU"}};
  auto it = type_str_map.find(device_type);
  if (it != type_str_map.end()) {
    stream << it->second;
  } else {
    stream << "[InvalidDeviceType: " << static_cast<int>(device_type) << "]";
  }
  return stream;
}

using GraphImplCreator = std::function<std::shared_ptr<GraphCell::GraphImpl>()>;

class MS_API DeviceTypeCreator {
 public:
  static DeviceType &get();
};

class MS_API GraphImplFactory {
 public:
  GraphImplFactory(const GraphImplFactory &) = delete;
  GraphImplFactory &operator=(const GraphImplFactory &) = delete;

  static GraphImplFactory &Instance();

  void Register(const std::string &device_name, GraphImplCreator &&creator);

  std::shared_ptr<GraphCell::GraphImpl> Create(enum DeviceType device_type);

  GraphImplFactory() = default;
  ~GraphImplFactory() = default;

 private:
  std::vector<GraphImplCreator> creators_;
};

class GraphImplRegistrar {
 public:
  explicit GraphImplRegistrar(const std::string &device_name, GraphImplCreator &&creator) {
    GraphImplFactory::Instance().Register(device_name, std::move(creator));
  }
  ~GraphImplRegistrar() = default;
};

using ModelImplCreator = std::function<std::shared_ptr<ModelImpl>()>;

class MS_API ModelImplFactory {
 public:
  ModelImplFactory(const ModelImplFactory &) = delete;
  ModelImplFactory &operator=(const ModelImplFactory &) = delete;

  static ModelImplFactory &Instance();

  void Register(const std::string &device_name, ModelImplCreator &&creator);

  std::shared_ptr<ModelImpl> Create(enum DeviceType device_type);

  ModelImplFactory() = default;
  ~ModelImplFactory() = default;

 private:
  std::vector<ModelImplCreator> creators_;
};

class ModelImplRegistrar {
 public:
  explicit ModelImplRegistrar(const std::string &device_name, ModelImplCreator &&creator) {
    ModelImplFactory::Instance().Register(device_name, std::move(creator));
  }
  ~ModelImplRegistrar() = default;
};

#define API_GRAPH_REG(DEVICE_NAME, DEVICE_CLASS)                           \
  static const GraphImplRegistrar graph_api_##DEVICE_NAME##_registrar_reg( \
    DEVICE_NAME, []() { return std::make_shared<DEVICE_CLASS>(); });

#define API_MODEL_REG(DEVICE_NAME, DEVICE_CLASS)                           \
  static const ModelImplRegistrar model_api_##DEVICE_NAME##_registrar_reg( \
    DEVICE_NAME, []() { return std::make_shared<DEVICE_CLASS>(); });
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXX_API_FACTORY_H
