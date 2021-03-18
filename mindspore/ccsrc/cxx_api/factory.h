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
#include "utils/utils.h"

namespace mindspore {
inline std::string g_device_target = "Default";

template <class T>
class Factory {
  using U = std::function<std::shared_ptr<T>()>;

 public:
  Factory(const Factory &) = delete;
  void operator=(const Factory &) = delete;

  static Factory &Instance() {
    static Factory instance;
    return instance;
  }

  void Register(const std::string &device_name, U &&creator) {
    if (creators_.find(device_name) == creators_.end()) {
      (void)creators_.emplace(device_name, creator);
    }
  }

  bool CheckModelSupport(const std::string &device_name) {
    return std::any_of(creators_.begin(), creators_.end(),
                       [&device_name](const std::pair<std::string, U> &item) { return item.first == device_name; });
  }

  std::shared_ptr<T> Create(const std::string &device_name) {
    auto iter = creators_.find(device_name);
    if (creators_.end() != iter) {
      MS_EXCEPTION_IF_NULL(iter->second);
      return (iter->second)();
    }

    MS_LOG(ERROR) << "Unsupported device target " << device_name;
    return nullptr;
  }

 private:
  Factory() = default;
  ~Factory() = default;
  std::map<std::string, U> creators_;
};

template <class T>
class Registrar {
  using U = std::function<std::shared_ptr<T>()>;

 public:
  Registrar(const std::string &device_name, U creator) {
    Factory<T>::Instance().Register(device_name, std::move(creator));
  }
  ~Registrar() = default;
};

#define API_FACTORY_REG(BASE_CLASS, DEVICE_NAME, DERIVE_CLASS)                             \
  static const Registrar<BASE_CLASS> g_api_##DERIVE_CLASS##_registrar_##DEVICE_NAME##_reg( \
    #DEVICE_NAME, []() { return std::make_shared<DERIVE_CLASS>(); });
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXX_API_FACTORY_H
