/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef PREDICT_COMMON_MODULE_REGISTRY_H_
#define PREDICT_COMMON_MODULE_REGISTRY_H_
#include <memory>
#include <string>
#include <unordered_map>
#include "common/mslog.h"

#define MSPREDICT_API __attribute__((visibility("default")))

namespace mindspore {
namespace predict {
class ModuleBase {
 public:
  virtual ~ModuleBase() = default;
};

template <typename T>
class Module;

class ModuleRegistry {
 public:
  ModuleRegistry() = default;

  virtual ~ModuleRegistry() = default;

  template <class T>
  bool Register(const std::string &name, const T &t) {
    modules[name] = &t;
    return true;
  }

  template <class T>
  std::shared_ptr<T> Create(const std::string &name) {
    auto it = modules.find(name);
    if (it == modules.end()) {
      return nullptr;
    }
    auto *module = (Module<T> *)it->second;
    if (module == nullptr) {
      return nullptr;
    } else {
      return module->Create();
    }
  }

  template <class T>
  T *GetInstance(const std::string &name) {
    auto it = modules.find(name);
    if (it == modules.end()) {
      return nullptr;
    }
    auto *module = (Module<T> *)it->second;
    if (module == nullptr) {
      return nullptr;
    } else {
      return module->GetInstance();
    }
  }

 protected:
  std::unordered_map<std::string, const ModuleBase *> modules;
};

ModuleRegistry *GetRegistryInstance() MSPREDICT_API;

template <class T>
class ModuleRegistrar {
 public:
  ModuleRegistrar(const std::string &name, const T &module) {
    auto registryInstance = GetRegistryInstance();
    if (registryInstance == nullptr) {
      MS_LOGW("registryInstance is nullptr.");
    } else {
      registryInstance->Register(name, module);
    }
  }
};
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_COMMON_MODULE_REGISTRY_H_
