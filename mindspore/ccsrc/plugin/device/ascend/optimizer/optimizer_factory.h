/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_FACTORY_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_FACTORY_H_

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mindspore::opt {
enum kPassType {
  kMindIRPass = 0,
  kIRFusionFissionPass,
  kUBFusionPass,
};

template <class C>
class Factory {
  using CreatorFunc = std::function<std::shared_ptr<C>()>;

 public:
  static Factory &Instance() {
    static Factory instance{};
    return instance;
  }

  void Register(kPassType pass_type, const std::string &name, CreatorFunc &&creator) {
    if (IsRegistered(pass_type, name)) {
      MS_LOG(WARNING) << "Pass " << name << " is already registered!";
    }
    (void)pass_creators_[pass_type].emplace(name, creator);
  }

  std::shared_ptr<C> Create(kPassType pass_type, const std::string &name) {
    auto iter = pass_creators_.find(pass_type);
    if (iter == pass_creators_.end()) {
      return nullptr;
    }
    auto iter_creator = iter->second.find(name);
    if (iter_creator != iter->second.end()) {
      return iter_creator->second();
    }
    return nullptr;
  }

  bool IsRegistered(kPassType pass_type, const std::string &name) {
    auto iter = pass_creators_.find(pass_type);
    if (iter == pass_creators_.end()) {
      return false;
    }
    return iter->second.find(name) != iter->second.end();
  }

  const std::map<std::string, CreatorFunc> &GetPassCreatorsByType(kPassType pass_type) {
    return pass_creators_[pass_type];
  }

 private:
  Factory() = default;
  ~Factory() = default;
  std::map<kPassType, std::map<std::string, CreatorFunc>> pass_creators_;
};

template <class C>
class PassRegister {
 public:
  explicit PassRegister(kPassType pass_type, const std::string &name, std::function<std::shared_ptr<C>()> creator) {
    Factory<C>::Instance().Register(pass_type, name, std::move(creator));
  }
  ~PassRegister() = default;
};

// Helper macro for factory registration.
#define MS_PASS_FACTORY_REG(BASE_CLASS, NAME, DERIVE_CLASS, PASS_TYPE)                                                 \
  static_assert(std::is_base_of<BASE_CLASS, DERIVE_CLASS>::value, #DERIVE_CLASS " must be derived from " #BASE_CLASS); \
  static const PassRegister<BASE_CLASS> g_##NAME##_##BASE_CLASS##_##PASS_TYPE##_reg(                                   \
    PASS_TYPE, #NAME, []() { return std::make_shared<DERIVE_CLASS>(); })
}  // namespace mindspore::opt
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_FACTORY_H_
