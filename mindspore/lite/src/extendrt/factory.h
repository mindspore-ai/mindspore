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
#ifndef MINDSPORE_LITE_EXTENDRT_FACTORY_H
#define MINDSPORE_LITE_EXTENDRT_FACTORY_H
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/common/utils/utils.h"

namespace mindspore {
inline enum DeviceType g_device_target = kInvalidDeviceType;

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

template <class T>
class Factory {
  using U = std::function<std::shared_ptr<T>()>;

 public:
  Factory(const Factory &) = delete;
  Factory &operator=(const Factory &) = delete;

  static Factory &Instance() {
    static Factory instance;
    return instance;
  }

  void Register(U &&creator) { creators_.push_back(creator); }

  std::shared_ptr<T> Create(enum DeviceType device_type) {
    for (auto &item : creators_) {
      MS_EXCEPTION_IF_NULL(item);
      auto val = item();
      if (val->CheckDeviceSupport(device_type)) {
        return val;
      }
    }
    MS_LOG(WARNING) << "Unsupported device target " << device_type;
    return nullptr;
  }

 private:
  Factory() = default;
  ~Factory() = default;
  std::vector<U> creators_;
};

template <class T>
class Registrar {
  using U = std::function<std::shared_ptr<T>()>;

 public:
  explicit Registrar(U creator) { Factory<T>::Instance().Register(std::move(creator)); }
  ~Registrar() = default;
};

#define API_FACTORY_CREATOR(DERIVE_CLASS) []() { return std::make_shared<DERIVE_CLASS>(); }

#define API_FACTORY_REG(BASE, DERIVE) static const Registrar<BASE> g_api_##DERIVE##_reg(API_FACTORY_CREATOR(DERIVE));
}  // namespace mindspore
#endif  // MINDSPORE_LITE_EXTENDRT_FACTORY_H
