/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_FACTORY_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_FACTORY_H_

#include <functional>
#include <string>
#include <memory>

#include "extendrt/delegate/graph_executor/delegate.h"
#include "extendrt/delegate/graph_executor/type.h"
#include "utils/hash_map.h"

namespace mindspore {
typedef std::shared_ptr<device::GraphExecutor> (*GraphExecutorCreator)(
  const std::shared_ptr<mindspore::DelegateConfig> &config);
class GraphExecutorRegistry {
 public:
  GraphExecutorRegistry() = default;
  virtual ~GraphExecutorRegistry() = default;

  static GraphExecutorRegistry *GetInstance() {
    static GraphExecutorRegistry instance;
    return &instance;
  }

  void RegGraphExecutor(const mindspore::DeviceType &device_type, const std::string &provider,
                        GraphExecutorCreator creator) {
    auto it = creator_map_.find(device_type);
    if (it == creator_map_.end()) {
      HashMap<std::string, GraphExecutorCreator> map;
      map[provider] = creator;
      creator_map_[device_type] = map;
      return;
    }
    it->second[provider] = creator;
  }

  std::shared_ptr<device::GraphExecutor> GetGraphExecutor(const mindspore::DeviceType &device_type,
                                                          const std::string &provider,
                                                          const std::shared_ptr<mindspore::DelegateConfig> &config) {
    auto it = creator_map_.find(device_type);
    if (it == creator_map_.end()) {
      return nullptr;
    }
    auto creator_it = it->second.find(provider);
    if (creator_it == it->second.end()) {
      return nullptr;
    }
    return creator_it->second(config);
  }

  std::shared_ptr<GraphExecutorDelegate> GetDelegate(const mindspore::DeviceType &device_type,
                                                     const std::string &provider,
                                                     const std::shared_ptr<mindspore::DelegateConfig> &config) {
    auto graph_executor = GetGraphExecutor(device_type, provider, config);
    if (graph_executor == nullptr) {
      return nullptr;
    }

    auto delegate = std::make_shared<mindspore::GraphExecutorDelegate>();
    delegate->SetGraphExecutor(graph_executor);
    return delegate;
  }

 private:
  mindspore::HashMap<DeviceType, mindspore::HashMap<std::string, GraphExecutorCreator>> creator_map_;
};

class GraphExecutorRegistrar {
 public:
  GraphExecutorRegistrar(const mindspore::DeviceType &device_type, const std::string &provider,
                         GraphExecutorCreator creator) {
    GraphExecutorRegistry::GetInstance()->RegGraphExecutor(device_type, provider, creator);
  }
  ~GraphExecutorRegistrar() = default;
};

#define REG_GRAPH_EXECUTOR(device_type, provider, creator) \
  static GraphExecutorRegistrar g_##device_type##provider##GraphExecutor(device_type, provider, creator);
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_FACTORY_H_
