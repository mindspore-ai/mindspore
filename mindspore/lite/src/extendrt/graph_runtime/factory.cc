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
#include "extendrt/graph_runtime/factory.h"
#include <functional>
#include <memory>

namespace mindspore {
GraphRuntimeRegistry &GraphRuntimeRegistry::GetInstance() {
  static GraphRuntimeRegistry instance;
  return instance;
}

void GraphRuntimeRegistry::RegRuntime(const mindspore::GraphRuntimeType &type, const GraphRuntimeRegFunc &creator) {
  graph_runtime_map_[type] = creator;
}

std::shared_ptr<infer::abstract::GraphRuntime> GraphRuntimeRegistry::GetRuntime(
  const mindspore::GraphRuntimeType &type) {
  auto it = graph_runtime_map_.find(type);
  if (it == graph_runtime_map_.end()) {
    return nullptr;
  }
  return it->second();
}
}  // namespace mindspore
