/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "extendrt/graph_compiler/factory.h"
#include <functional>
#include <memory>

namespace mindspore {
GraphCompilerRegistry &GraphCompilerRegistry::GetInstance() {
  static GraphCompilerRegistry instance;
  return instance;
}

void GraphCompilerRegistry::RegCompiler(const mindspore::GraphCompilerType &type, const GraphCompilerRegFunc &creator) {
  graph_compiler_map_[type] = creator;
}

std::shared_ptr<infer::abstract::GraphCompiler> GraphCompilerRegistry::GetCompiler(
  const mindspore::GraphCompilerType &type, const std::shared_ptr<Context> &context) {
  auto it = graph_compiler_map_.find(type);
  if (it == graph_compiler_map_.end()) {
    return nullptr;
  }
  return it->second(context);
}
}  // namespace mindspore
