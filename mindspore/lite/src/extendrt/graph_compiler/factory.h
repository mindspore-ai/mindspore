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
#ifndef MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_FACTORY_H_
#define MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_FACTORY_H_

#include <functional>
#include <memory>

#include "extendrt/graph_compiler/type.h"
#include "include/api/context.h"
#include "infer/graph_compiler.h"

namespace mindspore {
using GraphCompilerRegFunc =
  std::function<std::shared_ptr<infer::abstract::GraphCompiler>(const std::shared_ptr<Context> &)>;

class GraphCompilerRegistry {
 public:
  GraphCompilerRegistry() = default;
  virtual ~GraphCompilerRegistry() = default;

  static GraphCompilerRegistry &GetInstance();

  void RegCompiler(const mindspore::GraphCompilerType &graph_compiler_type, const GraphCompilerRegFunc &creator);

  std::shared_ptr<infer::abstract::GraphCompiler> GetCompiler(const mindspore::GraphCompilerType &type,
                                                              const std::shared_ptr<Context> &context);

 private:
  mindspore::HashMap<mindspore::GraphCompilerType, GraphCompilerRegFunc> graph_compiler_map_;
};

class GraphCompilerRegistrar {
 public:
  GraphCompilerRegistrar(const mindspore::GraphCompilerType &graph_compiler_type, const GraphCompilerRegFunc &creator) {
    GraphCompilerRegistry::GetInstance().RegCompiler(graph_compiler_type, creator);
  }
  ~GraphCompilerRegistrar() = default;
};

#define REG_GRAPH_COMPILER(type, creator) static GraphCompilerRegistrar g_##type##GraphCompiler(type, creator);
}  // namespace mindspore

#endif  // MINDSPORE_LITE_EXTENDRT_GRAPH_COMPILER_FACTORY_H_
