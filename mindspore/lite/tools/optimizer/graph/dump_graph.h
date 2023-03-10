/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_DUMP_GRAPH_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_DUMP_GRAPH_H_

#include <memory>
#include "include/backend/optimizer/pass.h"
#include "tools/converter/export_model.h"
#include "include/registry/pass_base.h"
#include "mindapi/ir/func_graph.h"

namespace mindspore {
namespace opt {
class DumpGraph : public registry::PassBase, public Pass {
 public:
  explicit DumpGraph(const std::shared_ptr<ConverterPara> &param) : Pass("DumpGraph"), param_(param) {}
  ~DumpGraph() = default;
  bool Run(const FuncGraphPtr &graph) override {
    MS_CHECK_TRUE_MSG(graph != nullptr, false, "funcGraph is a nullptr.");
    if (lite::ExportModel(graph, param_) != lite::RET_OK) {
      MS_LOG(ERROR) << "dump graph failed.";
      return false;
    }
    return true;
  }

  bool Execute(const api::FuncGraphPtr &func_graph) override {
    MS_CHECK_TRUE_MSG(func_graph != nullptr, false, "funcGraph is a nullptr.");
    auto impl = func_graph->impl();
    MS_CHECK_TRUE_MSG(impl != nullptr, false, "func_graph impl is a nullptr.");
    auto graph = std::dynamic_pointer_cast<FuncGraph>(impl);
    MS_CHECK_TRUE_MSG(graph != nullptr, false, "Graph is a nullptr.");
    return Run(graph);
  }

 private:
  const std::shared_ptr<ConverterPara> &param_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_DUMP_GRAPH_H_
