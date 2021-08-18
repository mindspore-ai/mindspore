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

#include "backend/optimizer/common/pass.h"
#include "tools/converter/export_model.h"

namespace mindspore {
namespace opt {
class DumpGraph : public Pass {
 public:
  explicit DumpGraph(const converter::Flags *flags = nullptr) : Pass("DumpGraph"), flags_(flags) {}
  ~DumpGraph() = default;
  bool Run(const FuncGraphPtr &graph) override {
    MS_ASSERT(graph != nullptr);
    if (lite::ExportModel(graph, flags_) != lite::RET_OK) {
      MS_LOG(ERROR) << "dump graph failed.";
      return false;
    }
    return true;
  }

 private:
  const converter::Flags *flags_{nullptr};
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_DUMP_GRAPH_H_
