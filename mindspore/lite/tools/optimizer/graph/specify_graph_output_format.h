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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SPECIFY_GRAPH_OUTPUT_FORMAT_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SPECIFY_GRAPH_OUTPUT_FORMAT_H_

#include "include/backend/optimizer/pass.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "include/api/format.h"
#include "include/api/types.h"
#include "include/registry/converter_context.h"

namespace mindspore {
namespace opt {
class SpecifyGraphOutputFormat : public Pass {
 public:
  explicit SpecifyGraphOutputFormat(mindspore::Format exp_graph_output_format)
      : Pass("SpecifyGraphOutputFormat"), exp_graph_output_format_(exp_graph_output_format) {}
  ~SpecifyGraphOutputFormat() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  STATUS HandleGraphOutput(const FuncGraphPtr &graph);
  mindspore::Format exp_graph_output_format_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SPECIFY_GRAPH_OUTPUT_FORMAT_H_
