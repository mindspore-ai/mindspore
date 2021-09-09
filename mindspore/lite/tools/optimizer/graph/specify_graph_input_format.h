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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SPECIFY_GRAPH_INPUT_FORMAT_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SPECIFY_GRAPH_INPUT_FORMAT_H_

#include "backend/optimizer/common/pass.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {
class SpecifyGraphInputFormat : public Pass {
 public:
  explicit SpecifyGraphInputFormat(mindspore::Format format = mindspore::NHWC)
      : Pass("SpecifyGraphInputFormat"), format_(format) {}
  ~SpecifyGraphInputFormat() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  STATUS HandleGraphInput(const FuncGraphPtr &graph);
  mindspore::Format format_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SPECIFY_GRAPH_INPUT_FORMAT_H_
