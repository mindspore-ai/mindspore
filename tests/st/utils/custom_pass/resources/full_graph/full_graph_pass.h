/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CUSTOM_PASS_FULL_GRAPH_PASS_H_
#define MINDSPORE_CUSTOM_PASS_FULL_GRAPH_PASS_H_

#include "mindspore/include/custom_pass_api.h"

namespace mindspore {
namespace opt {

/**
 * @brief Frontend custom pass for testing pass_type=ms.graph.CustomPassType.FULL_GRAPH
 *
 * This pass runs in the frontend pipeline before OrderEnforceAction
 */
class FullGraphPass : public Pass {
 public:
  explicit FullGraphPass(const std::string &name = "FullGraphPass") : Pass(name) {}
  ~FullGraphPass() override = default;

  bool Run(const FuncGraphPtr &func_graph) override;
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CUSTOM_PASS_FULL_GRAPH_PASS_H_
