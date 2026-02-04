/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_PASS_LABEL_1F1B_OVERLAP_NODE_PASS_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_PASS_LABEL_1F1B_OVERLAP_NODE_PASS_H_

#include <string>
#include <vector>
#include "include/backend/common/pass_manager/pass.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
class BACKEND_COMMON_EXPORT Label1F1BOverlapNode : public Pass {
 public:
  explicit Label1F1BOverlapNode(const std::string &name = "label_1f1b_overlap_node") : Pass(name) {}
  ~Label1F1BOverlapNode() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_PASS_LABEL_1F1B_OVERLAP_NODE_PASS_H_
