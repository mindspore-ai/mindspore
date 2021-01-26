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

#ifndef MINDSPORE_LITE_NESTED_LOOP_EXPAND_PASS_H
#define MINDSPORE_LITE_NESTED_LOOP_EXPAND_PASS_H

#include <vector>
#include <utility>
#include <set>
#include <memory>
#include "tools/converter/optimizer.h"

namespace mindspore {
namespace lite {
class NestedLoopExpandPass : public GraphPass {
 public:
  NestedLoopExpandPass() = default;

  ~NestedLoopExpandPass() override = default;

  STATUS Run(schema::MetaGraphT *graph) override;

 private:
  bool IsNestedPartial(const std::unique_ptr<CNodeT> &node);

  void ReplacePartialNodeWithSubgraph(const std::unique_ptr<SubGraphT> &main_graph);

  schema::MetaGraphT *graph_ = nullptr;

  std::vector<int> subgraph_to_drop_{};
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_PREDICT_ISOLATED_NODE_REMOVE_PASS_H
