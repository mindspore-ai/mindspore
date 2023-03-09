/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_BROADCASTTO_FISSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_BROADCASTTO_FISSION_H_

#include <vector>
#include <string>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class BroadcasttoFission : public PatternProcessPass {
 public:
  explicit BroadcasttoFission(bool multigraph = true) : PatternProcessPass("broadcastto_fission", multigraph) {}
  ~BroadcasttoFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  CNodePtr AddBroadCastToNode(const FuncGraphPtr &func_graph, const CNodePtr &input_node,
                              const std::vector<int64_t> &broad_shape) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_BROADCASTTO_FISSION_H_
