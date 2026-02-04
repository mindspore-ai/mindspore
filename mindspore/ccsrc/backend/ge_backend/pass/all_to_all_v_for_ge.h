/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_PASS_ALL_TO_ALL_V_FOR_GE_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_PASS_ALL_TO_ALL_V_FOR_GE_H_

#include <string>
#include <vector>
#include "include/backend/common/pass_manager/optimizer.h"

/* This pass adapts AlltoAllV node to GE prototype
 *                       sendcount  recvcount
 *     input           input  |senddispl| recvdispl
 *       |                 \  |    |    |  /
 *  [AlltoAllV]      ->      [AlltoAllVGE]
 *       |                        |
 *     output                   output
 */

namespace mindspore {
namespace opt {
class AlltoAllVForGE : public PatternProcessPass {
 public:
  explicit AlltoAllVForGE(bool multigraph = true) : PatternProcessPass("all_to_all_v_for_ge", multigraph) {}
  ~AlltoAllVForGE() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
  CNodePtr CreateAlltoAllVForGENode(const FuncGraphPtr &graph, const CNodePtr &origin_node) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_PASS_ALL_TO_ALL_V_FOR_GE_H_
