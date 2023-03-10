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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FORMAT_DELETE_REDUNDANT_TRANSPOSE_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FORMAT_DELETE_REDUNDANT_TRANSPOSE_H_

#include "include/backend/optimizer/pass.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {
class DeleteRedundantTranspose : public Pass {
 public:
  DeleteRedundantTranspose() : Pass("DeleteRedundantTranspose") {}
  ~DeleteRedundantTranspose() = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  STATUS DeleteNot4DTranspose(const FuncGraphPtr &func_graph);
  STATUS DeleteControlFlowTranspose(const CNodePtr &cnode);
  STATUS TransTransFusion(const FuncGraphPtr &func_graph);
  STATUS UpdateNodeFormat(const CNodePtr &node);
  STATUS DoTransTransFusion(const FuncGraphPtr &func_graph, const CNodePtr &node);

  FuncGraphManagerPtr manager_{nullptr};
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FORMAT_DELETE_REDUNDANT_TRANSPOSE_H_
