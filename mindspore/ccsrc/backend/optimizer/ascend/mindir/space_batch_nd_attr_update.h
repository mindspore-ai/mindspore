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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_SPACE_BATCH_ND_ATTR_UPDATE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_SPACE_BATCH_ND_ATTR_UPDATE_H_

#include <memory>
#include "backend/optimizer/common/optimizer.h"

namespace mindspore {
namespace opt {
class SpaceToBatchNDAttrUpdate : public PatternProcessPass {
 public:
  explicit SpaceToBatchNDAttrUpdate(bool multigraph = true)
      : PatternProcessPass("space_to_batch_nd_attr_update", multigraph) {}
  ~SpaceToBatchNDAttrUpdate() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};

class BatchToSpaceNDAttrUpdate : public PatternProcessPass {
 public:
  explicit BatchToSpaceNDAttrUpdate(bool multigraph = true)
      : PatternProcessPass("batch_to_space_nd_attr_update", multigraph) {}
  ~BatchToSpaceNDAttrUpdate() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_SPACE_BATCH_ND_ATTR_UPDATE_H_
