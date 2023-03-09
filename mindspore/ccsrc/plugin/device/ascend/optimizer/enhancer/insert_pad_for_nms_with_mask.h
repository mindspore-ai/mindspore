/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_ENHANCER_INSERT_PAD_FOR_NMS_WITH_MASK_H
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_ENHANCER_INSERT_PAD_FOR_NMS_WITH_MASK_H

#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pass.h"

namespace mindspore {
namespace opt {
class InsertPadForNMSWithMask : public PatternProcessPass {
 public:
  explicit InsertPadForNMSWithMask(bool multigraph = true)
      : PatternProcessPass("insert_pad_for_nms_with_mask", multigraph) {}
  ~InsertPadForNMSWithMask() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  AnfNodePtr InsertPadToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const TypeId &origin_type,
                              const abstract::BaseShapePtr &origin_shape) const;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_ENHANCER_INSERT_PAD_FOR_NMS_WITH_MASK_H
