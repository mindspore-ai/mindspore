/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_AVGPOOL_3D_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_AVGPOOL_3D_FUSION_H_
#include <memory>
#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
class AvgPool3DFusion : public PatternProcessPass {
 public:
  explicit AvgPool3DFusion(bool multigraph = true) : PatternProcessPass("avg_pool_3d_fusion", multigraph) {}
  ~AvgPool3DFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;
};

AnfNodePtr ConstructFilterValueNode(const FuncGraphPtr &func_graph, float val, const ShapeVector &assist_shape,
                                    const ShapeVector &infer_shape, int64_t cnt);
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_AVGPOOL_3D_FUSION_H_
