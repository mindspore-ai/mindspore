/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_PASS_SHAPE_GETITEM_RESHAPE_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_PASS_SHAPE_GETITEM_RESHAPE_FUSION_H_

#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class BACKEND_EXPORT ShapeGetItemReshapeFusion : public PatternProcessPass {
 public:
  explicit ShapeGetItemReshapeFusion(bool multigraph = true)
      : PatternProcessPass("shape_getitem_reshape", multigraph) {}
  ~ShapeGetItemReshapeFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                           const EquivPtr &equiv) const override;

 private:
  VarPtr x_ = std::make_shared<Var>();
  VarPtr y_ = std::make_shared<Var>();
  VarPtr index0_ = std::make_shared<Var>();
  VarPtr index1_ = std::make_shared<Var>();
  VarPtr var_ = std::make_shared<Var>();
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_PASS_SHAPE_GETITEM_RESHAPE_FUSION_H_
