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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFERENCE_SWIGLU_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFERENCE_SWIGLU_FUSION_H_

#include <memory>
#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "mindspore/core/ops/math_ops.h"

namespace mindspore {
namespace opt {
class InferenceSwiGLUFusion : public PatternProcessPass {
 public:
  explicit InferenceSwiGLUFusion(const std::string &name = "inference_swiglu_fusion", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {}

  ~InferenceSwiGLUFusion() override = default;

  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
  const BaseRef DefinePattern() const override;

 private:
  bool Init() const;
  CNodePtr CreateSwiGLUNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv) const;

 protected:
  mutable VarPtr input_ = nullptr;
  mutable VarPtr split_size_ = nullptr;
  mutable VarPtr axis_ = nullptr;
  mutable VarPtr split_prim_ = nullptr;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFERENCE_SWIGLU_FUSION_H_
