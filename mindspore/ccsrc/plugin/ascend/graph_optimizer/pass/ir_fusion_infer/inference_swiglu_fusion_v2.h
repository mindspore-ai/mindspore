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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFERENCE_SWIGLU_FUSION_V2_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFERENCE_SWIGLU_FUSION_V2_H_

#include <memory>
#include <vector>
#include <string>
#include "include/backend/common/pass_manager/optimizer.h"
#include "primitive/math_ops.h"

namespace mindspore {
namespace opt {
class InferenceSwiGLUFusionV2 : public PatternProcessPass {
 public:
  explicit InferenceSwiGLUFusionV2(const std::string &name = "inference_swiglu_fusion_v2", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {}

  ~InferenceSwiGLUFusionV2() override = default;

  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
  const BaseRef DefinePattern() const override;

 private:
  bool Init() const;
  CNodePtr CreateSwiGLUNodeV2(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv) const;
  std::vector<std::string> MustExistPrimitiveName() const override;

 protected:
  mutable VarPtr input_ = nullptr;
  mutable VarPtr split_size_ = nullptr;
  mutable VarPtr axis_ = nullptr;
  mutable VarPtr split_prim_ = nullptr;
  mutable VarPtr reshape_size_ = nullptr;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFERENCE_SWIGLU_FUSION_V2_H_
