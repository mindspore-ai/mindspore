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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFERENCE_QBMM_ALLREDUCE_ADD_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFERENCE_QBMM_ALLREDUCE_ADD_FUSION_H_

#include <memory>
#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "mindspore/core/ops/math_ops.h"

namespace mindspore {
namespace opt {
class QbmmAllReduceAddFusion : public PatternProcessPass {
 public:
  explicit QbmmAllReduceAddFusion(const std::string &name = "inference_qbmm_allreduce_add_fusion",
                                  bool multigraph = true)
      : PatternProcessPass(name, multigraph) {}

  ~QbmmAllReduceAddFusion() override = default;

  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
  const BaseRef DefinePattern() const override;

 private:
  bool Init() const;
  CNodePtr UpdateQbmmAllReduceAddNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                      const EquivPtr &equiv) const;

 protected:
  mutable VarPtr qbmm_prim_ = nullptr;
  mutable VarPtr x_ = nullptr;
  mutable VarPtr w_ = nullptr;
  mutable VarPtr scale_ = nullptr;
  mutable VarPtr unused_offset_ = nullptr;
  mutable VarPtr orig_bias_ = nullptr;
  mutable VarPtr trans_a_ = nullptr;
  mutable VarPtr trans_b_ = nullptr;
  mutable VarPtr out_dtype_ = nullptr;
  mutable VarPtr bias_tensor_ = nullptr;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFERENCE_QBMM_ALLREDUCE_ADD_FUSION_H_
