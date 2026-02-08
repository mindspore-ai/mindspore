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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_QBMM_ALLREDUCE_CONVERT_BIAS_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_QBMM_ALLREDUCE_CONVERT_BIAS_H_
#include <string>
#include "include/backend/common/pass_manager/optimizer.h"
#include "primitive/nn_ops.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace opt {
class QbmmAllReduceConvertBias : public PatternProcessPass {
 public:
  explicit QbmmAllReduceConvertBias(const std::string &name = "qbmm_allreduce_convert_bias", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {}
  ~QbmmAllReduceConvertBias() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const override;

 protected:
  bool Init() const;
  mutable VarPtr x_ = nullptr;
  mutable VarPtr w_ = nullptr;
  mutable VarPtr scale_ = nullptr;
  mutable VarPtr offset_ = nullptr;
  mutable VarPtr bias_ = nullptr;
  mutable VarPtr pertoken_scale_ = nullptr;
  mutable VarPtr trans_a_ = nullptr;
  mutable VarPtr trans_b_ = nullptr;
  mutable VarPtr out_dtype_ = nullptr;
  mutable VarPtr bias_tensor_ = nullptr;
  mutable VarPtr qbmm_prim_ = nullptr;
  mutable AnfNodePtr bias_node_ = nullptr;
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_QBMM_ALLREDUCE_CONVERT_BIAS_H_
