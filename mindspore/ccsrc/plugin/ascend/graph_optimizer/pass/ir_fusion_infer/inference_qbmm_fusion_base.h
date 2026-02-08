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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_INFERENCE_QBMM_FUSION_BASE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_INFERENCE_QBMM_FUSION_BASE_H_
#include <string>
#include "include/backend/common/pass_manager/optimizer.h"
#include "primitive/nn_ops.h"
#include "primitive/math_ops.h"
#include "primitive/framework_ops.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace opt {
class QbmmFusionBase : public PatternProcessPass {
 public:
  explicit QbmmFusionBase(const std::string &name, bool multigraph = true) : PatternProcessPass(name, multigraph) {}

 protected:
  bool PassEnable(const std::string &op_name) const;
  void SetNodes(const EquivPtr &equiv) const;
  bool CheckValid() const;
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
  mutable AnfNodePtr x_node_ = nullptr;
  mutable AnfNodePtr w_node_ = nullptr;
  mutable AnfNodePtr scale_node_ = nullptr;
  mutable AnfNodePtr offset_node_ = nullptr;
  mutable AnfNodePtr bias_node_ = nullptr;
  mutable AnfNodePtr pertoken_scale_node_ = nullptr;
  mutable AnfNodePtr bias_tensor_node_ = nullptr;
  mutable AnfNodePtr trans_a_node_ = nullptr;
  mutable AnfNodePtr trans_b_node_ = nullptr;
  mutable AnfNodePtr out_dtype_node_ = nullptr;
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFER_INFERENCE_QBMM_FUSION_BASE_H_
