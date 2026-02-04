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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_qbmm_fusion_base.h"
#include <memory>
#include <string>
#include "backend/common/pass/common/gllo_utils.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_weight_preprocess_utils.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"

namespace mindspore {
namespace opt {
bool QbmmFusionBase::Init() const {
  x_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(x_ != nullptr, false);
  w_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(w_ != nullptr, false);
  scale_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(scale_ != nullptr, false);
  offset_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(offset_ != nullptr, false);
  bias_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(bias_ != nullptr, false);
  pertoken_scale_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(pertoken_scale_ != nullptr, false);
  trans_a_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(trans_a_ != nullptr, false);
  trans_b_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(trans_b_ != nullptr, false);
  out_dtype_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(out_dtype_ != nullptr, false);
  qbmm_prim_ = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimQuantBatchMatmul>);
  MS_CHECK_TRUE_RET(qbmm_prim_ != nullptr, false);
  return true;
}

void QbmmFusionBase::SetNodes(const EquivPtr &equiv) const {
  x_node_ = utils::cast<AnfNodePtr>((*equiv)[x_]);
  MS_ASSERT(x_node_ != nullptr);
  w_node_ = utils::cast<AnfNodePtr>((*equiv)[w_]);
  MS_ASSERT(w_node_ != nullptr);
  scale_node_ = utils::cast<AnfNodePtr>((*equiv)[scale_]);
  MS_ASSERT(scale_node_ != nullptr);
  offset_node_ = utils::cast<AnfNodePtr>((*equiv)[offset_]);
  MS_ASSERT(offset_node_ != nullptr);
  bias_node_ = utils::cast<AnfNodePtr>((*equiv)[bias_]);
  MS_ASSERT(bias_node_ != nullptr);
  pertoken_scale_node_ = utils::cast<AnfNodePtr>((*equiv)[pertoken_scale_]);
  MS_ASSERT(pertoken_scale_node_ != nullptr);
  bias_tensor_node_ = utils::cast<AnfNodePtr>((*equiv)[bias_tensor_]);
  MS_ASSERT(bias_tensor_node_ != nullptr);
  trans_a_node_ = utils::cast<AnfNodePtr>((*equiv)[trans_a_]);
  MS_ASSERT(trans_a_node != nullptr);
  trans_b_node_ = utils::cast<AnfNodePtr>((*equiv)[trans_b_]);
  MS_ASSERT(trans_b_node != nullptr);
  out_dtype_node_ = utils::cast<AnfNodePtr>((*equiv)[out_dtype_]);
  MS_ASSERT(out_dtype_node_ != nullptr);
}

bool QbmmFusionBase::PassEnable(const std::string &op_name) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  auto enable_fusion = (std::find(enable_op_list.begin(), enable_op_list.end(), op_name) != enable_op_list.end());
  if (!enable_fusion) {
    return false;
  }
  return true;
}

bool QbmmFusionBase::CheckValid() const {
  if (!CheckSupportDataType(bias_tensor_node_, {kNumberTypeFloat16, kNumberTypeBFloat16}) ||
      !CheckSupportDataType(scale_node_, {kNumberTypeInt64, kNumberTypeFloat32}) ||
      !CheckSupportDataType(bias_node_, {kMetaTypeNone})) {
    return false;
  }
  auto dtype_value = GetValue<int64_t>(out_dtype_node_->cast<ValueNodePtr>()->value());
  if (dtype_value != static_cast<int64_t>(kNumberTypeFloat16) &&
      dtype_value != static_cast<int64_t>(kNumberTypeBFloat16)) {
    return false;
  }
  auto bias_shape = common::AnfAlgo::GetOutputInferShape(bias_tensor_node_, kIndex0);
  auto scale_shape = common::AnfAlgo::GetOutputInferShape(scale_node_, kIndex0);
  if (bias_shape.size() != 1 || scale_shape.size() != 1 || bias_shape[0] != scale_shape[0]) {
    return false;
  }
  auto scale_param = GetParamFromLoad(scale_node_->cast<CNodePtr>(), false);
  if (!scale_param) {
    return false;
  }
  auto bias_tensor_param = GetParamFromLoad(bias_tensor_node_->cast<CNodePtr>(), false);
  if (!bias_tensor_param) {
    return false;
  }
  return true;
}

}  // namespace opt
}  // namespace mindspore
