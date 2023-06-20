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
#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/conv_bn_fusion.h"
#include <memory>
#include "mindspore/core/ops/nn_ops.h"
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/fusion/batchnorm_to_scale_fusion.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
namespace {
bool IsBatchNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim::kPrimBatchNorm) ||
           CheckPrimitiveType(anf_node, prim::kPrimFusedBatchNorm);
  }
  return false;
}
}  // namespace
const BaseRef ConvBatchNormFusion::DefinePattern() const {
  auto is_conv = std::make_shared<CondVar>(IsConvNode);
  MS_CHECK_TRUE_RET(is_conv != nullptr, {});
  auto is_bn = std::make_shared<CondVar>(IsBatchNode);
  MS_CHECK_TRUE_RET(is_bn != nullptr, nullptr);
  auto is_param_bn_mean = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param_bn_mean != nullptr, nullptr);
  auto is_param_bn_var = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param_bn_var != nullptr, nullptr);
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, nullptr);
  return VectorRef({is_bn, is_conv, is_param_bn_mean, is_param_bn_var, is_seq_var});
}

int ConvBatchNormFusion::InitTransParam(const CNodePtr &bn_node, int kernel_num, float *trans_scale,
                                        float *trans_bias) const {
  MS_ASSERT(bn_node != nullptr && trans_bias != nullptr && trans_scale != nullptr);
  auto ret = CalculateScaleAndBiasFromBN(bn_node, kernel_num, trans_scale, trans_bias);
  return ret;
}
}  // namespace mindspore::opt
