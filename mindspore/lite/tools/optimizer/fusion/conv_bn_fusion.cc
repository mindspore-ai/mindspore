/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *conv_activation_fusion.h
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tools/optimizer/fusion/conv_bn_fusion.h"
#include <memory>
#include "ops/batch_norm.h"
#include "ops/fused_batch_norm.h"
#include "src/param_value_lite.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"

namespace mindspore::opt {
namespace {
constexpr size_t kCaffeBNMeanIndex = 2;
constexpr size_t kCaffeBNVarIndex = 3;
constexpr size_t kCaffeBNScaleFactorIndex = 4;
constexpr size_t kTFBNScaleIndex = 2;
constexpr size_t kTFBNBiasIndex = 3;
constexpr size_t kTFBNMeanIndex = 4;
constexpr size_t kTFBNVarIndex = 5;
constexpr const float EPS = 1e-8;
constexpr const float POW_NUM = 0.5;
constexpr const float DEFAULT_EPS = 1e-5;
bool IsBatchNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim::kPrimBatchNorm) ||
           CheckPrimitiveType(anf_node, prim::kPrimFusedBatchNorm);
  }
  return false;
}
void CalTransale(const AnfNodePtr &bn_scale_node, const AnfNodePtr &bn_var_node, float *trans_scale, float eps,
                 int kernel_num) {
  auto bn_var_param = bn_var_node->cast<ParameterPtr>()->default_param();
  auto bn_var_tensor = std::dynamic_pointer_cast<ParamValueLite>(bn_var_param);
  auto bn_var_data = reinterpret_cast<float *>(bn_var_tensor->tensor_addr());
  // cal transScale, tf : scale/sqrt(variance + eps); caffe : 1/sqrt(variance + eps)
  if (memcpy_s(trans_scale, kernel_num * sizeof(float), bn_var_data, kernel_num * sizeof(float)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s transScale error";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    return;
  }
  // 1/sqrt(variance + eps)
  for (int32_t i = 0; i < kernel_num; i++) {
    float tmp = trans_scale[i] + eps;
    tmp = pow(tmp, POW_NUM);
    if (tmp <= 0.0f) {
      MS_LOG(ERROR) << "divisor cannot be 0";
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_ERROR);
      return;
    }
    trans_scale[i] = 1 / tmp;
  }
  if (bn_scale_node != nullptr) {
    auto bn_scale_param = bn_scale_node->cast<ParameterPtr>()->default_param();
    auto bn_scale_tensor = std::dynamic_pointer_cast<ParamValueLite>(bn_scale_param);
    auto bn_scale_data = reinterpret_cast<float *>(bn_scale_tensor->tensor_addr());
    // scale/sqrt(variance + eps)
    for (int32_t i = 0; i < kernel_num; i++) {
      trans_scale[i] *= bn_scale_data[i];
    }
  }
}
void CalTransBias(const AnfNodePtr &bn_mean_node, const AnfNodePtr &bn_bias_node, const float *trans_scale,
                  float *trans_bias, int kernel_num) {
  auto bn_mean_param = bn_mean_node->cast<ParameterPtr>()->default_param();
  auto bn_mean_tensor = std::dynamic_pointer_cast<ParamValueLite>(bn_mean_param);
  auto bn_mean_data = reinterpret_cast<float *>(bn_mean_tensor->tensor_addr());
  // cal transBias, tf : -scale*mean/sqrt(variance + eps) + bias; caffe : -mean/sqrt(variance + eps)
  // -mean/sqrt(variance + eps)
  for (int32_t i = 0; i < kernel_num; i++) {
    trans_bias[i] = -bn_mean_data[i] * trans_scale[i];
  }

  if (bn_bias_node != nullptr) {
    auto bn_bias_param = bn_bias_node->cast<ParameterPtr>()->default_param();
    auto bn_bias_tensor = std::dynamic_pointer_cast<ParamValueLite>(bn_bias_param);
    auto bn_bias_data = reinterpret_cast<float *>(bn_bias_tensor->tensor_addr());
    // -scale*mean/sqrt(variance + eps) + bias
    for (int32_t i = 0; i < kernel_num; i++) {
      trans_bias[i] += bn_bias_data[i];
    }
  }
}

STATUS CalEstimatedData(const AnfNodePtr &origin_node, const AnfNodePtr &scale_factor_node) {
  if (origin_node == nullptr) {
    MS_LOG(ERROR) << "origin node is null";
    return RET_ERROR;
  }

  if (scale_factor_node == nullptr) {
    MS_LOG(ERROR) << "scale factor node is null";
    return RET_ERROR;
  }
  auto origin_param = origin_node->cast<ParameterPtr>()->default_param();
  auto origin_tensor = std::dynamic_pointer_cast<ParamValueLite>(origin_param);
  auto origin_data = reinterpret_cast<float *>(origin_tensor->tensor_addr());

  auto scale_factor_param = scale_factor_node->cast<ParameterPtr>()->default_param();
  auto scale_factor_tensor = std::dynamic_pointer_cast<ParamValueLite>(scale_factor_param);
  if (scale_factor_tensor->tensor_shape_size() < 1) {
    MS_LOG(ERROR) << "scale factor data size is not equal to 1";
    return RET_ERROR;
  }
  auto scale_factor_data = (reinterpret_cast<float *>(scale_factor_tensor->tensor_addr()))[0];
  float scale_factor = scale_factor_data == 0 ? 0 : 1 / scale_factor_data;
  for (int i = 0; i < origin_tensor->tensor_shape_size(); i++) {
    origin_data[i] = origin_data[i] * scale_factor;
  }
  return RET_OK;
}
}  // namespace
const BaseRef ConvBatchNormFusion::DefinePattern() const {
  auto conv_var = std::make_shared<CondVar>(IsConvNode);
  auto bn_var = std::make_shared<CondVar>(IsBatchNode);
  auto bn_mean_var = std::make_shared<CondVar>(IsParamNode);
  auto bn_variable_var = std::make_shared<CondVar>(IsParamNode);
  auto bn_other_var = std::make_shared<SeqVar>();
  return VectorRef({bn_var, conv_var, bn_mean_var, bn_variable_var, bn_other_var});
}
// BatchNorm weight Tensor definition:
// caffe
//   mean  --0
//   variance  --1
//   scale_factor  --2
// tensorflow
//   scale    -- 0
//   bias        --1
//   estimated_mean  --2
//   estimated_variance  --3
void ConvBatchNormFusion::InitTransParam(const CNodePtr &bn_node, int kernel_num, float *trans_scale,
                                         float *trans_bias) const {
  MS_ASSERT(bn_node != nullptr);
  MS_ASSERT(trans_bias != nullptr);
  MS_ASSERT(trans_scale != nullptr);
  AnfNodePtr bn_mean_node = nullptr;
  AnfNodePtr bn_variance_node = nullptr;
  AnfNodePtr bn_scale_node = nullptr;
  AnfNodePtr bn_bias_node = nullptr;
  float eps = 0;
  auto primitive_c = GetValueNode<PrimitiveCPtr>(bn_node->input(0));
  if (CheckPrimitiveType(bn_node, prim::kPrimBatchNorm)) {
    bn_mean_node = bn_node->input(kCaffeBNMeanIndex);
    bn_variance_node = bn_node->input(kCaffeBNVarIndex);
    AnfNodePtr bn_scale_factor_node = bn_node->input(kCaffeBNScaleFactorIndex);
    if (CheckIfNodeIsParam(bn_mean_node) != lite::RET_OK || CheckIfNodeIsParam(bn_variance_node) != lite::RET_OK ||
        CheckIfNodeIsParam(bn_scale_factor_node) != lite::RET_OK) {
      return;
    }
    MS_ASSERT(utils::isa<std::shared_ptr<mindspore::ops::BatchNorm>>(primitive_c));
    auto primc = utils::cast<std::shared_ptr<mindspore::ops::BatchNorm>>(primitive_c);
    MS_ASSERT(primc != nullptr);
    if (primc->GetAttr("epsilon") != nullptr) {
      eps = primc->get_epsilon();
    } else {
      eps = DEFAULT_EPS;
    }
    CalEstimatedData(bn_mean_node, bn_scale_factor_node);
    CalEstimatedData(bn_variance_node, bn_scale_factor_node);
  } else if (CheckPrimitiveType(bn_node, prim::kPrimFusedBatchNorm)) {
    bn_scale_node = bn_node->input(kTFBNScaleIndex);
    bn_bias_node = bn_node->input(kTFBNBiasIndex);
    bn_mean_node = bn_node->input(kTFBNMeanIndex);
    bn_variance_node = bn_node->input(kTFBNVarIndex);
    MS_ASSERT(utils::isa<std::shared_ptr<mindspore::ops::FusedBatchNorm>>(primitive_c));
    auto primc = utils::cast<std::shared_ptr<mindspore::ops::FusedBatchNorm>>(primitive_c);
    MS_ASSERT(primc != nullptr);
    if (primc->GetAttr("epsilon") != nullptr) {
      eps = primc->get_epsilon();
    } else {
      eps = DEFAULT_EPS;
    }
  } else {
    MS_LOG(ERROR) << "not caffe or tf batchnorm op.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return;
  }
  if (CheckIfNodeIsParam(bn_mean_node) != lite::RET_OK || CheckIfNodeIsParam(bn_variance_node) != lite::RET_OK) {
    return;
  }
  if (eps < EPS) {
    eps = EPS;
  }

  CalTransale(bn_scale_node, bn_variance_node, trans_scale, eps, kernel_num);
  CalTransBias(bn_mean_node, bn_bias_node, trans_scale, trans_bias, kernel_num);
}
}  // namespace mindspore::opt
