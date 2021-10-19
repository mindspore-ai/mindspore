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

#include "tools/optimizer/fusion/conv_bn_fusion.h"
#include <memory>
#include "ops/batch_norm.h"
#include "ops/fused_batch_norm.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
namespace {
constexpr size_t kCaffeBNMeanIndex = 2;
constexpr size_t kCaffeBNVarIndex = 3;
constexpr size_t kCaffeBNScaleFactorIndex = 4;
constexpr size_t kTFBNScaleIndex = 2;
constexpr size_t kTFBNBiasIndex = 3;
constexpr size_t kTFBNMeanIndex = 4;
constexpr size_t kTFBNVarIndex = 5;
constexpr float kEps = 1e-8;
constexpr float kPowNum = 0.5;
constexpr float kDefaultEps = 1e-5;
bool IsBatchNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim::kPrimBatchNorm) ||
           CheckPrimitiveType(anf_node, prim::kPrimFusedBatchNorm);
  }
  return false;
}
int CalTransale(const AnfNodePtr &bn_scale_node, const AnfNodePtr &bn_var_node, float *trans_scale, float eps,
                int kernel_num) {
  MS_ASSERT(bn_var_node != nullptr && trans_scale != nullptr);
  auto bn_var_param = bn_var_node->cast<ParameterPtr>()->default_param();
  MS_ASSERT(bn_var_param != nullptr);
  auto bn_var_tensor = std::dynamic_pointer_cast<tensor::Tensor>(bn_var_param);
  MS_ASSERT(bn_var_tensor != nullptr);
  auto bn_var_data = reinterpret_cast<float *>(bn_var_tensor->data_c());
  // cal transScale, tf : scale/sqrt(variance + eps); caffe : 1/sqrt(variance + eps)
  if (memcpy_s(trans_scale, kernel_num * sizeof(float), bn_var_data, bn_var_tensor->Size()) != EOK) {
    MS_LOG(ERROR) << "memcpy_s transScale error";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    return lite::RET_ERROR;
  }
  // 1/sqrt(variance + eps)
  for (int32_t i = 0; i < kernel_num; i++) {
    float tmp = trans_scale[i] + eps;
    tmp = pow(tmp, kPowNum);
    if (tmp <= 0.0f) {
      MS_LOG(ERROR) << "divisor cannot be 0";
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_ERROR);
      return lite::RET_ERROR;
    }
    trans_scale[i] = 1 / tmp;
  }
  if (bn_scale_node != nullptr) {
    auto bn_scale_param = bn_scale_node->cast<ParameterPtr>()->default_param();
    MS_ASSERT(bn_scale_param != nullptr);
    auto bn_scale_tensor = std::dynamic_pointer_cast<tensor::Tensor>(bn_scale_param);
    MS_ASSERT(bn_scale_tensor != nullptr);
    auto bn_scale_data = reinterpret_cast<float *>(bn_scale_tensor->data_c());
    // scale/sqrt(variance + eps)
    for (int32_t i = 0; i < kernel_num; i++) {
      trans_scale[i] *= bn_scale_data[i];
    }
  }
  return lite::RET_OK;
}

void CalTransBias(const AnfNodePtr &bn_mean_node, const AnfNodePtr &bn_bias_node, const float *trans_scale,
                  float *trans_bias, int kernel_num) {
  MS_ASSERT(bn_mean_node != nullptr && trans_scale != nullptr && trans_bias != nullptr);
  auto bn_mean_param = bn_mean_node->cast<ParameterPtr>()->default_param();
  MS_ASSERT(bn_mean_param != nullptr);
  auto bn_mean_tensor = std::dynamic_pointer_cast<tensor::Tensor>(bn_mean_param);
  MS_ASSERT(bn_mean_tensor != nullptr);
  auto bn_mean_data = reinterpret_cast<float *>(bn_mean_tensor->data_c());
  // cal transBias, tf : -scale*mean/sqrt(variance + eps) + bias; caffe : -mean/sqrt(variance + eps)
  // -mean/sqrt(variance + eps)
  for (int32_t i = 0; i < kernel_num; i++) {
    trans_bias[i] = -bn_mean_data[i] * trans_scale[i];
  }

  if (bn_bias_node != nullptr) {
    auto bn_bias_param = bn_bias_node->cast<ParameterPtr>()->default_param();
    MS_ASSERT(bn_bias_param != nullptr);
    auto bn_bias_tensor = std::dynamic_pointer_cast<tensor::Tensor>(bn_bias_param);
    MS_ASSERT(bn_bias_tensor != nullptr);
    auto bn_bias_data = reinterpret_cast<float *>(bn_bias_tensor->data_c());
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
  MS_CHECK_TRUE_RET(origin_param != nullptr, RET_ERROR);
  auto origin_tensor = std::dynamic_pointer_cast<tensor::Tensor>(origin_param);
  MS_CHECK_TRUE_RET(origin_tensor != nullptr, RET_ERROR);
  auto origin_data = reinterpret_cast<float *>(origin_tensor->data_c());

  auto scale_factor_param = scale_factor_node->cast<ParameterPtr>()->default_param();
  MS_CHECK_TRUE_RET(scale_factor_param != nullptr, RET_ERROR);
  auto scale_factor_tensor = std::dynamic_pointer_cast<tensor::Tensor>(scale_factor_param);
  MS_CHECK_TRUE_RET(scale_factor_tensor != nullptr, RET_ERROR);
  if (scale_factor_tensor->DataSize() != 1) {
    MS_LOG(ERROR) << "scale factor data size is not equal to 1";
    return RET_ERROR;
  }
  auto scale_factor_data = (reinterpret_cast<float *>(scale_factor_tensor->data_c()))[0];
  float scale_factor = scale_factor_data == 0 ? 0 : 1 / scale_factor_data;
  for (int i = 0; i < origin_tensor->DataSize(); i++) {
    origin_data[i] = origin_data[i] * scale_factor;
  }
  return RET_OK;
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
  /*
  BatchNorm weight Tensor definition:
   caffe
     mean  --0
     variance  --1
     scale_factor  --2
   tensorflow
     scale    -- 0
     bias        --1
     estimated_mean  --2
     estimated_variance  --3
  */
  MS_ASSERT(bn_node != nullptr);
  MS_ASSERT(trans_bias != nullptr);
  MS_ASSERT(trans_scale != nullptr);
  AnfNodePtr bn_mean_node = nullptr;
  AnfNodePtr bn_variance_node = nullptr;
  AnfNodePtr bn_scale_node = nullptr;
  AnfNodePtr bn_bias_node = nullptr;
  auto primitive_c = GetValueNode<PrimitiveCPtr>(bn_node->input(0));
  MS_ASSERT(primitive_c != nullptr);
  float eps = kDefaultEps;
  if (primitive_c->GetAttr(ops::kEpsilon) != nullptr) {
    eps = GetValue<float>(primitive_c->GetAttr(ops::kEpsilon));
  }
  if (CheckPrimitiveType(bn_node, prim::kPrimBatchNorm)) {
    bn_mean_node = bn_node->input(kCaffeBNMeanIndex);
    MS_ASSERT(bn_mean_node != nullptr);
    bn_variance_node = bn_node->input(kCaffeBNVarIndex);
    MS_ASSERT(bn_variance_node != nullptr);
    AnfNodePtr bn_scale_factor_node = bn_node->input(kCaffeBNScaleFactorIndex);
    if (!bn_mean_node->isa<Parameter>() || !bn_variance_node->isa<Parameter>() || !IsParamNode(bn_scale_factor_node)) {
      MS_LOG(DEBUG) << "bn op's input is dynamic.";
      return lite::RET_NO_CHANGE;
    }
    auto status = CalEstimatedData(bn_mean_node, bn_scale_factor_node);
    MS_CHECK_TRUE_RET(status == lite::RET_OK, status);
    status = CalEstimatedData(bn_variance_node, bn_scale_factor_node);
    MS_CHECK_TRUE_RET(status == lite::RET_OK, status);
  } else if (CheckPrimitiveType(bn_node, prim::kPrimFusedBatchNorm)) {
    bn_scale_node = bn_node->input(kTFBNScaleIndex);
    bn_bias_node = bn_node->input(kTFBNBiasIndex);
    bn_mean_node = bn_node->input(kTFBNMeanIndex);
    bn_variance_node = bn_node->input(kTFBNVarIndex);
  } else {
    MS_LOG(ERROR) << "not caffe or tf batchnorm op.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return lite::RET_NOT_SUPPORT;
  }
  if (!IsParamNode(bn_mean_node) || !IsParamNode(bn_variance_node)) {
    return lite::RET_NO_CHANGE;
  }
  if (eps < kEps) {
    eps = kEps;
  }

  if (CalTransale(bn_scale_node, bn_variance_node, trans_scale, eps, kernel_num) != lite::RET_OK) {
    return lite::RET_NO_CHANGE;
  }
  CalTransBias(bn_mean_node, bn_bias_node, trans_scale, trans_bias, kernel_num);
  return lite::RET_OK;
}
}  // namespace mindspore::opt
