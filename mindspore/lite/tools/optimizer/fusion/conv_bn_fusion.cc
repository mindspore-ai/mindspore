/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/param_value_lite.h"
#include "schema/inner/model_generated.h"
#include "src/ir/primitive_t_value.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"

namespace mindspore::opt {
namespace {
constexpr size_t kCaffeBNMeanIndex = 2;
constexpr size_t kCaffeBNVarIndex = 3;
constexpr size_t kTFBNScaleIndex = 2;
constexpr size_t kTFBNBiasIndex = 3;
constexpr size_t kTFBNMeanIndex = 4;
constexpr size_t kTFBNVarIndex = 5;
constexpr const float EPS = 1e-8;
constexpr const float POW_NUM = 0.5;
bool IsBatchNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_BatchNorm || type == schema::PrimitiveType_FusedBatchNorm;
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
    MS_LOG(EXCEPTION) << "memcpy_s transScale error";
    return;
  }
  // 1/sqrt(variance + eps)
  for (int32_t i = 0; i < kernel_num; i++) {
    float tmp = trans_scale[i] + eps;
    tmp = pow(tmp, POW_NUM);
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
}  // namespace
const BaseRef ConvBatchNormFusion::DefinePattern() const {
  auto conv_var = std::make_shared<CondVar>(IsConvNode);
  auto bn_var = std::make_shared<CondVar>(IsBatchNode);
  auto bn_mean_var = std::make_shared<CondVar>(IsParamNode);
  auto bn_variable_var = std::make_shared<CondVar>(IsParamNode);
  auto bn_other_var = std::make_shared<SeqVar>();
  return VectorRef({bn_var, conv_var, bn_mean_var, bn_variable_var, bn_other_var});;
}
// BatchNorm weight Tensor definition:
// caffe
//   estimated_mean  --0
//   estimated_variance  --1
// tensorflow
//   scale    -- 0
//   bias        --1
//   estimated_mean  --2
//   estimated_variance  --3
const void ConvBatchNormFusion::InitTransParam(const CNodePtr &bn_node, int kernel_num, float *trans_scale,
                                                       float *trans_bias) const {
  MS_ASSERT(bn_node != nullptr);
  AnfNodePtr bn_mean_node = nullptr;
  AnfNodePtr bn_variance_node = nullptr;
  AnfNodePtr bn_scale_node = nullptr;
  AnfNodePtr bn_bias_node = nullptr;
  float eps = 0;
  auto primitiveT_value = GetValueNode<std::shared_ptr<lite::PrimitiveTValue>>(bn_node->input(0));
  if (GetCNodeType(bn_node) == schema::PrimitiveType_BatchNorm) {
    bn_mean_node = bn_node->input(kCaffeBNMeanIndex);
    bn_variance_node = bn_node->input(kCaffeBNVarIndex);
    CheckIfNodeIsParam(bn_mean_node);
    CheckIfNodeIsParam(bn_variance_node);
    eps = primitiveT_value->GetPrimitiveT()->value.AsBatchNorm()->epsilon;
  } else if (GetCNodeType(bn_node) == schema::PrimitiveType_FusedBatchNorm) {
    bn_scale_node = bn_node->input(kTFBNScaleIndex);
    bn_bias_node = bn_node->input(kTFBNBiasIndex);
    bn_mean_node = bn_node->input(kTFBNMeanIndex);
    bn_variance_node = bn_node->input(kTFBNVarIndex);
    eps = primitiveT_value->GetPrimitiveT()->value.AsFusedBatchNorm()->epsilon;
  } else {
    MS_LOG(EXCEPTION) << "not caffe or tf batchnorm op.";
  }
  CheckIfNodeIsParam(bn_mean_node);
  CheckIfNodeIsParam(bn_variance_node);
  if (eps < EPS) {
    eps = EPS;
  }

  CalTransale(bn_scale_node, bn_variance_node, trans_scale, eps, kernel_num);
  CalTransBias(bn_mean_node, bn_bias_node, trans_scale, trans_bias, kernel_num);
}
}  // namespace mindspore::opt
