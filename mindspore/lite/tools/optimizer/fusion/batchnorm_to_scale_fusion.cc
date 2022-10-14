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
#include "tools/optimizer/fusion/batchnorm_to_scale_fusion.h"
#include <memory>
#include "ops/batch_norm.h"
#include "ops/fused_batch_norm.h"
#include "include/common/utils/utils.h"
#include "ops/fusion/scale_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/common/tensor_util.h"
#include "securec/include/securec.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::opt {
namespace {
constexpr float kEps = 1e-8;
constexpr float kDefaultEps = 1e-5;
constexpr float kPowNum = 0.5;

constexpr size_t kInputNodeIndex = 1;
constexpr size_t kCaffeBNMeanIndex = 2;
constexpr size_t kCaffeBNVarIndex = 3;
constexpr size_t kCaffeBNScaleFactorIndex = 4;
constexpr size_t kCaffeBNInputSize = 5;

constexpr size_t kTFBNScaleIndex = 2;
constexpr size_t kTFBNBiasIndex = 3;
constexpr size_t kTFBNMeanIndex = 4;
constexpr size_t kTFBNVarIndex = 5;
constexpr size_t kTFBNInputSize = 6;
}  // namespace

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

int CalTransBias(const AnfNodePtr &bn_mean_node, const AnfNodePtr &bn_bias_node, const float *trans_scale,
                 float *trans_bias, int kernel_num) {
  MS_ASSERT(bn_mean_node != nullptr && trans_scale != nullptr && trans_bias != nullptr);
  MS_ASSERT(bn_mean_node->cast<ParameterPtr>() != nullptr);
  auto bn_mean_param = bn_mean_node->cast<ParameterPtr>()->default_param();
  MS_CHECK_TRUE_RET(bn_mean_param != nullptr, lite::RET_ERROR);
  auto bn_mean_tensor = std::dynamic_pointer_cast<tensor::Tensor>(bn_mean_param);
  MS_CHECK_TRUE_RET(bn_mean_tensor != nullptr && bn_mean_tensor->data_c() != nullptr, lite::RET_ERROR);
  auto bn_mean_data = reinterpret_cast<float *>(bn_mean_tensor->data_c());
  // cal transBias, tf : -scale*mean/sqrt(variance + eps) + bias; caffe : -mean/sqrt(variance + eps)
  // -mean/sqrt(variance + eps)
  for (int32_t i = 0; i < kernel_num; i++) {
    trans_bias[i] = -bn_mean_data[i] * trans_scale[i];
  }

  if (bn_bias_node != nullptr) {
    MS_ASSERT(bn_bias_node->cast<ParameterPtr>() != nullptr);
    auto bn_bias_param = bn_bias_node->cast<ParameterPtr>()->default_param();
    MS_CHECK_TRUE_RET(bn_bias_param != nullptr, lite::RET_ERROR);
    auto bn_bias_tensor = std::dynamic_pointer_cast<tensor::Tensor>(bn_bias_param);
    MS_CHECK_TRUE_RET(bn_bias_tensor != nullptr && bn_bias_tensor->data_c() != nullptr, lite::RET_ERROR);
    auto bn_bias_data = reinterpret_cast<float *>(bn_bias_tensor->data_c());
    // -scale*mean/sqrt(variance + eps) + bias
    for (int32_t i = 0; i < kernel_num; i++) {
      trans_bias[i] += bn_bias_data[i];
    }
  }
  return lite::RET_OK;
}

int CalEstimatedData(const AnfNodePtr &origin_node, const AnfNodePtr &scale_factor_node) {
  if (origin_node == nullptr) {
    MS_LOG(ERROR) << "origin node is null";
    return lite::RET_ERROR;
  }

  if (scale_factor_node == nullptr) {
    MS_LOG(ERROR) << "scale factor node is null";
    return lite::RET_ERROR;
  }
  MS_ASSERT(origin_node->cast<ParameterPtr>() != nullptr);
  auto origin_param = origin_node->cast<ParameterPtr>()->default_param();
  MS_CHECK_TRUE_RET(origin_param != nullptr, lite::RET_ERROR);
  auto origin_tensor = std::dynamic_pointer_cast<tensor::Tensor>(origin_param);
  MS_CHECK_TRUE_RET(origin_tensor != nullptr, lite::RET_ERROR);
  auto origin_data = reinterpret_cast<float *>(origin_tensor->data_c());

  MS_ASSERT(scale_factor_node->cast<ParameterPtr>() != nullptr);
  auto scale_factor_param = scale_factor_node->cast<ParameterPtr>()->default_param();
  MS_CHECK_TRUE_RET(scale_factor_param != nullptr, lite::RET_ERROR);
  auto scale_factor_tensor = std::dynamic_pointer_cast<tensor::Tensor>(scale_factor_param);
  MS_CHECK_TRUE_RET(scale_factor_tensor != nullptr, lite::RET_ERROR);
  if (scale_factor_tensor->DataSize() != 1) {
    MS_LOG(ERROR) << "scale factor data size is not equal to 1";
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_RET(scale_factor_tensor->data_c() != nullptr, lite::RET_ERROR);
  auto scale_factor_data = (reinterpret_cast<float *>(scale_factor_tensor->data_c()))[0];
  float scale_factor = scale_factor_data == 0 ? 0 : 1 / scale_factor_data;
  for (size_t i = 0; i < origin_tensor->DataSize(); i++) {
    origin_data[i] = origin_data[i] * scale_factor;
  }
  return lite::RET_OK;
}

int CalculateScaleAndBiasFromBN(const CNodePtr &bn_node, int kernel_num, float *trans_scale, float *trans_bias) {
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
  MS_ASSERT(bn_node != nullptr && trans_bias != nullptr && trans_scale != nullptr);
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
  if (eps < kEps) {
    eps = kEps;
  }
  if (CheckPrimitiveType(bn_node, prim::kPrimBatchNorm) || bn_node->size() == kCaffeBNInputSize) {
    MS_CHECK_TRUE_RET(bn_node->size() == kCaffeBNInputSize, lite::RET_ERROR);
    bn_mean_node = bn_node->input(kCaffeBNMeanIndex);
    MS_CHECK_TRUE_RET(bn_mean_node != nullptr, lite::RET_ERROR);
    bn_variance_node = bn_node->input(kCaffeBNVarIndex);
    MS_CHECK_TRUE_RET(bn_variance_node != nullptr, lite::RET_ERROR);
    AnfNodePtr bn_scale_factor_node = bn_node->input(kCaffeBNScaleFactorIndex);
    MS_CHECK_TRUE_RET(bn_scale_factor_node != nullptr, lite::RET_ERROR);
    if (!bn_mean_node->isa<Parameter>() || !bn_variance_node->isa<Parameter>() || !IsParamNode(bn_scale_factor_node)) {
      MS_LOG(DEBUG) << "bn op's input is dynamic.";
      return lite::RET_NO_CHANGE;
    }
    if (CalEstimatedData(bn_mean_node, bn_scale_factor_node) != lite::RET_OK ||
        CalEstimatedData(bn_variance_node, bn_scale_factor_node) != lite::RET_OK) {
      MS_LOG(ERROR) << "Calculate esimate data failed.";
      return lite::RET_ERROR;
    }
  } else if (CheckPrimitiveType(bn_node, prim::kPrimFusedBatchNorm)) {
    MS_CHECK_TRUE_RET(bn_node->size() == kTFBNInputSize, lite::RET_ERROR);
    bn_scale_node = bn_node->input(kTFBNScaleIndex);
    MS_CHECK_TRUE_RET(bn_scale_node != nullptr, lite::RET_ERROR);
    bn_bias_node = bn_node->input(kTFBNBiasIndex);
    MS_CHECK_TRUE_RET(bn_bias_node != nullptr, lite::RET_ERROR);
    bn_mean_node = bn_node->input(kTFBNMeanIndex);
    MS_CHECK_TRUE_RET(bn_mean_node != nullptr, lite::RET_ERROR);
    bn_variance_node = bn_node->input(kTFBNVarIndex);
    MS_CHECK_TRUE_RET(bn_variance_node != nullptr, lite::RET_ERROR);
  } else {
    MS_LOG(ERROR) << "not caffe or tf batchnorm op.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return lite::RET_NOT_SUPPORT;
  }
  if (!IsParamNode(bn_mean_node) || !IsParamNode(bn_variance_node)) {
    return lite::RET_NO_CHANGE;
  }

  if (CalTransale(bn_scale_node, bn_variance_node, trans_scale, eps, kernel_num) != lite::RET_OK) {
    return lite::RET_NO_CHANGE;
  }
  if (CalTransBias(bn_mean_node, bn_bias_node, trans_scale, trans_bias, kernel_num) != lite::RET_OK) {
    return lite::RET_NO_CHANGE;
  }
  return lite::RET_OK;
}

bool BatchNormToScaleFusion::CheckBNCanFused(const AnfNodePtr &node) {
  MS_ASSERT(node != nullptr);
  if (!utils::isa<CNode>(node) ||
      (!CheckPrimitiveType(node, prim::kPrimBatchNorm) && !CheckPrimitiveType(node, prim::kPrimFusedBatchNorm))) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cnode != nullptr, false);
  if (IsMarkedTrainOp(cnode)) {
    return false;
  }
  auto abstract = GetCNodeInputAbstract(cnode, kInputNodeIndex);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Get abstract failed.";
    return false;
  }
  if (FetchShapeFromAbstract(abstract, &input_shape_) != lite::RET_OK || input_shape_.empty()) {
    return false;
  }
  return true;
}

bool BatchNormToScaleFusion::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    MS_ASSERT(node != nullptr);
    if (!CheckBNCanFused(node)) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    MS_ASSERT(cnode != nullptr);
    auto bn_mean_node =
      CheckPrimitiveType(cnode, prim::kPrimBatchNorm) ? cnode->input(kCaffeBNMeanIndex) : cnode->input(kTFBNMeanIndex);
    MS_CHECK_TRUE_RET(bn_mean_node != nullptr, false);
    auto bn_mean_param = bn_mean_node->cast<ParameterPtr>();
    MS_CHECK_TRUE_RET(bn_mean_param != nullptr, false);
    if (!bn_mean_param->has_default() || bn_mean_param->default_param() == nullptr) {
      MS_LOG(WARNING) << "The mean parameter of batchnorm has no data.";
      return false;
    }
    auto mean_tensor = bn_mean_param->default_param()->cast<tensor::TensorPtr>();
    MS_CHECK_TRUE_RET(mean_tensor != nullptr, false);
    auto channel = mean_tensor->ElementsNum();
    float *trans_scale = static_cast<float *>(malloc(mean_tensor->Size()));
    if (trans_scale == nullptr) {
      MS_LOG(ERROR) << "malloc data failed.";
      return false;
    }
    float *trans_bias = static_cast<float *>(malloc(mean_tensor->Size()));
    if (trans_bias == nullptr) {
      MS_LOG(ERROR) << "malloc data failed.";
      free(trans_scale);
      return false;
    }
    auto ret = CalculateScaleAndBiasFromBN(cnode, channel, trans_scale, trans_bias);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Calculate scale and bias failed.";
      free(trans_scale);
      free(trans_bias);
      return false;
    }

    auto weight_tensor =
      lite::CreateTensorInfo(trans_scale, mean_tensor->Size(), mean_tensor->shape_c(), mean_tensor->data_type());
    auto bias_tensor =
      lite::CreateTensorInfo(trans_bias, mean_tensor->Size(), mean_tensor->shape_c(), mean_tensor->data_type());
    free(trans_scale);
    free(trans_bias);
    MS_CHECK_TRUE_RET(weight_tensor != nullptr && bias_tensor != nullptr, false);
    auto new_weight_param = func_graph->add_parameter();
    auto new_bias_param = func_graph->add_parameter();
    MS_CHECK_TRUE_RET(new_weight_param != nullptr && new_bias_param != nullptr, false);

    if (lite::InitParameterFromTensorInfo(new_weight_param, weight_tensor) != lite::RET_OK ||
        lite::InitParameterFromTensorInfo(new_bias_param, bias_tensor) != lite::RET_OK) {
      MS_LOG(ERROR) << "Create parameter node failed.";
      return false;
    }
    new_weight_param->set_name(cnode->fullname_with_scope() + "_scale");
    new_bias_param->set_name(cnode->fullname_with_scope() + "_bias");

    auto manager = func_graph->manager();
    MS_ASSERT(manager != nullptr);
    auto scale_primitive = std::make_shared<ops::ScaleFusion>();
    if (scale_primitive == nullptr) {
      MS_LOG(ERROR) << "new scale primitive failed";
      return false;
    }
    MS_CHECK_TRUE_RET(!input_shape_.empty(), false);
    int64_t axis = input_shape_.size() == DIMENSION_4D ? -1 : 1;
    scale_primitive->set_axis(axis);
    scale_primitive->set_activation_type(ActivationType::NO_ACTIVATION);
    auto scale_primitive_c = scale_primitive->GetPrim();
    if (scale_primitive_c == nullptr) {
      MS_LOG(ERROR) << "new scale primitive_c failed";
      return false;
    }
    auto scale_node = func_graph->NewCNode(scale_primitive_c, {cnode->input(1), new_weight_param, new_bias_param});
    scale_node->set_fullname_with_scope(cnode->fullname_with_scope());
    scale_node->set_abstract(cnode->abstract());
    (void)manager->Replace(cnode, scale_node);
  }
  return false;
}
}  // namespace mindspore::opt
