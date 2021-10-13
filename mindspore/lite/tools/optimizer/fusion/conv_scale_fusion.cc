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

#include "tools/optimizer/fusion/conv_scale_fusion.h"
#include <memory>
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
namespace {
constexpr size_t kScaleWeightIndex = 2;
constexpr size_t kScaleBiasIndex = 3;
constexpr size_t kScaleNoBiasLen = 3;
constexpr size_t kScaleWithBiasLen = 4;
}  // namespace

const BaseRef ConvScaleFusion::DefinePattern() const {
  auto is_conv = std::make_shared<CondVar>(IsConvNode);
  MS_CHECK_TRUE_RET(is_conv != nullptr, {});
  auto is_scale = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimScaleFusion>);
  MS_CHECK_TRUE_RET(is_scale != nullptr, {});
  auto is_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  return VectorRef({is_scale, is_conv, is_param, is_seq_var});
}
int ConvScaleFusion::InitTransParam(const CNodePtr &scale_node, int kernel_num, float *trans_scale,
                                    float *trans_bias) const {
  MS_ASSERT(scale_node != nullptr);
  MS_ASSERT(trans_bias != nullptr);
  MS_ASSERT(trans_scale != nullptr);
  AnfNodePtr scale_weight_node;
  AnfNodePtr scale_bias_node;
  if (scale_node->inputs().size() == kScaleNoBiasLen) {
    scale_weight_node = scale_node->input(kScaleWeightIndex);
  } else if (scale_node->inputs().size() == kScaleWithBiasLen) {
    scale_weight_node = scale_node->input(kScaleWeightIndex);
    scale_bias_node = scale_node->input(kScaleBiasIndex);
  } else {
    MS_LOG(ERROR) << "Scale should has 2 or 3 input tensors, current inputs is" << scale_node->inputs().size();
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INPUT_TENSOR_ERROR);
    return lite::RET_ERROR;
  }
  if (!scale_weight_node->isa<Parameter>()) {
    MS_LOG(ERROR) << "scale weight node not parameter node";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return lite::RET_ERROR;
  }
  if (scale_bias_node != nullptr && !IsParamNode(scale_bias_node)) {
    MS_LOG(DEBUG) << "scale bias input is dynamic.";
    return lite::RET_NO_CHANGE;
  }
  auto scale_weight_param = scale_weight_node->cast<ParameterPtr>()->default_param();
  MS_ASSERT(scale_weight_param != nullptr);
  auto weight_value = std::dynamic_pointer_cast<tensor::Tensor>(scale_weight_param);
  MS_ASSERT(weight_value != nullptr);
  auto weight_data = reinterpret_cast<const float *>(weight_value->data_c());

  if (memcpy_s(trans_scale, kernel_num * sizeof(float), weight_data, weight_value->Size()) != EOK) {
    MS_LOG(ERROR) << "memcpy_s transScale failed";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    return lite::RET_ERROR;
  }

  if (scale_bias_node != nullptr) {
    auto scale_bias_param = scale_bias_node->cast<ParameterPtr>()->default_param();
    MS_ASSERT(scale_bias_param != nullptr);
    auto bias_value = std::dynamic_pointer_cast<tensor::Tensor>(scale_bias_param);
    MS_ASSERT(bias_value != nullptr);
    auto bias_data = reinterpret_cast<const float *>(bias_value->data_c());
    if (memcpy_s(trans_bias, kernel_num * sizeof(float), bias_data, bias_value->Size()) != EOK) {
      MS_LOG(ERROR) << "memcpy_s transScale failed";
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}
}  // namespace mindspore::opt
