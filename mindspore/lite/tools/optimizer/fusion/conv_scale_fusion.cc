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

#include "tools/optimizer/fusion/conv_scale_fusion.h"
#include <memory>
#include "src/param_value_lite.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"

namespace mindspore::opt {
namespace {
constexpr size_t kScaleWeightIndex = 2;
constexpr size_t kScaleBiasIndex = 3;
constexpr size_t kScaleNoBiasLen = 3;
constexpr size_t kScaleWithBiasLen = 4;
bool IsScaleNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim::kPrimScaleFusion);
  }
  return false;
}
}  // namespace

const BaseRef ConvScaleFusion::DefinePattern() const {
  auto conv_var = std::make_shared<CondVar>(IsConvNode);
  auto bn_var = std::make_shared<CondVar>(IsScaleNode);
  auto weight_var = std::make_shared<CondVar>(IsParamNode);
  auto bias_var = std::make_shared<SeqVar>();
  return VectorRef({bn_var, conv_var, weight_var, bias_var});
}
void ConvScaleFusion::InitTransParam(const CNodePtr &scale_node, int kernel_num, float *trans_scale,
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
    return;
  }
  if (!scale_weight_node->isa<Parameter>()) {
    MS_LOG(ERROR) << "scale weight node not parameter node";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return;
  }
  if (scale_bias_node != nullptr && !scale_bias_node->isa<Parameter>()) {
    MS_LOG(ERROR) << "scale bias node not parameter node";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return;
  }
  auto scale_weight_param = scale_weight_node->cast<ParameterPtr>()->default_param();
  auto weight_value = std::dynamic_pointer_cast<ParamValueLite>(scale_weight_param);
  auto weight_data = reinterpret_cast<const float *>(weight_value->tensor_addr());

  if (EOK != memcpy_s(trans_scale, kernel_num * sizeof(float), weight_data, kernel_num * sizeof(float))) {
    MS_LOG(ERROR) << "memcpy_s transScale failed";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    return;
  }

  if (scale_bias_node != nullptr) {
    auto scale_bias_param = scale_bias_node->cast<ParameterPtr>()->default_param();
    auto bias_value = std::dynamic_pointer_cast<ParamValueLite>(scale_bias_param);
    auto bias_data = reinterpret_cast<const float *>(bias_value->tensor_addr());
    if (EOK != memcpy_s(trans_bias, kernel_num * sizeof(float), bias_data, kernel_num * sizeof(float))) {
      MS_LOG(ERROR) << "memcpy_s transScale failed";
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    }
  }
}
}  // namespace mindspore::opt
