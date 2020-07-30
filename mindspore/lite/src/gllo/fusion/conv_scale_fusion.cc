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

#include "src/gllo/fusion/conv_scale_fusion.h"
#include <memory>
#include "schema/inner/model_generated.h"
#include "src/ir/primitive_t_value.h"
#include "src/param_value_lite.h"
#include "mindspore/ccsrc/utils/utils.h"
#include "src/gllo/common/utils.h"
#include "include/errorcode.h"

namespace mindspore {
namespace opt {
const BaseRef ConvScaleFusion::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  // conv2d inputs may be 2 or 3 inputs,match move to process
  auto prim = new schema::PrimitiveT();
  prim->value.type = schema::PrimitiveType_Scale;
  auto prim_value = std::make_shared<lite::PrimitiveTValue>(prim);

  return VectorRef({prim_value, X});
}

const AnfNodePtr ConvScaleFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &) const {
  MS_LOG(DEBUG) << "conv activation pass process";
  CheckIfFuncGraphIsNull(func_graph);

  CheckIfAnfNodeIsNull(node);
  auto scale_node = node->cast<CNodePtr>();
  CheckIfCNodeIsNull(scale_node);
  CheckInputSize(scale_node, 2);

  AnfNodePtr pre_node = scale_node->input(1);
  CheckIfAnfNodeIsNull(pre_node);
  if (pre_node != nullptr && pre_node->isa<CNode>()) {
    auto conv_node = pre_node->cast<CNodePtr>();
    auto node_type = GetCNodeType(conv_node);
    if (node_type == schema::PrimitiveType_Conv2D || node_type == schema::PrimitiveType_DepthwiseConv2D) {
      return DoFusion(conv_node, scale_node);
    }
  }
  return node;
}
const AnfNodePtr ConvScaleFusion::DoFusion(const CNodePtr &conv_node, const CNodePtr &scale_node) const {
  if (scale_node->inputs().size() == 3) {
    GetTransParam(scale_node->input(2), nullptr);
  } else if (scale_node->inputs().size() == 4) {
    // todo add bias fusion zhengjun10
    GetTransParam(scale_node->input(2), scale_node->input(3));
  } else {
    MS_LOG(ERROR) << "scale inputs size is error:" << scale_node->DebugString();
    return nullptr;
  }

  AnfNodePtr conv_weight_node;
  if (conv_node->inputs().size() == 3) {
     conv_weight_node = conv_node->input(2);
  } else {
    MS_LOG(ERROR) << "scale inputs size is error:" << scale_node->DebugString();
    return nullptr;
  }
  auto conv_weight_param = conv_weight_node->cast<ParameterPtr>()->default_param();
  auto weight_value = std::dynamic_pointer_cast<ParamValueLite>(conv_weight_param);
  auto old_conv_weight = reinterpret_cast<const float *>(weight_value->tensor_addr());

  auto new_conv_weight = new(std::nothrow) float[weight_value->tensor_shape_size()];
  CalNewWeightTensor(old_conv_weight, new_conv_weight, weight_value->tensor_shape_size());
  weight_value->set_tensor_addr(new_conv_weight);
  return conv_node;
}

const lite::STATUS ConvScaleFusion::GetTransParam(const AnfNodePtr &scale_weight_node,
                                                  const AnfNodePtr &scale_bias_node) const {
  if (!scale_weight_node->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "scale weight node not paramter node";
  }
  if (scale_bias_node != nullptr && !scale_bias_node->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "scale bias node not paramter node";
  }
  auto scale_weight_param = scale_weight_node->cast<ParameterPtr>()->default_param();
  auto weight_value = std::dynamic_pointer_cast<ParamValueLite>(scale_weight_param);
  auto weight_data = reinterpret_cast<const float *>(weight_value->tensor_addr());

  if (0 != memcpy_s(trans_scale, kernel_nums * sizeof(float), weight_data, kernel_nums * sizeof(float))) {
    MS_LOG(ERROR) << "memcpy_s transScale failed";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

const lite::STATUS ConvScaleFusion::CalNewWeightTensor(const float *oldWeightTensor, float *newWeightTensor,
                                                       const size_t tensor_shape_size) const {
  MS_ASSERT(oldWeightTensor != nullptr);
  if (0 != memset_s(newWeightTensor, tensor_shape_size * sizeof(float), 0, tensor_shape_size * sizeof(float))) {
    MS_LOG(ERROR) << "memset newWeightData failed";
    return lite::RET_ERROR;
  }
  if (kernel_nums == 0) {
    MS_LOG(ERROR) << "kernel nums is 0";
    return lite::RET_ERROR;
  }
  auto kernel_size = tensor_shape_size / kernel_nums;
  for (size_t i = 0; i < kernel_nums; i++) {
    for (size_t j = 0; j < kernel_size; j++) {
      newWeightTensor[i * kernel_size + j] = oldWeightTensor[i * kernel_size + j] * trans_scale[i];
    }
  }
  return lite::RET_OK;
}
}  // namespace opt
}  // namespace mindspore
