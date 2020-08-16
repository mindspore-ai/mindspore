/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/fusion/conv_biasadd_fusion.h"
#include <memory>
#include "src/param_value_lite.h"
#include "schema/inner/model_generated.h"
#include "src/ir/primitive_t_value.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"

namespace mindspore::opt {
namespace {
constexpr size_t kAddInputsLength = 3;
constexpr size_t kAddWEIGHTINDEX = 2;
constexpr size_t kConvWeightIndex = 2;
constexpr size_t kConvBiasIndex = 3;
constexpr size_t kConvNoBiasLen = 3;
constexpr size_t kConvWithBiasLen = 4;
bool IsConvExtendNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Conv2D || type == schema::PrimitiveType_DepthwiseConv2D
        || type == schema::PrimitiveType_DeConv2D;
  }
  return false;
}
bool IsAddNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Add || type == schema::PrimitiveType_BiasAdd;
  }
  return false;
}

int Get_Kenrnel_nums(const CNodePtr &conv_node) {
  MS_ASSERT(conv_node != nullptr);
  auto value_primitive = conv_node->input(0);
  auto value_node = value_primitive->cast<ValueNodePtr>();
  MS_ASSERT(value_node != nullptr);
  auto value = value_node->value();
  MS_ASSERT(value != nullptr);
  auto primitive = value->cast<PrimitiveTValuePtr>();
  MS_ASSERT(primitive != nullptr);
  auto type = primitive->GetPrimitiveT()->value.type;
  if (type == schema::PrimitiveType_Conv2D) {
    return primitive->GetPrimitiveT()->value.AsConv2D()->channelOut;
  } else if (type == schema::PrimitiveType_DepthwiseConv2D) {
    return primitive->GetPrimitiveT()->value.AsDepthwiseConv2D()->channelMultiplier
        * primitive->GetPrimitiveT()->value.AsDepthwiseConv2D()->channelIn;
  } else if (type == schema::PrimitiveType_DeConv2D) {
    return primitive->GetPrimitiveT()->value.AsDeConv2D()->channelOut;
  } else {
    MS_LOG(ERROR) << "Unsupported opType, " << type;
    return 0;
  }
}
void GenConvNewBias(const FuncGraphPtr &func_graph, const CNodePtr &conv_node, const CNodePtr &bias_node) {
  AnfNodePtr conv_bias_node = nullptr;
  AnfNodePtr conv_weight_node = nullptr;
  if (conv_node->inputs().size() == kConvNoBiasLen) {
    conv_weight_node = conv_node->input(kConvWeightIndex);
  } else if (conv_node->inputs().size() == kConvWithBiasLen) {
    conv_weight_node = conv_node->input(kConvWeightIndex);
    conv_bias_node = conv_node->input(kConvBiasIndex);
  } else {
    MS_LOG(EXCEPTION) << "conv node:" << conv_node->DebugString() << "inputs size must 3 or 4";
  }
  auto kernel_nums = Get_Kenrnel_nums(conv_node);
  if (kernel_nums <= 0) {
    MS_LOG(EXCEPTION) << "kernel num less than 0";
  }
  auto add_bias_data = new(std::nothrow) float[kernel_nums];
  auto bias_add_weight = bias_node->input(kAddWEIGHTINDEX);
  CheckIfNodeIsParam(bias_add_weight);
  auto add_weight_param = bias_add_weight->cast<ParameterPtr>()->default_param();
  auto add_weight_tensor = std::dynamic_pointer_cast<ParamValueLite>(add_weight_param);
  auto add_weight_data = reinterpret_cast<float *>(add_weight_tensor->tensor_addr());
  auto add_weight_shape = add_weight_tensor->tensor_shape();
  if (add_weight_shape.empty() || (add_weight_shape.size() == 1 && add_weight_shape[0] ==1)) {
      for (size_t i = 0; i < kernel_nums; i++) {
        add_bias_data[i] = *add_weight_data;
    }
  } else {
    if (EOK != memcpy_s(add_bias_data, kernel_nums * sizeof(float), add_weight_data, kernel_nums * sizeof(float))) {
      MS_LOG(EXCEPTION) << "memset_s conv_bias_data failed";
    }
  }
  if (conv_bias_node != nullptr) {
    CheckIfNodeIsParam(conv_bias_node);
    auto conv_bias_param = conv_bias_node->cast<ParameterPtr>()->default_param();
    auto conv_bias_tensor = std::dynamic_pointer_cast<ParamValueLite>(conv_bias_param);
    if (conv_bias_tensor->tensor_shape().empty() || conv_bias_tensor->tensor_shape()[0] != kernel_nums) {
      MS_LOG(EXCEPTION) << "conv_bias_node shape error";
    }
    auto conv_bias_data = reinterpret_cast<float *>(conv_bias_tensor->tensor_addr());
    for (size_t i = 0; i < kernel_nums; i++) {
      conv_bias_data[i] += add_bias_data[i];
    }
    delete[] add_bias_data;
  } else {
    auto conv_weight_param = conv_weight_node->cast<ParameterPtr>()->default_param();
    auto conv_weight_tensor = std::dynamic_pointer_cast<ParamValueLite>(conv_weight_param);
    auto conv_new_bias = AddNewBiasNode(add_bias_data, func_graph, kernel_nums, conv_weight_tensor);
    conv_node->add_input(conv_new_bias);
  }
}
}  // namespace
const BaseRef ConvBiasaddFusion::DefinePattern() const {
  auto conv_var = std::make_shared<CondVar>(IsConvExtendNode);
  auto add_var = std::make_shared<CondVar>(IsAddNode);
  auto weight_var = std::make_shared<CondVar>(IsParamNode);
  return VectorRef({add_var, conv_var, weight_var});
}

const AnfNodePtr ConvBiasaddFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const EquivPtr &) const {
  MS_LOG(DEBUG) << "Enter pass process";
  CheckIfFuncGraphIsNull(func_graph);

  CheckIfAnfNodeIsNull(node);
  auto add_node = node->cast<CNodePtr>();
  CheckIfCNodeIsNull(add_node);
  CheckInputSize(add_node, kAddInputsLength);

  AnfNodePtr conv_node_anf = add_node->input(1);
  CheckIfAnfNodeIsNull(conv_node_anf);
  if (IsMultiOutputTensors(func_graph, conv_node_anf)) {
    return add_node;
  }
  auto conv_node = conv_node_anf->cast<CNodePtr>();
  CheckIfCNodeIsNull(conv_node);
  GenConvNewBias(func_graph, conv_node, add_node);
  auto primitiveT_value = GetValueNode<std::shared_ptr<lite::PrimitiveTValue>>(conv_node->input(0));
  MS_ASSERT(primitiveT_value != nullptr);
  auto type = primitiveT_value->GetPrimitiveT()->value.type;
  if (type == schema::PrimitiveType_Conv2D) {
    primitiveT_value->GetPrimitiveT()->value.AsConv2D()->hasBias = true;
  } else if (type == schema::PrimitiveType_DepthwiseConv2D) {
    primitiveT_value->GetPrimitiveT()->value.AsDepthwiseConv2D()->hasBias = true;
  } else if (type == schema::PrimitiveType_DeConv2D) {
    primitiveT_value->GetPrimitiveT()->value.AsDeConv2D()->hasBias = true;
  } else {
    MS_LOG(EXCEPTION) << "Unsupported opType, " << type;
  }
  return conv_node;
}
}  // namespace mindspore::opt

