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
#include "tools/optimizer/fusion/conv_biasadd_fusion.h"
#include <memory>
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/conv2d_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "tools/common/tensor_util.h"
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
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim::kPrimConv2DFusion) ||
           CheckPrimitiveType(anf_node, prim::kPrimConv2dTransposeFusion);
  }
  return false;
}

bool IsAddNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim::kPrimAddFusion) || CheckPrimitiveType(anf_node, prim::kPrimBiasAdd);
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
  auto primitive = value->cast<PrimitiveCPtr>();
  MS_ASSERT(primitive != nullptr);
  if (primitive->isa<mindspore::ops::Conv2DFusion>()) {
    MS_ASSERT(utils::isa<std::shared_ptr<mindspore::ops::Conv2DFusion>>(primitive));
    auto primc = utils::cast<std::shared_ptr<mindspore::ops::Conv2DFusion>>(primitive);
    MS_ASSERT(primc != nullptr);
    return primc->get_out_channel();
  } else if (primitive->isa<mindspore::ops::Conv2dTransposeFusion>()) {
    MS_ASSERT(utils::isa<std::shared_ptr<mindspore::ops::Conv2dTransposeFusion>>(primitive));
    auto primc = utils::cast<std::shared_ptr<mindspore::ops::Conv2dTransposeFusion>>(primitive);
    MS_ASSERT(primc != nullptr);
    return primc->get_out_channel();
  } else {
    MS_LOG(ERROR) << "Unsupported opType, " << primitive->name();
    return 0;
  }
}

int GetAddBiasData(const AnfNodePtr &bias_add_weight_node, const int &kernel_nums, float **add_bias_data) {
  MS_ASSERT(bias_add_weight_node != nullptr);
  MS_ASSERT(add_bias_data != nullptr);
  MS_ASSERT(*add_bias_data != nullptr);
  float *add_weight_data = nullptr;
  ShapeVector add_weight_shape;
  if (utils::isa<Parameter>(bias_add_weight_node)) {
    auto add_weight_param_node = bias_add_weight_node->cast<ParameterPtr>();
    if (!add_weight_param_node->has_default() || add_weight_param_node->default_param() == nullptr) {
      MS_LOG(ERROR) << "The bias parameter of " << bias_add_weight_node->fullname_with_scope() << " is nullptr.";
      return lite::RET_ERROR;
    }
    auto add_weight_tensor = std::dynamic_pointer_cast<tensor::Tensor>(add_weight_param_node->default_param());
    if (add_weight_tensor == nullptr) {
      MS_LOG(ERROR) << "The bias data of parameter node " << bias_add_weight_node->fullname_with_scope()
                    << " is not tensorPtr.";
      return lite::RET_ERROR;
    }
    add_weight_data = reinterpret_cast<float *>(add_weight_tensor->data_c());
    MS_ASSERT(add_weight_data != nullptr);
    add_weight_shape = add_weight_tensor->shape();
  } else {
    MS_ASSERT(utils::isa<ValueNode>(bias_add_weight_node));
    auto add_weight_value_node = bias_add_weight_node->cast<ValueNodePtr>();
    auto add_weight_value = add_weight_value_node->value();
    MS_ASSERT(add_weight_value != nullptr);
    auto add_weight_tensor = add_weight_value->cast<tensor::TensorPtr>();
    if (add_weight_tensor == nullptr) {
      MS_LOG(ERROR) << "The bias data of value node " << bias_add_weight_node->fullname_with_scope()
                    << " is not tensorPtr.";
      return lite::RET_ERROR;
    }
    add_weight_data = reinterpret_cast<float *>(add_weight_tensor->data_c());
    MS_ASSERT(add_weight_data != nullptr);
    auto value_abstract = add_weight_value_node->abstract();
    auto value_abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(value_abstract);
    add_weight_shape = utils::cast<abstract::ShapePtr>(value_abstract_tensor->BuildShape())->shape();
  }
  if (add_weight_shape.empty() || (add_weight_shape.size() == 1 && add_weight_shape[0] == 1)) {
    for (int i = 0; i < kernel_nums; i++) {
      (*add_bias_data)[i] = *add_weight_data;
    }
  } else {
    if (EOK != memcpy_s(*add_bias_data, kernel_nums * sizeof(float), add_weight_data, kernel_nums * sizeof(float))) {
      MS_LOG(ERROR) << "memcpy_s conv_bias_data failed";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

int GetNewConvBiasData(const AnfNodePtr &conv_bias_node, const int &kernel_nums, const float *add_bias_data) {
  MS_ASSERT(add_bias_data != nullptr);
  MS_ASSERT(conv_bias_node != nullptr);
  if (utils::isa<Parameter>(conv_bias_node)) {
    auto conv_bias_param_node = conv_bias_node->cast<ParameterPtr>();
    if (!conv_bias_param_node->has_default() || conv_bias_param_node->default_param() == nullptr) {
      MS_LOG(ERROR) << "The bias parameter of " << conv_bias_node->fullname_with_scope() << " is nullptr.";
      return lite::RET_ERROR;
    }
    auto conv_bias_tensor = std::dynamic_pointer_cast<tensor::Tensor>(conv_bias_param_node->default_param());
    if (conv_bias_tensor == nullptr || conv_bias_tensor->shape().empty() ||
        conv_bias_tensor->shape()[0] != kernel_nums) {
      MS_LOG(ERROR) << "conv_bias_node shape error";
      return lite::RET_ERROR;
    }
    auto conv_bias_data = reinterpret_cast<float *>(conv_bias_tensor->data_c());
    MS_ASSERT(conv_bias_data != nullptr);
    for (int i = 0; i < kernel_nums; i++) {
      conv_bias_data[i] += add_bias_data[i];
    }
  } else {
    MS_ASSERT(utils::isa<ValueNode>(conv_bias_node));
    auto conv_bias_value_node = conv_bias_node->cast<ValueNodePtr>();
    auto conv_bias_value = conv_bias_value_node->value();
    MS_ASSERT(conv_bias_value != nullptr);
    auto conv_bias_tensor = conv_bias_value->cast<tensor::TensorPtr>();
    if (conv_bias_tensor == nullptr) {
      MS_LOG(ERROR) << "The bias data of value node " << conv_bias_node->fullname_with_scope() << "is not tensorPtr.";
      return lite::RET_ERROR;
    }
    auto conv_bias_data = reinterpret_cast<float *>(conv_bias_tensor->data_c());
    MS_ASSERT(conv_bias_data != nullptr);
    for (int i = 0; i < kernel_nums; i++) {
      conv_bias_data[i] += add_bias_data[i];
    }
  }
  return lite::RET_OK;
}

tensor::TensorPtr GetConvWeightTensor(const AnfNodePtr &conv_weight_node) {
  tensor::TensorPtr conv_weight_tensor;
  if (utils::isa<ValueNode>(conv_weight_node)) {
    auto conv_weight_value_node = conv_weight_node->cast<ValueNodePtr>();
    auto conv_weight_value = conv_weight_value_node->value();
    MS_ASSERT(conv_weight_value != nullptr);
    conv_weight_tensor = conv_weight_value->cast<tensor::TensorPtr>();
    MS_ASSERT(conv_weight_tensor != nullptr);
  } else {
    MS_ASSERT(utils::isa<Parameter>(conv_weight_node));
    auto conv_weight_param = conv_weight_node->cast<ParameterPtr>()->default_param();
    MS_ASSERT(conv_weight_param != nullptr);
    conv_weight_tensor = std::dynamic_pointer_cast<tensor::Tensor>(conv_weight_param);
    MS_ASSERT(conv_weight_tensor != nullptr);
  }
  return conv_weight_tensor;
}

int GenConvNewBias(const FuncGraphPtr &func_graph, const CNodePtr &conv_node, const CNodePtr &bias_node) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(conv_node != nullptr);
  MS_ASSERT(bias_node != nullptr);
  AnfNodePtr conv_bias_node = nullptr;
  AnfNodePtr conv_weight_node = nullptr;
  if (conv_node->inputs().size() == kConvNoBiasLen) {
    conv_weight_node = conv_node->input(kConvWeightIndex);
  } else if (conv_node->inputs().size() == kConvWithBiasLen) {
    conv_weight_node = conv_node->input(kConvWeightIndex);
    conv_bias_node = conv_node->input(kConvBiasIndex);
  } else {
    MS_LOG(ERROR) << "conv node:" << conv_node->DebugString() << "inputs size must 3 or 4";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  auto kernel_nums = Get_Kenrnel_nums(conv_node);
  if (kernel_nums <= 0) {
    MS_LOG(ERROR) << "kernel num less than 0";
    return lite::RET_INVALID_OP_ATTR;
  }
  auto add_bias_data = new (std::nothrow) float[kernel_nums];
  if (add_bias_data == nullptr) {
    MS_LOG(ERROR) << "tensor_data is nullptr";
    return lite::RET_MEMORY_FAILED;
  }
  auto bias_add_weight = bias_node->input(kAddWEIGHTINDEX);
  if (CheckIfNodeIsParamOrValue(bias_add_weight) != lite::RET_OK) {
    delete[] add_bias_data;
    return lite::RET_INVALID_OP_ATTR;
  }
  if (GetAddBiasData(bias_add_weight, kernel_nums, &add_bias_data) != lite::RET_OK) {
    delete[] add_bias_data;
    return lite::RET_INVALID_OP_ATTR;
  }
  if (conv_bias_node != nullptr) {
    if (CheckIfNodeIsParamOrValue(conv_bias_node) != lite::RET_OK) {
      delete[] add_bias_data;
      return lite::RET_INVALID_OP_ATTR;
    }
    if (GetNewConvBiasData(conv_bias_node, kernel_nums, add_bias_data) != lite::RET_OK) {
      delete[] add_bias_data;
      return lite::RET_INVALID_OP_ATTR;
    }
    delete[] add_bias_data;
  } else {
    if (CheckIfNodeIsParamOrValue(conv_weight_node) != lite::RET_OK) {
      delete[] add_bias_data;
      return lite::RET_INVALID_OP_ATTR;
    }
    tensor::TensorPtr conv_weight_tensor = GetConvWeightTensor(conv_weight_node);
    auto conv_new_bias = AddNewBiasNode(add_bias_data, func_graph, kernel_nums, conv_weight_tensor);
    conv_new_bias->set_name(conv_node->fullname_with_scope() + "_bias");
    conv_node->add_input(conv_new_bias);
  }
  return lite::RET_OK;
}
}  // namespace
const BaseRef ConvBiasaddFusion::DefinePattern() const {
  auto conv_var = std::make_shared<CondVar>(IsConvExtendNode);
  auto add_var = std::make_shared<CondVar>(IsAddNode);
  auto weight_var = std::make_shared<CondVar>(IsParamOrValueNodeWithData);
  return VectorRef({add_var, conv_var, weight_var});
}

const AnfNodePtr ConvBiasaddFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const EquivPtr &) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  MS_LOG(DEBUG) << "Enter pass process";
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
    return nullptr;
  }
  auto add_node = node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(add_node) != lite::RET_OK || CheckInputSize(add_node, kAddInputsLength) != lite::RET_OK) {
    return nullptr;
  }
  if (CheckPrimitiveType(add_node, prim::kPrimAddFusion)) {
    auto primitive_c = GetValueNode<PrimitiveCPtr>(add_node->input(0));
    MS_ASSERT(utils::isa<std::shared_ptr<mindspore::ops::AddFusion>>(primitive_c));
    auto primc = utils::cast<std::shared_ptr<mindspore::ops::AddFusion>>(primitive_c);
    MS_ASSERT(primc != nullptr);
    if (primc->GetAttr(ops::kActivationType) != nullptr && primc->get_activation_type() != mindspore::NO_ACTIVATION) {
      return add_node;
    }
  }

  AnfNodePtr conv_node_anf = add_node->input(1);
  if (CheckIfAnfNodeIsNull(conv_node_anf) != lite::RET_OK || IsMultiOutputTensors(func_graph, conv_node_anf)) {
    return nullptr;
  }
  auto conv_node = conv_node_anf->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(conv_node) != lite::RET_OK) {
    return nullptr;
  }
  int ret = GenConvNewBias(func_graph, conv_node, add_node);
  if (ret != lite::RET_OK) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return nullptr;
  }
  return conv_node;
}
}  // namespace mindspore::opt
