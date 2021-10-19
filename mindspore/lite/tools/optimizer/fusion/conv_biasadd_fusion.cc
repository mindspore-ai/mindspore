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
#include <functional>
#include <memory>
#include <vector>
#include "tools/anf_exporter/fetch_content.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
namespace {
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

bool FuseBias(const lite::DataInfo &add_bias, const lite::DataInfo &conv_bias, std::vector<float> *fusion_bias,
              int out_channel) {
  MS_ASSERT(fusion_bias != nullptr);
  if ((add_bias.data_type_ != TypeId::kNumberTypeFloat32 && add_bias.data_type_ != TypeId::kNumberTypeFloat) ||
      add_bias.data_.empty()) {
    return false;
  }
  if (out_channel <= 0) {
    return false;
  }
  std::vector<float> add_bias_data(add_bias.data_.size() / sizeof(float));
  if (memcpy_s(add_bias_data.data(), add_bias_data.size() * sizeof(float), add_bias.data_.data(),
               add_bias.data_.size()) != EOK) {
    return false;
  }
  fusion_bias->resize(static_cast<size_t>(out_channel), 0);
  if (!conv_bias.data_.empty()) {
    if (conv_bias.data_type_ != TypeId::kNumberTypeFloat32 && conv_bias.data_type_ != TypeId::kNumberTypeFloat &&
        conv_bias.data_.size() != static_cast<size_t>(out_channel) * sizeof(float)) {
      return false;
    }
    if (memcpy_s(fusion_bias->data(), fusion_bias->size() * sizeof(float), conv_bias.data_.data(),
                 conv_bias.data_.size()) != EOK) {
      return false;
    }
  }
  if (fusion_bias->size() % add_bias_data.size() != 0) {
    return false;
  }
  for (size_t i = 0; i < fusion_bias->size(); ++i) {
    fusion_bias->at(i) += add_bias_data[i % add_bias_data.size()];
  }
  return true;
}
}  // namespace
const BaseRef ConvBiasaddFusion::DefinePattern() const {
  auto is_conv = std::make_shared<CondVar>(IsConvExtendNode);
  MS_CHECK_TRUE_RET(is_conv != nullptr, {});
  auto is_add = std::make_shared<CondVar>(IsAddNode);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto is_const = std::make_shared<CondVar>(IsParamOrValueNodeWithData);
  MS_CHECK_TRUE_RET(is_const != nullptr, {});
  return VectorRef({is_add, is_conv, is_const});
}

CNodePtr ConvBiasaddFusion::GetAddCnode(const AnfNodePtr &node) const {
  MS_ASSERT(node != nullptr);
  if (!utils::isa<CNode>(node)) {
    return nullptr;
  }
  auto add_cnode = node->cast<CNodePtr>();
  if (add_cnode->size() != kInputSizeThree || IsMarkedTrainOp(add_cnode)) {
    return nullptr;
  }
  return add_cnode;
}

bool ConvBiasaddFusion::CheckCanFusion(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  MS_ASSERT(func_graph != nullptr);
  auto add_cnode = GetAddCnode(node);
  MS_CHECK_TRUE_RET(add_cnode != nullptr, false);
  auto prim_add = GetValueNode<PrimitivePtr>(add_cnode->input(0));
  MS_ASSERT(prim_add != nullptr);
  auto add_act_ptr = prim_add->GetAttr(ops::kActivationType);
  auto add_act = add_act_ptr == nullptr ? mindspore::NO_ACTIVATION
                                        : static_cast<mindspore::ActivationType>(GetValue<int64_t>(add_act_ptr));
  auto conv_cnode = add_cnode->input(1)->cast<CNodePtr>();
  if (conv_cnode == nullptr || IsMarkedTrainOp(conv_cnode)) {
    return false;
  }
  if (IsMultiOutputTensors(func_graph, conv_cnode)) {
    return false;
  }
  if (conv_cnode->size() == kInputSizeFour) {
    auto conv_bias = conv_cnode->input(kInputIndexThree);
    if (conv_bias == nullptr || conv_bias->isa<CNode>() || !IsParamNode(conv_bias)) {
      return false;
    }
  }
  auto prim_conv = GetValueNode<PrimitivePtr>(conv_cnode->input(0));
  MS_ASSERT(prim_conv != nullptr);
  auto conv_act_ptr = prim_add->GetAttr(ops::kActivationType);
  auto conv_act = add_act_ptr == nullptr ? mindspore::NO_ACTIVATION
                                         : static_cast<mindspore::ActivationType>(GetValue<int64_t>(conv_act_ptr));
  if (add_act != mindspore::NO_ACTIVATION) {
    if (conv_act != mindspore::NO_ACTIVATION || (add_act != mindspore::RELU && add_act != mindspore::RELU6)) {
      return false;
    }
  }

  if (prim_conv->GetAttr(ops::kOutChannel) == nullptr) {
    return false;
  }
  auto out_channel = GetValue<int64_t>(prim_conv->GetAttr(ops::kOutChannel));
  auto add_weight = add_cnode->input(kInputIndexTwo);
  MS_ASSERT(add_weight != nullptr);
  ShapeVector shape;
  if (FetchShapeFromAbstract(add_weight->abstract(), &shape) != lite::RET_OK) {
    return false;
  }
  if (std::count_if(shape.begin(), shape.end(), [](int64_t dim) { return dim > 1; }) > 1) {
    return false;
  }
  auto element_num = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  return out_channel % element_num == 0;
}

int ConvBiasaddFusion::DoFuison(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  MS_ASSERT(node != nullptr);
  auto add_cnode = node->cast<CNodePtr>();
  MS_ASSERT(add_cnode != nullptr);
  auto add_bias = add_cnode->input(kInputIndexTwo);
  lite::DataInfo add_bias_info;
  int status = lite::RET_ERROR;
  if (add_bias->isa<Parameter>()) {
    status = lite::FetchDataFromParameterNode(add_cnode, kInputIndexTwo, converter::kFmkTypeMs, false, &add_bias_info);
  } else if (add_bias->isa<ValueNode>()) {
    status = lite::FetchDataFromValueNode(add_cnode, kInputIndexTwo, converter::kFmkTypeMs, false, &add_bias_info);
  }
  if (status != lite::RET_OK) {
    MS_LOG(DEBUG) << "conv and add do fusion failed, please check";
    return status;
  }
  auto conv_cnode = add_cnode->input(1)->cast<CNodePtr>();
  MS_ASSERT(conv_cnode != nullptr);
  lite::DataInfo conv_bias_info;
  if (conv_cnode->size() > kInputSizeThree) {
    auto conv_bias = conv_cnode->input(kInputIndexThree);
    if (conv_bias->isa<Parameter>()) {
      status =
        lite::FetchDataFromParameterNode(conv_cnode, kInputIndexThree, converter::kFmkTypeMs, false, &conv_bias_info);
    } else if (conv_bias->isa<ValueNode>()) {
      status =
        lite::FetchDataFromValueNode(conv_cnode, kInputIndexThree, converter::kFmkTypeMs, false, &conv_bias_info);
    }
    if (status != lite::RET_OK) {
      MS_LOG(DEBUG) << "conv and add do fusion failed, please check";
      return status;
    }
  }
  auto prim = GetValueNode<PrimitivePtr>(conv_cnode->input(0));
  MS_ASSERT(prim != nullptr);
  if (prim->GetAttr(ops::kOutChannel) == nullptr) {
    return lite::RET_ERROR;
  }
  int out_channel = GetValue<int64_t>(prim->GetAttr(ops::kOutChannel));
  std::vector<float> fusion_data;
  if (!FuseBias(add_bias_info, conv_bias_info, &fusion_data, out_channel)) {
    MS_LOG(DEBUG) << "conv and add do fusion failed, please check";
    return lite::RET_ERROR;
  }
  auto conv_new_bias =
    AddNewBiasNode(fusion_data.data(), func_graph, out_channel, static_cast<TypeId>(add_bias_info.data_type_));
  MS_CHECK_TRUE_RET(conv_new_bias != nullptr, lite::RET_NULL_PTR);
  conv_new_bias->set_name(conv_cnode->fullname_with_scope() + "_bias");
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  if (conv_cnode->size() > kInputSizeThree) {
    manager->SetEdge(conv_cnode, kInputIndexThree, conv_new_bias);
  } else {
    manager->AddEdge(conv_cnode, conv_new_bias);
  }
  return lite::RET_OK;
}

const AnfNodePtr ConvBiasaddFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr) {
    return nullptr;
  }
  if (!CheckCanFusion(func_graph, node)) {
    return nullptr;
  }
  if (DoFuison(func_graph, node) != lite::RET_OK) {
    return nullptr;
  }
  auto add_cnode = node->cast<CNodePtr>();
  MS_ASSERT(add_cnode != nullptr);
  return add_cnode->input(1);
}
}  // namespace mindspore::opt
