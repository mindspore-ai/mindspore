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
#include "tools/optimizer/fusion/layer_norm_fusion.h"
#include <memory>
#include "ops/fusion/layer_norm_fusion.h"
#include "ops/fusion/reduce_fusion.h"
#include "ops/rsqrt.h"
#include "src/param_value_lite.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"

namespace mindspore {
namespace opt {
namespace {
bool IsAddNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimAddFusion);
  }
  return false;
}

bool IsSquaredDifferenceNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimSquaredDifference);
  }
  return false;
}

bool IsRsqrtNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimRsqrt);
  }
  return false;
}

bool IsMulNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimMulFusion);
  }
  return false;
}

bool IsSubNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    return CheckPrimitiveType(utils::cast<AnfNodePtr>(n), prim::kPrimSubFusion);
  }
  return false;
}

lite::STATUS GetReduceAxes(const BaseRef &n, std::vector<int> *axes) {
  MS_ASSERT(node != nullptr);
  if (utils::isa<ParameterPtr>(n)) {
    auto axes_param = utils::cast<ParameterPtr>(n);
    if (!axes_param->has_default() || axes_param->default_param() == nullptr) {
      return lite::RET_NOT_SUPPORT;
    }
    auto axes_value = axes_param->default_param()->cast<ParamValueLitePtr>();
    if (axes_value == nullptr) {
      return lite::RET_ERROR;
    }
    axes->resize(axes_value->tensor_shape()[0]);
    if (memcpy_s(axes->data(), axes_value->tensor_size(), axes_value->tensor_addr(), axes_value->tensor_size()) ==
        EOK) {
      return lite::RET_OK;
    }
  }
  if (utils::isa<ValueNodePtr>(n)) {
    auto axes_value_node = utils::cast<ValueNodePtr>(n);
    auto axes_content = CastToInt(axes_value_node->value());
    if (memcpy_s(axes->data(), axes_content.size() * sizeof(int), axes_content.data(),
                 axes_content.size() * sizeof(int)) == EOK) {
      return lite::RET_OK;
    }
  }
  return lite::RET_ERROR;
}

bool IsReduceNode(const EquivPtr &equiv, const VarPtr &input_prim, const VarPtr &input_axes, std::vector<int> *axes) {
  MS_ASSERT(equiv != nullptr && input_prim != nullptr);
  MS_ASSERT(input_axes != nullptr && axes != nullptr);
  auto reduce_value = utils::cast<AnfNodePtr>((*equiv)[input_prim]);
  MS_ASSERT(reduce_value != nullptr);
  auto mean2_primitive = GetValueNode<std::shared_ptr<ops::ReduceFusion>>(reduce_value);
  if (mean2_primitive == nullptr || mean2_primitive->GetAttr(ops::kMode) == nullptr ||
      mean2_primitive->get_mode() != mindspore::Reduce_Mean) {
    return false;
  }
  if (GetReduceAxes((*equiv)[input_axes], axes) != lite::RET_OK) {
    return false;
  }
  return true;
}
}  // namespace

const BaseRef LayerNormFusion::DefinePattern() const {
  VectorRef mean1_ref = VectorRef({mean1_, input_, mean1_axes_});
  auto squared_diffference1 = std::make_shared<CondVar>(IsSquaredDifferenceNode);
  VectorRef squared_diffference1_ref = VectorRef({squared_diffference1, input_, mean1_ref});
  auto mul1 = std::make_shared<CondVar>(IsMulNode);
  VectorRef mean2_ref = VectorRef({mean2_, squared_diffference1_ref, mean2_axes_});
  auto add1 = std::make_shared<CondVar>(IsAddNode);
  VectorRef add1_ref = VectorRef({add1, mean2_ref, epsilon_});
  auto rsqrt1 = std::make_shared<CondVar>(IsRsqrtNode);
  VectorRef rsqrt1_ref = VectorRef({rsqrt1, add1_ref});
  auto mul2 = std::make_shared<CondVar>(IsMulNode);
  VectorRef mul2_ref = VectorRef({mul2, rsqrt1_ref, gamma_});
  VectorRef mul1_ref = VectorRef({mul1, input_, mul2_ref});
  auto mul3 = std::make_shared<CondVar>(IsMulNode);
  VectorRef mul3_ref = VectorRef({mul3, mean1_ref, mul2_ref});
  auto sub1 = std::make_shared<CondVar>(IsSubNode);
  VectorRef sub1_ref = VectorRef({sub1, beta_, mul3_ref});
  auto add2 = std::make_shared<CondVar>(IsAddNode);
  VectorRef add2_ref = VectorRef({add2, mul1_ref, sub1_ref});
  return add2_ref;
}

CNodePtr LayerNormFusion::CreateLayerNormNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv, float epsilon,
                                              int begin_norm_axis, int begin_params_axis) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(equiv != nullptr);
  auto layer_norm_primitive = std::make_shared<ops::LayerNormFusion>();
  layer_norm_primitive->Init(begin_norm_axis, begin_params_axis, epsilon);
  auto value_node = NewValueNode(layer_norm_primitive);
  std::vector<AnfNodePtr> new_node_inputs = {value_node};
  auto input_node = utils::cast<AnfNodePtr>((*equiv)[input_]);
  MS_ASSERT(input_node != nullptr);
  new_node_inputs.push_back(input_node);
  auto gamma_node = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  MS_ASSERT(gamma_node != nullptr);
  new_node_inputs.push_back(gamma_node);
  auto beta_node = utils::cast<AnfNodePtr>((*equiv)[beta_]);
  MS_ASSERT(beta_node != nullptr);
  new_node_inputs.push_back(beta_node);
  auto new_node = func_graph->NewCNode(new_node_inputs);
  return new_node;
}

bool LayerNormFusion::GetAxis(const CNodePtr &input_cnode, const std::vector<int> &mean_axes,
                              const std::vector<int> &params_shape, int *begin_norm_axis,
                              int *begin_params_axis) const {
  MS_ASSERT(input_node != nullptr);
  MS_ASSERT(begin_norm_axis != nullptr);
  MS_ASSERT(begin_params_axis != nullptr);
  auto abstract = input_cnode->abstract();
  if (abstract == nullptr) {
    MS_LOG(DEBUG) << "abstract of input is nullptr";
    return false;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract)) {
    MS_LOG(DEBUG) << "Abstract should be abstract tensor";
    return false;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
  if (!utils::isa<abstract::ShapePtr>(abstract_tensor->BuildShape())) {
    MS_LOG(DEBUG) << "Shape of Abstract should be ShapePtr";
    return false;
  }
  auto shape = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  if (mean_axes.back() + 1 != static_cast<int>(shape.size())) {
    MS_LOG(DEBUG) << "mean node is not reduce to last axis";
    return false;
  }
  for (size_t i = 1; i < mean_axes.size(); ++i) {
    if (mean_axes[i] != mean_axes[i - 1] + 1) {
      MS_LOG(DEBUG) << "mean axes is not continuous";
      return false;
    }
  }
  // there is no need to check params_shape
  *begin_norm_axis = mean_axes.front();
  *begin_params_axis = static_cast<int>(shape.size()) - static_cast<int>(params_shape.size());
  if (*begin_params_axis < 0) {
    MS_LOG(DEBUG) << "LayerNorm begin_params_axis illegal, not fuse";
    return false;
  }
  return true;
}

bool LayerNormFusion::CheckPattern(const EquivPtr &equiv, float *epsilon, int *begin_norm_axis,
                                   int *begin_params_axis) const {
  MS_ASSERT(equiv != nullptr);
  MS_ASSERT(epsilon != nullptr);
  MS_ASSERT(begin_norm_axis != nullptr);
  MS_ASSERT(begin_params_axis != nullptr);
  // beta
  auto beta_node = utils::cast<AnfNodePtr>((*equiv)[beta_]);
  MS_ASSERT(beta_node != nullptr);
  if (CheckIfNodeIsParam(beta_node) != lite::RET_OK) {
    return false;
  }
  auto beta_param = beta_node->cast<ParameterPtr>()->default_param();
  auto beta_tensor = std::dynamic_pointer_cast<ParamValueLite>(beta_param);
  auto beta_shape = beta_tensor->tensor_shape();
  // gamma
  auto gamma_node = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  MS_ASSERT(gamma_node != nullptr);
  if (CheckIfNodeIsParam(gamma_node) != lite::RET_OK) {
    return false;
  }
  auto gamma_param = gamma_node->cast<ParameterPtr>()->default_param();
  auto gamma_tensor = std::dynamic_pointer_cast<ParamValueLite>(gamma_param);
  auto gamma_shape = gamma_tensor->tensor_shape();
  // epsilon
  auto epsilon_node = utils::cast<AnfNodePtr>((*equiv)[epsilon_]);
  MS_ASSERT(epsilon_node != nullptr);
  if (CheckIfNodeIsParam(epsilon_node) != lite::RET_OK) {
    return false;
  }
  auto epsilon_param = epsilon_node->cast<ParameterPtr>()->default_param();
  auto epsilon_tensor = std::dynamic_pointer_cast<ParamValueLite>(epsilon_param);
  auto epsilon_data = reinterpret_cast<float *>(epsilon_tensor->tensor_addr());
  auto epsilon_shape = epsilon_tensor->tensor_shape();
  // mean2
  std::vector<int> mean2_axes;
  if (!IsReduceNode(equiv, mean2_, mean2_axes_, &mean2_axes)) {
    return false;
  }
  // mean1
  std::vector<int> mean1_axes;
  if (!IsReduceNode(equiv, mean1_, mean1_axes_, &mean1_axes)) {
    return false;
  }
  auto input_node = utils::cast<AnfNodePtr>((*equiv)[input_]);
  MS_ASSERT(input_node != nullptr);
  if (!utils::isa<CNodePtr>(input_node)) {
    return false;
  }
  auto input_cnode = input_node->cast<CNodePtr>();
  if (mean1_axes != mean2_axes) {
    return false;
  }
  if (mean1_axes.size() != gamma_shape.size() || mean1_axes.size() != beta_shape.size()) {
    return false;
  }
  if (gamma_shape != beta_shape) {
    return false;
  }
  if (epsilon_shape.empty() || (epsilon_shape.size() == 1 && epsilon_shape[0] == 1)) {
    *epsilon = epsilon_data[0];
  } else {
    return false;
  }
  if (!GetAxis(input_cnode, mean1_axes, gamma_shape, begin_norm_axis, begin_params_axis)) {
    return false;
  }
  return true;
}

const AnfNodePtr LayerNormFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  MS_ASSERT(equiv != nullptr);
  MS_LOG(DEBUG) << "layer_norm_fusion pass";
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  auto add2_cnode = node->cast<CNodePtr>();
  float epsilon = 0.0f;
  int begin_norm_axis = 0;
  int begin_params_axis = 0;
  if (!CheckPattern(equiv, &epsilon, &begin_norm_axis, &begin_params_axis)) {
    return nullptr;
  }
  auto layer_norm_cnode = CreateLayerNormNode(func_graph, equiv, epsilon, begin_norm_axis, begin_params_axis);
  layer_norm_cnode->set_abstract(add2_cnode->abstract()->Clone());
  layer_norm_cnode->set_fullname_with_scope("layer_norm_" + add2_cnode->fullname_with_scope());
  MS_LOG(INFO) << "layernorm node:" << layer_norm_cnode->fullname_with_scope() << " fusion success";
  return layer_norm_cnode;
}
}  // namespace opt
}  // namespace mindspore
