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
#include "tools/optimizer/fusion/layer_norm_fusion.h"
#include <memory>
#include "src/ops/primitive_c.h"
#include "src/param_value_lite.h"
#include "schema/inner/model_generated.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"
#include "src/ops/add.h"
#include "src/ops/mul.h"
#include "src/ops/rsqrt.h"
#include "src/ops/reduce.h"
#include "src/ops/sub.h"

namespace mindspore {
namespace opt {
namespace {

bool IsAddNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Add;
  }
  return false;
}

bool IsSquaredDifferenceNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_SquaredDifference;
  }
  return false;
}

bool IsRsqrtNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Rsqrt;
  }
  return false;
}

bool IsMulNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Mul;
  }
  return false;
}

bool IsSubNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Sub;
  }
  return false;
}
}  // namespace

const BaseRef LayerNormFusion::DefinePattern() const {
  VectorRef mean1_ref = VectorRef({mean1_, input_});
  auto squared_diffference1 = std::make_shared<CondVar>(IsSquaredDifferenceNode);
  VectorRef squared_diffference1_ref = VectorRef({squared_diffference1, input_, mean1_ref});
  auto mul1 = std::make_shared<CondVar>(IsMulNode);
  VectorRef mean2_ref = VectorRef({mean2_, squared_diffference1_ref});
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
  auto layer_norm_primitive = std::make_unique<schema::PrimitiveT>();
  std::unique_ptr<schema::LayerNormT> attr = std::make_unique<schema::LayerNormT>();
  attr->epsilon = epsilon;
  attr->begin_norm_axis = begin_norm_axis;
  attr->begin_params_axis = begin_params_axis;
  layer_norm_primitive->value.type = schema::PrimitiveType_LayerNorm;
  layer_norm_primitive->value.value = attr.release();
  auto layer_norm_cvalue = lite::PrimitiveC::Create(layer_norm_primitive.release());
  auto value_node = NewValueNode(std::shared_ptr<lite::PrimitiveC>(layer_norm_cvalue));
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
  auto mean2_value = utils::cast<AnfNodePtr>((*equiv)[mean2_]);
  MS_ASSERT(mean2_value != nullptr);
  auto mean2_primitivec = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(mean2_value);
  if (!utils::isa<std::shared_ptr<mindspore::lite::Reduce>>(mean2_primitivec)) {
    return false;
  }
  auto mean2_op = utils::cast<std::shared_ptr<mindspore::lite::Reduce>>(mean2_primitivec);
  MS_ASSERT(mean2_op != nullptr);
  if (mean2_op->GetMode() != schema::ReduceMode_ReduceMean) {
    return false;
  }
  auto mean2_axes = mean2_op->GetAxes();
  // mean1
  auto mean1_value = utils::cast<AnfNodePtr>((*equiv)[mean1_]);
  MS_ASSERT(mean1_value != nullptr);
  auto mean1_primitivec = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(mean1_value);
  if (!utils::isa<std::shared_ptr<mindspore::lite::Reduce>>(mean1_primitivec)) {
    return false;
  }
  auto mean1_op = utils::cast<std::shared_ptr<mindspore::lite::Reduce>>(mean1_primitivec);
  MS_ASSERT(mean1_op != nullptr);
  if (mean1_op->GetMode() != schema::ReduceMode_ReduceMean) {
    return false;
  }
  auto mean1_axes = mean1_op->GetAxes();
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
