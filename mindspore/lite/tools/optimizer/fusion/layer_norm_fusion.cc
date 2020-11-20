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
constexpr size_t kAddInputsLength = 3;
constexpr size_t kSubInputsLength = 3;
constexpr size_t kMulInputsLength = 3;
constexpr size_t kRsqrtInputsLength = 2;
constexpr size_t kReduceInputsLength = 2;

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

bool IsReduceNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Reduce;
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
  auto mean1 = std::make_shared<CondVar>(IsReduceNode);
  VectorRef mean1_ref = VectorRef({mean1, input_});
  auto squared_diffference1 = std::make_shared<CondVar>(IsSquaredDifferenceNode);
  VectorRef squared_diffference1_ref = VectorRef({squared_diffference1, input_, mean1_ref});
  auto mul1 = std::make_shared<CondVar>(IsMulNode);
  auto mean2 = std::make_shared<CondVar>(IsReduceNode);
  VectorRef mean2_ref = VectorRef({mean2, squared_diffference1_ref});
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

CNodePtr LayerNormFusion::CreateLayerNormNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                              const std::vector<int> &shape, const float epsilon) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto layer_norm_primitive = std::make_unique<schema::PrimitiveT>();
  std::unique_ptr<schema::LayerNormT> attr = std::make_unique<schema::LayerNormT>();
  attr->normalizedShape = shape;
  attr->epsilon = epsilon;
  attr->elementwiseAffine = true;
  layer_norm_primitive->value.type = schema::PrimitiveType_LayerNorm;
  layer_norm_primitive->value.value = attr.release();
  auto layer_norm_cvalue = lite::PrimitiveC::Create(layer_norm_primitive.release());
  auto value_node = NewValueNode(std::shared_ptr<lite::PrimitiveC>(layer_norm_cvalue));
  std::vector<AnfNodePtr> new_node_inputs = {value_node};
  auto input_node = utils::cast<AnfNodePtr>((*equiv)[input_]);
  MS_EXCEPTION_IF_NULL(input_node);
  new_node_inputs.push_back(input_node);
  auto gamma_node = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  MS_EXCEPTION_IF_NULL(gamma_node);
  new_node_inputs.push_back(gamma_node);
  auto beta_node = utils::cast<AnfNodePtr>((*equiv)[beta_]);
  MS_EXCEPTION_IF_NULL(beta_node);
  new_node_inputs.push_back(beta_node);
  auto new_node = func_graph->NewCNode(new_node_inputs);
  return new_node;
}

const AnfNodePtr LayerNormFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  MS_LOG(DEBUG) << "layer_norm pass";
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }

  // add2
  auto add2_cnode = node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(add2_cnode) != lite::RET_OK || CheckInputSize(add2_cnode, kAddInputsLength) != lite::RET_OK) {
    return nullptr;
  }
  auto add2_primitivec = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(add2_cnode->input(0));
  MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::Add>>(add2_primitivec));
  auto add2_op = utils::cast<std::shared_ptr<mindspore::lite::Add>>(add2_primitivec);
  MS_ASSERT(add2_op != nullptr);
  AnfNodePtr sub1_node = add2_cnode->input(2);
  if (CheckIfAnfNodeIsNull(sub1_node) != lite::RET_OK) {
    return nullptr;
  }

  // sub1
  auto sub1_cnode = sub1_node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(sub1_cnode) != lite::RET_OK || CheckInputSize(sub1_cnode, kSubInputsLength) != lite::RET_OK) {
    return nullptr;
  }
  auto sub1_primitivec = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(sub1_cnode->input(0));
  MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::Sub>>(sub1_primitivec));
  auto sub1_op = utils::cast<std::shared_ptr<mindspore::lite::Sub>>(sub1_primitivec);
  MS_ASSERT(sub1_op != nullptr);
  AnfNodePtr beta_node = sub1_cnode->input(1);
  AnfNodePtr mul3_node = sub1_cnode->input(2);
  if (CheckIfAnfNodeIsNull(beta_node) != lite::RET_OK || CheckIfAnfNodeIsNull(mul3_node) != lite::RET_OK) {
    return nullptr;
  }

  // beta
  if (CheckIfNodeIsParam(beta_node) != lite::RET_OK) {
    return nullptr;
  }
  auto beta_param = beta_node->cast<ParameterPtr>()->default_param();
  auto beta_tensor = std::dynamic_pointer_cast<ParamValueLite>(beta_param);
  auto beta_shape = beta_tensor->tensor_shape();

  // mul3
  auto mul3_cnode = mul3_node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(mul3_cnode) != lite::RET_OK || CheckInputSize(mul3_cnode, kMulInputsLength) != lite::RET_OK) {
    return nullptr;
  }
  auto mul3_primitivec = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(mul3_cnode->input(0));
  MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::Mul>>(mul3_primitivec));
  auto mul3_op = utils::cast<std::shared_ptr<mindspore::lite::Mul>>(mul3_primitivec);
  MS_ASSERT(mul3_op != nullptr);
  AnfNodePtr mean1_node = mul3_cnode->input(1);
  AnfNodePtr mul2_node = mul3_cnode->input(2);
  if (CheckIfAnfNodeIsNull(mean1_node) != lite::RET_OK || CheckIfAnfNodeIsNull(mul2_node) != lite::RET_OK) {
    return nullptr;
  }

  // mul2
  auto mul2_cnode = mul2_node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(mul2_cnode) != lite::RET_OK || CheckInputSize(mul2_cnode, kMulInputsLength) != lite::RET_OK) {
    return nullptr;
  }
  auto mul2_primitivec = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(mul2_cnode->input(0));
  MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::Mul>>(mul2_primitivec));
  auto mul2_op = utils::cast<std::shared_ptr<mindspore::lite::Mul>>(mul2_primitivec);
  MS_ASSERT(mul2_op != nullptr);
  AnfNodePtr rsqrt_node = mul2_cnode->input(1);
  AnfNodePtr gamma_node = mul2_cnode->input(2);
  if (CheckIfAnfNodeIsNull(rsqrt_node) != lite::RET_OK || CheckIfAnfNodeIsNull(gamma_node) != lite::RET_OK) {
    return nullptr;
  }

  // gamma
  if (CheckIfNodeIsParam(gamma_node) != lite::RET_OK) {
    return nullptr;
  }
  auto gamma_param = gamma_node->cast<ParameterPtr>()->default_param();
  auto gamma_tensor = std::dynamic_pointer_cast<ParamValueLite>(gamma_param);
  auto gamma_shape = gamma_tensor->tensor_shape();

  // rsqrt
  auto rsqrt_cnode = rsqrt_node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(rsqrt_cnode) != lite::RET_OK ||
      CheckInputSize(rsqrt_cnode, kRsqrtInputsLength) != lite::RET_OK) {
    return nullptr;
  }
  auto rsqrt_primitivec = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(rsqrt_cnode->input(0));
  MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::Rsqrt>>(rsqrt_primitivec));
  auto rsqrt_op = utils::cast<std::shared_ptr<mindspore::lite::Rsqrt>>(rsqrt_primitivec);
  MS_ASSERT(rsqrt_op != nullptr);
  AnfNodePtr add1_node = rsqrt_cnode->input(1);
  if (CheckIfAnfNodeIsNull(add1_node) != lite::RET_OK) {
    return nullptr;
  }

  // add1
  auto add1_cnode = add1_node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(add1_cnode) != lite::RET_OK || CheckInputSize(add1_cnode, kAddInputsLength) != lite::RET_OK) {
    return nullptr;
  }
  auto add1_primitivec = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(add1_cnode->input(0));
  MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::Add>>(add1_primitivec));
  auto add1_op = utils::cast<std::shared_ptr<mindspore::lite::Add>>(add1_primitivec);
  MS_ASSERT(add1_op != nullptr);
  AnfNodePtr mean2_node = add1_cnode->input(1);
  AnfNodePtr epsilon_node = add1_cnode->input(2);
  if (CheckIfAnfNodeIsNull(mean2_node) != lite::RET_OK || CheckIfAnfNodeIsNull(epsilon_node) != lite::RET_OK) {
    return nullptr;
  }

  // epsilon
  if (CheckIfNodeIsParam(epsilon_node) != lite::RET_OK) {
    // delete[] add_bias_data;
    return nullptr;
  }
  auto epsilon_param = epsilon_node->cast<ParameterPtr>()->default_param();
  auto epsilon_tensor = std::dynamic_pointer_cast<ParamValueLite>(epsilon_param);
  auto epsilon_data = reinterpret_cast<float *>(epsilon_tensor->tensor_addr());
  auto epsilon_shape = epsilon_tensor->tensor_shape();

  // mean2
  auto mean2_cnode = mean2_node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(mean2_cnode) != lite::RET_OK ||
      CheckInputSize(mean2_cnode, kReduceInputsLength) != lite::RET_OK) {
    return nullptr;
  }
  auto mean2_primitivec = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(mean2_cnode->input(0));
  MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::Reduce>>(mean2_primitivec));
  auto mean2_op = utils::cast<std::shared_ptr<mindspore::lite::Reduce>>(mean2_primitivec);
  MS_ASSERT(mean2_op != nullptr);
  if (mean2_op->GetMode() != schema::ReduceMode_ReduceMean) {
    return nullptr;
  }
  auto mean2_axes = mean2_op->GetAxes();
  AnfNodePtr squared_difference_node = mean2_cnode->input(1);
  if (CheckIfAnfNodeIsNull(squared_difference_node) != lite::RET_OK) {
    return nullptr;
  }

  // mean1
  auto mean1_cnode = mean1_node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(mean1_cnode) != lite::RET_OK ||
      CheckInputSize(mean1_cnode, kReduceInputsLength) != lite::RET_OK) {
    return nullptr;
  }
  auto mean1_primitivec = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(mean1_cnode->input(0));
  MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::Reduce>>(mean1_primitivec));
  auto mean1_op = utils::cast<std::shared_ptr<mindspore::lite::Reduce>>(mean1_primitivec);
  MS_ASSERT(mean1_op != nullptr);
  if (mean1_op->GetMode() != schema::ReduceMode_ReduceMean) {
    return nullptr;
  }
  AnfNodePtr input3_node = mean1_cnode->input(1);
  auto mean1_axes = mean1_op->GetAxes();
  if (CheckIfAnfNodeIsNull(input3_node) != lite::RET_OK) {
    return nullptr;
  }

  // verify two mean ops have same axes
  if (mean1_axes.size() != mean2_axes.size()) {
    return nullptr;
  }
  for (size_t i = 0; i < mean1_axes.size(); ++i) {
    if (mean1_axes[i] != mean2_axes[i]) {
      return nullptr;
    }
  }
  // verify axes size and gamma/beta size are equal
  if (mean1_axes.size() != gamma_shape.size() || mean1_axes.size() != beta_shape.size()) {
    return nullptr;
  }
  // verify gamma and beta have same shape
  for (size_t i = 0; i < gamma_shape.size(); ++i) {
    if (gamma_shape[i] != beta_shape[i]) {
      return nullptr;
    }
  }
  // verify epsilon has exactly one element
  float epsilon;
  if (epsilon_shape.empty() || (epsilon_shape.size() == 1 && epsilon_shape[0] == 1)) {
    epsilon = epsilon_data[0];
  } else {
    return nullptr;
  }

  auto layer_norm_cnode = CreateLayerNormNode(func_graph, equiv, gamma_shape, epsilon);
  layer_norm_cnode->set_abstract(add2_cnode->abstract()->Clone());
  layer_norm_cnode->set_fullname_with_scope("layer_norm_" + add2_cnode->fullname_with_scope());
  MS_LOG(INFO) << "layernorm node:" << layer_norm_cnode->fullname_with_scope() << " fusion success";
  return layer_norm_cnode;
}
}  // namespace opt
}  // namespace mindspore
