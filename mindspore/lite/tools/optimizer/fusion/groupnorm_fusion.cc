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
#include "tools/optimizer/fusion/groupnorm_fusion.h"
#include <algorithm>
#include <vector>
#include <memory>
#include "ops/fusion/groupnorm_fusion.h"
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"
#include "nnacl/op_base.h"
#include "src/common/ops/ops_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace opt {
namespace {
STATUS GetAxis(const BaseRef &n, std::vector<int> *axes) {
  MS_ASSERT(axes != nullptr);
  if (utils::isa<ParameterPtr>(n)) {
    auto axes_param = utils::cast<ParameterPtr>(n);
    if (!axes_param->has_default() || axes_param->default_param() == nullptr) {
      return lite::RET_NOT_SUPPORT;
    }
    auto axes_value = axes_param->default_param()->cast<tensor::TensorPtr>();
    if (axes_value == nullptr) {
      return lite::RET_ERROR;
    }
    if (axes_value->data_type() != kNumberTypeInt && axes_value->data_type() != kNumberTypeInt32) {
      MS_LOG(ERROR) << "reduce's axes should be integer, now is " << axes_value->data_type();
      return lite::RET_ERROR;
    }
    if (axes_value->data_c() == nullptr) {
      return lite::RET_ERROR;
    }
    if (axes_value->shape().size() > 1) {
      return lite::RET_ERROR;
    }
    axes->resize(1);
    if (!axes_value->shape().empty()) {
      MS_CHECK_GE(axes_value->shape()[0], 0, lite::RET_ERROR);
      axes->resize(static_cast<size_t>(axes_value->shape()[0]));
    }
    if (memcpy_s(axes->data(), axes->size() * sizeof(int), axes_value->data_c(), axes_value->Size()) == EOK) {
      return lite::RET_OK;
    }
  }
  if (utils::isa<ValueNodePtr>(n)) {
    auto axes_value_node = utils::cast<ValueNodePtr>(n);
    *axes = CastToInt(axes_value_node->value());
    return lite::RET_OK;
  }
  return lite::RET_ERROR;
}

bool IsReduceSumNode(const EquivPtr &equiv, const VarPtr &input_prim, const VarPtr &input_axes,
                     std::vector<int> *axes) {
  MS_ASSERT(equiv != nullptr && input_prim != nullptr && input_axes != nullptr && axes != nullptr);
  auto reduce_value = utils::cast<AnfNodePtr>((*equiv)[input_prim]);
  MS_ASSERT(reduce_value != nullptr);
  auto mean2_primitive = ops::GetOperator<ops::ReduceFusion>(reduce_value);
  MS_CHECK_TRUE_RET(mean2_primitive != nullptr, false);
  auto mean2_primitive_c = mean2_primitive->GetPrim();
  if (mean2_primitive_c->GetAttr(ops::kMode) == nullptr || mean2_primitive->get_mode() != mindspore::Reduce_Sum) {
    return false;
  }
  if (GetAxis((*equiv)[input_axes], axes) != lite::RET_OK) {
    return false;
  }
  return true;
}

bool IsReduceMeanNode(const EquivPtr &equiv, const VarPtr &input_prim, const VarPtr &input_axes,
                      std::vector<int> *axes) {
  MS_ASSERT(equiv != nullptr && input_prim != nullptr && input_axes != nullptr && axes != nullptr);
  auto reduce_value = utils::cast<AnfNodePtr>((*equiv)[input_prim]);
  MS_ASSERT(reduce_value != nullptr);
  auto mean2_primitive = ops::GetOperator<ops::ReduceFusion>(reduce_value);
  MS_CHECK_TRUE_RET(mean2_primitive != nullptr, false);
  auto mean2_primitive_c = mean2_primitive->GetPrim();
  if (mean2_primitive_c->GetAttr(ops::kMode) == nullptr || mean2_primitive->get_mode() != mindspore::Reduce_Mean) {
    return false;
  }
  if (GetAxis((*equiv)[input_axes], axes) != lite::RET_OK) {
    return false;
  }
  return true;
}
}  // namespace

bool GroupNormFusion::Init() const {
  input_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_ != nullptr, false);
  mean1_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mean1_ != nullptr, false);
  mean1_axis_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mean1_axis_ != nullptr, false);
  sum1_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(sum1_ != nullptr, false);
  sum1_axis_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(sum1_axis_ != nullptr, false);
  reshape1_axis_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape1_axis_ != nullptr, false);
  reshape2_axis_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape2_axis_ != nullptr, false);
  gamma_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(gamma_ != nullptr, false);
  beta_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(beta_ != nullptr, false);
  epsilon_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(epsilon_ != nullptr, false);
  real_div_divider_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(real_div_divider_ != nullptr, false);

  return true;
}

bool GroupNormFusion::CheckPattern(const FuncGraphPtr &func_graph, const EquivPtr &equiv, int *num_groups,
                                   float *epsilon, bool *affine) const {
  MS_ASSERT(equiv != nullptr);
  MS_ASSERT(epsilon != nullptr);
  MS_ASSERT(num_groups != nullptr);
  MS_ASSERT(epsilon != nullptr);
  MS_ASSERT(affine != nullptr);

  // beta
  auto beta_node = utils::cast<AnfNodePtr>((*equiv)[beta_]);
  MS_ASSERT(beta_node != nullptr);
  if (!beta_node->isa<Parameter>()) {
    return false;
  }
  auto beta_param = beta_node->cast<ParameterPtr>()->default_param();
  MS_CHECK_TRUE_RET(beta_param != nullptr, false);
  auto beta_tensor = beta_param->cast<tensor::TensorPtr>();
  MS_CHECK_TRUE_RET(beta_tensor != nullptr, false);
  std::vector<int> beta_shape;
  (void)std::transform(beta_tensor->shape().begin(), beta_tensor->shape().end(), std::back_inserter(beta_shape),
                       [](int64_t val) { return static_cast<int>(val); });
  // gamma
  auto gamma_node = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  MS_ASSERT(gamma_node != nullptr);
  if (!gamma_node->isa<Parameter>()) {
    return false;
  }
  auto gamma_param = gamma_node->cast<ParameterPtr>()->default_param();
  MS_CHECK_TRUE_RET(gamma_param != nullptr, false);
  auto gamma_tensor = gamma_param->cast<tensor::TensorPtr>();
  MS_CHECK_TRUE_RET(gamma_tensor != nullptr, false);
  std::vector<int> gamma_shape;
  (void)std::transform(gamma_tensor->shape().begin(), gamma_tensor->shape().end(), std::back_inserter(gamma_shape),
                       [](int64_t val) { return static_cast<int>(val); });
  // epsilon
  auto epsilon_node = utils::cast<AnfNodePtr>((*equiv)[epsilon_]);
  MS_ASSERT(epsilon_node != nullptr);
  if (!epsilon_node->isa<ValueNode>()) {
    return false;
  }
  auto epsilon_value_node = epsilon_node->cast<ValueNodePtr>();
  MS_CHECK_TRUE_RET(epsilon_value_node != nullptr, false);
  auto epsilon_value = epsilon_value_node->value();
  MS_CHECK_TRUE_RET(epsilon_value != nullptr, false);
  if (!epsilon_value->isa<tensor::Tensor>()) {
    std::cout << "CheckPattern:epsilon_value_node not tensor" << std::endl;
    return false;
  }
  auto epsilon_tensor = epsilon_value->cast<tensor::TensorPtr>();
  MS_CHECK_TRUE_RET(epsilon_tensor != nullptr, false);
  TypeId tensor_type = epsilon_tensor->Dtype()->type_id();
  if (!(tensor_type == TypeId::kNumberTypeFloat32) || (tensor_type == TypeId::kNumberTypeFloat)) {
    std::cout << "CheckPattern:epsilon_value_node not float" << std::endl;

    return false;
  }
  auto epsilon_shape = epsilon_tensor->shape();
  // sum1
  std::vector<int> sum1_axes;
  if (!IsReduceSumNode(equiv, sum1_, sum1_axis_, &sum1_axes)) {
    return false;
  }
  // mean1
  std::vector<int> mean1_axes;
  if (!IsReduceMeanNode(equiv, mean1_, mean1_axis_, &mean1_axes)) {
    return false;
  }
  auto input_node = utils::cast<AnfNodePtr>((*equiv)[input_]);
  MS_ASSERT(input_node != nullptr);
  if (!utils::isa<CNodePtr>(input_node)) {
    return false;
  }
  if (mean1_axes != sum1_axes) {
    return false;
  }
  if (gamma_shape != beta_shape) {
    return false;
  }
  if (epsilon_shape.empty() || (epsilon_shape.size() == 1 && epsilon_shape[0] == 1)) {
    MS_CHECK_TRUE_RET(epsilon_tensor->data_c() != nullptr, false);
    auto epsilon_data = reinterpret_cast<float *>(epsilon_tensor->data_c());
    *epsilon = epsilon_data[0];
  } else {
    return false;
  }
  std::vector<int> reshape1_axes;
  if (GetAxis((*equiv)[reshape1_axis_], &reshape1_axes) != lite::RET_OK) {
    return false;
  }
  if (reshape1_axes.size() != C3NUM) {
    return false;
  }
  *num_groups = reshape1_axes.at(C1NUM);
  *affine = true;
  return true;
}

CNodePtr GroupNormFusion::CreateGroupNormNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv, int num_groups,
                                              float epsilon) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(equiv != nullptr);
  PrimitiveCPtr primitive_c = nullptr;

  auto layer_norm_primitive = std::make_shared<ops::GroupNormFusion>();
  MS_CHECK_TRUE_RET(layer_norm_primitive != nullptr, nullptr);
  layer_norm_primitive->Init(num_groups, epsilon, true);
  auto layer_norm_primitive_c = layer_norm_primitive->GetPrim();
  MS_CHECK_TRUE_RET(layer_norm_primitive_c != nullptr, nullptr);
  primitive_c = layer_norm_primitive_c;

  auto value_node = NewValueNode(primitive_c);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
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

const AnfNodePtr GroupNormFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    MS_LOG(ERROR) << "input param is nullptr, do group norm fusion failed.";
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  auto add2_cnode = node->cast<CNodePtr>();
  if (IsMarkedTrainOp(add2_cnode)) {
    return nullptr;
  }
  float epsilon = 0.0f;
  int num_groups = 0;
  bool affine = true;
  if (!CheckPattern(func_graph, equiv, &num_groups, &epsilon, &affine)) {
    return nullptr;
  }
  auto norm_cnode = CreateGroupNormNode(func_graph, equiv, num_groups, epsilon);
  if (norm_cnode == nullptr) {
    MS_LOG(DEBUG) << "create norm cnode failed";
    return nullptr;
  }
  MS_CHECK_TRUE_RET(add2_cnode->abstract() != nullptr, nullptr);
  norm_cnode->set_abstract(add2_cnode->abstract()->Clone());
  norm_cnode->set_fullname_with_scope("group_norm_" + add2_cnode->fullname_with_scope());
  MS_LOG(DEBUG) << "group_norm_ node:" << norm_cnode->fullname_with_scope() << " fusion success";
  return norm_cnode;
}

const BaseRef GroupNormFusion::DefinePattern() const {
  if (!Init()) {
    MS_LOG(ERROR) << "initial member failed.";
    return {};
  }

  auto is_reshape1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  VectorRef reshape_ref1 = VectorRef({is_reshape1, input_, reshape1_axis_});
  VectorRef mean1_ref = VectorRef({mean1_, reshape_ref1, mean1_axis_});
  auto is_sub1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSubFusion>);
  MS_CHECK_TRUE_RET(is_sub1 != nullptr, {});
  VectorRef sub1_ref = VectorRef({is_sub1, reshape_ref1, mean1_ref});

  auto is_sqare = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSquare>);
  MS_CHECK_TRUE_RET(is_sqare != nullptr, {});
  VectorRef square_ref = VectorRef({is_sqare, sub1_ref});
  VectorRef sum1_ref = VectorRef({sum1_, square_ref, sum1_axis_});
  auto is_realdiv1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimRealDiv>);
  MS_CHECK_TRUE_RET(is_realdiv1 != nullptr, {});
  VectorRef realdiv1_ref = VectorRef({is_realdiv1, sum1_ref, real_div_divider_});
  auto is_add1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add1 != nullptr, {});
  VectorRef add1_ref = VectorRef({is_add1, realdiv1_ref, epsilon_});
  auto is_sqrt = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSqrt>);
  MS_CHECK_TRUE_RET(is_sqrt != nullptr, {});
  VectorRef sqrt_ref = VectorRef({is_sqrt, add1_ref});
  auto is_realdiv2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimRealDiv>);
  MS_CHECK_TRUE_RET(is_realdiv2 != nullptr, {});
  VectorRef realdiv2_ref = VectorRef({is_realdiv2, sub1_ref, sqrt_ref});

  auto is_reshape2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape2 != nullptr, {});
  VectorRef reshape_ref2 = VectorRef({is_reshape2, realdiv2_ref, reshape2_axis_});
  auto is_mul1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul1 != nullptr, {});
  VectorRef mul1_ref = VectorRef({is_mul1, reshape_ref2, gamma_});
  auto is_add2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add2 != nullptr, {});
  VectorRef add2_ref = VectorRef({is_add2, mul1_ref, beta_});
  return add2_ref;
}
}  // namespace opt
}  // namespace mindspore
