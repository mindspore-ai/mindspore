/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/fusion/mul_add_fusion.h"
#include <memory>
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/scale_fusion.h"
#include "ops/op_utils.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
namespace {
constexpr size_t kMulInputsLength = 3;
constexpr size_t kAddInputsLength = 3;
}  // namespace

const BaseRef MulAddFusion::DefinePattern() const {
  auto mul_var = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  auto add_var = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  return VectorRef({add_var, mul_var});
}

bool MulAddFusion::ScaleInputShapeValid() const {
  MS_ASSERT(scale_tensor_ != nullptr);
  MS_ASSERT(bias_tensor_ != nullptr);
  auto scale_shape = scale_tensor_->shape_c();
  auto offset_shape = bias_tensor_->shape_c();
  if (mul_input_shape_.size() < scale_shape.size() || scale_shape.size() == 0) {
    return false;
  }
  size_t rank_diff = mul_input_shape_.size() - scale_shape.size();
  for (size_t i = 0; i < scale_shape.size(); ++i) {
    if (mul_input_shape_[i + rank_diff] != scale_shape[i]) {
      return false;
    }
  }
  if (scale_shape != offset_shape) {
    return false;
  }
  return true;
}

namespace {
constexpr int kInputIndex1 = 1;
constexpr int kInputIndex2 = 2;
}  // namespace

bool MulAddFusion::CheckMulNode(const FuncGraphPtr &func_graph) const {
  MS_ASSERT(func_graph != nullptr);
  if (mul_anode_ == nullptr) {
    return false;
  }
  if (IsMultiOutputTensors(func_graph, mul_anode_)) {
    MS_LOG(DEBUG) << "Mul op has multi-output";
    return false;
  }
  auto mul_node = mul_anode_->cast<CNodePtr>();
  if (!CheckPrimitiveType(mul_node, prim::kPrimMulFusion)) {
    MS_LOG(DEBUG) << "Mul add fusion pass match only mul or add";
    return false;
  }
  auto mul_primitive = GetValueNode<std::shared_ptr<ops::MulFusion>>(mul_node->input(0));
  MS_ASSERT(mul_primitive != nullptr);
  auto mul_act_type = mul_primitive->get_activation_type();
  if (mul_act_type != ActivationType::NO_ACTIVATION) {
    MS_LOG(DEBUG) << "Only support mul node with no activation";
    return false;
  }
  if (mul_node == nullptr || mul_node->size() != kInputSizeThree) {
    MS_LOG(DEBUG) << "Mul op is null or has error input size";
    return false;
  }
  // find mul's const input and mul input
  AnfNodePtr mul_pre_input_node = nullptr;
  AnfNodePtr mul_pre_const_node = nullptr;
  auto mul_pre_node_1 = mul_node->input(kInputIndex1);
  if (CheckIfAnfNodeIsNull(mul_pre_node_1) != lite::RET_OK) {
    MS_LOG(DEBUG) << "Pre-node of mul op is nullptr";
    return false;
  }
  auto mul_pre_node_2 = mul_node->input(kInputIndex2);
  if (CheckIfAnfNodeIsNull(mul_pre_node_2) != lite::RET_OK) {
    MS_LOG(DEBUG) << "Pre-node of mul op is nullptr";
    return false;
  }
  if (utils::isa<CNodePtr>(mul_pre_node_1) && !utils::isa<CNodePtr>(mul_pre_node_2)) {
    mul_pre_input_node = mul_pre_node_1;
    mul_pre_const_node = mul_pre_node_2;
  } else if (!utils::isa<CNodePtr>(mul_pre_node_1) && utils::isa<CNodePtr>(mul_pre_node_2)) {
    mul_pre_input_node = mul_pre_node_1;
    mul_pre_const_node = mul_pre_node_2;
  } else {
    MS_LOG(DEBUG) << "Mul op should has a cnode input and a const input";
    return false;
  }
  // check mul's const input
  tensor::TensorPtr mul_tensor = nullptr;
  if (utils::isa<ParameterPtr>(mul_pre_const_node)) {
    auto mul_bias_node = mul_pre_const_node->cast<ParameterPtr>();
    MS_ASSERT(mul_bias_node != nullptr);
    if (!mul_bias_node->has_default()) {
      MS_LOG(DEBUG) << "Const input of mul op should has data";
      return false;
    }
    mul_tensor = mul_bias_node->default_param()->cast<tensor::TensorPtr>();
  } else if (utils::isa<ValueNodePtr>(mul_pre_const_node)) {
    auto mul_bias_node = mul_pre_const_node->cast<ValueNodePtr>();
    MS_ASSERT(mul_bias_node != nullptr);
    if (mul_bias_node->value() == nullptr) {
      MS_LOG(DEBUG) << "Const input of mul op should has data";
      return false;
    }
    mul_tensor = mul_bias_node->value()->cast<tensor::TensorPtr>();
  } else {
    MS_ASSERT(false);
  }
  if (mul_tensor == nullptr) {
    MS_LOG(DEBUG) << "Const input of add op should has data";
    return false;
  }
  mul_input_anode_ = mul_pre_input_node;
  mul_const_anode_ = mul_pre_const_node;
  scale_tensor_ = mul_tensor;
  return true;
}

bool MulAddFusion::CheckAddNode() const {
  if (add_anode_ == nullptr) {
    return false;
  }
  auto add_cnode = add_anode_->cast<CNodePtr>();
  if (add_cnode == nullptr || add_cnode->size() != kInputSizeThree) {
    MS_LOG(DEBUG) << "Add op is null or has error input size";
    return false;
  }
  if (!CheckPrimitiveType(add_cnode, prim::kPrimAddFusion)) {
    MS_LOG(DEBUG) << "Mul add fusion pass match only mul or add";
    return false;
  }
  auto add_primitive = GetValueNode<std::shared_ptr<ops::AddFusion>>(add_cnode->input(0));
  MS_ASSERT(add_primitive != nullptr);
  auto add_act_type = add_primitive->get_activation_type();
  if (add_act_type != ActivationType::RELU && add_act_type != ActivationType::RELU6 &&
      add_act_type != ActivationType::NO_ACTIVATION) {
    MS_LOG(DEBUG) << "Only support add node with relu or relu6 or no activation";
    return false;
  }
  scale_act_type_ = add_act_type;
  // find add's const input and mul input
  AnfNodePtr add_pre_input_node = nullptr;
  AnfNodePtr add_pre_const_node = nullptr;
  auto add_pre_node_1 = add_cnode->input(kInputIndex1);
  if (CheckIfAnfNodeIsNull(add_pre_node_1) != lite::RET_OK) {
    MS_LOG(DEBUG) << "Pre-node of add op is nullptr";
    return false;
  }
  auto add_pre_node_2 = add_cnode->input(kInputIndex2);
  if (CheckIfAnfNodeIsNull(add_pre_node_2) != lite::RET_OK) {
    MS_LOG(DEBUG) << "Pre-node of add op is nullptr";
    return false;
  }
  if (utils::isa<CNodePtr>(add_pre_node_1) && !utils::isa<CNodePtr>(add_pre_node_2)) {
    add_pre_input_node = add_pre_node_1;
    add_pre_const_node = add_pre_node_2;
  } else if (!utils::isa<CNodePtr>(add_pre_node_1) && utils::isa<CNodePtr>(add_pre_node_2)) {
    add_pre_input_node = add_pre_node_2;
    add_pre_const_node = add_pre_node_1;
  } else {
    MS_LOG(DEBUG) << "Add op should has a cnode input and a const input";
    return false;
  }
  // check add's const input
  tensor::TensorPtr add_tensor = nullptr;
  if (utils::isa<ParameterPtr>(add_pre_const_node)) {
    auto add_bias_node = add_pre_const_node->cast<ParameterPtr>();
    MS_ASSERT(add_bias_node != nullptr);
    if (!add_bias_node->has_default()) {
      MS_LOG(DEBUG) << "Const input of add op should has data";
      return false;
    }
    add_tensor = add_bias_node->default_param()->cast<tensor::TensorPtr>();
  } else if (utils::isa<ValueNodePtr>(add_pre_const_node)) {
    auto add_bias_node = add_pre_const_node->cast<ValueNodePtr>();
    MS_ASSERT(add_bias_node != nullptr);
    if (add_bias_node->value() == nullptr) {
      MS_LOG(DEBUG) << "Const input of add op should has data";
      return false;
    }
    add_tensor = add_bias_node->value()->cast<tensor::TensorPtr>();
  } else {
    MS_ASSERT(false);
  }
  if (add_tensor == nullptr) {
    MS_LOG(DEBUG) << "Const input of add op should has data";
    return false;
  }
  mul_anode_ = add_pre_input_node;
  add_const_anode_ = add_pre_const_node;
  bias_tensor_ = add_tensor;
  return true;
}

bool MulAddFusion::GetMulInputShape() const {
  MS_ASSERT(mul_input_anode_ != nullptr);
  ShapeVector mul_input_shape;
  AbstractBasePtr mul_input_abstract = nullptr;
  if (utils::isa<ParameterPtr>(mul_input_anode_)) {
    auto mul_input_node = mul_input_anode_->cast<ParameterPtr>();
    MS_ASSERT(mul_bias_node != nullptr);
    mul_input_abstract = mul_input_node->abstract();
  } else if (utils::isa<ValueNodePtr>(mul_input_anode_)) {
    auto mul_input_node = mul_input_anode_->cast<ValueNodePtr>();
    MS_ASSERT(mul_input_node != nullptr);
    mul_input_abstract = mul_input_node->abstract();
  } else if (utils::isa<CNodePtr>(mul_input_anode_)) {
    auto mul_input_node = mul_input_anode_->cast<CNodePtr>();
    MS_ASSERT(mul_input_node != nullptr);
    mul_input_abstract = mul_input_node->abstract();
  } else {
    MS_ASSERT(false);
  }
  if (mul_input_abstract == nullptr) {
    MS_LOG(DEBUG) << "Mul input node has no abstract";
    return false;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(mul_input_abstract)) {
    MS_LOG(DEBUG) << "Abstract of mul input node should be AbstractTensor";
    return false;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(mul_input_abstract);
  MS_ASSERT(abstract_tensor != nullptr);
  MS_ASSERT(abstract_tensor->BuildShape() != nullptr);
  if (!utils::isa<abstract::ShapePtr>(abstract_tensor->BuildShape())) {
    MS_LOG(DEBUG) << "BuildShape of abstract of mul input node should be ShapePtr";
    return false;
  }
  mul_input_shape_ = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  return true;
}

const AnfNodePtr MulAddFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  add_anode_ = node;
  if (!CheckAddNode()) {
    MS_LOG(DEBUG) << "Add op is not suit for mul-add-fusion: " << node->fullname_with_scope();
    return nullptr;
  }
  MS_ASSERT(mul_anode_ != nullptr);
  MS_ASSERT(bias_tensor_ != nullptr);
  MS_ASSERT(add_const_anode_ != nullptr);
  if (!CheckMulNode(func_graph)) {
    MS_LOG(DEBUG) << "Mul op is not suit for mul-add-fusion: " << mul_anode_->fullname_with_scope();
    return nullptr;
  }
  MS_ASSERT(mul_input_anode_ != nullptr);
  MS_ASSERT(scale_tensor_ != nullptr);
  MS_ASSERT(mul_const_anode_ != nullptr);
  if (!GetMulInputShape()) {
    MS_LOG(DEBUG) << "Get input shape of mul op failed";
    return nullptr;
  }
  // scale requires scale shape tail sub of input shape, scale shape same as bias shape
  if (!ScaleInputShapeValid()) {
    MS_LOG(DEBUG) << "Check input shape, scale shape and bias shape failed";
    return nullptr;
  }
  // create scale primitive
  auto scale_primitive = new (std::nothrow) mindspore::ops::ScaleFusion();
  if (scale_primitive == nullptr) {
    MS_LOG(ERROR) << "new scale primitive failed";
    return nullptr;
  }
  scale_primitive->set_activation_type(scale_act_type_);
  scale_primitive->set_axis(-(bias_tensor_->shape_c().size()));
  // create scale op
  auto scale_node = func_graph->NewCNode(std::shared_ptr<ops::PrimitiveC>(scale_primitive),
                                         {mul_input_anode_, mul_const_anode_, add_const_anode_});
  return scale_node;
}
}  // namespace mindspore::opt
