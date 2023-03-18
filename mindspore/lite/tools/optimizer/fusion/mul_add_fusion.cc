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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/mul_add_fusion.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "nnacl/op_base.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/scale_fusion.h"
#include "ops/op_utils.h"
#include "tools/lite_exporter/fetch_content.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
VectorRef MulAddFusion::DefineMulFirstPattern() const {
  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto is_const = std::make_shared<CondVar>(IsParamOrValueNodeWithData);
  MS_CHECK_TRUE_RET(is_const != nullptr, {});
  return VectorRef({is_add, is_mul, is_const});
}

VectorRef MulAddFusion::DefineMulSecondPattern() const {
  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  auto is_const = std::make_shared<CondVar>(IsParamOrValueNodeWithData);
  MS_CHECK_TRUE_RET(is_const != nullptr, {});
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  return VectorRef({is_add, is_const, is_mul});
}

std::unordered_map<std::string, VectorRef> MulAddFusion::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns["MulFirstPatternName"] = DefineMulFirstPattern();
  patterns["MulSecondPatternName"] = DefineMulSecondPattern();
  return patterns;
}

bool MulAddFusion::CheckAddNode(const mindspore::CNodePtr &cnode) const {
  MS_CHECK_TRUE_RET(cnode != nullptr, false);
  if (cnode->size() != kInputSizeThree) {
    MS_LOG(DEBUG) << "Add op is null or has error input size";
    return false;
  }
  if (IsMarkedTrainOp(cnode)) {
    return false;
  }
  auto add_primitive = ops::GetOperator<ops::AddFusion>(cnode->input(0));
  MS_CHECK_TRUE_RET(add_primitive != nullptr, false);
  auto add_primitive_c = add_primitive->GetPrim();
  MS_CHECK_TRUE_RET(add_primitive_c != nullptr, false);
  auto quant_attr = add_primitive_c->GetAttr("quant_params");
  if (quant_attr != nullptr) {
    auto quant_param_holder = quant_attr->cast<lite::QuantParamHolderPtr>();
    MS_CHECK_TRUE_RET(quant_param_holder != nullptr, false);
    auto quant_params = quant_param_holder->get_input_quant_params();
    bool is_quant = std::any_of(quant_params.begin(), quant_params.end(), [](std::vector<schema::QuantParamT> &params) {
      return !params.empty() && params.front().inited;
    });
    if (is_quant) {
      return false;
    }
  }

  ActivationType add_act_type = ActivationType::NO_ACTIVATION;
  if (add_primitive_c->GetAttr(ops::kActivationType) != nullptr) {
    add_act_type = add_primitive->get_activation_type();
    if (add_act_type != ActivationType::RELU && add_act_type != ActivationType::RELU6 &&
        add_act_type != ActivationType::NO_ACTIVATION) {
      MS_LOG(DEBUG) << "Only support add node with relu or relu6 or no activation";
      return false;
    }
  }
  scale_act_type_ = add_act_type;
  return true;
}

bool MulAddFusion::CheckMulNode(const mindspore::FuncGraphPtr &func_graph, const mindspore::CNodePtr &cnode) const {
  MS_ASSERT(func_graph != nullptr);
  MS_CHECK_TRUE_RET(cnode != nullptr, false);
  if (IsMultiOutputTensors(func_graph, cnode)) {
    MS_LOG(DEBUG) << "Mul op has multi-output";
    return false;
  }
  if (IsMarkedTrainOp(cnode)) {
    return false;
  }
  auto mul_primitive = ops::GetOperator<ops::MulFusion>(cnode->input(0));
  MS_CHECK_TRUE_RET(mul_primitive != nullptr, false);
  auto mul_primitive_c = mul_primitive->GetPrim();
  MS_CHECK_TRUE_RET(mul_primitive_c != nullptr, false);
  auto quant_attr = mul_primitive_c->GetAttr("quant_params");
  if (quant_attr != nullptr) {
    auto quant_param_holder = quant_attr->cast<lite::QuantParamHolderPtr>();
    MS_CHECK_TRUE_RET(quant_param_holder != nullptr, false);
    auto quant_params = quant_param_holder->get_input_quant_params();
    bool is_quant = std::any_of(quant_params.begin(), quant_params.end(), [](std::vector<schema::QuantParamT> &params) {
      return !params.empty() && params.front().inited;
    });
    if (is_quant) {
      return false;
    }
  }

  if (mul_primitive_c->GetAttr(ops::kActivationType) != nullptr &&
      mul_primitive->get_activation_type() != ActivationType::NO_ACTIVATION) {
    MS_LOG(DEBUG) << "Only support mul node with no activation";
    return false;
  }
  if (cnode->size() != kInputSizeThree) {
    MS_LOG(DEBUG) << "Mul op is null or has error input size";
    return false;
  }
  return true;
}

bool MulAddFusion::AdjustScaleBiasTensorShape(size_t *axis_offset) const {
  MS_CHECK_TRUE_RET(axis_offset != nullptr, false);
  auto scale_shape = scale_tensor_->shape_c();
  if (mul_input_shape_ == scale_shape) {
    return true;
  }
  while (scale_shape.size() > DIMENSION_1D) {
    bool begin_with_value_one = scale_shape.at(FIRST_INPUT) == DIMENSION_1D ? true : false;
    bool end_with_value_one = scale_shape.at(scale_shape.size() - DIMENSION_1D) == DIMENSION_1D ? true : false;
    if (!begin_with_value_one && !end_with_value_one) {
      break;
    }
    if (begin_with_value_one) {
      scale_shape.erase(scale_shape.begin());
    }
    if (end_with_value_one) {
      scale_shape.erase(scale_shape.end() - DIMENSION_1D);
      *axis_offset += DIMENSION_1D;
    }
  }
  (void)scale_tensor_->set_shape(scale_shape);
  (void)bias_tensor_->set_shape(scale_shape);

  // set shape for abstract
  auto mul_abstract = mul_const_anode_->abstract();
  MS_CHECK_TRUE_RET(mul_abstract != nullptr, false);
  auto new_shape = std::make_shared<abstract::Shape>(scale_shape);
  MS_CHECK_TRUE_RET(new_shape != nullptr, false);
  mul_abstract->set_shape(new_shape);

  auto add_abstract = add_const_anode_->abstract();
  MS_CHECK_TRUE_RET(add_abstract != nullptr, false);
  auto new_add_shape = std::make_shared<abstract::Shape>(scale_shape);
  MS_CHECK_TRUE_RET(new_add_shape != nullptr, false);
  add_abstract->set_shape(new_add_shape);
  return true;
}

bool MulAddFusion::ScaleInputShapeValid(size_t *axis_offset) const {
  MS_ASSERT(scale_tensor_ != nullptr && bias_tensor_ != nullptr && axis_offset != nullptr);
  if (scale_tensor_->shape_c() != bias_tensor_->shape_c()) {
    return false;
  }
  // remove value 1 which is in the begin or the end of shape vector.
  if (!AdjustScaleBiasTensorShape(axis_offset)) {
    MS_LOG(ERROR) << "Adjust scale shape and bias shape failed.";
    return false;
  }
  auto scale_shape = scale_tensor_->shape_c();
  if (mul_input_shape_.size() < scale_shape.size() || scale_shape.size() == 0) {
    return false;
  }
  size_t rank_diff = mul_input_shape_.size() - scale_shape.size();
  for (size_t i = 0; i < scale_shape.size(); ++i) {
    if (i + rank_diff < *axis_offset) {
      MS_LOG(ERROR) << "Sub overflow occur may cause index out of range.";
      return false;
    }
    if (mul_input_shape_[i + rank_diff - *axis_offset] != scale_shape[i]) {
      return false;
    }
  }
  return true;
}

bool MulAddFusion::MulInputAnodeIsInferred(const AnfNodePtr &mul_input_anode) const {
  auto mul_input_cnode = mul_input_anode->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul_input_cnode);
  auto prim = GetValueNode<PrimitivePtr>(mul_input_cnode->input(0));
  MS_CHECK_TRUE_RET(prim != nullptr, false);
  auto is_inferred = prim->GetAttr(kInferDone) != nullptr && GetValue<bool>(prim->GetAttr(kInferDone));
  return is_inferred;
}

bool MulAddFusion::CopyNodeFormat(CNodePtr node, mindspore::ops::PrimitiveCPtr prim) const {
  auto src_prim = GetValueNode<PrimitiveCPtr>(node->input(0));
  if (src_prim->GetAttr(mindspore::ops::kFormat) != nullptr) {
    auto value = src_prim->GetAttr(mindspore::ops::kFormat);
    if (value->isa<mindspore::Int64Imm>()) {
      auto format = GetValue<int64_t>(value);
      prim->AddAttr(mindspore::ops::kFormat, MakeValue(format));
    }
  }
  return true;
}

AnfNodePtr MulAddFusion::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
                                 const mindspore::AnfNodePtr &node, const mindspore::EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto add_cnode = node->cast<CNodePtr>();
  if (!CheckAddNode(add_cnode)) {
    MS_LOG(DEBUG) << "Add op is not suit for mul-add-fusion: " << node->fullname_with_scope();
    return nullptr;
  }

  auto mul_node = utils::isa<CNodePtr>(add_cnode->input(SECOND_INPUT)) ? add_cnode->input(SECOND_INPUT)
                                                                       : add_cnode->input(THIRD_INPUT);
  MS_CHECK_TRUE_RET(mul_node != nullptr, nullptr);
  auto mul_cnode = mul_node->cast<CNodePtr>();
  if (!CheckMulNode(func_graph, mul_cnode)) {
    MS_LOG(DEBUG) << "Mul op is not suit for mul-add-fusion: " << mul_cnode->fullname_with_scope();
    return nullptr;
  }

  auto mul_input_anode = mul_cnode->input(SECOND_INPUT);
  if (utils::isa<ParameterPtr>(mul_input_anode)) {
    auto param_node = mul_input_anode->cast<ParameterPtr>();
    MS_CHECK_TRUE_RET(param_node != nullptr, nullptr);
    mul_const_anode_ = param_node->has_default() ? mul_input_anode : mul_cnode->input(THIRD_INPUT);
    mul_input_anode = param_node->has_default() ? mul_cnode->input(THIRD_INPUT) : mul_input_anode;
  } else if (utils::isa<CNodePtr>(mul_input_anode)) {
    mul_const_anode_ = mul_cnode->input(THIRD_INPUT);
  }
  size_t add_const_idx = utils::isa<CNodePtr>(add_cnode->input(SECOND_INPUT)) ? THIRD_INPUT : SECOND_INPUT;
  add_const_anode_ = add_cnode->input(add_const_idx);
  MS_CHECK_TRUE_RET(mul_const_anode_ != nullptr && add_const_anode_ != nullptr, nullptr);
  bias_tensor_ = GetTensorInfo(add_const_anode_);
  scale_tensor_ = GetTensorInfo(mul_const_anode_);
  MS_CHECK_TRUE_RET(bias_tensor_ != nullptr, nullptr);
  MS_CHECK_TRUE_RET(scale_tensor_ != nullptr, nullptr);
  MS_CHECK_TRUE_RET(mul_input_anode != nullptr, nullptr);
  if (mul_input_anode->isa<CNode>()) {
    if (!MulInputAnodeIsInferred(mul_input_anode)) {
      MS_LOG(DEBUG) << "mul_input_anode is not inferred, don't perform the ScaleInputShapeValid method.";
      return nullptr;
    }
  }
  if (FetchShapeFromAbstract(mul_input_anode->abstract(), &mul_input_shape_) != lite::RET_OK) {
    return nullptr;
  }
  // scale requires scale shape tail sub of input shape, scale shape same as bias shape
  size_t axis_offset = 0;
  if (!ScaleInputShapeValid(&axis_offset)) {
    MS_LOG(DEBUG) << "Check input shape, scale shape and bias shape failed";
    return nullptr;
  }
  // create scale primitive
  auto scale_primitive = std::make_shared<ops::ScaleFusion>();
  if (scale_primitive == nullptr) {
    MS_LOG(ERROR) << "new scale primitive failed";
    return nullptr;
  }
  scale_primitive->set_activation_type(scale_act_type_);
  auto scale_primitive_c = scale_primitive->GetPrim();
  MS_CHECK_TRUE_RET(scale_primitive_c != nullptr, nullptr);
  if (INT_ADD_OVERFLOW_THRESHOLD(bias_tensor_->shape_c().size(), axis_offset, SIZE_MAX)) {
    MS_LOG(ERROR) << "Add overflow: " << bias_tensor_->shape_c().size() << " + " << axis_offset;
    return nullptr;
  }
  scale_primitive->set_axis(-(static_cast<int64_t>(bias_tensor_->shape_c().size() + axis_offset)));

  // copy the format of add node to scale node
  if (CopyNodeFormat(add_cnode, scale_primitive_c)) {
    MS_LOG(WARNING) << "Copy original node format failed";
  }

  // create scale op
  auto scale_node = func_graph->NewCNode(scale_primitive_c, {mul_input_anode, mul_const_anode_, add_const_anode_});
  scale_node->set_abstract(add_cnode->abstract());
  return scale_node;
}
}  // namespace mindspore::opt
