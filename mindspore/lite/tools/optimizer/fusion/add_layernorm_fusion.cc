/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/fusion/add_layernorm_fusion.h"
#include <memory>
#include <vector>
#include <string>
#include "ops/nn_ops.h"
#include "ops/fusion/activation.h"
#include "ops/lite_ops.h"
#include "ops/add_layernorm.h"
#include "ops/auto_generate/gen_lite_ops.h"
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
constexpr int kAxis = -1;
constexpr float kPow = 2.0;
constexpr float kEps = 1e-5;
constexpr float kDiffThreshold = 1e-6;
constexpr auto kAddLayerNormPattern = "AddLayerNormFusion";
constexpr auto kLayerNormV3Pattern = "LayerNormV3Fusion";
constexpr int kReduceAxisNum = 1;
constexpr int kInputIndex1 = 1;
constexpr int kInputIndex2 = 2;
constexpr int kInvalidDim = -1;

std::vector<int64_t> GetTensorShape(CNodePtr cnode, size_t index) {
  auto abstract = GetCNodeInputAbstract(cnode, index);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Fail to GetCNodeInputAbstract.";
    return {};
  }
  std::vector<int64_t> shape = {};
  if (FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "FetchShapeFromAbstract failed.";
    return {};
  }
  return shape;
}

int GetInputDims(const AnfNodePtr &node) {
  auto add3 = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add3 != nullptr, kInvalidDim);
  auto mul = add3->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(mul != nullptr, kInvalidDim);
  auto div = mul->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(div != nullptr, kInvalidDim);
  auto sub = div->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(sub != nullptr, kInvalidDim);
  auto sub_a_shape = GetTensorShape(sub, 1);
  if (!(sub_a_shape.size() == 1 && sub_a_shape[0] == -1)) {
    return sub_a_shape.size();
  }
  // In dynamic shape scene, sub_a_shape is {-1}. In order to get real dims,
  // go backward from sub_a and find Reshape + Concat,
  // the num of inputs of concat is just the dim of sub_a.
  // Sub <-- [Add] <-- [Add] <-- Reshape <-- Concat
  auto sub_input_0 = sub->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(sub_input_0 != nullptr, kInvalidDim);
  auto reshape = sub_input_0;
  if (CheckPrimitiveType(sub_input_0, prim::kPrimAddFusion)) {
    reshape = sub_input_0->input(kInputIndex2)->cast<CNodePtr>();
  }

  if (CheckPrimitiveType(reshape, prim::kPrimAddFusion)) {
    reshape = reshape->input(kInputIndex2)->cast<CNodePtr>();
  }

  if (CheckPrimitiveType(reshape, prim::kPrimReshape)) {
    auto concat = reshape->input(kInputIndex2)->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(concat != nullptr, kInvalidDim);
    if (!CheckPrimitiveType(concat, prim::kPrimConcat)) {
      MS_LOG(INFO) << "The second input of cnode " << reshape->fullname_with_scope() << " is not Concat.";
      return kInvalidDim;
    }
    size_t dim = concat->size();
    if (dim == 0) {
      MS_LOG(ERROR) << "node " << concat->fullname_with_scope() << "has no inputs.";
      return kInvalidDim;
    }
    return dim - 1;
  }
  MS_LOG(INFO) << "The first input of sub is " << sub_input_0->fullname_with_scope();
  return kInvalidDim;
}

CNodePtr NewCNodeInner(const CNodePtr &cnode, const PrimitivePtr &primitive, const std::vector<AnfNodePtr> &inputs,
                       const abstract::AbstractBasePtr &abstract, const std::string &name) {
  auto func_graph = cnode->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Failed to NewCNode, funcGraph cannot be nullptr";
    return nullptr;
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "Failed to NewCNode, FuncGraph manager cannot be nullptr";
    return nullptr;
  }
  auto new_node = func_graph->NewCNode(primitive, inputs);
  if (new_node == nullptr) {
    MS_LOG(ERROR) << "Failed to create node " << name << " for node " << cnode->fullname_with_scope();
    return nullptr;
  }
  new_node->set_fullname_with_scope(name);
  for (size_t i = 0; i < inputs.size(); i++) {
    manager->SetEdge(new_node, i + 1, inputs[i]);
  }
  new_node->set_abstract(abstract);
  return new_node;
}

CNodePtr AddLayerNormFusion::CreateAddLayerNormNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  auto add_layernorm_prim = std::make_shared<ops::AddLayerNorm>();
  MS_CHECK_TRUE_RET(add_layernorm_prim != nullptr, nullptr);

  add_layernorm_prim->AddAttr("additional_output", api::MakeValue(true));

  auto add_layernorm_prim_c = add_layernorm_prim->GetPrim();
  MS_CHECK_TRUE_RET(add_layernorm_prim_c != nullptr, nullptr);

  auto x1 = utils::cast<AnfNodePtr>((*equiv)[add_1_a_]);
  MS_CHECK_TRUE_RET(x1 != nullptr, nullptr);
  auto x2 = utils::cast<AnfNodePtr>((*equiv)[add_1_b_]);
  MS_CHECK_TRUE_RET(x2 != nullptr, nullptr);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[mul_b_]);
  MS_CHECK_TRUE_RET(gamma != nullptr, nullptr);
  auto beta = utils::cast<AnfNodePtr>((*equiv)[add_3_b_]);
  MS_CHECK_TRUE_RET(beta != nullptr, nullptr);

  auto add_layernorm_cnode = func_graph->NewCNode(add_layernorm_prim_c, {x1, x2, gamma, beta});
  MS_CHECK_TRUE_RET(add_layernorm_cnode != nullptr, nullptr);
  add_layernorm_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_add_layernorm_fusion");
  if (node->abstract() != nullptr) {
    add_layernorm_cnode->set_abstract(node->abstract()->Clone());
  }
  return add_layernorm_cnode;
}

CNodePtr AddLayerNormFusion::CreateLayerNormV3Node(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                   const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  auto lnv3_prim = std::make_shared<ops::LayerNormV3>();
  MS_CHECK_TRUE_RET(lnv3_prim != nullptr, nullptr);

  auto input_dims = GetInputDims(node);
  MS_CHECK_TRUE_RET(input_dims != -1, nullptr);
  MS_LOG(INFO) << "Input dims of LayerNormV3: " << input_dims;
  lnv3_prim->AddAttr("begin_norm_axis", api::MakeValue(input_dims - kReduceAxisNum));
  lnv3_prim->AddAttr("begin_params_axis", api::MakeValue(input_dims - kReduceAxisNum));

  auto lnv3_prim_c = lnv3_prim->GetPrim();
  MS_CHECK_TRUE_RET(lnv3_prim_c != nullptr, nullptr);

  auto x = utils::cast<AnfNodePtr>((*equiv)[reduce_1_x_]);
  MS_CHECK_TRUE_RET(x != nullptr, nullptr);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[mul_b_]);
  MS_CHECK_TRUE_RET(gamma != nullptr, nullptr);
  auto beta = utils::cast<AnfNodePtr>((*equiv)[add_3_b_]);
  MS_CHECK_TRUE_RET(beta != nullptr, nullptr);

  auto lnv3_cnode = func_graph->NewCNode(lnv3_prim_c, {x, gamma, beta});
  MS_CHECK_TRUE_RET(lnv3_cnode != nullptr, nullptr);
  lnv3_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_layernormv3_fusion");
  if (node->abstract() != nullptr) {
    lnv3_cnode->set_abstract(node->abstract()->Clone());
  }

  auto add_cnode = lnv3_cnode->input(1)->cast<CNodePtr>();
  if (!CheckPrimitiveType(add_cnode, prim::kPrimAdd) && !CheckPrimitiveType(add_cnode, prim::kPrimAddFusion)) {
    MS_LOG(INFO) << "No add before LNV3, skip insert cast: " << add_cnode->fullname_with_scope();
    return lnv3_cnode;
  }
  // Insert cast to make the dtype of Add before LayerNormV3 is fp16,
  // otherwise, pattern (Add, LayerNormV3) will not work.
  auto add_in1 = add_cnode->input(kInputIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add_in1 != nullptr, nullptr);
  auto add_in2 = add_cnode->input(kInputIndex2)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add_in2 != nullptr, nullptr);
  auto add_input1_cast =
    NewCNodeInner(add_in1, prim::kPrimCast, {add_in1, NewValueNode(TypeIdToType(kNumberTypeFloat16))},
                  add_in1->abstract(), add_in1->fullname_with_scope() + "_input1_cast");
  MS_CHECK_TRUE_RET(add_input1_cast != nullptr, nullptr);
  add_cnode->set_input(kInputIndex1, add_input1_cast);

  auto add_input2_cast =
    NewCNodeInner(add_in2, prim::kPrimCast, {add_in2, NewValueNode(TypeIdToType(kNumberTypeFloat16))},
                  add_in2->abstract(), add_in2->fullname_with_scope() + "_input2_cast");
  MS_CHECK_TRUE_RET(add_input2_cast != nullptr, nullptr);
  add_cnode->set_input(kInputIndex2, add_input2_cast);

  return lnv3_cnode;
}

bool AddLayerNormFusion::Init() const {
  add_1_a_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add_1_a_ != nullptr, false);

  add_1_b_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add_1_b_ != nullptr, false);

  reduce_1_x_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reduce_1_x_ != nullptr, false);

  sub_a_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(sub_a_ != nullptr, false);

  reduce_1_axis_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reduce_1_axis_ != nullptr, false);

  pow_y_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(pow_y_ != nullptr, false);

  reduce_2_axis_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reduce_2_axis_ != nullptr, false);

  add_2_b_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add_2_b_ != nullptr, false);

  mul_b_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul_b_ != nullptr, false);

  add_3_b_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add_3_b_ != nullptr, false);
  return true;
}

const VectorRef AddLayerNormFusion::DefineAddlayerNormPattern() const {
  MS_LOG(INFO) << "start define add layernorm fusion patterns.";
  if (!Init()) {
    MS_LOG(ERROR) << "AddlayerNormFusion pattern Init Failed.";
    return {};
  }

  auto is_add_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  VectorRef add_1_ref({is_add_1, add_1_a_, add_1_b_});

  auto is_reduce_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReduceFusion>);
  VectorRef reduce_1_ref({is_reduce_1, add_1_ref, reduce_1_axis_});

  auto is_sub = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSubFusion>);
  VectorRef sub_ref({is_sub, add_1_ref, reduce_1_ref});

  auto is_pow = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimPowFusion>);
  VectorRef pow_ref({is_pow, sub_ref, pow_y_});

  auto is_reduce_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReduceFusion>);
  VectorRef reduce_2_ref({is_reduce_2, pow_ref, reduce_2_axis_});

  auto is_add_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  VectorRef add_2_ref({is_add_2, reduce_2_ref, add_2_b_});

  auto is_sqrt = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSqrt>);
  VectorRef sqrt_ref({is_sqrt, add_2_ref});

  auto is_div = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDivFusion>);
  VectorRef div_ref({is_div, sub_ref, sqrt_ref});

  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  VectorRef mul_ref({is_mul, div_ref, mul_b_});

  auto is_add_3 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  VectorRef add_3_ref({is_add_3, mul_ref, add_3_b_});
  return add_3_ref;
}

const VectorRef AddLayerNormFusion::DefineLayerNormV3Pattern() const {
  MS_LOG(INFO) << "start define LayerNormV3 fusion patterns.";
  if (!Init()) {
    MS_LOG(ERROR) << "LayerNormV3 pattern Init Failed.";
    return {};
  }

  auto is_reduce_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReduceFusion>);
  VectorRef reduce_1_ref({is_reduce_1, reduce_1_x_, reduce_1_axis_});

  auto is_sub = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSubFusion>);
  VectorRef sub_ref({is_sub, reduce_1_x_, reduce_1_ref});

  auto is_pow = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimPowFusion>);
  VectorRef pow_ref({is_pow, sub_ref, pow_y_});

  auto is_reduce_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReduceFusion>);
  VectorRef reduce_2_ref({is_reduce_2, pow_ref, reduce_2_axis_});

  auto is_add_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  VectorRef add_2_ref({is_add_2, reduce_2_ref, add_2_b_});

  auto is_sqrt = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSqrt>);
  VectorRef sqrt_ref({is_sqrt, add_2_ref});

  auto is_div = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDivFusion>);
  VectorRef div_ref({is_div, sub_ref, sqrt_ref});

  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  VectorRef mul_ref({is_mul, div_ref, mul_b_});

  auto is_add_3 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  VectorRef add_3_ref({is_add_3, mul_ref, add_3_b_});
  return add_3_ref;
}

bool AddLayerNormFusion::CheckPattern(const EquivPtr &equiv) const {
  MS_ASSERT(equiv != nullptr);
  int reduce_1_axis = GetIntParameterValue(equiv, reduce_1_axis_);
  if (reduce_1_axis == INT_MIN) {
    MS_LOG(ERROR) << "not supported axis: " << reduce_1_axis;
    return false;
  }

  float pow_y = GetFloatParameterValue(equiv, pow_y_);
  if (pow_y <= 0 || fabs(pow_y - kPow) > kDiffThreshold) {
    MS_LOG(ERROR) << "not supported pow: " << pow_y;
    return false;
  }

  int reduce_2_axis = GetIntParameterValue(equiv, reduce_2_axis_);
  if (reduce_2_axis == INT_MIN) {
    MS_LOG(ERROR) << "not supported axis: " << reduce_2_axis;
    return false;
  }

  float add_2_b = GetFloatParameterValue(equiv, add_2_b_);
  if (add_2_b <= 0 || fabs(add_2_b - kEps) > kDiffThreshold) {
    MS_LOG(ERROR) << "not supported bias: " << add_2_b;
    return false;
  }
  return true;
}

std::unordered_map<std::string, VectorRef> AddLayerNormFusion::DefinePatterns() const {
  MS_LOG(INFO) << "start define add layernorm fusion patterns.";
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kLayerNormV3Pattern] = DefineLayerNormV3Pattern();
  return patterns;
}

AnfNodePtr AddLayerNormFusion::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
                                       const mindspore::AnfNodePtr &node, const mindspore::EquivPtr &equiv) const {
  MS_LOG(INFO) << "do fusion, pattern name: " << pattern_name;
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  if (IsMarkedTrainOp(utils::cast<CNodePtr>(node))) {
    return nullptr;
  }
  if (!CheckPattern(equiv)) {
    return nullptr;
  }

  CNodePtr cnode = nullptr;
  if (pattern_name == kAddLayerNormPattern) {
    MS_LOG(INFO) << "start create add layernorm fusion";
    cnode = CreateAddLayerNormNode(func_graph, node, equiv);
  } else if (pattern_name == kLayerNormV3Pattern) {
    MS_LOG(INFO) << "start create layernormv3 fusion";
    cnode = CreateLayerNormV3Node(func_graph, node, equiv);
  } else {
    MS_LOG(ERROR) << "not supported pattern: " << pattern_name;
  }

  if (cnode == nullptr) {
    MS_LOG(INFO) << "new fusion node failed under " << pattern_name;
    return nullptr;
  }
  MS_LOG(INFO) << pattern_name << " fusion success, fusion node name: " << cnode->fullname_with_scope();
  return cnode;
}
}  // namespace opt
}  // namespace mindspore
