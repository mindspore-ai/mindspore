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
#include <algorithm>
#include "ops/nn_ops.h"
#include "ops/fusion/activation.h"
#include "ops/lite_ops.h"
#include "ops/add_layernorm.h"
#include "ops/fusion/layer_norm_fusion.h"
#include "ops/auto_generate/gen_lite_ops.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "tools/optimizer/graph/node_infershape.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int kAxis = -1;
constexpr float kPow = 2.0;
constexpr float kEps = 1e-5;
constexpr float kDiffThreshold = 1e-6;
constexpr auto kLayerNormV3Pattern1 = "LayerNormV3Fusion1";
constexpr auto kLayerNormV3Pattern2 = "LayerNormV3Fusionf2";
constexpr int kReduceAxisNum = -1;
constexpr int kInputIndex1 = 1;
constexpr int kInputIndex2 = 2;
constexpr int kInvalidDim = -1;
constexpr int kAdditionalOutIdx = 3;
constexpr int kStructureNum = 2;
constexpr int kLayerNormOutNum = 3;
}  // namespace

bool LayerNormFusionInferShape(const AnfNodePtr &layernorm_node, const AnfNodePtr &layernorm_input) {
  auto node_prim = std::make_shared<ops::LayerNormFusion>()->GetPrim();
  if (!IsPrimitiveCNode(layernorm_node, node_prim)) {
    MS_LOG(WARNING) << "This func can only infer LayerNormFusion op, but got " << node_prim->ToString();
    return true;
  }
  auto cnode = layernorm_node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return true;
  }
  AbstractBasePtrList abstracts;
  if (layernorm_input->abstract() != nullptr) {
    abstracts.emplace_back(layernorm_input->abstract()->Clone());
    abstracts.emplace_back(layernorm_input->abstract()->Clone());
    abstracts.emplace_back(layernorm_input->abstract()->Clone());
    auto new_abstract_list = std::make_shared<abstract::AbstractTuple>(abstracts);
    layernorm_node->set_abstract(new_abstract_list->Clone());
  }
  NodeInferShape infer;
  (void)infer.InferShape(cnode);
  return true;
}

AnfNodePtr LayerNormV3Fusion::CreateLayerNormV3Node(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  auto lnfusion_prim = std::make_shared<ops::LayerNormFusion>();
  MS_CHECK_TRUE_RET(lnfusion_prim != nullptr, nullptr);

  lnfusion_prim->AddAttr("begin_norm_axis", api::MakeValue(kReduceAxisNum));
  lnfusion_prim->AddAttr("begin_params_axis", api::MakeValue(kReduceAxisNum));

  auto lnfusion_prim_c = lnfusion_prim->GetPrim();
  MS_CHECK_TRUE_RET(lnfusion_prim_c != nullptr, nullptr);

  auto x = utils::cast<AnfNodePtr>((*equiv)[reduce_1_x_[index_]]);
  MS_CHECK_TRUE_RET(x != nullptr, nullptr);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[mul_b_[index_]]);
  MS_CHECK_TRUE_RET(gamma != nullptr, nullptr);
  auto beta = utils::cast<AnfNodePtr>((*equiv)[add_3_b_[index_]]);
  MS_CHECK_TRUE_RET(beta != nullptr, nullptr);

  auto ln_cnode = func_graph->NewCNode(lnfusion_prim_c, {x, gamma, beta});
  if (!LayerNormFusionInferShape(ln_cnode, node)) {
    return node;
  }
  MS_CHECK_TRUE_RET(ln_cnode != nullptr, nullptr);
  ln_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_layernormv3_fusion");

  auto pre_node = ln_cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(pre_node != nullptr, nullptr);
  MS_LOG(INFO) << "cnode before LayerNormV3: " << pre_node->fullname_with_scope();

  auto infer_abs = ln_cnode->abstract()->Clone();
  MS_CHECK_TRUE_RET(infer_abs != nullptr, nullptr);
  ln_cnode->set_abstract(infer_abs);
  auto infer_abs_tuple = infer_abs->cast<abstract::AbstractTuplePtr>();
  MS_CHECK_TRUE_RET(infer_abs_tuple != nullptr, nullptr);
  // change LayerNormFusion to LayerNormV3
  auto lnv3_new_prim = std::make_shared<ops::LayerNormV3>();
  lnv3_new_prim->AddAttr("begin_norm_axis", api::MakeValue(kReduceAxisNum));
  lnv3_new_prim->AddAttr("begin_params_axis", api::MakeValue(kReduceAxisNum));

  ln_cnode->set_input(0, NewValueNode(lnv3_new_prim->GetPrim()));
  auto prim_getitem = std::make_shared<Primitive>("TupleGetItem");
  auto get_item_result = func_graph->NewCNode(prim_getitem, {ln_cnode, NewValueNode(static_cast<int64_t>(0))});
  constexpr const size_t kNumZero = 0;
  get_item_result->set_abstract(infer_abs_tuple->elements()[kNumZero]);
  return get_item_result;
}

bool LayerNormV3Fusion::Init() const {
  reduce_1_x_.resize(kStructureNum);
  reduce_1_axis_.resize(kStructureNum);
  sub_a_.resize(kStructureNum);
  pow_y_.resize(kStructureNum);
  reduce_2_axis_.resize(kStructureNum);
  add_2_b_.resize(kStructureNum);
  mul_b_.resize(kStructureNum);
  add_3_b_.resize(kStructureNum);
  cast_to_.resize(kStructureNum);
  for (int i = 0; i < kStructureNum; ++i) {
    reduce_1_x_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(reduce_1_x_[i] != nullptr, false);
    reduce_1_axis_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(reduce_1_axis_[i] != nullptr, false);
    sub_a_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(sub_a_[i] != nullptr, false);
    pow_y_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(pow_y_[i] != nullptr, false);
    reduce_2_axis_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(reduce_2_axis_[i] != nullptr, false);
    add_2_b_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(add_2_b_[i] != nullptr, false);
    mul_b_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(mul_b_[i] != nullptr, false);
    add_3_b_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(add_3_b_[i] != nullptr, false);
    cast_to_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(cast_to_[i] != nullptr, false);
  }

  return true;
}

const VectorRef LayerNormV3Fusion::DefineLayerNormV3Pattern1() const {
  const int index = 0;
  auto is_reduce_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReduceFusion>);
  VectorRef reduce_1_ref({is_reduce_1, reduce_1_x_[index], reduce_1_axis_[index]});

  auto is_sub = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSubFusion>);
  VectorRef sub_ref({is_sub, reduce_1_x_[index], reduce_1_ref});

  auto is_pow = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimPowFusion>);
  VectorRef pow_ref({is_pow, sub_ref, pow_y_[index]});

  auto is_reduce_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReduceFusion>);
  VectorRef reduce_2_ref({is_reduce_2, pow_ref, reduce_2_axis_[index]});

  auto is_add_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  VectorRef add_2_ref({is_add_2, reduce_2_ref, add_2_b_[index]});

  auto is_sqrt = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSqrt>);
  VectorRef sqrt_ref({is_sqrt, add_2_ref});

  auto is_div = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDivFusion>);
  VectorRef div_ref({is_div, sub_ref, sqrt_ref});

  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  VectorRef mul_ref({is_mul, div_ref, mul_b_[index]});

  auto is_add_3 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  VectorRef add_3_ref({is_add_3, mul_ref, add_3_b_[index]});
  return add_3_ref;
}

const VectorRef LayerNormV3Fusion::DefineLayerNormV3Pattern2() const {
  const int index = 1;
  auto is_reduce_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReduceFusion>);
  VectorRef reduce_1_ref({is_reduce_1, reduce_1_x_[index], reduce_1_axis_[index]});

  auto is_sub = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSubFusion>);
  VectorRef sub_ref({is_sub, reduce_1_x_[index], reduce_1_ref});

  auto is_pow = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimPowFusion>);
  VectorRef pow_ref({is_pow, sub_ref, pow_y_[index]});

  auto is_reduce_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReduceFusion>);
  VectorRef reduce_2_ref({is_reduce_2, pow_ref, reduce_2_axis_[index]});

  auto is_add_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  VectorRef add_2_ref({is_add_2, reduce_2_ref, add_2_b_[index]});

  auto is_sqrt = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSqrt>);
  VectorRef sqrt_ref({is_sqrt, add_2_ref});

  auto is_div = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDivFusion>);
  VectorRef div_ref({is_div, sub_ref, sqrt_ref});

  auto is_cast = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  VectorRef cast_ref({is_cast, div_ref, cast_to_[index]});

  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  VectorRef mul_ref({is_mul, mul_b_[index], cast_ref});

  auto is_add_3 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  VectorRef add_3_ref({is_add_3, mul_ref, add_3_b_[index]});
  return add_3_ref;
}

bool LayerNormV3Fusion::CheckPattern(const EquivPtr &equiv) const {
  MS_ASSERT(equiv != nullptr);
  int reduce_1_axis = GetIntParameterValue(equiv, reduce_1_axis_[index_]);
  if (reduce_1_axis == INT_MIN) {
    MS_LOG(INFO) << "not supported axis: " << reduce_1_axis;
    return false;
  }

  float pow_y = GetFloatParameterValue(equiv, pow_y_[index_]);
  if (pow_y <= 0 || fabs(pow_y - kPow) > kDiffThreshold) {
    MS_LOG(INFO) << "not supported pow: " << pow_y;
    return false;
  }

  int reduce_2_axis = GetIntParameterValue(equiv, reduce_2_axis_[index_]);
  if (reduce_2_axis == INT_MIN) {
    MS_LOG(INFO) << "not supported axis: " << reduce_2_axis;
    return false;
  }

  float add_2_b = GetFloatParameterValue(equiv, add_2_b_[index_]);
  if (add_2_b <= 0 || fabs(add_2_b - kEps) > kDiffThreshold) {
    MS_LOG(INFO) << "not supported bias: " << add_2_b;
    return false;
  }
  return true;
}

std::unordered_map<std::string, VectorRef> LayerNormV3Fusion::DefinePatterns() const {
  MS_LOG(INFO) << "start define LayerNormV3 fusion patterns.";
  if (!Init()) {
    MS_LOG(ERROR) << "LayerNormV3 pattern Init Failed.";
    return {};
  }
  MS_LOG(INFO) << "start define add layernorm fusion patterns.";
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kLayerNormV3Pattern1] = DefineLayerNormV3Pattern1();
  patterns[kLayerNormV3Pattern2] = DefineLayerNormV3Pattern2();
  return patterns;
}

AnfNodePtr LayerNormV3Fusion::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
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
  std::unordered_map<std::string, int> pattern2index = {{kLayerNormV3Pattern1, 0}, {kLayerNormV3Pattern2, 1}};
  if (pattern2index.find(pattern_name) == pattern2index.end()) {
    MS_LOG(WARNING) << "Current pattern doesn't support.";
    return nullptr;
  }
  index_ = pattern2index[pattern_name];
  if (!CheckPattern(equiv)) {
    return nullptr;
  }

  AnfNodePtr cnode = nullptr;
  cnode = CreateLayerNormV3Node(func_graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(INFO) << "new fusion node failed under " << pattern_name;
    return nullptr;
  }
  MS_LOG(INFO) << pattern_name << " fusion success, fusion node name: " << cnode->fullname_with_scope();
  return cnode;
}

const BaseRef FuseAddAndLayernorm::DefinePattern() const {
  VectorRef add_layer_norm = VectorRef({layer_norm_, VectorRef({prim::kPrimAddFusion, x1_, x2_}), gamma_, beta_});
  return add_layer_norm;
}

const AnfNodePtr FuseAddAndLayernorm::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                              const EquivPtr &equiv) const {
  MS_LOG(INFO) << "FuseAddAndLayernorm start processing";
  MS_CHECK_TRUE_RET(graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  MS_CHECK_TRUE_RET(equiv != nullptr, nullptr);
  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto x2 = utils::cast<AnfNodePtr>((*equiv)[x2_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto beta = utils::cast<AnfNodePtr>((*equiv)[beta_]);
  MS_CHECK_TRUE_RET(x1 != nullptr, nullptr);
  MS_CHECK_TRUE_RET(x2 != nullptr, nullptr);
  MS_CHECK_TRUE_RET(gamma != nullptr, nullptr);
  MS_CHECK_TRUE_RET(beta != nullptr, nullptr);
  // input 1 is add node
  auto cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  auto tensor_add = cnode->input(kInputIndex1);

  FuncGraphManagerPtr manager = graph->manager();
  auto cnode_outputs = manager->node_users()[tensor_add];

  auto add_layernorm_op = std::make_shared<ops::AddLayerNorm>();
  MS_CHECK_TRUE_RET(add_layernorm_op != nullptr, nullptr);

  MS_LOG(INFO) << "cnode_outputs.size: " << cnode_outputs.size();
  bool additional_output = cnode_outputs.size() > 1 ? true : false;
  add_layernorm_op->AddAttr("additional_output", api::MakeValue(additional_output));

  auto prim = add_layernorm_op->GetPrim();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x1, x2, gamma, beta};
  auto add_layernorm = graph->NewCNode(inputs);
  MS_CHECK_TRUE_RET(add_layernorm != nullptr, nullptr);
  auto layernorm_node = opt::GetAnfNodeByVar(equiv, layer_norm_);
  auto layernorm_abs = layernorm_node->abstract();
  if (!layernorm_abs->isa<abstract::AbstractTuple>()) {
    return node;
  }

  auto layernorm_abs_tuple = utils::cast<abstract::AbstractTuplePtr>(layernorm_abs);
  auto layer_norm_output_size = layernorm_abs_tuple->size();
  AbstractBasePtrList abstracts;
  std::transform(layernorm_abs_tuple->elements().begin(), layernorm_abs_tuple->elements().end(),
                 std::back_inserter(abstracts), [](const AbstractBasePtr &abs) { return (abs->Clone()); });
  if (layer_norm_output_size < kLayerNormOutNum) {
    if (!LayerNormFusionInferShape(layernorm_node, tensor_add)) {
      return node;
    }
  }
  abstracts.emplace_back(tensor_add->abstract()->Clone());
  auto new_abstract_list = std::make_shared<abstract::AbstractTuple>(abstracts);
  add_layernorm->set_abstract(new_abstract_list);

  if (cnode_outputs.size() > 1) {
    auto prim_getitem = std::make_shared<Primitive>("TupleGetItem");
    // the add output
    auto add_result =
      graph->NewCNode(prim_getitem, {add_layernorm, NewValueNode(static_cast<int64_t>(kAdditionalOutIdx))});
    add_result->set_abstract(tensor_add->abstract()->Clone());
    (void)manager->Replace(tensor_add, add_result);
  }

  MS_LOG(INFO) << "FuseAddAndLayernorm process success";
  return add_layernorm;
}

}  // namespace opt
}  // namespace mindspore
