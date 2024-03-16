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
#include "tools/optimizer/fusion/groupnormsilu_fusion.h"
#include <memory>
#include "ops/array_ops.h"
#include "ops/nn_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/core/ops/lite_ops.h"
#include "ops/auto_generate/gen_lite_ops.h"
#include "ops/custom.h"
#include "ops/group_norm_silu.h"

namespace mindspore::opt {
namespace {
constexpr auto kNameGroupNormSiluPatternForSD15 = "GroupNormSiluPatternForSD15";
constexpr auto kNameGroupNormSiluPatternForSDWithCast = "GroupNormSiluPatternForSDWithCast";
constexpr size_t kNumIndex1 = 1;
constexpr size_t kNumIndex2 = 2;
constexpr float kNumEps = 0.00001;

std::vector<int64_t> GetTensorShape(CNodePtr cnode, size_t input_index) {
  auto abstract = GetCNodeInputAbstract(cnode, input_index);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "GetCNodeInputAbstract in GroupNormSilu fusion.";
    return {};
  }
  std::vector<int64_t> shape = {};
  if (FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "FetchShapeFromAbstract failed.";
    return {};
  }
  return shape;
}

int64_t GetInstanceNormGroups(const AnfNodePtr &instance_norm_node) {
  auto instance_norm_cnode = instance_norm_node->cast<CNodePtr>();
  if (instance_norm_cnode == nullptr) {
    MS_LOG(WARNING) << "instance_norm_cnode is nullptr.";
    return -1;
  }
  auto instance_norm_input2 = instance_norm_cnode->input(kNumIndex2);
  if (instance_norm_input2 == nullptr) {
    MS_LOG(WARNING) << "instance_norm_input2 is nullptr.";
    return -1;
  }
  auto scale_param = instance_norm_input2->cast<ParameterPtr>();
  if (scale_param == nullptr) {
    MS_LOG(WARNING) << "scale_param is nullptr.";
    return -1;
  }
  auto scale_default_param = scale_param->default_param();
  auto scale_value = std::dynamic_pointer_cast<tensor::Tensor>(scale_default_param);
  if (scale_value == nullptr) {
    MS_LOG(WARNING) << "scale_value is nullptr.";
    return -1;
  }
  return static_cast<int64_t>(scale_value->ElementsNum());
}
}  // namespace

std::unordered_map<std::string, VectorRef> GroupNormSiluFusion::DefinePatterns() const {
  MS_LOG(INFO) << "start define flash attention fusion patterns.";
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kNameGroupNormSiluPatternForSD15] = DefineGroupNormSiluPatternForSD15();
  patterns[kNameGroupNormSiluPatternForSDWithCast] = DefineGroupNormSiluPatternForSDWithCast();
  return patterns;
}

const VectorRef GroupNormSiluFusion::DefineGroupNormSiluPatternForSD15() const {
  // reshape
  auto reshape_1_input_1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_1_input_1 != nullptr, {});
  auto reshape_1_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_1_input_2 != nullptr, {});
  auto is_reshape_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape_1 != nullptr, {});
  auto reshape_1 = VectorRef({is_reshape_1, reshape_1_input_1, reshape_1_input_2});

  // instanceNormalization
  auto instance_norm_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(instance_norm_input_2 != nullptr, {});
  auto instance_norm_input_3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(instance_norm_input_3 != nullptr, {});
  auto is_instance_norm = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimInstanceNorm>);
  MS_CHECK_TRUE_RET(is_instance_norm != nullptr, {});
  auto instance_norm = VectorRef({is_instance_norm, reshape_1, instance_norm_input_2, instance_norm_input_3});

  // reshape
  auto reshape_2_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_2_input_2 != nullptr, {});
  auto is_reshape_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape_2 != nullptr, {});
  auto reshape_2 = VectorRef({is_reshape_2, instance_norm, reshape_2_input_2});

  // mul
  auto mul_1_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul_1_input_2 != nullptr, {});
  auto is_mul_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul_1 != nullptr, {});
  auto mul_1 = VectorRef({is_mul_1, reshape_2, mul_1_input_2});

  // add
  auto add_input_2 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(add_input_2 != nullptr, {});
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto add = VectorRef({is_add, mul_1, add_input_2});

  // sigmoid
  auto is_sigmoid = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  MS_CHECK_TRUE_RET(is_sigmoid != nullptr, {});
  auto sigmoid = VectorRef({is_sigmoid, add});

  // mul
  auto is_mul_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul_2 != nullptr, {});
  auto mul_2 = VectorRef({is_mul_2, add, sigmoid});
  return mul_2;
}

const VectorRef GroupNormSiluFusion::DefineGroupNormSiluPatternForSDWithCast() const {
  // cast
  auto cast_1_input = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(cast_1_input != nullptr, {});
  auto is_cast_1_param = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_cast_1_param != nullptr, {});
  auto is_cast_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast_1 != nullptr, {});
  auto cast_1 = VectorRef({is_cast_1, cast_1_input, is_cast_1_param});

  // reshape
  auto reshape_1_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_1_input_2 != nullptr, {});
  auto is_reshape_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape_1 != nullptr, {});
  auto reshape_1 = VectorRef({is_reshape_1, cast_1, reshape_1_input_2});

  // instanceNormalization
  auto instance_norm_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(instance_norm_input_2 != nullptr, {});
  auto instance_norm_input_3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(instance_norm_input_3 != nullptr, {});
  auto is_instance_norm = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimInstanceNorm>);
  MS_CHECK_TRUE_RET(is_instance_norm != nullptr, {});
  auto instance_norm = VectorRef({is_instance_norm, reshape_1, instance_norm_input_2, instance_norm_input_3});

  // reshape
  auto reshape_2_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_2_input_2 != nullptr, {});
  auto is_reshape_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape_2 != nullptr, {});
  auto reshape_2 = VectorRef({is_reshape_2, instance_norm, reshape_2_input_2});

  // mul
  auto mul_1_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul_1_input_2 != nullptr, {});
  auto is_mul_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul_1 != nullptr, {});
  auto mul_1 = VectorRef({is_mul_1, reshape_2, mul_1_input_2});

  // add
  auto add_input_2 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(add_input_2 != nullptr, {});
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto add = VectorRef({is_add, mul_1, add_input_2});

  // cast
  auto is_cast_2_param = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_cast_2_param != nullptr, {});
  auto is_cast_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast_2 != nullptr, {});
  auto cast_2 = VectorRef({is_cast_2, add, is_cast_2_param});

  // sigmoid
  auto is_sigmoid = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  MS_CHECK_TRUE_RET(is_sigmoid != nullptr, {});
  auto sigmoid = VectorRef({is_sigmoid, cast_2});

  // mul
  auto is_mul_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul_2 != nullptr, {});
  auto mul_2 = VectorRef({is_mul_2, cast_2, sigmoid});
  return mul_2;
}

CNodePtr GroupNormSiluFusion::ReshapeCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  // reshape [x, 1, 1] to [x]
  std::vector<int32_t> shape_1d = {0};
  auto shape_param_node = BuildIntVecParameterNode(func_graph, shape_1d, node->fullname_with_scope() + "_shape_param");
  MS_CHECK_TRUE_MSG(shape_param_node != nullptr, nullptr, "create shape_param_node return nullptr");

  std::vector<AnfNodePtr> op_inputs;
  if (utils::isa<ParameterPtr>(node)) {
    auto reshape_input_1 = node->cast<ParameterPtr>();
    op_inputs = {reshape_input_1, shape_param_node};
  } else {
    MS_LOG(ERROR) << "node is not ParameterPtr.";
    return nullptr;
  }

  auto reshape_prim = std::make_shared<ops::Reshape>();
  MS_CHECK_TRUE_MSG(reshape_prim != nullptr, nullptr, "create reshape_prim return nullptr");
  auto reshape_prim_c = reshape_prim->GetPrim();
  MS_CHECK_TRUE_MSG(reshape_prim_c != nullptr, nullptr, "create prim_c return nullptr");
  auto reshape_node = func_graph->NewCNode(reshape_prim_c, op_inputs);
  MS_CHECK_TRUE_MSG(reshape_node != nullptr, nullptr, "create node return nullptr");

  reshape_node->set_fullname_with_scope(node->fullname_with_scope() + "_GNS_reshape");
  if (node->abstract() != nullptr) {
    reshape_node->set_abstract(node->abstract()->Clone());
  }
  auto manager = Manage(func_graph);
  (void)manager->Replace(node, reshape_node);
  MS_LOG(INFO) << "GroupNormSiluFusion create reshape node end.";
  return reshape_node;
}

CNodePtr GroupNormSiluFusion::CreateGroupNormSiluNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                      const AnfNodePtr &conv, const AnfNodePtr &gamma_3D,
                                                      const AnfNodePtr &beta_3D, int64_t num_groups) const {
  MS_LOG(INFO) << "create GroupNormSilu node";

  auto gamma_1D = ReshapeCNode(func_graph, gamma_3D);
  MS_CHECK_TRUE_RET(gamma_1D != nullptr, nullptr);
  auto beta_1D = ReshapeCNode(func_graph, beta_3D);
  MS_CHECK_TRUE_RET(beta_1D != nullptr, nullptr);

  auto groupnorm_silu_prim = std::make_shared<ops::GroupNormSilu>();
  if (groupnorm_silu_prim == nullptr) {
    MS_LOG(ERROR) << "new GroupNormSilu prim failed.";
    return nullptr;
  }
  groupnorm_silu_prim->AddAttr("num_groups", api::MakeValue(num_groups));
  groupnorm_silu_prim->AddAttr("eps", api::MakeValue(kNumEps));

  auto GNS_prim_c = groupnorm_silu_prim->GetPrim();
  if (GNS_prim_c == nullptr) {
    MS_LOG(ERROR) << "GNS_prim_c is nullptr.";
    return nullptr;
  }

  auto groupnorm_silu_cnode = func_graph->NewCNode(GNS_prim_c, {conv, gamma_1D, beta_1D});
  if (groupnorm_silu_cnode == nullptr) {
    MS_LOG(ERROR) << "new groupnormsilu cnode failed.";
    return nullptr;
  }
  groupnorm_silu_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_groupnormsilu_sd");
  if (node->abstract() != nullptr) {
    groupnorm_silu_cnode->set_abstract(node->abstract()->Clone());
  }
  return groupnorm_silu_cnode;
}

CNodePtr GroupNormSiluFusion::CreateGroupNormSiluNodeForSD15(const std::string &pattern_name,
                                                             const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                             const EquivPtr &equiv) const {
  MS_LOG(INFO) << "GroupNormSilu for SD15";
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  auto cnode = node->cast<CNodePtr>();  // mul_2
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);

  auto add = cnode->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add != nullptr, nullptr);

  auto mul_1 = add->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(mul_1 != nullptr, nullptr);

  auto reshape_2 = mul_1->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_2 != nullptr, nullptr);

  auto instance_normalization = reshape_2->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(instance_normalization != nullptr, nullptr);

  auto reshape_1 = instance_normalization->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_1 != nullptr, nullptr);

  auto conv = reshape_1->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(conv != nullptr, nullptr);

  auto gamma_3D = mul_1->input(kNumIndex2);
  MS_CHECK_TRUE_RET(gamma_3D != nullptr, nullptr);

  auto beta_3D = add->input(kNumIndex2);
  MS_CHECK_TRUE_RET(beta_3D != nullptr, nullptr);

  // get instancenorm input2 scale shape
  auto num_groups = GetInstanceNormGroups(instance_normalization);
  if (num_groups == -1) {
    MS_LOG(ERROR) << "get num_groups failed";
    return nullptr;
  }
  auto conv_output_shape = GetTensorShape(reshape_1, kNumIndex1);
  MS_LOG(INFO) << "num_groups: " << num_groups << ", conv_output_shape: " << conv_output_shape;
  if (std::find(conv_output_shape.begin(), conv_output_shape.end(), -1) != conv_output_shape.end()) {
    MS_LOG(WARNING) << "GroupNormSilu is not support dynamic shape in CANN";
    return nullptr;
  }

  auto groupnorm_silu_cnode = CreateGroupNormSiluNode(func_graph, node, conv, gamma_3D, beta_3D, num_groups);
  if (groupnorm_silu_cnode == nullptr) {
    MS_LOG(WARNING) << "create groupnorm_silu_cnode failed";
    return nullptr;
  }

  auto manager = Manage(func_graph);
  (void)manager->Replace(cnode, groupnorm_silu_cnode);
  MS_LOG(INFO) << "create GroupNormSilu for SD15 success.";
  return groupnorm_silu_cnode;
}

CNodePtr GroupNormSiluFusion::CreateGroupNormSiluNodeForSDWithCast(const std::string &pattern_name,
                                                                   const FuncGraphPtr &func_graph,
                                                                   const AnfNodePtr &node,
                                                                   const EquivPtr &equiv) const {
  MS_LOG(INFO) << "GroupNormSilu with cast";
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  auto cnode = node->cast<CNodePtr>();  // mul_2

  auto cast = cnode->input(1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cast != nullptr, nullptr);

  auto add = cast->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add != nullptr, nullptr);

  auto mul_1 = add->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(mul_1 != nullptr, nullptr);

  auto reshape_2 = mul_1->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_2 != nullptr, nullptr);

  auto instance_normalization = reshape_2->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(instance_normalization != nullptr, nullptr);

  auto reshape_1 = instance_normalization->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_1 != nullptr, nullptr);

  auto cast_1 = reshape_1->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cast_1 != nullptr, nullptr);

  auto conv = cast_1->input(kNumIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(conv != nullptr, nullptr);

  auto gamma_3D = mul_1->input(kNumIndex2);
  MS_CHECK_TRUE_RET(gamma_3D != nullptr, nullptr);

  auto beta_3D = add->input(kNumIndex2);
  MS_CHECK_TRUE_RET(beta_3D != nullptr, nullptr);

  // get instancenorm input2 scale shape
  auto num_groups = GetInstanceNormGroups(instance_normalization);
  if (num_groups == -1) {
    MS_LOG(ERROR) << "get num_groups failed";
    return nullptr;
  }
  auto conv_output_shape = GetTensorShape(reshape_1, kNumIndex1);
  MS_LOG(INFO) << "num_groups: " << num_groups << ", conv_output_shape: " << conv_output_shape;
  if (std::find(conv_output_shape.begin(), conv_output_shape.end(), -1) != conv_output_shape.end()) {
    MS_LOG(WARNING) << "GroupNormSilu is not support dynamic shape in CANN";
    return nullptr;
  }

  auto groupnorm_silu_cnode = CreateGroupNormSiluNode(func_graph, node, conv, gamma_3D, beta_3D, num_groups);
  if (groupnorm_silu_cnode == nullptr) {
    MS_LOG(WARNING) << "create groupnorm_silu_cnode failed";
    return nullptr;
  }

  auto manager = Manage(func_graph);
  (void)manager->Replace(cnode, groupnorm_silu_cnode);
  MS_LOG(INFO) << "create GroupNormSilu with cast success.";
  return groupnorm_silu_cnode;
}

AnfNodePtr GroupNormSiluFusion::Process(const std::string &patten_name, const FuncGraphPtr &func_graph,
                                        const AnfNodePtr &node, const EquivPtr &equiv) const {
  MS_LOG(INFO) << "do GroupNormSilu fusion, pattern name: " << patten_name;
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    MS_LOG(ERROR) << "function graph, node or equiv is nullptr.";
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    MS_LOG(ERROR) << "this node is not cnode, node name: " << node->fullname_with_scope();
    return nullptr;
  }
  if (IsMarkedTrainOp(utils::cast<CNodePtr>(node))) {
    MS_LOG(ERROR) << "node is train op, can not fusion.";
    return nullptr;
  }
  auto manager = Manage(func_graph);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return nullptr;
  }
  CNodePtr groupnormsilu_node = nullptr;
  if (patten_name == kNameGroupNormSiluPatternForSD15) {
    MS_LOG(INFO) << "start create GroupNormSilu node for SD15.";
    groupnormsilu_node = CreateGroupNormSiluNodeForSD15(patten_name, func_graph, node, equiv);
  } else if (patten_name == kNameGroupNormSiluPatternForSDWithCast) {
    MS_LOG(INFO) << "start create GroupNormSilu node for SD15 with cast.";
    groupnormsilu_node = CreateGroupNormSiluNodeForSDWithCast(patten_name, func_graph, node, equiv);
  } else {
    MS_LOG(ERROR) << " not pattern.";
  }
  if (groupnormsilu_node == nullptr) {
    MS_LOG(INFO) << "GroupNormSilu not fusion.";
    return nullptr;
  }
  MS_LOG(INFO) << "GroupNormSilu node fusion success, fusion node name: " << groupnormsilu_node->fullname_with_scope();
  return groupnormsilu_node;
}
}  // namespace mindspore::opt
