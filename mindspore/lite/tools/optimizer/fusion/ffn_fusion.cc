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
#include "tools/optimizer/fusion/ffn_fusion.h"
#include <vector>
#include <unordered_map>
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/core/ops/lite_ops.h"
#include "mindspore/core/ops/custom.h"
#include "ops/f_f_n.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int kStructureNum = 2;
constexpr int DIV1_Y = 2;
constexpr int MUL2_Y = 2;
constexpr float DIFF_THRESHOLD = 0.0001;
constexpr float DIV2_Y = 1.41421;
constexpr float ADD3_Y = 1.0;
constexpr float MUL4_y = 0.5;
constexpr auto kFFNFusion = "FFN_Fusion";
constexpr auto kFFNPatternForConstFolding = "FFNPatternForConstFolding";
constexpr auto kFFNPatternForDynamicDims = "FFNPatternForDynamicDims";
}  // namespace

CNodePtr FFNFusion::CreateFFNFusionNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv,
                                        int index) const {
  auto ffn_fusion_prim = std::make_shared<ops::FFN>();
  MS_CHECK_TRUE_RET(ffn_fusion_prim != nullptr, nullptr);

  ffn_fusion_prim->AddAttr("activation", api::MakeValue("geglu"));
  // "inner_precise" must be 1 when "activation" is "geglu"
  ffn_fusion_prim->AddAttr("inner_precise", api::MakeValue(1));

  auto ffn_fusion_prim_c = ffn_fusion_prim->GetPrim();
  MS_CHECK_TRUE_RET(ffn_fusion_prim_c != nullptr, nullptr);
  auto input = (*equiv)[input_[index]];
  MS_CHECK_TRUE_RET(input != nullptr, nullptr);
  auto input_node = utils::cast<AnfNodePtr>(input);
  MS_CHECK_TRUE_RET(input_node != nullptr, nullptr);
  auto param1 = utils::cast<AnfNodePtr>((*equiv)[matmul1_b_[index]]);
  MS_CHECK_TRUE_RET(param1 != nullptr, nullptr);
  auto param2 = utils::cast<AnfNodePtr>((*equiv)[add1_x_[index]]);
  MS_CHECK_TRUE_RET(param2 != nullptr, nullptr);
  auto param3 = utils::cast<AnfNodePtr>((*equiv)[matmul2_b_[index]]);
  MS_CHECK_TRUE_RET(param3 != nullptr, nullptr);

  auto none_value_node = NewValueNode(std::make_shared<None>());
  none_value_node->set_abstract(std::make_shared<abstract::AbstractNone>());

  auto ffn_fusion_cnode =
    func_graph->NewCNode(ffn_fusion_prim_c, {input_node, param1, param3, none_value_node, param2});
  MS_CHECK_TRUE_RET(ffn_fusion_cnode != nullptr, nullptr);
  ffn_fusion_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_ffn_fusion");
  if (node->abstract() != nullptr) {
    ffn_fusion_cnode->set_abstract(node->abstract()->Clone());
  }
  return ffn_fusion_cnode;
}

bool FFNFusion::Init() const {
  for (int i = 0; i < kMaxPatternNum; i++) {
    input_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(input_[i] != nullptr, false);
    div2_y_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(div2_y_[i] != nullptr, false);
    add3_y_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(add3_y_[i] != nullptr, false);
    mul4_y_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(mul4_y_[i] != nullptr, false);
    matmul1_b_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(matmul1_b_[i] != nullptr, false);
    add1_x_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(add1_x_[i] != nullptr, false);
    matmul2_b_[i] = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(matmul2_b_[i] != nullptr, false);
  }
  gather_y_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(gather_y_ != nullptr, false);
  add2_y_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add2_y_ != nullptr, false);
  div1_y_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(div1_y_ != nullptr, false);
  mul1_y_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul1_y_ != nullptr, false);
  mul2_y_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul2_y_ != nullptr, false);
  return true;
}

const VectorRef FFNFusion::DefineFFNPatternForDynamicDims() const {
  MS_LOG(INFO) << "start define FFN fusion pattern for dynamic dims.";
  const size_t param_num = 6;
  std::vector<CondVarPtr> params(param_num);
  for (size_t i = 0; i < params.size(); ++i) {
    params[i] = std::make_shared<CondVar>(IsParamNode);
    MS_CHECK_TRUE_RET(params[i] != nullptr, {});
  }
  size_t index = 0;
  auto is_matmul1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul1 != nullptr, {});
  VectorRef matmul1_ref({is_matmul1, input_[kDynamicDims], matmul1_b_[kDynamicDims]});
  auto is_add1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add1 != nullptr, {});
  VectorRef add1_ref({is_add1, add1_x_[kDynamicDims], matmul1_ref});
  auto is_shape = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimShape>);
  MS_CHECK_TRUE_RET(is_shape != nullptr, {});
  VectorRef shape_ref({is_shape, add1_ref});
  auto is_gather = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimGather>);
  MS_CHECK_TRUE_RET(is_gather != nullptr, {});
  VectorRef gather_ref({is_gather, shape_ref, gather_y_, params[index++]});
  auto is_add2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add2 != nullptr, {});
  VectorRef add2_ref({is_add2, gather_ref, add2_y_});
  auto is_div1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDivFusion>);
  MS_CHECK_TRUE_RET(is_div1 != nullptr, {});
  VectorRef div1_ref({is_div1, add2_ref, div1_y_});
  auto is_mul1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul1 != nullptr, {});
  VectorRef mul1_ref({is_mul1, div1_ref, mul1_y_});
  auto is_stridedslice1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimStridedSlice>);
  MS_CHECK_TRUE_RET(is_stridedslice1 != nullptr, {});
  VectorRef stridedslice1_ref(
    {is_stridedslice1, add1_ref, params[index++], mul1_ref, params[index++], params[index++]});
  auto is_mul2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul2 != nullptr, {});
  VectorRef mul2_ref({is_mul2, div1_ref, mul2_y_});
  auto is_stridedslice2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimStridedSlice>);
  MS_CHECK_TRUE_RET(is_stridedslice2 != nullptr, {});
  VectorRef stridedslice2_ref({is_stridedslice2, add1_ref, mul1_ref, mul2_ref, params[index++], params[index++]});
  auto is_div2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDivFusion>);
  MS_CHECK_TRUE_RET(is_div2 != nullptr, {});
  VectorRef div2_ref({is_div2, stridedslice2_ref, div2_y_[kDynamicDims]});
  auto is_erf = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimErf>);
  MS_CHECK_TRUE_RET(is_erf != nullptr, {});
  VectorRef erf_ref({is_erf, div2_ref});
  auto is_add3 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add3 != nullptr, {});
  VectorRef add3_ref({is_add3, erf_ref, add3_y_[kDynamicDims]});
  auto is_mul3 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul3 != nullptr, {});
  VectorRef mul3_ref({is_mul3, stridedslice2_ref, add3_ref});
  auto is_mul4 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul4 != nullptr, {});
  VectorRef mul4_ref({is_mul4, mul3_ref, mul4_y_[kDynamicDims]});
  auto is_mul5 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul5 != nullptr, {});
  VectorRef mul5_ref({is_mul5, stridedslice1_ref, mul4_ref});
  auto is_matmul2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul2 != nullptr, {});
  VectorRef matmul2_ref({is_matmul2, mul5_ref, matmul2_b_[kDynamicDims]});
  return matmul2_ref;
}

const VectorRef FFNFusion::DefineFFNPatternForConstFolding() const {
  MS_LOG(INFO) << "start define FFN fusion pattern for const folding.";
  const size_t param_num = 8;
  std::vector<CondVarPtr> params(param_num);
  for (size_t i = 0; i < params.size(); ++i) {
    params[i] = std::make_shared<CondVar>(IsParamNode);
    MS_CHECK_TRUE_RET(params[i] != nullptr, {});
  }
  size_t index = 0;
  auto is_matmul1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul1 != nullptr, {});
  VectorRef matmul1_ref({is_matmul1, input_[kConstFold], matmul1_b_[kConstFold]});
  auto is_add1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add1 != nullptr, {});
  VectorRef add1_ref({is_add1, add1_x_[kConstFold], matmul1_ref});

  auto is_stridedslice1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimStridedSlice>);
  MS_CHECK_TRUE_RET(is_stridedslice1 != nullptr, {});
  VectorRef stridedslice1_ref(
    {is_stridedslice1, add1_ref, params[index++], params[index++], params[index++], params[index++]});

  auto is_stridedslice2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimStridedSlice>);
  MS_CHECK_TRUE_RET(is_stridedslice2 != nullptr, {});
  VectorRef stridedslice2_ref(
    {is_stridedslice2, add1_ref, params[index++], params[index++], params[index++], params[index++]});

  auto is_div2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDivFusion>);
  MS_CHECK_TRUE_RET(is_div2 != nullptr, {});
  VectorRef div2_ref({is_div2, stridedslice2_ref, div2_y_[kConstFold]});
  auto is_erf = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimErf>);
  MS_CHECK_TRUE_RET(is_erf != nullptr, {});
  VectorRef erf_ref({is_erf, div2_ref});
  auto is_add3 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add3 != nullptr, {});
  VectorRef add3_ref({is_add3, erf_ref, add3_y_[kConstFold]});
  auto is_mul3 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul3 != nullptr, {});
  VectorRef mul3_ref({is_mul3, stridedslice2_ref, add3_ref});
  auto is_mul4 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul4 != nullptr, {});
  VectorRef mul4_ref({is_mul4, mul3_ref, mul4_y_[kConstFold]});
  auto is_mul5 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul5 != nullptr, {});
  VectorRef mul5_ref({is_mul5, stridedslice1_ref, mul4_ref});
  auto is_matmul2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMulFusion>);
  MS_CHECK_TRUE_RET(is_matmul2 != nullptr, {});
  VectorRef matmul2_ref({is_matmul2, mul5_ref, matmul2_b_[kConstFold]});
  return matmul2_ref;
}

std::unordered_map<std::string, VectorRef> FFNFusion::DefinePatterns() const {
  MS_LOG(INFO) << "start define FFN fusion patterns.";
  if (!Init()) {
    MS_LOG(ERROR) << "DefinePatterns Init Failed.";
    return {};
  }
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kFFNPatternForConstFolding] = DefineFFNPatternForConstFolding();
  patterns[kFFNPatternForDynamicDims] = DefineFFNPatternForDynamicDims();
  return patterns;
}

bool FFNFusion::CheckPattern(const std::string &pattern_name, const EquivPtr &equiv) const {
  int index = pattern_name == kFFNPatternForDynamicDims ? kDynamicDims : kConstFold;

  float div2_y = GetFloatParameterValue(equiv, div2_y_[index]);
  if (div2_y < 0 || fabs(div2_y - DIV2_Y) > DIFF_THRESHOLD) {
    return false;
  }
  float add3_y = GetFloatParameterValue(equiv, add3_y_[index]);
  if (add3_y < 0 || fabs(add3_y - ADD3_Y) > DIFF_THRESHOLD) {
    return false;
  }
  float mul4_y = GetFloatParameterValue(equiv, mul4_y_[index]);
  if (mul4_y < 0 || fabs(mul4_y - MUL4_y) > DIFF_THRESHOLD) {
    return false;
  }
  if (pattern_name == kFFNPatternForConstFolding) {
    return true;
  }
  // if pattern is for const folding, there are no nodes below, so no need checking.
  int gather_index = GetIntParameterValue(equiv, gather_y_);
  if (gather_index == INT_MIN) {
    return false;
  }
  int add2_y = GetIntParameterValue(equiv, add2_y_);
  if (add2_y != 1) {
    return false;
  }
  int div1_y = GetIntParameterValue(equiv, div1_y_);
  if (div1_y != DIV1_Y) {
    return false;
  }
  int mul1_y = GetIntParameterValue(equiv, mul1_y_);
  if (mul1_y != 1) {
    return false;
  }
  int mul2_y = GetIntParameterValue(equiv, mul2_y_);
  if (mul2_y != MUL2_Y) {
    return false;
  }
  return true;
}

AnfNodePtr FFNFusion::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
                              const mindspore::AnfNodePtr &node, const mindspore::EquivPtr &equiv) const {
  MS_LOG(INFO) << "do fusion, pattern name: " << pattern_name;
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
  if (!CheckPattern(pattern_name, equiv)) {
    MS_LOG(ERROR) << "CheckPattern failed.";
    return nullptr;
  }
  int index = pattern_name == kFFNPatternForDynamicDims ? kDynamicDims : kConstFold;
  auto cnode = CreateFFNFusionNode(func_graph, node, equiv, index);
  if (cnode == nullptr) {
    MS_LOG(INFO) << "new FFN node failed.";
    return nullptr;
  }
  MS_LOG(INFO) << "FFN fusion success, fusion node name: " << cnode->fullname_with_scope();
  return cnode;
}
}  // namespace opt
}  // namespace mindspore
