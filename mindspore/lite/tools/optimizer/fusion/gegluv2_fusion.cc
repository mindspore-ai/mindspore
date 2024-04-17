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
#include "tools/optimizer/fusion/gegluv2_fusion.h"
#include <memory>
#include <utility>
#include "ops/op_utils.h"
#include "ops/array_ops.h"
#include "ops/nn_ops.h"
#include "ops/custom.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/core/ops/lite_ops.h"
#include "ops/slice.h"
#include "ops/auto_generate/gen_lite_ops.h"

namespace mindspore::opt {
namespace {
constexpr auto kNameGeGluV2Pattern = "GeGluV2Pattern";

constexpr int kNumDim = -1;
constexpr size_t kNumUseTanh = 1;
}  // namespace

std::unordered_map<std::string, VectorRef> GeGluV2Fusion::DefinePatterns() const {
  MS_LOG(INFO) << "start define GeGluV2 fusion patterns.";
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kNameGeGluV2Pattern] = DefineGeGluV2Pattern();
  return patterns;
}

const VectorRef GeGluV2Fusion::DefineGeGluV2Pattern() const {
  auto add_input_1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add_input_1 != nullptr, {});
  auto add_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add_input_2 != nullptr, {});
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddFusion>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto add_node = VectorRef({is_add, add_input_1, add_input_2});
  MS_CHECK_TRUE_RET(add_node != nullptr, {});

  auto is_seq_var1 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var1 != nullptr, {});
  auto is_stride_slice_node = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimStridedSlice>);
  MS_CHECK_TRUE_RET(is_stride_slice_node != nullptr, {});
  auto stride_slice_node = VectorRef({is_stride_slice_node, add_node, is_seq_var1});
  MS_CHECK_TRUE_RET(stride_slice_node != nullptr, {});

  auto mul_param = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul_param != nullptr, {});
  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  // activate_left=false means mul's input_1 is x, and input_2 is gelu.
  auto is_mul_node = VectorRef({is_mul, stride_slice_node, mul_param});
  MS_CHECK_TRUE_RET(is_mul_node != nullptr, {});
  return is_mul_node;
}

CNodePtr GeGluV2Fusion::CreateGeGluV2Node(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const AnfNodePtr &add_output) const {
  MS_LOG(INFO) << "CreateGeGluV2Cnode";
  auto gegluv2_prim = std::make_shared<ops::Custom>();
  if (gegluv2_prim == nullptr) {
    MS_LOG(ERROR) << "new gegluv2 prim failed.";
    return nullptr;
  }
  gegluv2_prim->AddAttr("dim", api::MakeValue(kNumDim));
  gegluv2_prim->AddAttr("approximate", api::MakeValue(kNumUseTanh));
  gegluv2_prim->AddAttr("activate_left", api::MakeValue(false));
  std::vector<std::string> input_names = {"x"};
  std::vector<std::string> output_names = {"y", "gelu"};
  gegluv2_prim->set_type("GeGluV2");
  gegluv2_prim->AddAttr("input_names", api::MakeValue(input_names));
  gegluv2_prim->AddAttr("output_names", api::MakeValue(output_names));
  gegluv2_prim->AddAttr("reg_op_name", api::MakeValue("GeGluV2"));

  auto gegluv2_prim_c = gegluv2_prim->GetPrim();
  if (gegluv2_prim_c == nullptr) {
    MS_LOG(ERROR) << "gegluv2_prim_c is nullptr.";
    return nullptr;
  }
  CNodePtr gegluv2_cnode = nullptr;
  gegluv2_cnode = func_graph->NewCNode(gegluv2_prim_c, {add_output});
  if (gegluv2_cnode == nullptr) {
    MS_LOG(ERROR) << "new cnode failed.";
    return nullptr;
  }
  gegluv2_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_gegluv2");
  if (node->abstract() != nullptr) {
    gegluv2_cnode->set_abstract(node->abstract()->Clone());
  }
  MS_LOG(INFO) << "create gegluv2 success.";
  return gegluv2_cnode;
}

AnfNodePtr GeGluV2Fusion::Process(const std::string &patten_name, const FuncGraphPtr &func_graph,
                                  const AnfNodePtr &node, const EquivPtr &equiv) const {
  MS_LOG(INFO) << "do GeGluV2 fusion, pattern name: " << patten_name << "   " << node->fullname_with_scope();
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
  CNodePtr GeGluV2_node = nullptr;
  if (patten_name == kNameGeGluV2Pattern) {
    MS_LOG(INFO) << "start create GeGluV2";
    auto mul_node = node->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(mul_node != nullptr, nullptr);
    auto slice_node = mul_node->input(1)->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(slice_node != nullptr, nullptr);
    auto add_node = slice_node->input(1)->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(add_node != nullptr, nullptr);
    GeGluV2_node = CreateGeGluV2Node(func_graph, node, add_node);
    if (GeGluV2_node == nullptr) {
      MS_LOG(ERROR) << "new_node is nullptr.";
      return nullptr;
    }
    manager->Replace(node, GeGluV2_node);
  } else {
    MS_LOG(INFO) << " not pattern.";
  }
  if (GeGluV2_node == nullptr) {
    MS_LOG(INFO) << "GeGluV2 op not fusion.";
    return nullptr;
  }
  MS_LOG(INFO) << "GeGluV2 node fusion success, fusion node name: " << GeGluV2_node->fullname_with_scope();
  return GeGluV2_node;
}

}  // namespace mindspore::opt
