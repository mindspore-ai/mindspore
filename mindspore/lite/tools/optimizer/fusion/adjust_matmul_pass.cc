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
#include "tools/optimizer/fusion/adjust_matmul_pass.h"
#include <memory>
#include <vector>
#include "ops/resize.h"
#include "ops/op_utils.h"
#include "mindspore/lite/tools/common/tensor_util.h"
#include "mindspore/core/ops/lite_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/auto_generate/gen_lite_ops.h"
#include "ops/squeeze.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/range_v2.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int32_t kShapeMinus_1 = -1;
constexpr size_t kShape_1 = 1;
constexpr size_t kInputIndex_0 = 0;
constexpr size_t kInputIndex_1 = 1;
constexpr size_t kInputIndex_2 = 2;
constexpr size_t kAxis_0 = 0;
constexpr size_t kSize_3 = 3;
constexpr size_t kSize_4 = 4;

void SetMatMulTransposeAttr(const PrimitivePtr &src_prim, const PrimitivePtr &dst_prim) {
  auto transpose_a = src_prim->GetAttr(mindspore::ops::kTransposeA);
  auto transpose_b = src_prim->GetAttr(mindspore::ops::kTransposeB);
  if (transpose_a != nullptr) {
    dst_prim->AddAttr("transpose_a", transpose_a);
  } else {
    dst_prim->AddAttr("transpose_a", MakeValue(false));
  }
  if (transpose_b != nullptr) {
    dst_prim->AddAttr("transpose_b", transpose_b);
  } else {
    dst_prim->AddAttr("transpose_b", MakeValue(false));
  }
}

CNodePtr CreateSqueezeCnode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  auto squeeze_op = std::make_unique<ops::Squeeze>();
  if (squeeze_op == nullptr) {
    MS_LOG(ERROR) << "New Squeeze op failed, squeeze_op is nullptr!";
    return nullptr;
  }
  squeeze_op->set_axis({0});

  auto squeeze_prim_c = squeeze_op->GetPrim();
  if (squeeze_prim_c == nullptr) {
    MS_LOG(ERROR) << "squeeze_prim_c is nullptr!";
    return nullptr;
  }
  std::vector<AnfNodePtr> inputs = {cnode};
  auto squeeze_node = func_graph->NewCNode(squeeze_prim_c, inputs);
  if (squeeze_node == nullptr) {
    MS_LOG(ERROR) << "new squeeze cnode failed, squeeze_node is nullptr!";
    return nullptr;
  }
  squeeze_node->set_fullname_with_scope(cnode->fullname_with_scope() + "_data_squeeze");
  if (cnode->abstract() != nullptr) {
    squeeze_node->set_abstract(cnode->abstract()->Clone());
  }
  MS_LOG(INFO) << "Create squeeze node end.";
  return squeeze_node;
}

CNodePtr CreateShapeCNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  auto shape_prim_c = mindspore::prim::kPrimShape;
  MS_CHECK_TRUE_RET(shape_prim_c != nullptr, nullptr);
  std::vector<AnfNodePtr> inputs = {cnode};
  auto shape_cnode = func_graph->NewCNode(shape_prim_c, inputs);
  if (shape_cnode == nullptr) {
    MS_LOG(ERROR) << "New shape cnode failed, shape_cnode is nullptr!";
    return nullptr;
  }
  shape_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_shape");
  if (cnode->abstract() != nullptr) {
    shape_cnode->set_abstract(cnode->abstract()->Clone());
  }
  MS_LOG(INFO) << "Create shape node end.";
  return shape_cnode;
}

CNodePtr CreateRangeV2Cnode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  auto range_op = std::make_unique<ops::RangeV2>();
  if (range_op == nullptr) {
    MS_LOG(ERROR) << "New RangeV2 op failed, range_op is nullptr!";
    return nullptr;
  }

  auto range_prim_c = range_op->GetPrim();
  if (range_prim_c == nullptr) {
    MS_LOG(ERROR) << "range_prim_c is nullptr!";
    return nullptr;
  }

  auto start_num = opt::BuildIntValueParameterNode(func_graph, 0, cnode->fullname_with_scope() + "_start", false);
  MS_CHECK_TRUE_RET(start_num != nullptr, nullptr);
  auto delta_num = opt::BuildIntValueParameterNode(func_graph, 1, cnode->fullname_with_scope() + "_delta", false);
  MS_CHECK_TRUE_RET(delta_num != nullptr, nullptr);
  std::vector<AnfNodePtr> inputs = {start_num, cnode, delta_num};
  auto range_node = func_graph->NewCNode(range_prim_c, inputs);
  if (range_node == nullptr) {
    MS_LOG(ERROR) << "New range cnode failed, range_node is nullptr!";
    return nullptr;
  }
  range_node->set_fullname_with_scope(cnode->fullname_with_scope() + "_range");
  if (cnode->abstract() != nullptr) {
    range_node->set_abstract(cnode->abstract()->Clone());
  }
  MS_LOG(INFO) << "Create squeeze node end.";
  return range_node;
}

CNodePtr CreateSubCnode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  auto sub_op = std::make_unique<ops::Sub>();
  if (sub_op == nullptr) {
    MS_LOG(ERROR) << "New Sub op failed, sub_op is nullptr!";
    return nullptr;
  }

  auto sub_prim_c = sub_op->GetPrim();
  if (sub_prim_c == nullptr) {
    MS_LOG(ERROR) << "sub_prim_c is nullptr!";
    return nullptr;
  }

  auto sub_vale_parameter =
    opt::BuildIntValueParameterNode(func_graph, 1, cnode->fullname_with_scope() + "_sub_param", false);
  MS_CHECK_TRUE_RET(sub_vale_parameter != nullptr, nullptr);
  std::vector<AnfNodePtr> inputs = {cnode, sub_vale_parameter};
  auto sub_node = func_graph->NewCNode(sub_prim_c, inputs);
  if (sub_node == nullptr) {
    MS_LOG(ERROR) << "New sub cnode failed, sub_node is nullptr!";
    return nullptr;
  }
  sub_node->set_fullname_with_scope(cnode->fullname_with_scope() + "_sub");
  if (cnode->abstract() != nullptr) {
    sub_node->set_abstract(cnode->abstract()->Clone());
  }
  MS_LOG(INFO) << "Create Sub node end.";
  return sub_node;
}

CNodePtr CreateAfterReshapeNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const CNodePtr &shape_node) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  if (shape_node == nullptr) {
    MS_LOG(ERROR) << "Input shape cnode is nullptr!";
    return nullptr;
  }

  auto reshape_prim_c = mindspore::prim::kPrimReshape;
  if (reshape_prim_c == nullptr) {
    MS_LOG(ERROR) << "New Reshape prim failed, reshape_prim_c is nullptr!";
    return nullptr;
  }
  std::vector<AnfNodePtr> inputs = {cnode, shape_node};
  auto reshape_node = func_graph->NewCNode(reshape_prim_c, inputs);
  if (reshape_node == nullptr) {
    MS_LOG(ERROR) << "New reshape cnode failed, reshape_node is nullptr!";
    return nullptr;
  }
  reshape_node->set_fullname_with_scope(cnode->fullname_with_scope() + "_reshape_after");
  if (cnode->abstract() != nullptr) {
    reshape_node->set_abstract(cnode->abstract()->Clone());
  }
  MS_LOG(INFO) << "Create reshape node end.";
  return reshape_node;
}

bool AdjustBMMToMM(const FuncGraphPtr &func_graph, const CNodePtr &batch_matmul_cnode) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  MS_CHECK_TRUE_RET(batch_matmul_cnode != nullptr, false);
  MS_LOG(INFO) << "Adjust BatchMatMul node to MatMul node.";
  if (batch_matmul_cnode->size() < kSize_3 || batch_matmul_cnode->size() > kSize_4) {
    MS_LOG(ERROR) << "batch_matmul_cnode->size() < 3 or size() > 4!";
    return false;
  }
  if (batch_matmul_cnode->size() == kSize_4) {
    MS_LOG(INFO) << "Now not support MM with bias.";
    return true;
  }
  if (batch_matmul_cnode->abstract() == nullptr) {
    MS_LOG(ERROR) << "batch_matmul_cnode abstract is nullptr!";
    return false;
  }
  if (!utils::isa<CNodePtr>(batch_matmul_cnode->input(kInputIndex_1))) {
    MS_LOG(INFO) << "Input_1 cnode is not CNode, return true!";
    return true;
  }
  if (!utils::isa<ParameterPtr>(batch_matmul_cnode->input(kInputIndex_2))) {
    MS_LOG(INFO) << "Input_2 cnode is not ParameterPtr, return true!";
    return true;
  }
  auto batch_matmul_input_1 = batch_matmul_cnode->input(kInputIndex_1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(batch_matmul_input_1 != nullptr, false);
  auto matmul_weight_input = batch_matmul_cnode->input(kInputIndex_2);
  MS_CHECK_TRUE_RET(matmul_weight_input != nullptr, false);

  auto data_shape_cnode = CreateShapeCNode(func_graph, batch_matmul_input_1);
  MS_CHECK_TRUE_RET(data_shape_cnode != nullptr, false);
  auto data_shape_gather_node =
    opt::GenGatherNode(func_graph, data_shape_cnode, {kShapeMinus_1},
                       data_shape_cnode->fullname_with_scope() + "_data_shape_gather", {kAxis_0});
  MS_CHECK_TRUE_RET(data_shape_gather_node != nullptr, false);
  data_shape_gather_node->set_abstract(batch_matmul_cnode->abstract()->Clone());

  auto data_concat_parm = opt::BuildIntVecParameterNode(func_graph, {kShape_1, kShapeMinus_1},
                                                        batch_matmul_cnode->fullname_with_scope() + "_const_minus_2");
  MS_CHECK_TRUE_RET(data_concat_parm != nullptr, false);
  data_concat_parm->set_abstract(batch_matmul_cnode->abstract()->Clone());

  auto data_concat_cnode = opt::GenConcatNode(func_graph, {data_concat_parm, data_shape_gather_node},
                                              batch_matmul_cnode->fullname_with_scope() + "_concat", 0);
  MS_CHECK_TRUE_RET(data_concat_cnode != nullptr, false);
  data_concat_cnode->set_abstract(batch_matmul_cnode->abstract()->Clone());

  // create reshape node, Data reshape to (1,ab,c), weight shape is (c,d)
  auto reshape_data_node = CreateAfterReshapeNode(func_graph, batch_matmul_input_1, data_concat_cnode);
  MS_CHECK_TRUE_RET(reshape_data_node != nullptr, false);
  reshape_data_node->set_abstract(batch_matmul_cnode->abstract()->Clone());

  auto squeeze_cnode = CreateSqueezeCnode(func_graph, reshape_data_node);
  MS_CHECK_TRUE_RET(squeeze_cnode != nullptr, false);

  ops::MatMul matmul;
  auto dst_prim = matmul.GetPrim();
  MS_CHECK_TRUE_RET(dst_prim != nullptr, false);
  auto matmul_cnode = func_graph->NewCNode(dst_prim, {squeeze_cnode, matmul_weight_input});
  if (matmul_cnode == nullptr) {
    MS_LOG(ERROR) << "New matmul_cnode is nullptr!";
    return false;
  }
  auto abstract = lite::CreateTensorAbstract({kShapeMinus_1, kShapeMinus_1}, kNumberTypeFloat32);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstract failed!";
    return false;
  }
  matmul_cnode->set_abstract(abstract);
  matmul_cnode->set_fullname_with_scope(batch_matmul_cnode->fullname_with_scope() + "_bmm2mm");

  auto prim = GetValueNode<PrimitivePtr>(batch_matmul_cnode->input(kInputIndex_0));
  MS_CHECK_TRUE_RET(prim != nullptr, false);
  SetMatMulTransposeAttr(prim, dst_prim);

  auto data_shape_dim_cnode = CreateShapeCNode(func_graph, data_shape_cnode);
  MS_CHECK_TRUE_RET(data_shape_dim_cnode != nullptr, false);

  auto range_limit_cnode = CreateSubCnode(func_graph, data_shape_dim_cnode);
  MS_CHECK_TRUE_RET(range_limit_cnode != nullptr, false);

  auto range_cnode = CreateRangeV2Cnode(func_graph, range_limit_cnode);
  MS_CHECK_TRUE_RET(range_cnode != nullptr, false);

  auto shape_gather_node = opt::GenGatherNodeDynamicIndex(
    func_graph, data_shape_cnode, range_cnode, data_shape_cnode->fullname_with_scope() + "_gather", {kAxis_0});
  MS_CHECK_TRUE_RET(shape_gather_node != nullptr, false);

  auto concat_parm = opt::BuildIntValueParameterNode(
    func_graph, kShapeMinus_1, batch_matmul_cnode->fullname_with_scope() + "_const_minus_1", false);
  MS_CHECK_TRUE_RET(concat_parm != nullptr, false);
  concat_parm->set_abstract(batch_matmul_cnode->abstract()->Clone());

  auto concat_cnode = opt::GenConcatNode(func_graph, {shape_gather_node, concat_parm},
                                         batch_matmul_cnode->fullname_with_scope() + "_concat", 0);
  MS_CHECK_TRUE_RET(concat_cnode != nullptr, false);
  concat_cnode->set_abstract(batch_matmul_cnode->abstract()->Clone());

  // reshape(MM, (a,b,d))
  auto reshape_output_cnode = CreateAfterReshapeNode(func_graph, matmul_cnode, concat_cnode);
  MS_CHECK_TRUE_RET(reshape_output_cnode != nullptr, false);

  auto graph_manager = func_graph->manager();
  MS_CHECK_TRUE_RET(graph_manager != nullptr, false);

  if (!graph_manager->Replace(batch_matmul_cnode, reshape_output_cnode)) {
    MS_LOG(ERROR) << "Failed to replace MatMul with BatchMatMul! cnode " << batch_matmul_cnode->fullname_with_scope()
                  << ", input size " << batch_matmul_cnode->size();
    return false;
  }
  return true;
}

}  // namespace

bool AdjustMatmulPass::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  MS_LOG(INFO) << "AdjustResizeDimsPass start.";
  auto node_list = TopoSort(func_graph->get_return());
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "Manager is nullptr!";
    return false;
  }
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!opt::CheckPrimitiveType(node, prim::kPrimMatMulFusion)) {
      continue;
    }
    auto mm_cnode = node->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(mm_cnode != nullptr, false);
    if (!AdjustBMMToMM(func_graph, mm_cnode)) {
      MS_LOG(ERROR) << "This node run AdjustMatmulPass failed! Node_name is: " << mm_cnode->fullname_with_scope();
      return false;
    }
    MS_LOG(INFO) << "This node run AdjustMatmulPass success : " << mm_cnode->fullname_with_scope();
  }
  MS_LOG(INFO) << "AdjustMatmulPass end.";
  return true;
}
}  // namespace opt
}  // namespace mindspore
