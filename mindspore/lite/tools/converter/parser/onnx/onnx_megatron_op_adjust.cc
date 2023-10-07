/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/onnx/onnx_megatron_op_adjust.h"
#include <vector>
#include <string>
#include <memory>
#include "ops/all_gather.h"
#include "ops/all_reduce.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "ops/softmax.h"
#include "ops/where.h"
#include "include/errorcode.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/ops/ops_def.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::lite {
namespace {
constexpr auto kAttrParallel = "parallel";
constexpr auto kAttrParallelNum = "parallel_num";

bool AdjustMegatronAllReduce(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_RET(value_node != nullptr, false);
  auto prim = GetValueNode<PrimitivePtr>(value_node);
  MS_CHECK_TRUE_RET(prim != nullptr, false);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);
  auto parallel_num_attr = prim->GetAttr(kAttrParallelNum);
  auto replace_node = cnode->input(1);
  if (parallel_num_attr != nullptr && GetValue<int64_t>(parallel_num_attr) > 1) {
    auto all_reduce_prim = std::make_shared<ops::AllReduce>();
    MS_CHECK_TRUE_RET(all_reduce_prim != nullptr, false);
    auto all_reduce_prim_c = all_reduce_prim->GetPrim();
    MS_CHECK_TRUE_RET(all_reduce_prim_c != nullptr, false);
    all_reduce_prim_c->AddAttr(kAttrFusion, MakeValue(static_cast<int64_t>(0)));
    all_reduce_prim_c->AddAttr(kAttrOp, MakeValue("sum"));
    all_reduce_prim_c->AddAttr(kAttrGroup, MakeValue("hccl_world_group"));
    replace_node = func_graph->NewCNode(all_reduce_prim_c, {cnode->input(1)});
  }
  MS_CHECK_TRUE_RET(replace_node != nullptr, false);
  return manager->Replace(cnode, replace_node);
}

bool AdjustMegatronLinearAllGather(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(func_graph != nullptr && cnode != nullptr, false);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);

  auto matmul_prim = std::make_shared<ops::MatMulFusion>();
  MS_CHECK_TRUE_RET(matmul_prim != nullptr, false);
  matmul_prim->set_transpose_b(static_cast<bool>(true));
  auto matmul_prim_c = matmul_prim->GetPrim();
  MS_CHECK_TRUE_RET(matmul_prim_c != nullptr, false);
  auto inputs = std::vector<AnfNodePtr>(cnode->inputs().begin() + 1, cnode->inputs().end());
  auto matmul_cnode = func_graph->NewCNode(matmul_prim_c, inputs);
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, false);
  manager->Replace(cnode, matmul_cnode);

  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_RET(value_node != nullptr, false);
  auto prim = GetValueNode<PrimitivePtr>(value_node);
  MS_CHECK_TRUE_RET(prim != nullptr, false);
  auto parallel_attr = prim->GetAttr(kAttrParallel);
  MS_CHECK_TRUE_RET(parallel_attr != nullptr, false);
  auto parallel = GetValue<bool>(parallel_attr);
  if (parallel) {
    auto all_gather_prim = std::make_shared<ops::AllGather>();
    MS_CHECK_TRUE_RET(all_gather_prim != nullptr, false);
    auto all_gather_prim_c = all_gather_prim->GetPrim();
    MS_CHECK_TRUE_RET(all_gather_prim_c != nullptr, false);
    auto input = matmul_cnode->input(1);
    MS_CHECK_TRUE_RET(input != nullptr, false);
    auto gather_cnode = func_graph->NewCNode(all_gather_prim_c, {input});
    manager->SetEdge(matmul_cnode, 1, gather_cnode);
  }
  return true;
}

/*
 *                                                             input   scale
 *                                                                 \   /
 *     input         mask                mask    const(-10000.f)    mul
 *       \            /                      \          |          /
 *   ScaledMaskedSoftmax<scale>    -->                where
 *              |                                       |
 *            output                                 SoftMax
 *                                                      |
 *                                                    output
 */
bool AdjustMegatronScaledMaskedSoftmax(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);
  auto node_name = cnode->fullname_with_scope();
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_RET(value_node != nullptr, false);
  auto prim = GetValueNode<PrimitivePtr>(value_node);
  MS_CHECK_TRUE_RET(prim != nullptr, false);
  auto scale_attr = prim->GetAttr(kAttrScales);
  MS_CHECK_TRUE_RET(scale_attr != nullptr, false);
  auto scale = GetValue<float>(scale_attr);
  auto scale_param = opt::BuildFloatValueParameterNode(func_graph, scale, node_name + "MaskedSoftmax/Mul_scale", true);
  if (scale_param == nullptr) {
    MS_LOG(ERROR) << "Build scale parameter for MaskedSoftmax failed.";
    return false;
  }

  auto mul_prim = std::make_shared<ops::MulFusion>();
  MS_CHECK_TRUE_RET(mul_prim != nullptr, false);
  auto mul_prim_c = mul_prim->GetPrim();
  MS_CHECK_TRUE_RET(mul_prim_c != nullptr, false);
  auto mul_cnode = func_graph->NewCNode(mul_prim_c, {cnode->input(1), scale_param});
  MS_CHECK_TRUE_RET(mul_cnode != nullptr, false);
  mul_cnode->set_fullname_with_scope(node_name + "MaskedSoftmax/Mul");

  float neg_value = -10000.f;  // whose e-base exponent is 0 approximately.
  auto neg_value_param =
    opt::BuildFloatValueParameterNode(func_graph, neg_value, node_name + "MaskedSoftmax/Where_negval", true);
  if (neg_value_param == nullptr) {
    MS_LOG(ERROR) << "Build neg_value parameter for MaskedSoftmax failed.";
    return false;
  }
  auto where_prim = std::make_shared<ops::Where>();
  MS_CHECK_TRUE_RET(mul_prim != nullptr, false);
  auto where_prim_c = where_prim->GetPrim();
  MS_CHECK_TRUE_RET(where_prim_c != nullptr, false);
  auto where_cnode = func_graph->NewCNode(where_prim_c, {cnode->input(2), neg_value_param, mul_cnode});
  MS_CHECK_TRUE_RET(where_cnode != nullptr, false);
  where_cnode->set_fullname_with_scope(node_name + "MaskedSoftmax/Where");

  auto softmax_prim = std::make_shared<ops::Softmax>();
  MS_CHECK_TRUE_RET(softmax_prim != nullptr, false);
  softmax_prim->set_axis({-1});
  auto softmax_prim_c = softmax_prim->GetPrim();
  MS_CHECK_TRUE_RET(softmax_prim_c != nullptr, false);
  auto softmax_cnode = func_graph->NewCNode(softmax_prim_c, {where_cnode});
  MS_CHECK_TRUE_RET(softmax_cnode != nullptr, false);
  softmax_cnode->set_fullname_with_scope(node_name + "ScaledMaskedSoftmax/Softmax");
  return manager->Replace(cnode, softmax_cnode);
}
}  // namespace

bool OnnxMegatronOpAdjust::Adjust(const FuncGraphPtr &func_graph, const converter::ConverterParameters &flag) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return lite::RET_NULL_PTR;
  }
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      MS_LOG(DEBUG) << "node is not cnode.";
      continue;
    }
    bool ret = true;
    if (opt::CheckPrimitiveType(node, std::make_unique<Primitive>(lite::kNameMegatronMakeViewlessTensor))) {
      ret = manager->Replace(cnode, cnode->input(1));
    } else if (opt::CheckPrimitiveType(node, std::make_unique<Primitive>(lite::kNameMegatronAllReduce))) {
      ret = AdjustMegatronAllReduce(func_graph, cnode);
    } else if (opt::CheckPrimitiveType(node, std::make_unique<Primitive>(lite::kNameMegatronLinearAllGather))) {
      ret = AdjustMegatronLinearAllGather(func_graph, cnode);
    } else if (opt::CheckPrimitiveType(node, std::make_unique<Primitive>(lite::kNameMegatronScaledMaskedSoftmax))) {
      ret = AdjustMegatronScaledMaskedSoftmax(func_graph, cnode);
    }
    if (!ret) {
      MS_LOG(ERROR) << "Adjust megatron op failed: " << cnode->fullname_with_scope();
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::lite
