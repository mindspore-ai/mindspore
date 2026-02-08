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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_qbmm_add_fusion.h"
#include <string>
#include <vector>
#include <memory>
#include "backend/common/pass/common/gllo_utils.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_weight_preprocess_utils.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"

namespace mindspore {
namespace opt {
CNodePtr QbmmAddFusion::CreateQbmmAddNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &equiv) const {
  MS_LOG(DEBUG) << "start create QbmmAddFusion node";
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  std::string prim_name = "QuantBatchMatmul";
  auto qbmm_prim = std::make_shared<Primitive>(prim_name);
  const bool with_allreduce = false;
  TypeId dtype = static_cast<TypeId>(GetValue<int64_t>(out_dtype_node_->cast<ValueNodePtr>()->value()));
  auto bias_int32_node = ConvertBiasToInt32(bias_tensor_node_, scale_node_, with_allreduce, dtype);
  if (!bias_int32_node) {
    return nullptr;
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  kernel_graph->AddValueNodeToGraph(bias_int32_node);

  std::vector<AnfNodePtr> inputs = {std::make_shared<ValueNode>(qbmm_prim),
                                    x_node_,
                                    w_node_,
                                    scale_node_,
                                    offset_node_,
                                    bias_int32_node,
                                    pertoken_scale_node_,
                                    trans_a_node_,
                                    trans_b_node_,
                                    out_dtype_node_};
  auto new_qbmm_node = NewCNode(inputs, func_graph);
  MS_CHECK_TRUE_RET(new_qbmm_node != nullptr, nullptr);
  new_qbmm_node->set_scope(node->scope());
  if (node->abstract() != nullptr) {
    new_qbmm_node->set_abstract(node->abstract()->Clone());
  }
  MS_LOG(DEBUG) << "create QbmmAddFusion node success.";
  return new_qbmm_node;
}

std::vector<std::string> QbmmAddFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimQuantBatchMatmul->name(), prim::kPrimAdd->name()};
  return ret;
}

const BaseRef QbmmAddFusion::DefinePattern() const {
  if (!Init()) {
    MS_LOG(DEBUG) << "initial member failed.";
    return {};
  }
  VectorRef qbmm_ref({qbmm_prim_, x_, w_, scale_, offset_, bias_, pertoken_scale_, trans_a_, trans_b_, out_dtype_});
  bias_tensor_ = std::make_shared<Var>();
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAdd>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  VectorRef add_ref({is_add, qbmm_ref, bias_tensor_});
  return add_ref;
}

const AnfNodePtr QbmmAddFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &equiv) const {
  constexpr auto kQbmmAddName = "QbmmAdd";
  if (func_graph == nullptr || node == nullptr || equiv == nullptr || !PassEnable(kQbmmAddName)) {
    return nullptr;
  }
  SetNodes(equiv);
  if (!IsValueNode<None>(pertoken_scale_node_)) {
    MS_LOG(INFO) << "Currently, do not support to fuse qbmm(pertoken) with add.";
    return nullptr;
  }
  if (!CheckValid()) {
    return nullptr;
  }
  auto cnode = CreateQbmmAddNode(func_graph, node, equiv);
  return cnode;
}

}  // namespace opt
}  // namespace mindspore
