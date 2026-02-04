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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_qbmm_allreduce_add_fusion.h"
#include <vector>
#include <string>
#include <memory>
#include "primitive/other_ops.h"
#include "backend/common/pass/common/gllo_utils.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_weight_preprocess_utils.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"

namespace mindspore {
namespace opt {
CNodePtr QbmmAllReduceAddFusion::CreateQbmmAllReduceAddNode(const FuncGraphPtr &func_graph, const AnfNodePtr &add_node,
                                                            const EquivPtr &equiv) const {
  MS_LOG(DEBUG) << "start create QbmmAllReduceAddFusion node";
  MS_ASSERT(func_graph != nullptr && add_node != nullptr && equiv != nullptr);
  // quantbatchmatmul -> allreduce -> add
  const bool with_allreduce = true;
  TypeId dtype = static_cast<TypeId>(GetValue<int64_t>(out_dtype_node_->cast<ValueNodePtr>()->value()));
  auto bias_int32_node = ConvertBiasToInt32(bias_tensor_node_, scale_node_, with_allreduce, dtype);
  if (!bias_int32_node) {
    return nullptr;
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  kernel_graph->AddValueNodeToGraph(bias_int32_node);
  auto allreduce_node = add_node->cast<CNodePtr>()->input(kIndex1)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(allreduce_node);
  auto quantbatchmatmul_node = allreduce_node->input(kIndex1)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(quantbatchmatmul_node);
  quantbatchmatmul_node->set_input(kIndex5, bias_int32_node);
  if (add_node->abstract() != nullptr) {
    allreduce_node->set_abstract(add_node->abstract()->Clone());
  }
  MS_LOG(DEBUG) << "create QbmmAllReduceAddFusion node success.";
  return allreduce_node;
}

std::vector<std::string> QbmmAllReduceAddFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimQuantBatchMatmul->name(), prim::kPrimAllReduce->name(),
                               prim::kPrimAdd->name()};
  return ret;
}

const BaseRef QbmmAllReduceAddFusion::DefinePattern() const {
  if (!Init()) {
    MS_LOG(DEBUG) << "initial member failed.";
    return {};
  }
  VectorRef qbmm_ref({qbmm_prim_, x_, w_, scale_, offset_, bias_, pertoken_scale_, trans_a_, trans_b_, out_dtype_});
  auto is_allreduce = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAllReduce>);
  MS_CHECK_TRUE_RET(is_allreduce != nullptr, {});
  VectorRef allreduce_ref({is_allreduce, qbmm_ref});
  bias_tensor_ = std::make_shared<Var>();
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAdd>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  VectorRef add_ref({is_add, allreduce_ref, bias_tensor_});
  return add_ref;
}

const AnfNodePtr QbmmAllReduceAddFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &equiv) const {
  constexpr auto kQbmmAllReduceAddName = "QbmmAllReduceAdd";
  if (func_graph == nullptr || node == nullptr || equiv == nullptr || !PassEnable(kQbmmAllReduceAddName)) {
    return nullptr;
  }
  SetNodes(equiv);
  if (!IsValueNode<None>(pertoken_scale_node_)) {
    MS_LOG(INFO) << "Currently, do not support to fuse qbmm(pertoken) with communication.";
    return nullptr;
  }
  if (!CheckValid()) {
    return nullptr;
  }
  auto cnode = CreateQbmmAllReduceAddNode(func_graph, node, equiv);
  return cnode;
}

}  // namespace opt
}  // namespace mindspore
