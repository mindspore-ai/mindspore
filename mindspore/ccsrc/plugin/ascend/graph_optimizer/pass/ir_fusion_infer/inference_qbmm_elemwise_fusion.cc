/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_qbmm_elemwise_fusion.h"
#include <vector>
#include <string>
#include "backend/common/pass/common/gllo_utils.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_weight_preprocess_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"
#include "primitive/nn_ops.h"
#include "primitive/math_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {

CNodePtr InferenceQbmmElemwiseFusion::CreateQbmmElemNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                         const EquivPtr &equiv) const {
  const int trans_a_index = kIndex7;
  const int trans_b_index = kIndex8;
  MS_LOG(DEBUG) << "start CreateQbmmElemNode";
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  auto qbmm_prim = std::make_shared<Primitive>("QuantBatchMatmul");
  qbmm_prim->AddAttr("ElemwiseType", MakeValue("fastgelu"));
  std::vector<AnfNodePtr> inputs = {std::make_shared<ValueNode>(qbmm_prim),
                                    x_node_,
                                    w_node_,
                                    scale_node_,
                                    offset_node_,
                                    bias_node_,
                                    pertoken_scale_node_,
                                    trans_a_node_,
                                    trans_b_node_,
                                    out_dtype_node_};

  auto qbmm_node = common::AnfAlgo::GetInputNode(node->cast<CNodePtr>(), kIndex0);
  MS_CHECK_TRUE_RET(qbmm_node != nullptr, nullptr);

  auto qbmm_cnode = qbmm_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(qbmm_cnode != nullptr, nullptr);
  auto cnode_name = common::AnfAlgo::GetCNodeName(qbmm_cnode);
  MS_CHECK_TRUE_RET(cnode_name == prim::kPrimQuantBatchMatmul->name(), nullptr);
  auto trans_a = qbmm_cnode->input(trans_a_index)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_RET(trans_a != nullptr, nullptr);
  auto trans_b = qbmm_cnode->input(trans_b_index)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_RET(trans_b != nullptr, nullptr);
  bool is_trans_a = GetValue<bool>(trans_a->value());
  bool is_trans_b = GetValue<bool>(trans_b->value());
  MS_LOG(DEBUG) << "the transpose format of matmul node is: trans_a=" << is_trans_a << " trans_b=" << is_trans_b;
  if (is_trans_a || !is_trans_b) {
    return nullptr;
  }

  auto new_qbmm_node = NewCNode(inputs, func_graph);
  MS_CHECK_TRUE_RET(new_qbmm_node != nullptr, nullptr);
  new_qbmm_node->set_scope(node->scope());

  if (node->abstract() != nullptr) {
    new_qbmm_node->set_abstract(node->abstract()->Clone());
  }
  MS_LOG(DEBUG) << "create QbmmElem node success.";
  return new_qbmm_node;
}

std::vector<std::string> InferenceQbmmElemwiseFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimQuantBatchMatmul->name(), prim::kPrimFastGeLU->name()};
  return ret;
}

void InferenceQbmmElemwiseFusion::SetInternalNodes(const EquivPtr &equiv) const {
  x_node_ = utils::cast<AnfNodePtr>((*equiv)[x_]);
  MS_ASSERT(x_node_ != nullptr);
  w_node_ = utils::cast<AnfNodePtr>((*equiv)[w_]);
  MS_ASSERT(w_node_ != nullptr);
  scale_node_ = utils::cast<AnfNodePtr>((*equiv)[scale_]);
  MS_ASSERT(scale_node_ != nullptr);
  offset_node_ = utils::cast<AnfNodePtr>((*equiv)[offset_]);
  MS_ASSERT(offset_node != nullptr);
  bias_node_ = utils::cast<AnfNodePtr>((*equiv)[bias_]);
  MS_ASSERT(bias_node_ != nullptr);
  pertoken_scale_node_ = utils::cast<AnfNodePtr>((*equiv)[pertoken_scale_]);
  MS_ASSERT(pertoken_scale_node_ != nullptr);
  trans_a_node_ = utils::cast<AnfNodePtr>((*equiv)[trans_a_]);
  MS_ASSERT(trans_a_node != nullptr);
  trans_b_node_ = utils::cast<AnfNodePtr>((*equiv)[trans_b_]);
  MS_ASSERT(trans_b_node != nullptr);
  out_dtype_node_ = utils::cast<AnfNodePtr>((*equiv)[out_dtype_]);
  MS_ASSERT(out_dtype_node_ != nullptr);
}

const BaseRef InferenceQbmmElemwiseFusion::DefinePattern() const {
  if (!Init()) {
    MS_LOG(DEBUG) << "initial member failed.";
    return {};
  }
  VectorRef qbmm_ref({qbmm_prim_, x_, w_, scale_, offset_, bias_, pertoken_scale_, trans_a_, trans_b_, out_dtype_});
  bias_tensor_ = std::make_shared<Var>();
  auto is_fast = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimFastGeLU>);
  MS_CHECK_TRUE_RET(is_fast != nullptr, {});
  VectorRef fast_ref({is_fast, qbmm_ref});
  return fast_ref;
}

const AnfNodePtr InferenceQbmmElemwiseFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                      const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto const &soc_version = ms_context->ascend_soc_version();
  if (!soc_version.empty() && soc_version != "ascend310p") {
    return nullptr;
  }
  constexpr auto kQbmmElemName = "MatMulElemwise";
  if (func_graph == nullptr || node == nullptr || equiv == nullptr || !PassEnable(kQbmmElemName)) {
    return nullptr;
  }

  SetInternalNodes(equiv);
  if (!IsValueNode<None>(pertoken_scale_node_)) {
    MS_LOG(INFO) << "Currently, do not support to fuse qbmm(pertoken) with add.";
    return nullptr;
  }
  if (!CheckIOValid()) {
    MS_LOG(ERROR) << "Node type don't match";
    return nullptr;
  }
  auto cnode = CreateQbmmElemNode(func_graph, node, equiv);
  return cnode;
}

bool InferenceQbmmElemwiseFusion::CheckIOValid() const {
  if (!CheckSupportDataType(scale_node_, {kNumberTypeInt64}) || !CheckSupportDataType(bias_node_, {kNumberTypeInt32})) {
    return false;
  }
  auto dtype_value = GetValue<int64_t>(out_dtype_node_->cast<ValueNodePtr>()->value());
  if (dtype_value != static_cast<int64_t>(kNumberTypeFloat16)) {
    return false;
  }
  auto bias_shape = common::AnfAlgo::GetOutputInferShape(bias_node_, kIndex0);
  auto scale_shape = common::AnfAlgo::GetOutputInferShape(scale_node_, kIndex0);
  if (bias_shape.size() != 1 || scale_shape.size() != 1 || bias_shape[0] != scale_shape[0]) {
    return false;
  }
  auto scale_param = GetParamFromLoad(scale_node_->cast<CNodePtr>(), false);
  if (!scale_param) {
    return false;
  }
  auto bias_param = GetParamFromLoad(bias_node_->cast<CNodePtr>(), false);
  if (!bias_param) {
    return false;
  }
  return true;
}

}  // namespace opt
}  // namespace mindspore
