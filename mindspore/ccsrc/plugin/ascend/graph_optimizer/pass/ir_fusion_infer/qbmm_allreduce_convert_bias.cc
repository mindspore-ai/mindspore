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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/qbmm_allreduce_convert_bias.h"
#include <vector>
#include <string>
#include <memory>
#include "primitive/other_ops.h"
#include "backend/common/pass/common/gllo_utils.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_weight_preprocess_utils.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"

namespace mindspore {
namespace opt {
bool QbmmAllReduceConvertBias::Init() const {
  x_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(x_ != nullptr, false);
  w_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(w_ != nullptr, false);
  scale_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(scale_ != nullptr, false);
  offset_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(offset_ != nullptr, false);
  bias_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(bias_ != nullptr, false);
  pertoken_scale_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(pertoken_scale_ != nullptr, false);
  trans_a_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(trans_a_ != nullptr, false);
  trans_b_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(trans_b_ != nullptr, false);
  out_dtype_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(out_dtype_ != nullptr, false);
  qbmm_prim_ = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimQuantBatchMatmul>);
  MS_CHECK_TRUE_RET(qbmm_prim_ != nullptr, false);
  return true;
}

const BaseRef QbmmAllReduceConvertBias::DefinePattern() const {
  if (!Init()) {
    MS_LOG(INFO) << "initial member failed.";
    return {};
  }
  VectorRef qbmm_ref({qbmm_prim_, x_, w_, scale_, offset_, bias_, pertoken_scale_, trans_a_, trans_b_, out_dtype_});
  auto is_allreduce = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAllReduce>);
  MS_CHECK_TRUE_RET(is_allreduce != nullptr, {});
  VectorRef allreduce_ref({is_allreduce, qbmm_ref});
  return allreduce_ref;
}

const AnfNodePtr QbmmAllReduceConvertBias::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                   const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    MS_LOG(INFO) << "ms_context not enable ms infer boost.";
    return nullptr;
  }

  const std::string fusion_op_name = "QbmmAllReduceConvertBias";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_convert_bias =
    (std::find(enable_op_list.begin(), enable_op_list.end(), fusion_op_name) != enable_op_list.end());
  if (!enable_convert_bias) {
    MS_LOG(INFO) << "ms_context internal_enable_custom_kernel_list doesn't contain QbmmAllReduceConvertBias.";
    return nullptr;
  }

  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    MS_LOG(INFO) << "func_graph == nullptr || node == nullptr || equiv == nullptr.";
    return nullptr;
  }

  bias_node_ = utils::cast<AnfNodePtr>((*equiv)[bias_]);
  MS_ASSERT(bias_node_ != nullptr);
  if (!CheckSupportDataType(bias_node_, {kNumberTypeInt32})) {
    MS_LOG(INFO) << node->fullname_with_scope() << " type of bias is not int32.";
    return nullptr;
  }

  auto bias_tensor_param = GetParamFromLoad(bias_node_->cast<CNodePtr>(), false);
  if (!bias_tensor_param) {
    MS_LOG(INFO) << "GetParamFromLoad bias_node is nullptr.";
    return nullptr;
  }
  auto converted_bias_node = ConvertInt32BiasForMultiRank(bias_node_);
  MS_EXCEPTION_IF_NULL(converted_bias_node);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->AddValueNodeToGraph(converted_bias_node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto quantbatchmatmul_node = cnode->input(kIndex1)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(quantbatchmatmul_node);
  quantbatchmatmul_node->set_input(kIndex5, converted_bias_node);

  return node;
}
}  // namespace opt
}  // namespace mindspore
