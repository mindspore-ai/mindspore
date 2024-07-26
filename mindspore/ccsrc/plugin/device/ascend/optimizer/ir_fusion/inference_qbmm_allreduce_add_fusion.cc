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
#include "plugin/device/ascend/optimizer/ir_fusion/inference_qbmm_allreduce_add_fusion.h"
#include <vector>
#include <algorithm>
#include <limits>
#include "plugin/device/ascend/optimizer/common/gllo_utils.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "include/api/data_type.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"
#include "plugin/device/ascend/optimizer/ir_fusion/inference_weight_preprocess_utils.h"

namespace mindspore {
namespace opt {

CNodePtr QbmmAllReduceAddFusion::UpdateQbmmAllReduceAddNode(const FuncGraphPtr &func_graph, const AnfNodePtr &add_node,
                                                            const EquivPtr &equiv) const {
  MS_LOG(DEBUG) << "start create QbmmAllReduceAddFusion node";
  MS_ASSERT(func_graph != nullptr && add_node != nullptr && equiv != nullptr);
  auto scale_node = utils::cast<AnfNodePtr>((*equiv)[scale_]);
  MS_ASSERT(scale_node != nullptr);
  auto bias_tensor_node = utils::cast<AnfNodePtr>((*equiv)[bias_tensor_]);
  MS_ASSERT(bias_tensor_node != nullptr);

  // add -> allreduce -> quantbatchmatmul
  const bool with_allreduce = true;
  auto bias_int32_node = ConvertFp16BiasToInt32(bias_tensor_node, scale_node, with_allreduce);
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

bool QbmmAllReduceAddFusion::Init() const {
  qbmm_prim_ = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimQuantBatchMatmul>);
  MS_CHECK_TRUE_RET(qbmm_prim_ != nullptr, false);
  x_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(x_ != nullptr, false);
  w_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(w_ != nullptr, false);
  scale_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(scale_ != nullptr, false);
  unused_offset_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(unused_offset_ != nullptr, false);
  orig_bias_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(orig_bias_ != nullptr, false);
  trans_a_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(trans_a_ != nullptr, false);
  trans_b_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(trans_b_ != nullptr, false);
  out_dtype_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(out_dtype_ != nullptr, false);
  return true;
}

const BaseRef QbmmAllReduceAddFusion::DefinePattern() const {
  if (!Init()) {
    MS_LOG(DEBUG) << "initial member failed.";
    return {};
  }
  VectorRef qbmm_ref({qbmm_prim_, x_, w_, scale_, unused_offset_, orig_bias_, trans_a_, trans_b_, out_dtype_});
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
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }

  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  constexpr auto kQbmmAllReduceAddName = "QbmmAllReduceAdd";
  auto enable_fusion =
    (std::find(enable_op_list.begin(), enable_op_list.end(), kQbmmAllReduceAddName) != enable_op_list.end());
  if (!enable_fusion) {
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  auto scale_node = utils::cast<AnfNodePtr>((*equiv)[scale_]);
  auto bias_node = utils::cast<AnfNodePtr>((*equiv)[orig_bias_]);
  auto bias_tensor_node = utils::cast<AnfNodePtr>((*equiv)[bias_tensor_]);
  auto out_dtype_node = utils::cast<AnfNodePtr>((*equiv)[out_dtype_]);
  if (!CheckSupportDataType(bias_tensor_node, {kNumberTypeFloat16}) ||
      !CheckSupportDataType(scale_node, {kNumberTypeInt64}) || !CheckSupportDataType(bias_node, {kMetaTypeNone})) {
    return nullptr;
  }
  auto bias_shape = common::AnfAlgo::GetOutputInferShape(bias_tensor_node, kIndex0);
  auto scale_shape = common::AnfAlgo::GetOutputInferShape(scale_node, kIndex0);
  if (bias_shape.size() != 1 || scale_shape.size() != 1 || bias_shape[0] != scale_shape[0]) {
    return nullptr;
  }
  auto bias_tensor_param = GetParamFromLoad(bias_tensor_node->cast<CNodePtr>(), false);
  if (!bias_tensor_param) {
    return nullptr;
  }
  auto scale_param = GetParamFromLoad(scale_node->cast<CNodePtr>(), false);
  if (!scale_param) {
    return nullptr;
  }
  auto dtype_value = GetValue<int64_t>(out_dtype_node->cast<ValueNodePtr>()->value());
  if (dtype_value != static_cast<int64_t>(kNumberTypeFloat16)) {
    return nullptr;
  }
  auto cnode = UpdateQbmmAllReduceAddNode(func_graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(DEBUG) << "create QbmmAllReduceAddFusion node failed.";
    return nullptr;
  }
  return cnode;
}

}  // namespace opt
}  // namespace mindspore
