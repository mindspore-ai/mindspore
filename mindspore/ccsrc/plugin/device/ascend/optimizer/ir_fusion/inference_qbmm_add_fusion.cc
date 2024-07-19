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
#include "plugin/device/ascend/optimizer/ir_fusion/inference_qbmm_add_fusion.h"
#include <vector>
#include <algorithm>
#include <limits>
#include "plugin/device/ascend/optimizer/common/gllo_utils.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "include/api/data_type.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"
#include "plugin/device/ascend/optimizer/ir_fusion/inference_weight_preprocess_utils.h"
#include "include/backend/distributed/collective/collective_manager.h"

namespace mindspore {
namespace opt {

CNodePtr QbmmAddFusion::CreateQbmmAddNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &equiv) const {
  MS_LOG(WARNING) << "start create QbmmAddFusion node";
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  std::string prim_name = "QuantBatchMatmul";
  auto qbmm_prim = std::make_shared<Primitive>(prim_name);
  auto x_node = utils::cast<AnfNodePtr>((*equiv)[x_]);
  MS_ASSERT(x_node != nullptr);
  auto w_node = utils::cast<AnfNodePtr>((*equiv)[w_]);
  MS_ASSERT(w_node != nullptr);
  auto scale_node = utils::cast<AnfNodePtr>((*equiv)[scale_]);
  MS_ASSERT(scale_node != nullptr);
  auto offset_node = utils::cast<AnfNodePtr>((*equiv)[offset_]);
  MS_ASSERT(offset_node != nullptr);
  auto bias_tensor_node = utils::cast<AnfNodePtr>((*equiv)[bias_tensor_]);
  MS_ASSERT(bias_tensor_node != nullptr);
  auto trans_a_node = utils::cast<AnfNodePtr>((*equiv)[trans_a_]);
  MS_ASSERT(trans_a_node != nullptr);
  auto trans_b_node = utils::cast<AnfNodePtr>((*equiv)[trans_b_]);
  MS_ASSERT(trans_b_node != nullptr);
  auto out_dtype_node = utils::cast<AnfNodePtr>((*equiv)[out_dtype_]);
  MS_ASSERT(out_dtype_node != nullptr);

  const bool with_allreduce = false;
  auto bias_int32_node = ConvertFp16BiasToInt32(bias_tensor_node, scale_node, with_allreduce);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  kernel_graph->AddValueNodeToGraph(bias_int32_node);

  std::vector<AnfNodePtr> inputs = {x_node,          w_node,       scale_node,   offset_node,
                                    bias_int32_node, trans_a_node, trans_b_node, out_dtype_node};
  auto new_qbmm_node = func_graph->NewCNode(qbmm_prim, inputs);
  MS_CHECK_TRUE_RET(new_qbmm_node != nullptr, nullptr);
  new_qbmm_node->set_fullname_with_scope(node->fullname_with_scope() + "-QbmmAddFusion");
  if (node->abstract() != nullptr) {
    new_qbmm_node->set_abstract(node->abstract()->Clone());
  }
  MS_LOG(WARNING) << "create QbmmAddFusion node success.";
  return new_qbmm_node;
}

bool QbmmAddFusion::Init() const {
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

const BaseRef QbmmAddFusion::DefinePattern() const {
  if (!Init()) {
    MS_LOG(WARNING) << "initial member failed.";
    return {};
  }

  VectorRef qbmm_ref({qbmm_prim_, x_, w_, scale_, offset_, bias_, trans_a_, trans_b_, out_dtype_});
  bias_tensor_ = std::make_shared<Var>();
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAdd>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  VectorRef add_ref({is_add, qbmm_ref, bias_tensor_});

  return add_ref;
}

const AnfNodePtr QbmmAddFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }

  constexpr auto kQbmmAddName = "QbmmAdd";

  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  auto enable_fusion = (std::find(enable_op_list.begin(), enable_op_list.end(), kQbmmAddName) != enable_op_list.end());
  if (!enable_fusion) {
    return nullptr;
  }

  auto bias_node = utils::cast<AnfNodePtr>((*equiv)[bias_]);
  auto bias_tensor_node = utils::cast<AnfNodePtr>((*equiv)[bias_tensor_]);
  auto out_dtype_node = utils::cast<AnfNodePtr>((*equiv)[out_dtype_]);
  auto scale_node = utils::cast<AnfNodePtr>((*equiv)[scale_]);
  if (!CheckSupportDataType(bias_tensor_node, {kNumberTypeFloat16}) ||
      !CheckSupportDataType(scale_node, {kNumberTypeInt64}) || !CheckSupportDataType(bias_node, {kMetaTypeNone})) {
    return nullptr;
  }
  auto dtype_value = GetValue<int64_t>(out_dtype_node->cast<ValueNodePtr>()->value());
  if (dtype_value != static_cast<int64_t>(kNumberTypeFloat16)) {
    return nullptr;
  }
  auto bias_shape = common::AnfAlgo::GetOutputInferShape(bias_tensor_node, kIndex0);
  auto scale_shape = common::AnfAlgo::GetOutputInferShape(scale_node, kIndex0);
  if (bias_shape.size() != 1 || scale_shape.size() != 1 || bias_shape[0] != scale_shape[0]) {
    return nullptr;
  }
  auto scale_param = GetParamFromLoad(scale_node->cast<CNodePtr>(), false);
  if (!scale_param) {
    return nullptr;
  }
  auto bias_tensor_param = GetParamFromLoad(bias_tensor_node->cast<CNodePtr>(), false);
  if (!bias_tensor_param) {
    return nullptr;
  }

  MS_LOG(WARNING) << "QbmmAddFusion pass";
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }

  auto cnode = CreateQbmmAddNode(func_graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(WARNING) << "create QbmmAddFusion node failed.";
    return nullptr;
  }
  return cnode;
}

}  // namespace opt
}  // namespace mindspore
