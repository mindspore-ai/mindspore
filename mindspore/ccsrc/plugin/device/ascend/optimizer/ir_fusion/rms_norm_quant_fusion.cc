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
#include "plugin/device/ascend/optimizer/ir_fusion/rms_norm_quant_fusion.h"

#include <vector>
#include <string>

#include "mindspore/core/utils/ms_context.h"
#include "mindspore/core/ops/nn_ops.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
const BaseRef RmsNormQuantFusion::DefinePattern() const {
  VarPtr index0 = std::make_shared<CondVar>(IsConstant);
  VectorRef rms_norm = VectorRef({prim::kPrimRmsNorm, x1_, gamma_, eps_});
  VectorRef tuple_get_item_0 = VectorRef({prim::kPrimTupleGetItem, rms_norm, index0});
  VarPtr x0 = std::make_shared<CondVar>(IsConstant);
  VarPtr x1 = std::make_shared<CondVar>(IsConstant);
  VarPtr x2 = std::make_shared<CondVar>(IsConstant);
  VectorRef rms_norm_quant = VectorRef({prim::kPrimQuantV2, tuple_get_item_0, scale_, offset_, x0, x1, x2});
  return rms_norm_quant;
}

const AnfNodePtr RmsNormQuantFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }

  const std::string fusion_op_name = "RmsNormQuant";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_add_rmsnorm =
    (std::find(enable_op_list.begin(), enable_op_list.end(), fusion_op_name) != enable_op_list.end());
  if (!enable_add_rmsnorm) {
    return nullptr;
  }

  auto tuple_get_item_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  auto rms_norm_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(tuple_get_item_node), 0);
  MS_EXCEPTION_IF_NULL(rms_norm_node);
  auto x_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(rms_norm_node, 0);
  auto gamma_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(rms_norm_node, 1);
  auto scale_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 1);
  auto offset_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 2);
  if (x_dtype != kNumberTypeFloat16 || gamma_dtype != kNumberTypeFloat16 || scale_dtype != kNumberTypeFloat16 ||
      offset_dtype != kNumberTypeInt8) {
    return nullptr;
  }

  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto beta = NewValueNode(kNone);
  beta->set_abstract(kNone->ToAbstract());
  auto scale = utils::cast<AnfNodePtr>((*equiv)[scale_]);
  auto offset = utils::cast<AnfNodePtr>((*equiv)[offset_]);
  auto epsilon_value = GetValueNode(utils::cast<AnfNodePtr>((*equiv)[eps_]));
  MS_EXCEPTION_IF_NULL(epsilon_value);

  auto prim = std::make_shared<Primitive>("RmsNormQuant");
  prim->set_attr("epsilon", epsilon_value);

  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x1, gamma, gamma, scale, offset};
  auto rms_norm_quant = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(rms_norm_quant);

  std::vector<TypeId> types;
  std::vector<BaseShapePtr> shapes;
  size_t output_num = AnfAlgo::GetOutputElementNum(node);
  for (size_t i = 0; i < output_num; i++) {
    types.push_back(common::AnfAlgo::GetOutputInferDataType(node, i));
    shapes.push_back(AnfAlgo::GetOutputDetailShape(node, i));
  }

  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, rms_norm_quant.get());
  rms_norm_quant->set_scope(node->scope());

  auto build_info = GenerateKernelBuildInfo(rms_norm_quant);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, rms_norm_quant.get());

  return rms_norm_quant;
}

}  // namespace opt
}  // namespace mindspore
