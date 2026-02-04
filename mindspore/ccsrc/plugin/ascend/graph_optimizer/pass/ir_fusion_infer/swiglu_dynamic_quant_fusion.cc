
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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/swiglu_dynamic_quant_fusion.h"

#include <string>
#include <vector>
#include <set>

#include "backend/common/pass/common/gllo_utils.h"
#include "utils/ms_context.h"
#include "primitive/nn_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/pass_manager/helper.h"
#include "utils/shape_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
std::vector<std::string> SwiGLUDynamicQuantFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimDynamicQuantExt->name(), prim::kPrimSwiglu->name()};
  return ret;
}

const BaseRef SwiGLUDynamicQuantFusion::DefinePattern() const {
  auto is_swiglu = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSwiglu>);
  MS_CHECK_TRUE_RET(is_swiglu != nullptr, {});
  VectorRef swiglu_ref({is_swiglu, x_, axis_});
  auto is_dynamic_quant = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDynamicQuantExt>);
  MS_CHECK_TRUE_RET(is_dynamic_quant != nullptr, {});
  VectorRef dynamic_quant_ref({is_dynamic_quant, swiglu_ref, smooth_scale_});
  return dynamic_quant_ref;
}

CNodePtr SwiGLUDynamicQuantFusion::CreateSwiGLUDynamicQuantNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                                const EquivPtr &equiv) const {
  MS_LOG(DEBUG) << "start create SwiGLU DynamicQuant node";
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  std::string prim_name = "SwiGLUDynamicQuant";
  auto quant_prim = std::make_shared<Primitive>(prim_name);
  auto input_node = utils::cast<AnfNodePtr>((*equiv)[x_]);
  MS_EXCEPTION_IF_NULL(input_node);
  auto axis_node = utils::cast<AnfNodePtr>((*equiv)[axis_]);
  MS_EXCEPTION_IF_NULL(axis_node);
  if (!axis_node->isa<ValueNode>()) {
    MS_LOG(DEBUG) << "axis node is not a value node";
    return nullptr;
  }
  auto axis_vnode = axis_node->cast<ValueNodePtr>();
  quant_prim->AddAttr("dim", axis_vnode->value());
  auto smooth_scale_node = utils::cast<AnfNodePtr>((*equiv)[smooth_scale_]);
  MS_EXCEPTION_IF_NULL(smooth_scale_node);

  const std::set<TypeId> support_dtype = {kNumberTypeFloat16, kNumberTypeBFloat16};
  if (!CheckSupportDataType(input_node, support_dtype)) {
    return nullptr;
  }

  auto quant_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(quant_cnode != nullptr, nullptr);
  auto swiglu_cnode = quant_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(swiglu_cnode != nullptr, nullptr);
  auto swiglu_prim = common::AnfAlgo::GetCNodePrimitive(swiglu_cnode);
  MS_EXCEPTION_IF_NULL(swiglu_prim);
  auto attr_value = swiglu_prim->GetAttr("FusionType");
  quant_prim->AddAttr("FusionType", attr_value);

  std::vector<AnfNodePtr> quant_inputs = {std::make_shared<ValueNode>(quant_prim), input_node, smooth_scale_node};
  auto swiglu_dynamic_quant = NewCNode(quant_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(swiglu_dynamic_quant);

  std::vector<TypeId> swiglu_dymaimc_quant_out_types;
  std::vector<BaseShapePtr> swiglu_dymaimc_quant_out_shapes;
  auto quant_out_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  auto quant_out_shape = AnfAlgo::GetOutputDetailShape(node, 0);
  auto out_scale_type = common::AnfAlgo::GetOutputInferDataType(node, 1);
  auto out_scale_shape = AnfAlgo::GetOutputDetailShape(node, 1);
  swiglu_dymaimc_quant_out_types.push_back(quant_out_type);
  swiglu_dymaimc_quant_out_shapes.push_back(quant_out_shape);
  swiglu_dymaimc_quant_out_types.push_back(out_scale_type);
  swiglu_dymaimc_quant_out_shapes.push_back(out_scale_shape);

  common::AnfAlgo::SetOutputTypeAndDetailShape(swiglu_dymaimc_quant_out_types, swiglu_dymaimc_quant_out_shapes,
                                               swiglu_dynamic_quant.get());
  swiglu_dynamic_quant->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(swiglu_dynamic_quant);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, swiglu_dynamic_quant.get());
  MS_LOG(DEBUG) << "create SwiGLU DynamicQuant node success.";
  return swiglu_dynamic_quant;
}

const AnfNodePtr SwiGLUDynamicQuantFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                   const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }

  auto soc = ms_context->ascend_soc_version();
  if (!soc.empty() && soc.find("ascend910_93") == std::string::npos && soc.find("ascend910b") == std::string::npos) {
    MS_LOG(INFO) << "SwiGLUDynamicQuant does not support " << soc;
    return nullptr;
  }

  const std::string fusion_op_name = "SwiGLUDynamicQuant";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_swiglu_quant =
    (std::find(enable_op_list.begin(), enable_op_list.end(), fusion_op_name) != enable_op_list.end());
  if (!enable_swiglu_quant) {
    return nullptr;
  }

  auto cnode = CreateSwiGLUDynamicQuantNode(graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(DEBUG) << "create swiglu dynamic quant node failed.";
    return nullptr;
  }
  return cnode;
}
}  // namespace opt
}  // namespace mindspore
