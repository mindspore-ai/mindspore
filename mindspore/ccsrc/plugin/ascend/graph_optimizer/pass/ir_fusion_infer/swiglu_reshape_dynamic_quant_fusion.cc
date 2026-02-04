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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/swiglu_reshape_dynamic_quant_fusion.h"

#include <string>
#include <vector>
#include <set>

#include "backend/common/pass/common/gllo_utils.h"
#include "primitive/nn_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/pass_manager/helper.h"
#include "utils/shape_utils.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
std::vector<std::string> SwiGLUReshapeDynamicQuantFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimDynamicQuantExt->name(), prim::kPrimSwiglu->name()};
  return ret;
}

const BaseRef SwiGLUReshapeDynamicQuantFusion::DefinePattern() const {
  auto is_swiglu = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSwiglu>);
  MS_CHECK_TRUE_RET(is_swiglu != nullptr, {});
  VectorRef swiglu_ref({is_swiglu, x_, axis_});
  auto is_reshape = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReshape>);
  MS_CHECK_TRUE_RET(is_reshape != nullptr, {});
  auto is_reshape_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_reshape_var != nullptr, {});
  VectorRef reshape_ref({is_reshape, swiglu_ref, is_reshape_var});
  auto is_dynamic_quant = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDynamicQuantExt>);
  MS_CHECK_TRUE_RET(is_dynamic_quant != nullptr, {});
  VectorRef dynamic_quant_ref({is_dynamic_quant, reshape_ref, smooth_scale_});
  return dynamic_quant_ref;
}

CNodePtr SwiGLUReshapeDynamicQuantFusion::CreateSwiGLUReshapeDynamicQuantNode(const FuncGraphPtr &func_graph,
                                                                              const AnfNodePtr &node,
                                                                              const EquivPtr &equiv) const {
  MS_LOG(DEBUG) << "start create SwiGLU Reshape DynamicQuant node";
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  std::string prim_name = "SwiGLUDynamicQuant";
  auto dyn_quant_prim = std::make_shared<Primitive>(prim_name);
  auto x_node = utils::cast<AnfNodePtr>((*equiv)[x_]);
  MS_EXCEPTION_IF_NULL(x_node);
  auto axis_node = utils::cast<AnfNodePtr>((*equiv)[axis_]);
  MS_EXCEPTION_IF_NULL(axis_node);
  if (!axis_node->isa<ValueNode>()) {
    MS_LOG(DEBUG) << "axis node is not a value node";
    return nullptr;
  }
  auto axis_vnode = axis_node->cast<ValueNodePtr>();
  dyn_quant_prim->AddAttr("dim", axis_vnode->value());
  auto smooth_scale_node = utils::cast<AnfNodePtr>((*equiv)[smooth_scale_]);
  MS_EXCEPTION_IF_NULL(smooth_scale_node);

  const std::set<TypeId> support_dtype = {kNumberTypeFloat16, kNumberTypeBFloat16};
  if (!CheckSupportDataType(x_node, support_dtype)) {
    return nullptr;
  }

  auto quant_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(quant_cnode != nullptr, nullptr);
  auto reshape_cnode = quant_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reshape_cnode != nullptr, nullptr);
  auto swiglu_cnode = reshape_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(swiglu_cnode != nullptr, nullptr);
  auto swiglu_shape = common::AnfAlgo::GetOutputInferShape(swiglu_cnode, kIndex0);
  auto reshape_shape = common::AnfAlgo::GetOutputInferShape(reshape_cnode, kIndex0);
  if (swiglu_shape.size() != kDim3 || reshape_shape.size() != kDim2) {
    MS_LOG(DEBUG) << "The shapes of input swiglu and reshape are expected to be 3 and 2 dim respectively,"
                  << "but got swiglu shape: " << swiglu_shape.size() << ", reshape shape: " << reshape_shape.size();
    return nullptr;
  }
  auto swiglu_prim = common::AnfAlgo::GetCNodePrimitive(swiglu_cnode);
  MS_EXCEPTION_IF_NULL(swiglu_prim);
  auto attr_value = swiglu_prim->GetAttr("FusionType");
  dyn_quant_prim->AddAttr("FusionType", attr_value);
  dyn_quant_prim->AddAttr("HasReshape", MakeValue<bool>(true));

  std::vector<AnfNodePtr> quant_inputs = {std::make_shared<ValueNode>(dyn_quant_prim), x_node, smooth_scale_node};
  auto swiglu_dyn_quant = NewCNode(quant_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(swiglu_dyn_quant);

  std::vector<TypeId> swiglu_dyn_quant_out_types;
  std::vector<BaseShapePtr> swiglu_dyn_quant_out_shapes;
  auto quant_out_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  auto quant_out_shape = AnfAlgo::GetOutputDetailShape(node, 0);
  auto out_scale_type = common::AnfAlgo::GetOutputInferDataType(node, 1);
  auto out_scale_shape = AnfAlgo::GetOutputDetailShape(node, 1);
  swiglu_dyn_quant_out_types.push_back(quant_out_type);
  swiglu_dyn_quant_out_shapes.push_back(quant_out_shape);
  swiglu_dyn_quant_out_types.push_back(out_scale_type);
  swiglu_dyn_quant_out_shapes.push_back(out_scale_shape);

  common::AnfAlgo::SetOutputTypeAndDetailShape(swiglu_dyn_quant_out_types, swiglu_dyn_quant_out_shapes,
                                               swiglu_dyn_quant.get());
  swiglu_dyn_quant->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(swiglu_dyn_quant);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, swiglu_dyn_quant.get());
  MS_LOG(DEBUG) << "create SwiGLU Reshape DynamicQuant node success.";
  return swiglu_dyn_quant;
}

const AnfNodePtr SwiGLUReshapeDynamicQuantFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                          const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }

  auto soc = ms_context->ascend_soc_version();
  if (!soc.empty() && soc.find("ascend910_93") == std::string::npos && soc.find("ascend910b") == std::string::npos) {
    MS_LOG(INFO) << "SwiGLUReshapeDynamicQuant does not support " << soc;
    return nullptr;
  }

  const std::string fusion_op_name = "SwiGLUReshapeDynamicQuant";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_swiglu_quant =
    (std::find(enable_op_list.begin(), enable_op_list.end(), fusion_op_name) != enable_op_list.end());
  if (!enable_swiglu_quant) {
    return nullptr;
  }

  auto cnode = CreateSwiGLUReshapeDynamicQuantNode(graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(DEBUG) << "create swiglu reshape dynamic quant node failed.";
    return nullptr;
  }
  return cnode;
}
}  // namespace opt
}  // namespace mindspore
