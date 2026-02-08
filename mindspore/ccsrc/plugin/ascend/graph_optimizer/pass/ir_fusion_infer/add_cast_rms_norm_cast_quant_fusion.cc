/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/add_cast_rms_norm_cast_quant_fusion.h"

#include <string>
#include <vector>

#include "utils/ms_context.h"
#include "primitive/nn_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/pass_manager/optimizer.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_weight_preprocess_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
std::vector<std::string> AddCastRmsNormCastQuantFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimRmsNorm->name(), prim::kPrimAdd->name(), prim::kPrimQuantV2->name()};
  return ret;
}

const BaseRef AddCastRmsNormCastQuantFusion::DefinePattern() const {
  VarPtr index0 = std::make_shared<CondVar>(IsConstant);
  VarPtr x0 = std::make_shared<Var>();
  VarPtr x1 = std::make_shared<Var>();
  VectorRef add_cast = VectorRef({prim::kPrimCast, VectorRef({prim::kPrimAdd, x1_, x2_}), x0});
  VectorRef add_rms_norm = VectorRef({prim::kPrimRmsNorm, add_cast, gamma_, eps_});
  VectorRef tuple_get_item_0 = VectorRef({prim::kPrimTupleGetItem, add_rms_norm, index0});
  sqrt_mode_ = std::make_shared<CondVar>(IsConstant);
  rounding_mode_ = std::make_shared<CondVar>(IsConstant);
  dst_type_ = std::make_shared<CondVar>(IsConstant);
  VectorRef add_cast_rms_norm_cast = VectorRef({prim::kPrimCast, tuple_get_item_0, x1});
  VectorRef add_cast_rms_norm_cast_quant =
    VectorRef({prim::kPrimQuantV2, add_cast_rms_norm_cast, scale_, offset_, sqrt_mode_, rounding_mode_, dst_type_});
  return add_cast_rms_norm_cast_quant;
}

const AnfNodePtr AddCastRmsNormCastQuantFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                        const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const std::string fusion_op_name = "AddCastRmsNormCastQuantV2";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_add_rmsnorm =
    (std::find(enable_op_list.begin(), enable_op_list.end(), fusion_op_name) != enable_op_list.end());
  if (!enable_add_rmsnorm) {
    return nullptr;
  }
  auto cast_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  auto tuple_get_item_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(cast_node), 0);
  auto rms_norm_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(tuple_get_item_node), 0);
  auto tensor_add_cast = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_node), 0);
  auto tensor_add = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(tensor_add_cast), 0);
  auto shape1 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 0);
  auto shape2 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 1);
  FuncGraphManagerPtr mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  if (shape1 != shape2 || mng->node_users()[tuple_get_item_node].size() != 1) {
    return nullptr;
  }

  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto x2 = utils::cast<AnfNodePtr>((*equiv)[x2_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto scale = utils::cast<AnfNodePtr>((*equiv)[scale_]);
  auto offset = utils::cast<AnfNodePtr>((*equiv)[offset_]);
  auto eps = utils::cast<AnfNodePtr>((*equiv)[eps_]);
  auto scale_fp32 = ConvertWeightsToNewType(scale);
  auto offset_int32 = ConvertWeightsToNewType(offset);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  kernel_graph->AddValueNodeToGraph(scale_fp32);
  kernel_graph->AddValueNodeToGraph(offset_int32);
  auto sqrt_mode = GetValueNode(utils::cast<AnfNodePtr>((*equiv)[sqrt_mode_]));
  auto rounding_mode = GetValueNode(utils::cast<AnfNodePtr>((*equiv)[rounding_mode_]));
  auto dst_type = GetValueNode(utils::cast<AnfNodePtr>((*equiv)[dst_type_]));

  auto prim = std::make_shared<Primitive>("AddRmsNormQuantV2");
  prim->set_attr("sqrt_mode", sqrt_mode);
  prim->set_attr("rounding_mode", rounding_mode);
  prim->set_attr("dst_type", dst_type);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x1, x2, gamma, scale_fp32, offset_int32, eps};
  auto add_rms_norm_quant = NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(add_rms_norm_quant);

  std::vector<TypeId> types;
  std::vector<BaseShapePtr> shapes;
  std::vector<TypeId> add_result_types;
  std::vector<BaseShapePtr> add_result_shapes;
  std::vector<TypeId> quant_result_types;
  std::vector<BaseShapePtr> quant_result_shapes;
  size_t output_num = AnfAlgo::GetOutputElementNum(node);
  if (output_num != 1) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << node->cast<CNodePtr>()->fullname_with_scope() << " output_num " << output_num
                                      << " != 1.";
  }
  auto tensor_quant_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  auto tensor_quant_shape = AnfAlgo::GetOutputDetailShape(node, 0);
  auto tensor_add_type = common::AnfAlgo::GetOutputInferDataType(tensor_add, 0);
  auto tensor_add_shape = AnfAlgo::GetOutputDetailShape(tensor_add, 0);

  for (size_t i = 0; i < output_num + 1; i++) {
    types.push_back(tensor_quant_type);
    shapes.push_back(tensor_quant_shape);
  }
  types.push_back(tensor_add_type);
  shapes.push_back(tensor_add_shape);
  quant_result_types.push_back(tensor_quant_type);
  quant_result_shapes.push_back(tensor_quant_shape);
  add_result_types.push_back(tensor_add_type);
  add_result_shapes.push_back(tensor_add_shape);
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, add_rms_norm_quant.get());
  add_rms_norm_quant->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(add_rms_norm_quant);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_rms_norm_quant.get());

  auto prim_getitem_2 = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> add_result_inputs = {NewValueNode(prim_getitem_2), add_rms_norm_quant,
                                               NewValueNode(static_cast<int64_t>(2))};
  auto add_result = NewCNode(add_result_inputs, graph);
  common::AnfAlgo::SetOutputTypeAndDetailShape(add_result_types, add_result_shapes, add_result.get());
  add_result->set_scope(tensor_add->scope());
  build_info = GenerateKernelBuildInfo(add_result);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_result.get());
  (void)mng->Replace(tensor_add, add_result);

  auto prim_getitem_0 = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> quant_result_inputs = {NewValueNode(prim_getitem_0), add_rms_norm_quant,
                                                 NewValueNode(static_cast<int64_t>(0))};
  auto quant_result = NewCNode(quant_result_inputs, graph);
  common::AnfAlgo::SetOutputTypeAndDetailShape(quant_result_types, quant_result_shapes, quant_result.get());
  quant_result->set_scope(node->scope());
  build_info = GenerateKernelBuildInfo(quant_result);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, quant_result.get());

  return quant_result;
}

}  // namespace opt
}  // namespace mindspore
