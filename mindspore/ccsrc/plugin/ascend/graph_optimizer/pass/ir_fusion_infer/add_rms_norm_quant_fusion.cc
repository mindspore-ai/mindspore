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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/add_rms_norm_quant_fusion.h"

#include <string>
#include <vector>

#include "utils/ms_context.h"
#include "primitive/nn_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/pass_manager/helper.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_weight_preprocess_utils.h"
#include "utils/shape_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kRmsNormOutUserSize1 = 1;

bool UnSupportedType(const AnfNodePtr &node, const AnfNodePtr &rms_norm_node) {
  auto rms_x_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(rms_norm_node, 0);
  auto rms_gamma_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(rms_norm_node, 1);
  auto scale_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 1);
  auto offset_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 2);
  auto is_offset_dtype_ok = (offset_dtype == kNumberTypeInt8) || (offset_dtype == scale_dtype);
  bool is_unsupported_type = (rms_x_dtype != kNumberTypeFloat16 && rms_x_dtype != kNumberTypeBFloat16) ||
                             (rms_gamma_dtype != kNumberTypeFloat16 && rms_gamma_dtype != kNumberTypeBFloat16) ||
                             (scale_dtype != kNumberTypeFloat16 && scale_dtype != kNumberTypeBFloat16) ||
                             !is_offset_dtype_ok;
  return is_unsupported_type;
}
}  // namespace
std::vector<std::string> AddRmsNormQuantFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimRmsNorm->name(), prim::kPrimAdd->name(), prim::kPrimQuantV2->name()};
  return ret;
}

const BaseRef AddRmsNormQuantFusion::DefinePattern() const {
  VectorRef add_rms_norm = VectorRef({prim::kPrimRmsNorm, VectorRef({prim::kPrimAdd, x1_, x2_}), gamma_, eps_});
  VarPtr index0 = std::make_shared<CondVar>(IsConstant);
  VectorRef tuple_get_item_0 = VectorRef({prim::kPrimTupleGetItem, add_rms_norm, index0});
  sqrt_mode_ = std::make_shared<CondVar>(IsConstant);
  rounding_mode_ = std::make_shared<CondVar>(IsConstant);
  dst_type_ = std::make_shared<CondVar>(IsConstant);
  VectorRef add_rms_norm_quant =
    VectorRef({prim::kPrimQuantV2, tuple_get_item_0, scale_, offset_, sqrt_mode_, rounding_mode_, dst_type_});
  return add_rms_norm_quant;
}

inline bool CheckSupport(size_t rms_norm_out0_users_size, const AnfNodePtr &node, const AnfNodePtr &tensor_add,
                         const AnfNodePtr &rms_norm_node, const BaseShapePtr &tensor_quant_shape) {
  auto shape1 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 0);
  auto shape2 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 1);
  bool is_unsupported_type = UnSupportedType(node, rms_norm_node);
  constexpr size_t kMaxRmsNormOutUserSize = 3;
  if (shape1 != shape2 || rms_norm_out0_users_size > kMaxRmsNormOutUserSize || is_unsupported_type) {
    MS_LOG(INFO) << "AddRmsNormQuant fused failed. shape1: " << shape1 << ", shape2: " << shape2
                 << ", rms_norm_out0_users_size: " << rms_norm_out0_users_size
                 << ", is_unsupported_type: " << is_unsupported_type;
    return false;
  }

  if ((rms_norm_out0_users_size > kRmsNormOutUserSize1)) {
    auto shape = tensor_quant_shape->GetShapeVector();
    constexpr auto kMaxDimLimited = 8192;
    constexpr auto kMinDimLimited = 4000;
    if (shape.empty() || shape[shape.size() - 1] == abstract::Shape::kShapeDimAny ||
        shape[shape.size() - 1] > kMaxDimLimited || shape[shape.size() - 1] < kMinDimLimited) {
      MS_LOG(INFO) << "AddRmsNormQuant fused failed when need_rms_norm_out is true and shape is: " << shape
                   << ", because the kernel does not support.";
      return false;
    }
  }

  return true;
}

const AnfNodePtr AddRmsNormQuantFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                const EquivPtr &equiv) const {
  auto tuple_get_item_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  auto rms_norm_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(tuple_get_item_node), 0);
  auto tensor_add = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_node), 0);
  FuncGraphManagerPtr mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto rms_norm_out0_users_size = mng->node_users()[tuple_get_item_node].size();
  auto tensor_quant_shape = AnfAlgo::GetOutputDetailShape(node, 0);

  if (!CheckSupport(rms_norm_out0_users_size, node, tensor_add, rms_norm_node, tensor_quant_shape)) {
    MS_LOG(INFO) << "AddRmsNormQuant fused failed.";
    return nullptr;
  }

  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto x2 = utils::cast<AnfNodePtr>((*equiv)[x2_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto scale = utils::cast<AnfNodePtr>((*equiv)[scale_]);
  auto offset = utils::cast<AnfNodePtr>((*equiv)[offset_]);
  auto eps = utils::cast<AnfNodePtr>((*equiv)[eps_]);
  auto scale_fp32 = ConvertWeightsToNewType(scale);

  auto kernel_graph = graph->cast<KernelGraphPtr>();
  kernel_graph->AddValueNodeToGraph(scale_fp32);
  auto prim = std::make_shared<Primitive>("AddRmsNormQuantV2");

  std::vector<AnfNodePtr> inputs;
  static const auto kOffSetIndex = 2;
  auto offset_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, kOffSetIndex);
  if (offset_dtype == kNumberTypeInt8) {
    auto offset_int32 = ConvertWeightsToNewType(offset);
    kernel_graph->AddValueNodeToGraph(offset_int32);
    inputs = {NewValueNode(prim), x1, x2, gamma, scale_fp32, offset_int32, eps};
  } else {
    inputs = {NewValueNode(prim), x1, x2, gamma, scale_fp32, offset, eps};
  }

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

  bool need_rms_norm_out = (rms_norm_out0_users_size > kRmsNormOutUserSize1);
  types[1] = need_rms_norm_out ? tensor_add_type : tensor_quant_type;
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

  if (need_rms_norm_out) {
    prim->set_attr("need_rms_norm_out", std::make_shared<BoolImm>(true));
    auto prim_getitem_rms_norm_out = std::make_shared<Primitive>("TupleGetItem");
    auto rms_norm_out = NewCNode(
      {NewValueNode(prim_getitem_rms_norm_out), add_rms_norm_quant, NewValueNode(static_cast<int64_t>(1))}, graph);
    common::AnfAlgo::SetOutputTypeAndDetailShape(add_result_types, quant_result_shapes, rms_norm_out.get());
    rms_norm_out->set_scope(node->scope());
    (void)mng->Replace(tuple_get_item_node, rms_norm_out);
  }

  MS_LOG(INFO) << "AddRmsNormQuant fused successfully with need_rms_norm_out: " << need_rms_norm_out;

  return quant_result;
}

const BaseRef AddRmsNormDynamicQuantFusion::DefinePattern() const {
  auto add_rms_norm = VectorRef({prim::kPrimRmsNorm, VectorRef({prim::kPrimAdd, x1_, x2_}), gamma_, eps_});
  auto index0 = std::make_shared<CondVar>(IsConstant);
  auto tuple_get_item_0 = VectorRef({prim::kPrimTupleGetItem, add_rms_norm, index0});
  auto add_rms_norm_dynamic_quant = VectorRef(
    {prim::kPrimDynamicQuantExt, VectorRef({prim::kPrimReshape, tuple_get_item_0, new_shape_}), smooth_scale_});
  return add_rms_norm_dynamic_quant;
}

bool IsSupport(const AnfNodePtr &node, const FuncGraphPtr &graph) {
  auto reshape_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  auto rms_norm_out_getitem_0 = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(reshape_node), 0);
  auto rms_norm_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_out_getitem_0), 0);
  auto tensor_add = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_node), 0);
  auto shape1 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 0);
  auto shape2 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 1);
  auto rms_x_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(rms_norm_node, 0);
  auto rms_gamma_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(rms_norm_node, 1);
  auto smooth_scale_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 1);
  FuncGraphManagerPtr mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);

  auto reshape_users_num = GetRealNodeUsedList(graph, reshape_node)->size();
  if (reshape_users_num > 1) {
    MS_LOG(INFO) << "The number of users of reshape_node is more than one: " << reshape_users_num;
    return false;
  }

  const size_t kUserNum = 2;
  if (mng->node_users()[rms_norm_out_getitem_0].size() != kUserNum) {
    MS_LOG(INFO) << "The number of users of the rms_norm's first output must be 2, but got: "
                 << mng->node_users()[rms_norm_out_getitem_0].size();
    return false;
  }

  if (shape1 != shape2) {
    MS_LOG(INFO) << "AddRmsNormDynamicQuant does not support broadcast, but got shape1: " << shape1
                 << ", shape2: " << shape2;
    return false;
  }

  if ((rms_x_dtype != kNumberTypeFloat16 && rms_x_dtype != kNumberTypeBFloat16) ||
      (rms_gamma_dtype != kNumberTypeFloat16 && rms_gamma_dtype != kNumberTypeBFloat16) ||
      (smooth_scale_dtype != kMetaTypeNone && smooth_scale_dtype != kNumberTypeFloat16 &&
       smooth_scale_dtype != kNumberTypeBFloat16)) {
    MS_LOG(INFO) << "Dtype not match, x_dtype: " << rms_x_dtype << ", rms_gamma_dtype: " << rms_gamma_dtype
                 << ", smooth_scale_dtype: " << smooth_scale_dtype;
    return false;
  }

  auto dynamic_quant_outlist = GetRealNodeUsedList(graph, node);
  for (const auto &out : *dynamic_quant_outlist) {
    auto item_index_input = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(out.first), 1);
    if (!item_index_input->isa<ValueNode>()) {
      MS_LOG(INFO) << "This is not a valuenode: " << item_index_input->fullname_with_scope();
      return false;
    }
  }

  // check the user of rstd
  return true;
}

AnfNodePtr GetShapeInputNode(const FuncGraphPtr &graph, const AnfNodePtr &rms_norm_out_getitem_0) {
  auto rms_norm_out_getitem_0_users = GetRealNodeUsedList(graph, rms_norm_out_getitem_0);
  AnfNodePtr rms_norm_out_getitem_0_shape_user = nullptr;
  for (const auto &user : *rms_norm_out_getitem_0_users) {
    if (IsPrimitiveCNode(user.first, prim::kPrimShape)) {
      rms_norm_out_getitem_0_shape_user = user.first;
      break;
    }
  }

  if (rms_norm_out_getitem_0_shape_user == nullptr) {
    MS_LOG(INFO) << "The users of the output0 of RmsNorm are not Shape and DynamicQuant";
    return nullptr;
  }

  return common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_out_getitem_0_shape_user), 0);
}

void ReplaceAddResult(const FuncGraphPtr &graph, const FuncGraphManagerPtr &mng,
                      const AnfNodePtr &add_rms_norm_dynamic_quant, const AnfNodePtr &ori_add_node,
                      const AnfNodePtr &shape_input_node, const std::vector<BaseShapePtr> &add_result_shapes,
                      const std::vector<TypeId> &add_result_types) {
  auto constexpr kNewAddOutIdx = 2;
  auto getitem_for_add = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> add_result_inputs = {NewValueNode(getitem_for_add), add_rms_norm_dynamic_quant,
                                               NewValueNode(static_cast<int64_t>(kNewAddOutIdx))};
  auto add_result = graph->NewCNode(add_result_inputs);
  common::AnfAlgo::SetOutputTypeAndDetailShape(add_result_types, add_result_shapes, add_result.get());
  add_result->set_scope(ori_add_node->scope());
  auto build_info = GenerateKernelBuildInfo(add_result);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_result.get());
  (void)mng->Replace(ori_add_node, add_result);
  (void)mng->Replace(shape_input_node, add_result);
}

AnfNodePtr NewReshapedQuantOut(const FuncGraphPtr &graph, const AnfNodePtr &add_rms_norm_dynamic_quant,
                               const AnfNodePtr &node, const BaseShapePtr &raw_quant_shape, const TypeId &quant_type,
                               const BaseShapePtr &reshaped_shape, const AnfNodePtr &ori_shape_node) {
  auto constexpr kNewQuantOutIdx = 0;
  std::vector<TypeId> getitem_for_quant_types{quant_type};
  std::vector<BaseShapePtr> getitem_for_quant_shapes{raw_quant_shape};
  auto getitem_for_quant = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> quant_inputs = {NewValueNode(getitem_for_quant), add_rms_norm_dynamic_quant,
                                          NewValueNode(static_cast<int64_t>(kNewQuantOutIdx))};
  auto quant_item = graph->NewCNode(quant_inputs);
  common::AnfAlgo::SetOutputTypeAndDetailShape(getitem_for_quant_types, getitem_for_quant_shapes, quant_item.get());
  quant_item->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(quant_item);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, quant_item.get());

  std::vector<TypeId> reshape_quant_types{quant_type};
  std::vector<BaseShapePtr> reshape_quant_shapes{reshaped_shape};
  auto reshape_quant_prim = std::make_shared<Primitive>("Reshape");
  std::vector<AnfNodePtr> reshape_quant_inputs = {NewValueNode(reshape_quant_prim), quant_item, ori_shape_node};
  auto reshape_quant = graph->NewCNode(reshape_quant_inputs);
  common::AnfAlgo::SetOutputTypeAndDetailShape(reshape_quant_types, reshape_quant_shapes, reshape_quant.get());
  reshape_quant->set_scope(node->scope());
  build_info = GenerateKernelBuildInfo(reshape_quant);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, reshape_quant.get());

  return reshape_quant;
}

AnfNodePtr NewReshapedScale1Out(const FuncGraphPtr &graph, const AnfNodePtr &add_rms_norm_dynamic_quant,
                                const AnfNodePtr &node, const BaseShapePtr &raw_scale_shape,
                                const TypeId &out_scale_type, const BaseShapePtr &ori_out_scale_shape) {
  auto constexpr kNewScale1OutIdx = 3;
  std::vector<TypeId> getitem_for_scale1_types{out_scale_type};
  std::vector<BaseShapePtr> getitem_for_scale1_shapes{raw_scale_shape};
  auto getitem_for_scale1 = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> scale1_inputs = {NewValueNode(getitem_for_scale1), add_rms_norm_dynamic_quant,
                                           NewValueNode(static_cast<int64_t>(kNewScale1OutIdx))};
  auto scale1_item = graph->NewCNode(scale1_inputs);
  common::AnfAlgo::SetOutputTypeAndDetailShape(getitem_for_scale1_types, getitem_for_scale1_shapes, scale1_item.get());
  scale1_item->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(scale1_item);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, scale1_item.get());

  std::vector<TypeId> reshape_scale1_types{out_scale_type};
  std::vector<BaseShapePtr> reshape_scale1_shapes{ori_out_scale_shape};

  auto reshape_scale1_prim = std::make_shared<Primitive>("Reshape");
  ShapeVector new_shape{-1};
  std::vector<AnfNodePtr> reshape_scale_inputs = {NewValueNode(reshape_scale1_prim), scale1_item,
                                                  CreateShapeValueNode(graph, new_shape, false)};
  auto reshape_scale = graph->NewCNode(reshape_scale_inputs);
  common::AnfAlgo::SetOutputTypeAndDetailShape(reshape_scale1_types, reshape_scale1_shapes, reshape_scale.get());
  reshape_scale->set_scope(node->scope());
  build_info = GenerateKernelBuildInfo(reshape_scale);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, reshape_scale.get());

  return reshape_scale;
}

void ReplaceDynamicQuantOut(const FuncGraphPtr &graph, const FuncGraphManagerPtr &mng, const AnfNodePtr &node,
                            const AnfNodePtr &reshape_quant, const AnfNodePtr &reshape_scale) {
  auto dynamic_quant_outlist = GetRealNodeUsedList(graph, node);
  auto constexpr kOriQuantOutIdx = 0;
  for (const auto &out : *dynamic_quant_outlist) {
    auto &ori_getitem = out.first;
    auto item_index_input = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(ori_getitem), 1);
    auto item_index_input_value_ptr = item_index_input->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(item_index_input_value_ptr);
    auto idx = GetValue<int64_t>(item_index_input_value_ptr->value());
    if (idx == kOriQuantOutIdx) {
      (void)mng->Replace(ori_getitem, reshape_quant);
    } else {
      (void)mng->Replace(ori_getitem, reshape_scale);
    }
  }
}

const AnfNodePtr AddRmsNormDynamicQuantFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                       const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto soc = ms_context->ascend_soc_version();
  if (soc.find("ascend910_93") == std::string::npos && soc.find("ascend910b") == std::string::npos) {
    MS_LOG(INFO) << "AddRmsNormDynamicQuant does not support " << soc;
    return nullptr;
  }

  const std::string fusion_op_name = "AddRmsNormDynamicQuant";
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_add_rmsnorm =
    (std::find(enable_op_list.begin(), enable_op_list.end(), fusion_op_name) != enable_op_list.end());
  if (!enable_add_rmsnorm) {
    return nullptr;
  }

  auto reshape_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  auto rms_norm_out_getitem_0 = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(reshape_node), 0);
  auto rms_norm_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_out_getitem_0), 0);
  auto tensor_add = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_node), 0);
  FuncGraphManagerPtr mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);

  if (!IsSupport(node, graph)) {
    return nullptr;
  }

  auto shape_input_node = GetShapeInputNode(graph, rms_norm_out_getitem_0);
  if (shape_input_node == nullptr) {
    return nullptr;
  }

  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto x2 = utils::cast<AnfNodePtr>((*equiv)[x2_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto smooth_scale1 = utils::cast<AnfNodePtr>((*equiv)[smooth_scale_]);
  auto eps = utils::cast<AnfNodePtr>((*equiv)[eps_]);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  auto new_shape = utils::cast<AnfNodePtr>((*equiv)[new_shape_]);

  auto quant_out_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  auto quant_out_shape = AnfAlgo::GetOutputDetailShape(node, 0);
  auto rms_out_shape = AnfAlgo::GetOutputDetailShape(rms_norm_node, 0);
  auto tensor_add_type = common::AnfAlgo::GetOutputInferDataType(tensor_add, 0);
  auto tensor_add_shape = AnfAlgo::GetOutputDetailShape(tensor_add, 0);

  MS_EXCEPTION_IF_NULL(rms_out_shape);
  auto shape_vec = rms_out_shape->GetShapeVector();
  if (IsDynamicRank(shape_vec)) {
    MS_LOG(INFO) << "AddRmsNormDynamicQuantFusion does not support dynamic rank.";
    return nullptr;
  }

  auto out_scale_type = common::AnfAlgo::GetOutputInferDataType(node, 1);
  ShapeVector out_scale_shape_vec(shape_vec.begin(), shape_vec.end() - 1);
  auto out_scale_shape = std::make_shared<abstract::Shape>(out_scale_shape_vec);
  auto ori_out_scale_shape = AnfAlgo::GetOutputDetailShape(node, 1);

  auto prim = std::make_shared<Primitive>("AddRmsNormDynamicQuant");
  auto smooth_scale2 = kernel_graph->NewValueNode(kNone->ToAbstract(), kNone);
  kernel_graph->AddValueNodeToGraph(smooth_scale2);

  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x1, x2, gamma, smooth_scale1, smooth_scale2, eps};
  auto add_rms_norm_dynamic_quant = NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(add_rms_norm_dynamic_quant);
  std::vector<TypeId> add_rms_norm_dymaimc_quant_out_types;
  std::vector<BaseShapePtr> add_rms_norm_dymaimc_quant_out_shapes;
  std::vector<TypeId> add_result_types;
  std::vector<BaseShapePtr> add_result_shapes;

  add_rms_norm_dymaimc_quant_out_types.push_back(quant_out_type);
  add_rms_norm_dymaimc_quant_out_shapes.push_back(rms_out_shape);
  add_rms_norm_dymaimc_quant_out_types.push_back(quant_out_type);
  add_rms_norm_dymaimc_quant_out_shapes.push_back(rms_out_shape);
  add_rms_norm_dymaimc_quant_out_types.push_back(tensor_add_type);
  add_rms_norm_dymaimc_quant_out_shapes.push_back(tensor_add_shape);
  add_rms_norm_dymaimc_quant_out_types.push_back(out_scale_type);
  add_rms_norm_dymaimc_quant_out_shapes.push_back(out_scale_shape);
  add_rms_norm_dymaimc_quant_out_types.push_back(out_scale_type);
  add_rms_norm_dymaimc_quant_out_shapes.push_back(out_scale_shape);

  add_result_types.push_back(tensor_add_type);
  add_result_shapes.push_back(tensor_add_shape);

  common::AnfAlgo::SetOutputTypeAndDetailShape(add_rms_norm_dymaimc_quant_out_types,
                                               add_rms_norm_dymaimc_quant_out_shapes, add_rms_norm_dynamic_quant.get());
  add_rms_norm_dynamic_quant->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(add_rms_norm_dynamic_quant);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_rms_norm_dynamic_quant.get());

  ReplaceAddResult(graph, mng, add_rms_norm_dynamic_quant, tensor_add, shape_input_node, add_result_shapes,
                   add_result_types);
  auto reshape_quant = NewReshapedQuantOut(graph, add_rms_norm_dynamic_quant, node, rms_out_shape, quant_out_type,
                                           quant_out_shape, new_shape);
  auto reshape_scale =
    NewReshapedScale1Out(graph, add_rms_norm_dynamic_quant, node, out_scale_shape, out_scale_type, ori_out_scale_shape);

  ReplaceDynamicQuantOut(graph, mng, node, reshape_quant, reshape_scale);

  return add_rms_norm_dynamic_quant;
}
}  // namespace opt
}  // namespace mindspore
