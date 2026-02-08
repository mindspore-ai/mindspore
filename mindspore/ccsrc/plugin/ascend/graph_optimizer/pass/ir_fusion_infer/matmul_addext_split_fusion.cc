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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_addext_split_fusion.h"
#include <memory>
#include <vector>
#include <set>
#include "backend/common/pass/common/gllo_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "utils/ms_context.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"

namespace mindspore {
namespace opt {
std::vector<std::string> MatmulAddExtSplitFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret = {prim::kPrimMatMul->name(), prim::kPrimSplitWithSize->name(),
                                  prim::kPrimAddExt->name()};
  return ret;
}

const BaseRef MatmulAddExtSplitFusion::DefinePattern() const {
  auto matmul_ref = GetMatmulPattern();
  auto add_input = std::make_shared<Var>();
  auto add_value = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add_input != nullptr, {});
  MS_CHECK_TRUE_RET(add_value != nullptr, {});
  auto is_addext = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAddExt>);
  MS_CHECK_TRUE_RET(is_addext != nullptr, {});
  auto addext_ref = VectorRef({is_addext, matmul_ref, add_input, add_value});
  auto split_with_size_ref = GetSplitWithSizePattern(addext_ref);
  return split_with_size_ref;
}

const AnfNodePtr MatmulAddExtSplitFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                  const EquivPtr &equiv) const {
  if (!IsEnableMatmulSplit()) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  MS_EXCEPTION_IF_NULL(graph);
  auto [split_cnode, split_size_node] = GetSplitSizeNode(node);
  auto add_cnode = split_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add_cnode != nullptr, {});
  auto [matmul_cnode, input_x, input_w, input_trans_a, input_trans_b] = GetMatmulNode(add_cnode);
  MS_CHECK_TRUE_RET(matmul_cnode->func_graph() == split_cnode->func_graph(), {});
  auto input_bias = add_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(input_bias);
  if (!CheckMatmulSplit(input_x, input_w, input_trans_a, input_trans_b, split_size_node) ||
      !CheckSupportDataType(input_bias, kSupportDataType)) {
    return nullptr;
  }
  PrimitivePtr matmul_add_split_prim = GetMatmulSplitPrimitive(split_size_node);
  AnfNodePtrList matmul_add_split_inputs = GetMatmulSplitInputs(input_x, input_w, input_bias, graph, matmul_cnode);
  auto matmul_add_split_cnode =
    GetMatmulSplitCNode(matmul_add_split_prim, matmul_add_split_inputs, graph, matmul_cnode, split_cnode);
  return matmul_add_split_cnode;
}

AnfNodePtrList MatmulAddExtSplitFusion::GetMatmulSplitInputs(const AnfNodePtr &input_x, const AnfNodePtr &input_w,
                                                             const AnfNodePtr &input_bias, const FuncGraphPtr &graph,
                                                             const CNodePtr &matmul_cnode) const {
  MS_EXCEPTION_IF_NULL(input_x);
  MS_EXCEPTION_IF_NULL(input_w);
  MS_EXCEPTION_IF_NULL(input_bias);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(matmul_cnode);
  AnfNodePtrList matmul_add_split_inputs = {input_x, input_w, GetReshapeTupleNode(graph)};
  const std::set<TypeId> bias_add_bf16_dtype = {kNumberTypeBFloat16};
  if (!CheckSupportDataType(input_bias, bias_add_bf16_dtype)) {
    matmul_add_split_inputs.push_back(input_bias);
    return matmul_add_split_inputs;
  }
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  auto type_value_f32 = std::make_shared<Int64Imm>(static_cast<int64_t>(TypeId::kNumberTypeFloat32));
  auto type_node_f32 = kernel_graph->NewValueNode(type_value_f32);
  std::vector<AnfNodePtr> casted_bias_inputs = {NewValueNode(prim::kPrimCast), input_bias, type_node_f32};
  auto bias_cast_cnode = NewCNode(casted_bias_inputs, graph);
  MS_EXCEPTION_IF_NULL(bias_cast_cnode);
  auto type_fp32 = TypeIdToType(TypeId::kNumberTypeFloat32);
  auto cast_abs = std::make_shared<abstract::AbstractTensor>(type_fp32, input_bias->Shape());
  bias_cast_cnode->set_abstract(cast_abs);
  bias_cast_cnode->set_scope(matmul_cnode->scope());

  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  TypeId type_id = common::AnfAlgo::GetOutputInferDataType(matmul_cnode, kIndex0);
  builder.SetInputsDeviceType({type_id, TypeId::kNumberTypeInt});
  builder.SetInputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder.SetOutputsDeviceType({TypeId::kNumberTypeFloat32});
  builder.SetOutputsFormat({kOpFormat_DEFAULT});
  auto build_info = builder.Build();
  AnfAlgo::SetSelectKernelBuildInfo(build_info, bias_cast_cnode.get());

  matmul_add_split_inputs.push_back(bias_cast_cnode);
  return matmul_add_split_inputs;
}

std::string MatmulAddExtSplitFusion::GetFfnSplitPriName() const { return kMatmulFfnBiasSplitPrimName; }

std::string MatmulAddExtSplitFusion::GetQkvSplitPriName() const { return kMatmulQkvBiasSplitPrimName; }

void MatmulAddExtSplitFusion::SetMatmulSplitPrimitiveAttr(const PrimitivePtr &matmul_split_prim,
                                                          const ValueNodePtr &split_size_node) const {
  MS_EXCEPTION_IF_NULL(matmul_split_prim);
  MS_EXCEPTION_IF_NULL(split_size_node);
  matmul_split_prim->AddAttr(kNLength, split_size_node->value());
  matmul_split_prim->AddAttr(kWithBias, MakeValue<bool>(true));
}
}  // namespace opt
}  // namespace mindspore
