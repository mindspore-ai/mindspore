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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_split_base.h"
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include "utils/convert_utils.h"
#include "backend/common/pass/common/gllo_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "utils/ms_context.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
BaseRef MatmulSplitBase::GetMatmulPattern() const {
  auto input_x = std::make_shared<Var>();
  auto input_w = std::make_shared<Var>();
  auto trans_a = std::make_shared<Var>();
  auto trans_b = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_x != nullptr, {});
  MS_CHECK_TRUE_RET(input_w != nullptr, {});
  MS_CHECK_TRUE_RET(trans_a != nullptr, {});
  MS_CHECK_TRUE_RET(trans_b != nullptr, {});
  auto is_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMul>);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  auto matmul_ref = VectorRef({is_matmul, input_x, input_w, trans_a, trans_b});
  return matmul_ref;
}

BaseRef MatmulSplitBase::GetSplitWithSizePattern(const BaseRef &pre_pattern_ref) const {
  MS_EXCEPTION_IF_NULL(pre_pattern_ref);
  auto split_size = std::make_shared<Var>();
  auto dim = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(split_size != nullptr, {});
  MS_CHECK_TRUE_RET(dim != nullptr, {});
  auto is_split_with_size = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSplitWithSize>);
  MS_CHECK_TRUE_RET(is_split_with_size != nullptr, {});
  auto split_with_size_ref = VectorRef({is_split_with_size, pre_pattern_ref, split_size, dim});
  return split_with_size_ref;
}

bool MatmulSplitBase::IsEnableMatmulSplit() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  auto enable_matmul_split = enable_op_list.find(kInferenceMatmulSplitName) != enable_op_list.end();
  return enable_matmul_split;
}

bool MatmulSplitBase::CheckMatmulSplit(const AnfNodePtr &input_x, const AnfNodePtr &input_w,
                                       const ValueNodePtr &input_trans_a, const ValueNodePtr &input_trans_b,
                                       const ValueNodePtr &split_size_node) const {
  return CheckSupportDataType(input_x, kSupportDataType) && CheckSupportDataType(input_w, kSupportDataType) &&
         CheckMatmulDataFormat(input_trans_a, input_trans_b) && CheckSplitSize(input_w, split_size_node);
}

bool MatmulSplitBase::CheckMatmulDataFormat(const ValueNodePtr &input_trans_a,
                                            const ValueNodePtr &input_trans_b) const {
  MS_EXCEPTION_IF_NULL(input_trans_a);
  MS_EXCEPTION_IF_NULL(input_trans_b);
  bool is_trans_a = GetValue<bool>(input_trans_a->value());
  bool is_trans_b = GetValue<bool>(input_trans_b->value());
  MS_LOG(DEBUG) << "the transpose format of matmul node is: trans_a=" << is_trans_a << ", trans_b:" << is_trans_b;
  return !is_trans_a && is_trans_b;
}

bool MatmulSplitBase::CheckSplitSize(const AnfNodePtr &input_w, const ValueNodePtr &split_size_node) const {
  MS_EXCEPTION_IF_NULL(input_w);
  MS_EXCEPTION_IF_NULL(split_size_node);
  auto split_size_shape = GetSplitSizeShape(split_size_node);
  if (split_size_shape.empty()) {
    return false;
  }
  size_t split_size_length = split_size_shape.size();
  if (split_size_length != kMatmulFfnSplitSizeLen && split_size_length != kMatmulQkvSplitSizeLen) {
    MS_LOG(DEBUG) << "split size length only support 2 or 3";
    return false;
  }
  uint32_t total_size = 0;
  for (size_t i = 0; i < split_size_length; i++) {
    auto split_size_num = split_size_shape[i];
    if (split_size_num % kValidShape != 0) {
      MS_LOG(DEBUG) << "split size should be a multiple of 16";
      return false;
    }
    total_size += split_size_num;
  }
  auto weight_shape = BaseShapeToShape(AnfAlgo::GetOutputDetailShape(input_w, kIndex0));
  if (weight_shape[kIndex0] != total_size) {
    MS_LOG(DEBUG) << "check split size failed, wight_shape_n: " << weight_shape[kIndex0]
                  << ", total_size:" << total_size;
    return false;
  }
  return true;
}

std::vector<int64_t> MatmulSplitBase::GetSplitSizeShape(const ValueNodePtr &split_size_node) const {
  if (split_size_node == nullptr) {
    return {};
  }
  auto split_size_shape = GetValue<std::vector<int64_t>>(split_size_node->value());
  return split_size_shape;
}

PrimitivePtr MatmulSplitBase::GetMatmulSplitPrimitive(const ValueNodePtr &split_size_node) const {
  MS_EXCEPTION_IF_NULL(split_size_node);
  std::string prim_name = GetMatmulSplitPrimName(split_size_node);
  MS_CHECK_TRUE_RET(!prim_name.empty(), {});
  PrimitivePtr matmul_split_prim = std::make_shared<Primitive>(prim_name);
  MS_CHECK_TRUE_RET(matmul_split_prim != nullptr, {});
  SetMatmulSplitPrimitiveAttr(matmul_split_prim, split_size_node);
  return matmul_split_prim;
}

std::string MatmulSplitBase::GetMatmulSplitPrimName(const ValueNodePtr &split_size_node) const {
  MS_EXCEPTION_IF_NULL(split_size_node);
  std::string prim_name = "";
  size_t split_size_length = GetSplitSizeShape(split_size_node).size();
  if (split_size_length == kMatmulFfnSplitSizeLen) {
    prim_name = GetFfnSplitPriName();
  }
  if (split_size_length == kMatmulQkvSplitSizeLen) {
    prim_name = GetQkvSplitPriName();
  }
  return prim_name;
}

std::tuple<CNodePtr, ValueNodePtr> MatmulSplitBase::GetSplitSizeNode(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto split_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(split_cnode != nullptr, {});
  auto split_size_node = split_cnode->input(kIndex2)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_RET(split_size_node != nullptr, {});
  return std::make_tuple(split_cnode, split_size_node);
}

std::tuple<CNodePtr, AnfNodePtr, AnfNodePtr, ValueNodePtr, ValueNodePtr> MatmulSplitBase::GetMatmulNode(
  const CNodePtr &pre_cnode) const {
  MS_EXCEPTION_IF_NULL(pre_cnode);
  auto matmul_cnode = pre_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, {});
  auto input_x = matmul_cnode->input(kIndex1);
  MS_CHECK_TRUE_RET(input_x != nullptr, {});
  auto input_w = matmul_cnode->input(kIndex2);
  MS_CHECK_TRUE_RET(input_w != nullptr, {});
  auto trans_a = matmul_cnode->input(kIndex3)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_RET(trans_a != nullptr, {});
  auto trans_b = matmul_cnode->input(kIndex4)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_RET(trans_b != nullptr, {});
  return std::make_tuple(matmul_cnode, input_x, input_w, trans_a, trans_b);
}

ValueNodePtr MatmulSplitBase::GetReshapeTupleNode(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  auto reshape_tuple_node = kernel_graph->NewValueNode(MakeValue((int64_t)kTuplePlaceHolderNum));
  kernel_graph->AddValueNodeToGraph(reshape_tuple_node);
  return reshape_tuple_node;
}

CNodePtr MatmulSplitBase::GetMatmulSplitCNode(const PrimitivePtr &matmul_split_prim,
                                              const AnfNodePtrList &matmul_split_inputs, const FuncGraphPtr &graph,
                                              const CNodePtr &matmul_cnode, const CNodePtr &split_cnode) const {
  MS_EXCEPTION_IF_NULL(matmul_split_prim);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(matmul_cnode);
  MS_EXCEPTION_IF_NULL(split_cnode);
  auto inputs = matmul_split_inputs;
  (void)inputs.insert(inputs.begin(), std::make_shared<ValueNode>(matmul_split_prim));
  auto matmul_split_cnode = NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(matmul_split_cnode);
  matmul_split_cnode->set_scope(matmul_cnode->scope());
  if (split_cnode->abstract() != nullptr) {
    matmul_split_cnode->set_abstract(split_cnode->abstract());
  }
  return matmul_split_cnode;
}
}  // namespace opt
}  // namespace mindspore
