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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/transpose_batch_matmul_transpose_fusion.h"
#include <algorithm>
#include <vector>
#include <string>
#include "backend/common/pass/common/gllo_utils.h"
#include "primitive/nn_ops.h"
#include "primitive/math_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
constexpr auto kTransposeBatchMatmulTransposeOpName = "TransposeBatchMatmulTranspose";

std::vector<std::string> TransposeBatchMatmulTranspose::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimBatchMatMul->name()};
  return ret;
}

ShapeVector TransposeBatchMatmulTranspose::GetPermValue(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto perm_node = cnode->input(kIndex2)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(perm_node);

  auto perm_node_value = perm_node->value();
  MS_EXCEPTION_IF_NULL(perm_node_value);
  auto perm_ptr = perm_node_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(perm_ptr);
  auto perm_vector = perm_ptr->value();
  auto perm_size = perm_vector.size();

  ShapeVector perm_value;
  (void)std::transform(perm_vector.begin(), perm_vector.end(), std::back_inserter(perm_value), [&perm_size](auto v) {
    auto value = GetValue<int64_t>(v);
    return v < 0 ? SizeToLong(perm_size) + value : value;
  });
  return perm_value;
}

const BaseRef TransposeBatchMatmulTranspose::DefinePattern() const {
  auto x = std::make_shared<Var>();
  auto perm_input = std::make_shared<Var>();
  auto bmm_y = std::make_shared<Var>();
  auto bmm_bool_a = std::make_shared<Var>();
  auto bmm_bool_b = std::make_shared<Var>();
  auto perm_out = std::make_shared<Var>();
  // Pattern: Transpose --> BatchMatmul --> Transpose
  auto input_transpose = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  auto output_transpose = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(input_transpose != nullptr, {});
  MS_CHECK_TRUE_RET(output_transpose != nullptr, {});
  auto transpose_in = VectorRef({input_transpose, x, perm_input});
  auto bmm = VectorRef({prim::kPrimBatchMatMul, transpose_in, bmm_y, bmm_bool_a, bmm_bool_b});
  VectorRef pattern({output_transpose, bmm, perm_out});
  return pattern;
}

const AnfNodePtr TransposeBatchMatmulTranspose::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                        const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }
  auto const &soc_version = ms_context->ascend_soc_version();
  if (!soc_version.empty() && soc_version != "ascend910b" && soc_version != "ascend910_93") {
    return nullptr;
  }

  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_fusion =
    (std::find(enable_op_list.begin(), enable_op_list.end(), "TransposeBatchMatmulTranspose") != enable_op_list.end());
  if (!enable_fusion) {
    return nullptr;
  }

  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);
  // TransposeIn --> BMM --> TransposeOut
  auto transpose_out = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(transpose_out != nullptr, {});
  auto bmm_cnode = transpose_out->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(bmm_cnode != nullptr, {});
  MS_CHECK_TRUE_RET(bmm_cnode->func_graph() == transpose_out->func_graph(), {});
  auto transpose_in = bmm_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(transpose_in != nullptr, {});

  // check transpose_perm
  auto transpose_a = bmm_cnode->input(kIndex3);
  auto trans_a_ptr = transpose_a->cast<ValueNodePtr>();
  auto is_transpose_a = GetValue<bool>(trans_a_ptr->value());
  if (is_transpose_a) {
    return nullptr;
  }

  ShapeVector perm_in_value = GetPermValue(transpose_in);
  ShapeVector perm_out_value = GetPermValue(transpose_out);
  if (perm_in_value != perm_out_value) {
    return nullptr;
  }

  auto input_x = common::AnfAlgo::GetPrevNodeOutputInferShape(transpose_in, kIndex0);
  const ShapeVector perm_3d = {1, 0, 2};
  const ShapeVector perm_4d = {0, 2, 1, 3};
  if (!(input_x.size() == perm_3d.size() && perm_in_value == perm_3d) &&
      !(input_x.size() == perm_4d.size() && perm_in_value == perm_4d)) {
    return nullptr;
  }

  // create op
  PrimitivePtr transpose_batch_matmul_transpose_prim =
    std::make_shared<Primitive>(kTransposeBatchMatmulTransposeOpName);
  MS_CHECK_TRUE_RET(transpose_batch_matmul_transpose_prim, {});

  CNodePtr fusion_cnode = NewCNode(
    {
      NewValueNode(transpose_batch_matmul_transpose_prim),
      transpose_in->input(kIndex1),
      bmm_cnode->input(kIndex2),
      transpose_in->input(kIndex2),
      transpose_out->input(kIndex2),
      transpose_a,
      bmm_cnode->input(kIndex4),
    },
    func_graph);

  fusion_cnode->set_scope(transpose_out->scope());
  if (node->abstract() != nullptr) {
    fusion_cnode->set_abstract(transpose_out->abstract()->Clone());
  }

  return fusion_cnode;
}
}  // namespace opt
}  // namespace mindspore
