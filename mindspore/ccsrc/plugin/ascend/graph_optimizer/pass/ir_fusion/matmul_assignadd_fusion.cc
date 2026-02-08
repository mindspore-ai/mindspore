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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion/matmul_assignadd_fusion.h"
#include <vector>
#include <string>
#include <functional>
#include "include/utils/anfalgo.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "primitive/nn_optimizer_op_name.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace opt {
CNodePtr MatmulAssignaddFusion::GetMatmul(const CNodePtr &assign_add) const {
  AnfNodePtr matmul;
  auto assign_value_input = common::AnfAlgo::GetInputNode(assign_add, 1);
  // value input of AssignAdd should be Matmul or Matmul-Cast
  if (common::AnfAlgo::CheckPrimitiveType(assign_value_input, prim::kPrimMatMul)) {
    matmul = assign_value_input;
  } else if (common::AnfAlgo::CheckPrimitiveType(assign_value_input, prim::kPrimCast)) {
    auto cast_input = common::AnfAlgo::GetInputNode(assign_value_input->cast<CNodePtr>(), 0);
    if (common::AnfAlgo::CheckPrimitiveType(cast_input, prim::kPrimMatMul)) {
      matmul = cast_input;
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
  return matmul->cast<CNodePtr>();
}

bool MatmulAssignaddFusion::CheckFusion(const CNodePtr &matmul) const {
  // check matmul input number
  const size_t kMatmulInputSize = 4;
  if (common::AnfAlgo::GetInputNum(matmul) != kMatmulInputSize) {
    MS_LOG(INFO) << "The input size of " << matmul->fullname_with_scope() << " is not " << kMatmulInputSize
                 << ", skip fusion.";
    return false;
  }
  // check transpose_a=true, transpose_b=false
  const size_t kTransposeaInputIdx = 2;
  const size_t kTransposebInputIdx = 3;
  auto transpose_a = common::AnfAlgo::GetInputNode(matmul, kTransposeaInputIdx);
  auto transpose_b = common::AnfAlgo::GetInputNode(matmul, kTransposebInputIdx);
  if (!transpose_a->isa<ValueNode>() || !transpose_b->isa<ValueNode>()) {
    MS_LOG(INFO) << "Expect transpose_a and transpose_b input should be ValueNode, but not, skip fusion.";
    return false;
  }
  auto transpose_a_value = transpose_a->cast<ValueNodePtr>()->value();
  auto transpose_b_value = transpose_b->cast<ValueNodePtr>()->value();
  if (!transpose_a_value->isa<BoolImm>() || !transpose_b_value->isa<BoolImm>()) {
    MS_LOG(INFO) << "Expect the datatype of transpose_a and transpose_b input should be bool, but not, skip fusion.";
    return false;
  }
  auto transpose_a_bool = GetValue<bool>(transpose_a_value);
  auto transpose_b_bool = GetValue<bool>(transpose_b_value);
  if (transpose_a_bool != true || transpose_b_bool != false) {
    MS_LOG(INFO) << "Expect transpose_a input is true and transpose_b input is false, but got transpose_a["
                 << transpose_a_bool << "] and transpose_b[" << transpose_b_bool << "], skip fusion.";
    return false;
  }
  // check output shape, should less than 1000w, otherwise performance may decay
  auto out_shape = common::AnfAlgo::GetOutputInferShape(matmul, 0);
  int64_t shape_size = std::accumulate(out_shape.begin(), out_shape.end(), int64_t(1), std::multiplies<int64_t>());
  const int64_t shape_limit = 10000000;
  if (shape_size > shape_limit) {
    MS_LOG(INFO) << "Expect shape size should be less than " << shape_limit << ", but got shape " << out_shape
                 << " and shape size[" << shape_size << "], skip fusion.";
    return false;
  }

  return true;
}

bool MatmulAssignaddFusion::CheckDataType(const AnfNodePtr &input_x, const AnfNodePtr &weight,
                                          const AnfNodePtr &out) const {
  auto input_x_dtype = common::AnfAlgo::GetOutputInferDataType(input_x, 0);
  auto weight_dtype = common::AnfAlgo::GetOutputInferDataType(weight, 0);
  auto out_dtype = common::AnfAlgo::GetOutputInferDataType(out, 0);
  if (input_x_dtype != kNumberTypeFloat16 && input_x_dtype != kNumberTypeBFloat16) {
    MS_LOG(INFO) << "Input x's datatype is not float16 or bfloat16, skip fusion.";
    return false;
  }
  if (weight_dtype != kNumberTypeFloat16 && weight_dtype != kNumberTypeBFloat16) {
    MS_LOG(INFO) << "Input weight's datatype is not float16 or bfloat16, skip fusion.";
    return false;
  }
  if (out_dtype != kNumberTypeFloat32) {
    MS_LOG(INFO) << "Input out's datatype is not float32, skip fusion.";
    return false;
  }
  return true;
}

void MatmulAssignaddFusion::ReplaceMatmulForDepend(const FuncGraphPtr &graph, const AnfNodePtr &matmul,
                                                   const AnfNodePtr &matmul_add) const {
  // replace matmul with matmuladd when matmul is depend's 2nd input.(depend is created by dw masking)
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &matmul_users = manager->node_users()[matmul];
  for (const auto &[node, index] : matmul_users) {
    const int depend_idx = 2;
    if (IsPrimitiveCNode(node, prim::kPrimDepend) && index == depend_idx) {
      (void)manager->SetEdge(node, kIndex2, matmul_add);
    }
  }
}

std::vector<std::string> MatmulAssignaddFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimMatMul->name(), kAssignAddOpName};
  return ret;
}

const BaseRef MatmulAssignaddFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto assignadd = VectorRef({std::make_shared<Primitive>(kAssignAddOpName), Xs});
  return assignadd;
}

const AnfNodePtr MatmulAssignaddFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto assign_add = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(assign_add);

  if (AnfAlgo::GetBackend(graph) != kBackendMSBackend) {
    return nullptr;
  }
  auto matmul = GetMatmul(assign_add);
  if (matmul == nullptr) {
    return nullptr;
  }
  std::vector<AnfNodePtr> orig_nodes{matmul, assign_add};
  if (!CheckStreamAndCoreAttrWithOrigNodes(graph, orig_nodes)) {
    return nullptr;
  }
  if (!CheckFusion(matmul)) {
    return nullptr;
  }
  auto input_x = common::AnfAlgo::GetInputNode(matmul, 0);
  auto weight = common::AnfAlgo::GetInputNode(matmul, 1);
  auto out = common::AnfAlgo::GetInputNode(assign_add, 0);
  if (!CheckDataType(input_x, weight, out)) {
    return nullptr;
  }

  // create InplaceMatmulAdd
  const std::string fusion_op_name = "InplaceMatmulAdd";
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(fusion_op_name)), input_x, weight, out};
  const size_t assign_add_monad_idx = 3;
  if (common::AnfAlgo::GetInputNum(assign_add) == assign_add_monad_idx) {
    // add monad input
    (void)inputs.emplace_back(assign_add->input(assign_add_monad_idx));
  }
  auto matmul_add = opt::NewCNode(inputs, graph, orig_nodes);
  matmul_add->set_scope(matmul->scope());
  matmul_add->set_abstract(assign_add->abstract());
  if (common::AnfAlgo::HasNodeAttr(GRAPH_FLAG_SIDE_EFFECT_MEM, assign_add)) {
    common::AnfAlgo::CopyNodeAttr(GRAPH_FLAG_SIDE_EFFECT_MEM, assign_add, matmul_add);
  }
  ReplaceMatmulForDepend(graph, matmul, matmul_add);
  return matmul_add;
}
}  // namespace opt
}  // namespace mindspore
