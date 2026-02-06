/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/cpu/graph_optimizer/pass/train/matmul_biasadd_relu_fusion.h"

#include <memory>
#include <vector>
#include "primitive/ascend_op_name.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "primitive/nn_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace opt {
namespace {}

MatMulBiasAddReluFusionCPU::MatMulBiasAddReluFusionCPU(bool multigraph)
    : PatternProcessPass("matmul_biasadd_relu_fusion_cpu", multigraph) {
  x0_ = std::make_shared<Var>();
  x1_ = std::make_shared<Var>();
  x2_ = std::make_shared<Var>();
  matmul_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimMatMul->name()));
}

const BaseRef MatMulBiasAddReluFusionCPU::DefinePattern() const {
  VectorRef matmul({matmul_var_, x0_, x1_});
  VectorRef biasadd({prim::kPrimBiasAdd, matmul, x2_});
  VectorRef pattern({prim::kPrimReLU, biasadd});
  return pattern;
}

AnfNodePtr MatMulBiasAddReluFusionCPU::CreateMatmulWithBias(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                            const EquivPtr &equiv) const {
  auto matmul = GetAnfNodeByVar(equiv, matmul_var_);
  if (matmul == nullptr || !matmul->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Get CNode MatMul failed!" << trace::DumpSourceLines(node);
  }

  std::vector<AnfNodePtr> inputs;
  (void)inputs.emplace_back(NewValueNode(std::make_shared<Primitive>(kMatMulBiasAddReluFusionOpName)));
  (void)inputs.emplace_back(GetAnfNodeByVar(equiv, x0_));
  (void)inputs.emplace_back(GetAnfNodeByVar(equiv, x1_));
  (void)inputs.emplace_back(GetAnfNodeByVar(equiv, x2_));
  auto new_node = NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(node->scope());
  new_node->set_abstract(node->abstract());
  common::AnfAlgo::CopyNodeAttrs(matmul, new_node);

  auto prim = GetValueNode<PrimitivePtr>(new_node->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  (void)prim->AddAttr(kAttrWithBiasAdd, MakeValue(true));
  (void)prim->AddAttr(kAttrWithRelu, MakeValue(true));
  new_node->AddAttr(kAttrWithBiasAdd, MakeValue(true));
  new_node->AddAttr(kAttrWithRelu, MakeValue(true));

  return new_node;
}

const AnfNodePtr MatMulBiasAddReluFusionCPU::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                     const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  dnnl::cpu_isa current_cpu_isa = dnnl::get_effective_cpu_isa();
  if (current_cpu_isa == dnnl::cpu_isa::sse41 || current_cpu_isa == dnnl::cpu_isa::avx ||
      current_cpu_isa == dnnl::cpu_isa::avx2 || current_cpu_isa == dnnl::cpu_isa::avx2_vnni) {
    MS_LOG(INFO) << "matmul and biasadd fusion is only supported on aarch or x86 with avx512, disabled here";
    return nullptr;
  }
  if (common::AnfAlgo::IsDynamicShape(node)) {
    return nullptr;
  }
  auto matmul = GetAnfNodeByVar(equiv, matmul_var_);
  MS_EXCEPTION_IF_NULL(matmul);
  auto matmul_node = matmul->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(matmul_node);
  auto dtype = common::AnfAlgo::GetOutputInferDataType(matmul_node, 0);
  if (dtype != kNumberTypeFloat32) {
    MS_LOG(INFO) << kMatMulBiasAddReluFusionOpName << " cpu kernel only supports float32 currently.";
    return nullptr;
  }
  return CreateMatmulWithBias(graph, node, equiv);
}
}  // namespace opt
}  // namespace mindspore
