/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/buffer_fusion/matmul_confusiontranspose_fusion_pass.h"
#include "kernel/kernel_fusion.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "base/core_ops.h"
#include "utils/ms_context.h"
#include "backend/common/optimizer/fusion_id_allocator.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kAttrTransposeX1 = "transpose_x1";
constexpr auto kAttrTransposeX2 = "transpose_x2";

struct WrongCase {
  std::vector<size_t> matmul_input0_shape;
  std::vector<size_t> matmul_input1_shape;
  std::vector<size_t> transpose_output_shape;
  bool transpose_x1;
  bool transpose_x2;
};

bool CheckWrongShape(const AnfNodePtr &matmul, const AnfNodePtr &confusion_transpose) {
  std::vector<WrongCase> wrong_cases;

  // add wrong cases
  WrongCase wrong_case1;
  wrong_case1.matmul_input0_shape = {128, 1024};
  wrong_case1.matmul_input1_shape = {1024, 1024};
  wrong_case1.transpose_output_shape = {1, 16, 128, 64};
  wrong_case1.transpose_x1 = false;
  wrong_case1.transpose_x2 = true;
  wrong_cases.push_back(std::move(wrong_case1));

  // get node shape
  auto matmul_input0_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(matmul, 0);
  auto matmul_input1_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(matmul, 1);
  auto transpose_output_shape = common::AnfAlgo::GetOutputInferShape(confusion_transpose, 0);
  auto transpose_x1 = common::AnfAlgo::GetBooleanAttr(matmul, kAttrTransposeX1);
  auto transpose_x2 = common::AnfAlgo::GetBooleanAttr(matmul, kAttrTransposeX2);

  // check
  return std::any_of(wrong_cases.begin(), wrong_cases.end(),
                     [matmul_input0_shape, matmul_input1_shape, transpose_output_shape, transpose_x1,
                      transpose_x2](WrongCase wrong_case) {
                       return wrong_case.matmul_input0_shape == matmul_input0_shape &&
                              wrong_case.matmul_input1_shape == matmul_input1_shape &&
                              wrong_case.transpose_output_shape == transpose_output_shape &&
                              wrong_case.transpose_x1 == transpose_x1 && wrong_case.transpose_x2 == transpose_x2;
                     });
}
}  // namespace

void MatmulConfusionTranposeFusionPass::MatchMatmulConfusionTranpose(const CNodePtr &cnode,
                                                                     const session::KernelGraph & /* kernel_graph */,
                                                                     FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  auto matmul = cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(matmul);
  if (matmul->isa<CNode>() && (common::AnfAlgo::CheckPrimitiveType(matmul, prim::kPrimMatMul) ||
                               common::AnfAlgo::CheckPrimitiveType(matmul, prim::kPrimBatchMatMul))) {
    if (CheckWrongShape(matmul, cnode)) {
      return;
    }
    mindspore::HashSet<AnfNodePtr> record{cnode, matmul};
    candidate_fusion->push_back(record);
    SetRecordFusionId(record);
  }
}

void MatmulConfusionTranposeFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
                                                                 FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  const auto &node_list = TopoSort(kernel_graph.get_return());
  for (auto &node : node_list) {
    if (!AnfUtils::IsRealCNodeKernel(node) || fusion_id_allocator->HasFusionIdAttr(node) ||
        common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);

    if (common::AnfAlgo::GetCNodeName(cnode) == kConfusionTransposeDOpName) {
      MatchMatmulConfusionTranpose(cnode, kernel_graph, candidate_fusion);
    }
  }
}
}  // namespace opt
}  // namespace mindspore
