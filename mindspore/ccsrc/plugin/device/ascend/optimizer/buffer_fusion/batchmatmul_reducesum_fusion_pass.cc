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
#include "plugin/device/ascend/optimizer/buffer_fusion/batchmatmul_reducesum_fusion_pass.h"
#include "kernel/kernel_fusion.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/core_ops.h"
#include "utils/ms_context.h"
#include "backend/common/optimizer/fusion_id_allocator.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kHWSize = 2;
}

void BatchMatmulReduceSumFusionPass::MatchBatchMatmulReduceSum(const CNodePtr &reduce_sum, const session::KernelGraph &,
                                                               FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(reduce_sum);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  auto batch_matmul = reduce_sum->input(kIndex1);
  MS_EXCEPTION_IF_NULL(batch_matmul);
  const PrimitiveSet batch_matmul_prims{prim::kPrimBatchMatMul, prim::kPrimBatchMatMulV2};
  if (!batch_matmul->isa<CNode>() || AnfAlgo::GetFusionType(batch_matmul) != kernel::kPatternBatchMatmul ||
      !IsOneOfPrimitiveCNode(batch_matmul, batch_matmul_prims)) {
    return;
  }
  // check batch_matmul
  auto out_shape = common::AnfAlgo::GetOutputInferShape(batch_matmul, 0);
  if (out_shape.size() > kHWSize + 1) {
    MS_LOG(DEBUG) << "Only support cases that the batch dim size of BatchMatmul is 1, but got "
                  << out_shape.size() - kHWSize;
    return;
  }
  auto x1_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(batch_matmul, 0);
  if (x1_shape.size() > kHWSize && x1_shape[0] == 1) {
    MS_LOG(DEBUG) << "Quit fusion when the input batch dim of BatchMatmul is 1.";
    return;
  }
  auto x2_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(batch_matmul, 1);
  if (x2_shape.size() > kHWSize && x2_shape[0] == 1) {
    MS_LOG(DEBUG) << "Quit fusion when the input batch dim of BatchMatmul is 1.";
    return;
  }
  // check reduce sum
  if (AnfAlgo::GetOutputDeviceDataType(reduce_sum, 0) != kNumberTypeFloat32) {
    MS_LOG(DEBUG) << "Quit fusion when the output device datatype of ReduceSum is not float32.";
    return;
  }
  if (common::AnfAlgo::GetBooleanAttr(reduce_sum, kAttrKeepDims)) {
    MS_LOG(DEBUG) << "Quit fusion when the keep_dims attr of ReduceSum is true.";
    return;
  }

  mindspore::HashSet<AnfNodePtr> record{reduce_sum, batch_matmul};
  candidate_fusion->push_back(record);
  SetRecordFusionId(record);
}

void BatchMatmulReduceSumFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
                                                              FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  const auto &node_list = TopoSort(kernel_graph.get_return());
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!AnfUtils::IsRealCNodeKernel(node) || fusion_id_allocator->HasFusionIdAttr(node) ||
        common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);

    if (AnfAlgo::GetKernelType(cnode) == KernelType::TBE_KERNEL &&
        AnfAlgo::GetFusionType(cnode) == kernel::kPatternCommReduce &&
        common::AnfAlgo::GetCNodeName(cnode) == kReduceSumDOpName) {
      MatchBatchMatmulReduceSum(cnode, kernel_graph, candidate_fusion);
    }
  }
}
}  // namespace opt
}  // namespace mindspore
