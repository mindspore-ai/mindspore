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
#include "plugin/device/ascend/optimizer/buffer_fusion/batchmatmul_eltwise_fusion_pass.h"
#include <set>
#include <string>
#include "kernel/kernel_fusion.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/core_ops.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/optimizer/fusion_id_allocator.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kAttrNoFusion = "no_fusion";

CNodePtr FindInputNode(const CNodePtr &cnode, const string &node_type, const std::string &fusion_type) {
  auto input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t i = 1; i <= input_num; ++i) {
    auto input = cnode->input(i);
    MS_EXCEPTION_IF_NULL(input);
    if (input->isa<CNode>() && common::AnfAlgo::GetCNodeName(input) == node_type &&
        AnfAlgo::GetFusionType(input) == fusion_type) {
      return input->cast<CNodePtr>();
    }
  }
  return nullptr;
}
}  // namespace

bool BatchMatmulEltwiseFusionPass::MatchPattern1(const CNodePtr &eltwise1,
                                                 mindspore::HashSet<AnfNodePtr> *record) const {
  // bmm - eltwise - eltwise1
  const std::set<string> kElem1TypeList = {kAddOpName, kReluOpName, kFusedMulAddOpName};
  if (kElem1TypeList.find(common::AnfAlgo::GetCNodeName(eltwise1)) == kElem1TypeList.end()) {
    return false;
  }

  auto input_num = common::AnfAlgo::GetInputTensorNum(eltwise1);
  for (size_t i = 1; i <= input_num; ++i) {
    auto eltwise1_input = eltwise1->input(i);
    MS_EXCEPTION_IF_NULL(eltwise1_input);
    if (eltwise1_input->isa<CNode>() && MatchPattern2(eltwise1_input->cast<CNodePtr>(), record)) {
      record->insert(eltwise1);
      return true;
    }
  }
  return false;
}

bool BatchMatmulEltwiseFusionPass::MatchPattern2(const CNodePtr &eltwise,
                                                 mindspore::HashSet<AnfNodePtr> *record) const {
  // bmm - eltwise
  const std::set<string> kElemTypeList = {kFusedMulAddOpName, kAddOpName,  kTruncateDivOpName,
                                          kRealDivOpName,     kReluOpName, kReluGradOpName};
  if (kElemTypeList.find(common::AnfAlgo::GetCNodeName(eltwise)) == kElemTypeList.end()) {
    return false;
  }

  CNodePtr bmm = FindInputNode(eltwise, kBatchMatMulOpName, kernel::kPatternBatchMatmul);
  if (bmm == nullptr || common::AnfAlgo::IsDynamicShape(bmm) || common::AnfAlgo::GetBooleanAttr(bmm, kAttrNoFusion)) {
    return false;
  }

  record->insert(eltwise);
  record->insert(bmm);
  return true;
}

bool BatchMatmulEltwiseFusionPass::MatchPattern3(const CNodePtr &eltwise,
                                                 mindspore::HashSet<AnfNodePtr> *record) const {
  // bmm - eltwise1(mul) - eltwise2(sigmoid) - eltwise(mul)
  if (common::AnfAlgo::GetCNodeName(eltwise) != kMulOpName) {
    return false;
  }

  CNodePtr eltwise2 = FindInputNode(eltwise, kSigmoidOpName, kernel::kPatternElemWise);
  if (eltwise2 == nullptr) {
    return false;
  }

  CNodePtr eltwise1 = FindInputNode(eltwise2, kMulOpName, kernel::kPatternElemWise);
  if (eltwise1 == nullptr) {
    return false;
  }

  CNodePtr bmm = FindInputNode(eltwise1, kBatchMatMulOpName, kernel::kPatternMatmul);
  if (bmm == nullptr || common::AnfAlgo::IsDynamicShape(bmm)) {
    return false;
  }

  record->insert(eltwise);
  record->insert(eltwise2);
  record->insert(eltwise1);
  record->insert(bmm);
  return true;
}

void BatchMatmulEltwiseFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
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
    if (AnfAlgo::GetKernelType(cnode) == KernelType::TBE_KERNEL &&
        (AnfAlgo::GetFusionType(cnode) == kernel::kPatternElemWise ||
         AnfAlgo::GetFusionType(cnode) == kernel::kPatternBroadcast)) {
      mindspore::HashSet<AnfNodePtr> record;
      if (MatchPattern1(cnode, &record) || MatchPattern2(cnode, &record) || MatchPattern3(cnode, &record)) {
        candidate_fusion->push_back(record);
        SetRecordFusionId(record);
      }
    }
  }
}
}  // namespace opt
}  // namespace mindspore
