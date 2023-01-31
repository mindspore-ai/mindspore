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

#include "plugin/device/ascend/optimizer/mindir/aicpu_lib_select.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_util.h"
#include <set>
#include <string>
#include "include/common/utils/utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const AnfNodePtr AICpuLibSelectPass::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);

  static const std::set<std::string> kAICpuOpNames = {kDropoutGenMaskOpName,
                                                      kEnvironCreateOpName,
                                                      kEnvironSetOpName,
                                                      kEnvironGetOpName,
                                                      kEnvironDestroyAllOpName,
                                                      kPriorityReplayBufferCreate,
                                                      kPriorityReplayBufferPush,
                                                      kPriorityReplayBufferSample,
                                                      kPriorityReplayBufferUpdate,
                                                      kPriorityReplayBufferDestroy,
                                                      kReservoirReplayBufferCreate,
                                                      kReservoirReplayBufferPush,
                                                      kReservoirReplayBufferSample,
                                                      kReservoirReplayBufferDestroy,
                                                      kGatherDGradV2OpName,
                                                      kSliceGradOpName};
  static const std::set<std::string> kMigrateAicpuKernelOps = {
    mindspore::kScatterNdOpName, mindspore::kScatterNdUpdateOpName, mindspore::kTensorScatterUpdateOpName,
    mindspore::kGatherNdOpName};

  static const std::string kEnvOpSoNames = "mindspore_aicpu_kernels";
  static const std::string kCpuKernelSoName = "mindspore_cpu_kernels";

  if (!node->isa<CNode>()) {
    return node;
  }
  auto kernel_name = common::AnfAlgo::GetCNodeName(node);
  if (kernel::kOpNameToAicpuOpNameMap.find(kernel_name) != kernel::kOpNameToAicpuOpNameMap.end()) {
    kernel_name = kernel::kOpNameToAicpuOpNameMap.at(kernel_name);
  }
  if (kAICpuOpNames.find(kernel_name) != kAICpuOpNames.end()) {
    common::AnfAlgo::SetNodeAttr(kAttrCustAicpu, MakeValue(kEnvOpSoNames), node);
  }
  if (kMigrateAicpuKernelOps.find(kernel_name) != kMigrateAicpuKernelOps.end()) {
    common::AnfAlgo::SetNodeAttr(kAttrCustAicpu, MakeValue(kCpuKernelSoName), node);
  }

  return node;
}
}  // namespace opt
}  // namespace mindspore
