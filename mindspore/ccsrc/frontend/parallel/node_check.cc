/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/node_check.h"

#include <set>
#include <string>

#include "frontend/parallel/ops_info/ops_utils.h"
#include "primitive/other_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace parallel {
const std::set<std::string> BATCH_PARALLEL_BLACK_LIST = {STACK, TENSOR_SCATTER_UPDATE, MESHGRID};

bool IsInBatchParallelBlackList(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  return (BATCH_PARALLEL_BLACK_LIST.find(prim->name()) != BATCH_PARALLEL_BLACK_LIST.end());
}

bool IsFromParallelOptimizerRs(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimReduceScatter)) {
    return false;
  }
  auto prim = GetCNodePrimitive(node->cast<CNodePtr>());
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->instance_name().find("grad_parallel_optimizer") == std::string::npos) {
    return false;
  }
  return true;
}

bool IsFromGradMirrorAR(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimAllReduce)) {
    return false;
  }
  auto prim = GetCNodePrimitive(node->cast<CNodePtr>());
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->instance_name().find("grad_mirror") == std::string::npos) {
    return false;
  }
  return true;
}

bool IsTFTAllReduce(const AnfNodePtr &node) {
  auto tftEnv = common::GetEnv("MS_ENABLE_TFT");
  constexpr std::string_view optUCE = "UCE:1";
  constexpr std::string_view optTTP = "TTP:1";
  constexpr std::string_view optARF = "ARF:1";
  if (tftEnv.empty() || (tftEnv.find(optUCE) == std::string::npos && tftEnv.find(optTTP) == std::string::npos &&
                         tftEnv.find(optARF) == std::string::npos)) {
    return false;
  }

  bool isTFTAllReduce = false;
  if (IsPrimitiveCNode(node, prim::kPrimAllReduce)) {
    auto tftAttr = "tft_report_before";
    auto nodePrim = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(nodePrim);
    isTFTAllReduce = nodePrim->HasAttr(tftAttr) && GetValue<bool>(nodePrim->GetAttr(tftAttr));
    MS_LOG(INFO) << "Found TFT allreduce, is enable:" << isTFTAllReduce;
  }
  return isTFTAllReduce;
}

}  // namespace parallel
}  // namespace mindspore
