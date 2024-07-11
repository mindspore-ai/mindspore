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
#include "mindspore/core/ops/other_ops.h"

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
  if (prim->instance_name().find("grad_mirror") == std::string::npos) {
    return false;
  }
  return true;
}
}  // namespace parallel
}  // namespace mindspore
