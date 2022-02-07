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

#include "backend/optimizer/ascend/mindir/env_op_attr_update.h"
#include <set>
#include <string>
#include "utils/utils.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
const AnfNodePtr EnvOpAttrUpdate::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                          const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);

  static const std::set<std::string> kEnvOpNames = {kEnvironCreateOpName, kEnvironSetOpName, kEnvironGetOpName,
                                                    kEnvironDestroyAllOpName};
  static const std::string kEnvOpSoNames = "mindspore_aicpu_kernels";

  if (!node->isa<CNode>()) {
    return node;
  }
  auto kernel_name = AnfAlgo::GetCNodeName(node);
  if (kEnvOpNames.find(kernel_name) != kEnvOpNames.end()) {
    AnfAlgo::SetNodeAttr(kAttrCustAicpu, MakeValue(kEnvOpSoNames), node);
  }

  return node;
}
}  // namespace opt
}  // namespace mindspore
