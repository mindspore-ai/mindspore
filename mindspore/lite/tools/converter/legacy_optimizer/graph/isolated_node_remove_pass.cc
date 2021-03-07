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

#include <queue>
#include <algorithm>
#include <cassert>

#include "tools/converter/legacy_optimizer/graph/isolated_node_remove_pass.h"
#include "src/common/log_adapter.h"
#include "tools/common/graph_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
STATUS IsolatedNodeRemovePass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  bool ifChanged = false;
  for (auto iter = graph->nodes.begin(); iter != graph->nodes.end();) {
    if ((*iter)->inputIndex.empty() && (*iter)->outputIndex.empty()) {
      ifChanged = true;
      iter = graph->nodes.erase(iter);
    } else {
      iter++;
    }
  }
  return ifChanged ? RET_OK : RET_NO_CHANGE;
}
}  // namespace lite
}  // namespace mindspore
