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

#include <vector>
#include <algorithm>
#include <memory>
#include "tools/converter/legacy_optimizer/graph/subgraph_tensor_pass.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "tools/common/graph_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
bool SubgraphTensorPass::IsUsing(schema::MetaGraphT *graph, const uint32_t &tensor_idx) {
  for (const auto &node : graph->nodes) {
    if (IsContain<uint32_t>(node->inputIndex, tensor_idx)) {
      return true;
    }
    if (IsContain<uint32_t>(node->outputIndex, tensor_idx)) {
      return true;
    }
  }
  return false;
}

STATUS SubgraphTensorPass::UpdateTensorIdx(schema::MetaGraphT *graph, const uint32_t &tensor_idx) {
  for (const auto &subgraph : graph->subGraph) {
    UpdateVec<uint32_t>(&(subgraph->inputIndices), tensor_idx);
    UpdateVec<uint32_t>(&(subgraph->outputIndices), tensor_idx);
  }
  for (const auto &node : graph->nodes) {
    UpdateVec<uint32_t>(&(node->inputIndex), tensor_idx);
    UpdateVec<uint32_t>(&(node->outputIndex), tensor_idx);
  }
  UpdateVec<uint32_t>(&(graph->inputIndex), tensor_idx);
  UpdateVec<uint32_t>(&(graph->outputIndex), tensor_idx);
  return RET_OK;
}

STATUS SubgraphTensorPass::RemoveUselessTensors(schema::MetaGraphT *graph) {
  for (auto it = graph->allTensors.begin(); it != graph->allTensors.end();) {
    uint32_t idx = it - graph->allTensors.begin();
    if (IsUsing(graph, idx)) {
      it++;
    } else {
      it = graph->allTensors.erase(it);
      UpdateTensorIdx(graph, idx);
    }
  }
  return RET_OK;
}

STATUS SubgraphTensorPass::SyncMainGraphInputAndOutput(schema::MetaGraphT *graph) {
  MS_ASSERT(graph->subGraph.size() > 0);
  graph->subGraph[0]->inputIndices.assign(graph->inputIndex.begin(), graph->inputIndex.end());
  graph->subGraph[0]->outputIndices.assign(graph->outputIndex.begin(), graph->outputIndex.end());
  return RET_OK;
}

STATUS SubgraphTensorPass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);

  int ret = RemoveUselessTensors(graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "RemoveUselessTensors failed, ret: " << ret;
    return ret;
  }

  ret = SetSubgraphTensorIndices(graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetSubgraphTensorIndices failed, ret: " << ret;
    return ret;
  }

  ret = SyncMainGraphInputAndOutput(graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetSubgraphTensorIndices failed, ret: " << ret;
    return ret;
  }

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
