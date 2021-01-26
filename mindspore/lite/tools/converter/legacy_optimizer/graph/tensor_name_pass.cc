/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "tools/converter/legacy_optimizer/graph/tensor_name_pass.h"
#include "tools/converter/converter_context.h"
#include "tools/common/tensor_util.h"

namespace mindspore::lite {
STATUS TensorNamePass::Run(schema::MetaGraphT *graph) {
  if (graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr";
    return RET_NULL_PTR;
  }

  for (auto &node : graph->nodes) {
    if (node == nullptr || node->primitive == nullptr) {
      MS_LOG(ERROR) << "node or node->primitive is nullptr";
      return RET_NULL_PTR;
    }

    for (int i = 0; i < static_cast<int>(node->inputIndex.size()); i++) {
      auto tensor_id = node->inputIndex.at(i);
      auto &tensor = graph->allTensors.at(tensor_id);
      if (tensor->name.empty()) {
        MS_LOG(DEBUG) << "input tensor (id = " << tensor_id << ") name is null";
        tensor->name = node->name + "/input-" + std::to_string(i);
      }
    }

    for (int i = 0; i < static_cast<int>(node->outputIndex.size()); i++) {
      auto tensor_id = node->outputIndex.at(i);
      auto &tensor = graph->allTensors.at(tensor_id);
      if (tensor->name.empty()) {
        tensor->name = node->name + "/output-" + std::to_string(i);
      }
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
