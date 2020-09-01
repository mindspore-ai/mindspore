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

#include "tools/converter/legacy_optimizer/graph/trans_format_remove_pass.h"
#include <vector>
#include "utils/log_adapter.h"
#include "include/errorcode.h"
#include "tools/common/graph_util.h"
#include "src/ir/tensor.h"
#include "src/ops/primitive_c.h"

using mindspore::lite::tensor::Tensor;
using mindspore::lite::PrimitiveC;
namespace mindspore {
namespace lite {
STATUS TransOpRemovePass::Run(MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
    auto &node = *iter;
    auto type = node->primitive->value.type;
    if (type == schema::PrimitiveType_Nchw2Nhwc || type == schema::PrimitiveType_Nhwc2Nchw) {
      auto &input_tensor = graph->allTensors.at(node->inputIndex.at(0));
      // less than 4 dims can delete
      if (!input_tensor->dims.empty() && input_tensor->dims.size() < 4) {
        auto status = IsolateOneWayNode(graph, node.get(), true);
        if (status != RET_OK) {
          MS_LOG(ERROR) << "IsolateOneWayNode failed, node: " << node->name.c_str() << ", error: " << status;
          return status;
        }
      }
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
