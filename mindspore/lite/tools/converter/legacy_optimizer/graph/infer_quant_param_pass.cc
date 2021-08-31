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

#include "tools/converter/legacy_optimizer/graph/infer_quant_param_pass.h"
#include <vector>
#include <memory>
#include "src/common/utils.h"
#include "tools/converter/quantizer/quant_helper/quant_node_helper.h"
#include "tools/common/node_util.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
STATUS InferQuantParamPass::Run(schema::MetaGraphT *graph) {
  if (graph == nullptr) {
    MS_LOG(ERROR) << "graph is null";
    return RET_NULL_PTR;
  }

  // forward infer nodes' quant params
  for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
    auto &node = *iter;
    if (node == nullptr) {
      MS_LOG(ERROR) << "node is null";
      return RET_NULL_PTR;
    }

    auto quant_helper = QuantHelperRegister::GetInstance()->GetQuantHelper(node->primitive->value.type);
    MS_CHECK_TRUE_MSG(quant_helper != nullptr, RET_ERROR, "Find QuantHelper return nullptr");
    quant_helper->NodeQuantPreprocess(graph, node.get());
  }

  // backward infer nodes' quant params
  for (auto iter = graph->nodes.rbegin(); iter != graph->nodes.rend(); iter++) {
    auto &node = *iter;
    if (node == nullptr) {
      MS_LOG(ERROR) << "node is null";
      return RET_NULL_PTR;
    }

    if (!node->primitive) {
      continue;
    }

    auto quant_helper = QuantHelperRegister::GetInstance()->GetQuantHelper(node->primitive->value.type);
    MS_CHECK_TRUE_MSG(quant_helper != nullptr, RET_ERROR, "Find QuantHelper return nullptr");
    quant_helper->NodeQuantPreprocess(graph, node.get());
  }
  return RET_OK;
}
}  // namespace mindspore::lite
