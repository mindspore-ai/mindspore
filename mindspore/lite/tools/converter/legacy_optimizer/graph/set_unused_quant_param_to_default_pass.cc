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
#include "tools/converter/legacy_optimizer/graph/set_unused_quant_param_to_default_pass.h"
#include "tools/common/tensor_util.h"
#include "src/common/log_util.h"

namespace mindspore::lite {
STATUS SetUnusedQuantParamToDefaultPass::Run(schema::MetaGraphT *graph) {
  CHECK_NULL_RETURN(graph);
  for (auto &tensor : graph->allTensors) {
    bool has_quant_param = false;
    for (auto &quant_param : tensor->quantParams) {
      quant_param->min = 0.0;
      quant_param->max = 0.0;
      quant_param->narrowRange = true;
      if (quant_param->inited) {
        has_quant_param = true;
        quant_param->inited = false;
      } else {
        quant_param = nullptr;
      }
    }
    if (!has_quant_param) {
      tensor->quantParams.clear();
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
