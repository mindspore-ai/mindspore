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

#include <vector>
#include "tools/converter/legacy_optimizer/graph/model_input_format_preprocess_pass.h"
#include "utils/log_adapter.h"
#include "tools/common/converter_op_utils.h"
#include "tools/common/node_util.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
STATUS ModelInputFormatPreProcessPass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  for (auto inputIndex : graph->inputIndex) {
    if (graph->allTensors[inputIndex]->dims.size() == 4) {
      std::vector<int32_t> tmpDims(graph->allTensors[inputIndex]->dims);
      auto status =
        NodeUtils::ConvertDims(schema::Format_NCHW, tmpDims, schema::Format_NHWC, &graph->allTensors[inputIndex]->dims);
      if (status == RET_OK) {
        graph->allTensors[inputIndex]->format = schema::Format_NHWC;
      } else {
        MS_LOG(ERROR) << "ConvertDims from NHWC to NCHW error: " << status;
        return RET_ERROR;
      }
    } else {
      graph->allTensors[inputIndex]->format = schema::Format_NHWC;
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
