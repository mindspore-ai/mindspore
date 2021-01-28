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
 * distributed under the License is distributed on an AS
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tools/converter/parser/tflite/tflite_space_to_batch_nd_parser.h"
#include <vector>
#include <memory>
#include "ops/space_to_batch_nd.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteSpaceToBatchNDParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                   const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::SpaceToBatchND>();

  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  const auto &tflite_subgraph = tflite_model->subgraphs.front();
  if (tflite_subgraph == nullptr) {
    MS_LOG(ERROR) << "tflite_subgraph is nullptr";
    return nullptr;
  }
  std::vector<int64_t> blockShape;
  if (GetTfliteData(tflite_op->inputs.at(1), tflite_subgraph->tensors, tflite_model->buffers, blockShape)) {
    MS_LOG(ERROR) << "get spaceToBatchND -> blockShape failed";
    return nullptr;
  }
  prim->set_block_shape(blockShape);
  std::vector<std::vector<int64_t>> paddings;
  if (TransTfliteDataToVec2D(tflite_op->inputs.at(2), tflite_subgraph->tensors, tflite_model->buffers, paddings)) {
    MS_LOG(ERROR) << "get spaceToBatchND -> paddings failed";
    return nullptr;
  }
  prim->set_paddings(paddings);

  return prim.release();
}

TfliteNodeRegister g_tfliteSpaceToBatchNDParser(tflite::BuiltinOperator_SPACE_TO_BATCH_ND,
                                                new TfliteSpaceToBatchNDParser());
}  // namespace lite
}  // namespace mindspore
