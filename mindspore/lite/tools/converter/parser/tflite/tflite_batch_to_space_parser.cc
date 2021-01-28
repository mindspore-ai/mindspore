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

#include "tools/converter/parser/tflite/tflite_batch_to_space_parser.h"
#include <vector>
#include <memory>
#include <string>
#include "ops/batch_to_space.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteBatchToSpaceParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                 const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::BatchToSpace>();

  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  const auto &tflite_subgraph = tflite_model->subgraphs.front();
  if (tflite_subgraph == nullptr) {
    MS_LOG(ERROR) << "tflite_subgraph is nullptr";
    return nullptr;
  }
  std::vector<int64_t> blockShape;
  if (GetTfliteData(tflite_op->inputs.at(1), tflite_subgraph->tensors, tflite_model->buffers, blockShape)) {
    MS_LOG(ERROR) << "get batchToSpace -> blockShape failed";
    return nullptr;
  }
  prim->set_block_size(blockShape);

  std::vector<std::vector<int64_t>> crops;
  if (TransTfliteDataToVec2D(tflite_op->inputs.at(2), tflite_subgraph->tensors, tflite_model->buffers, crops)) {
    MS_LOG(ERROR) << "get batchToSpace -> crops failed";
    return nullptr;
  }
  prim->set_crops(crops);

  return prim.release();
}

TfliteNodeRegister g_tfliteBatchToSpaceNDParser(tflite::BuiltinOperator_BATCH_TO_SPACE_ND,
                                                new TfliteBatchToSpaceParser());
}  // namespace lite
}  // namespace mindspore
