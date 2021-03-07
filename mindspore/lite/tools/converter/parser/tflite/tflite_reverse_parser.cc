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

#include "tools/converter/parser/tflite/tflite_reverse_parser.h"
#include <vector>
#include <memory>
#include "ops/reverse_v2.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteReverseParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                            const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::ReverseV2>();

  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  const auto &tflite_subgraph = tflite_model->subgraphs.front();
  if (tflite_subgraph == nullptr) {
    MS_LOG(ERROR) << "tflite_subgraph is nullptr";
    return nullptr;
  }
  std::vector<int64_t> axis;
  if (GetTfliteData(tflite_op->inputs.at(1), tflite_subgraph->tensors, tflite_model->buffers, axis)) {
    MS_LOG(ERROR) << "get reverse -> axis failed";
    return nullptr;
  }
  prim->set_axis(axis);

  return prim.release();
}

TfliteNodeRegister g_tfliteReverseParser(tflite::BuiltinOperator_REVERSE_V2, new TfliteReverseParser());
}  // namespace lite
}  // namespace mindspore
