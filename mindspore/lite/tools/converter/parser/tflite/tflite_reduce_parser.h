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

#ifndef PREDICT_TFLITE_REDUCE_MAX_PARSER_H
#define PREDICT_TFLITE_REDUCE_MAX_PARSER_H

#include <memory>
#include <vector>
#include "tools/converter/parser/tflite/tflite_node_parser.h"
#include "tools/converter/parser/tflite/tflite_node_parser_registry.h"

namespace mindspore {
namespace lite {
class TfliteReduceParser : public TfliteNodeParser {
 public:
  TfliteReduceParser() : TfliteNodeParser("node_name") {}

  STATUS Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
               const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
               const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
               const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet, schema::CNodeT *op,
               TensorCache *tensor_cache,
               bool quantizedModel) override;
};

class TfliteReduceMaxParser : public TfliteReduceParser {
 public:
  TfliteReduceMaxParser() : TfliteReduceParser() {}
};

class TfliteReduceMinParser : public TfliteReduceParser {
 public:
  TfliteReduceMinParser() : TfliteReduceParser() {}
};

class TfliteReduceProdParser : public TfliteReduceParser {
 public:
  TfliteReduceProdParser() : TfliteReduceParser() {}
};

class TfliteSumParser : public TfliteReduceParser {
 public:
  TfliteSumParser() : TfliteReduceParser() {}
};

class TfliteMeanParser : public TfliteReduceParser {
 public:
  TfliteMeanParser() : TfliteReduceParser() {}
};

class TfliteReduceAnyParser : public TfliteReduceParser {
 public:
  TfliteReduceAnyParser() : TfliteReduceParser() {}
};

}  // namespace lite
}  // namespace mindspore

#endif  // PREDICT_TFLITE_REDUCE_MAX_PARSER_H
