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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_ACTIVATION_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_ACTIVATION_PARSER_H

#include <vector>
#include <memory>
#include <map>
#include "tools/converter/parser/tflite/tflite_node_parser.h"
#include "tools/converter/parser/tflite/tflite_node_parser_registry.h"

namespace mindspore {
namespace lite {
class TfliteReluParser : public TfliteNodeParser {
 public:
  TfliteReluParser() : TfliteNodeParser("Relu") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteRelu6Parser : public TfliteNodeParser {
 public:
  TfliteRelu6Parser() : TfliteNodeParser("Relu6") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteLeakyReluParser : public TfliteNodeParser {
 public:
  TfliteLeakyReluParser() : TfliteNodeParser("LeakyRelu") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TflitePReLUParser : public TfliteNodeParser {
 public:
  TflitePReLUParser() : TfliteNodeParser("PReLU") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteTanhParser : public TfliteNodeParser {
 public:
  TfliteTanhParser() : TfliteNodeParser("Tanh") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteHardSwishParser : public TfliteNodeParser {
 public:
  TfliteHardSwishParser() : TfliteNodeParser("HardSwish") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteLogisticParser : public TfliteNodeParser {
 public:
  TfliteLogisticParser() : TfliteNodeParser("Logistic") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_ACTIVATION_PARSER_H
