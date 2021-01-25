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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_ARITHMETIC_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_ARITHMETIC_PARSER_H

#include <memory>
#include <vector>
#include <map>
#include "tools/converter/parser/tflite/tflite_node_parser.h"
#include "tools/converter/parser/tflite/tflite_node_parser_registry.h"

namespace mindspore {
namespace lite {
class TfliteAddParser : public TfliteNodeParser {
 public:
  TfliteAddParser() : TfliteNodeParser("Add") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteSubParser : public TfliteNodeParser {
 public:
  TfliteSubParser() : TfliteNodeParser("Sub") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteMulParser : public TfliteNodeParser {
 public:
  TfliteMulParser() : TfliteNodeParser("Mul") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteDivParser : public TfliteNodeParser {
 public:
  TfliteDivParser() : TfliteNodeParser("Div") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteFloorDivParser : public TfliteNodeParser {
 public:
  TfliteFloorDivParser() : TfliteNodeParser("FloorDiv") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteFloorModParser : public TfliteNodeParser {
 public:
  TfliteFloorModParser() : TfliteNodeParser("FloorMod") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TflitePowParser : public TfliteNodeParser {
 public:
  TflitePowParser() : TfliteNodeParser("PowFusion") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteSquaredDifferenceParser : public TfliteNodeParser {
 public:
  TfliteSquaredDifferenceParser() : TfliteNodeParser("SquaredDifference") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteMaximumParser : public TfliteNodeParser {
 public:
  TfliteMaximumParser() : TfliteNodeParser("Maximum") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteMinimumParser : public TfliteNodeParser {
 public:
  TfliteMinimumParser() : TfliteNodeParser("Minimum") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteAbsParser : public TfliteNodeParser {
 public:
  TfliteAbsParser() : TfliteNodeParser("Abs") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteExpParser : public TfliteNodeParser {
 public:
  TfliteExpParser() : TfliteNodeParser("Exp") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteSqrtParser : public TfliteNodeParser {
 public:
  TfliteSqrtParser() : TfliteNodeParser("Sqrt") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteRsqrtParser : public TfliteNodeParser {
 public:
  TfliteRsqrtParser() : TfliteNodeParser("Rsqrt") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteSquareParser : public TfliteNodeParser {
 public:
  TfliteSquareParser() : TfliteNodeParser("Square") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteSinParser : public TfliteNodeParser {
 public:
  TfliteSinParser() : TfliteNodeParser("Sin") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteCosParser : public TfliteNodeParser {
 public:
  TfliteCosParser() : TfliteNodeParser("Cos") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteLogParser : public TfliteNodeParser {
 public:
  TfliteLogParser() : TfliteNodeParser("Log") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteRoundParser : public TfliteNodeParser {
 public:
  TfliteRoundParser() : TfliteNodeParser("Round") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteCeilParser : public TfliteNodeParser {
 public:
  TfliteCeilParser() : TfliteNodeParser("Ceil") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteFloorParser : public TfliteNodeParser {
 public:
  TfliteFloorParser() : TfliteNodeParser("Floor") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteNegParser : public TfliteNodeParser {
 public:
  TfliteNegParser() : TfliteNodeParser("Neg") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteEqualParser : public TfliteNodeParser {
 public:
  TfliteEqualParser() : TfliteNodeParser("Equal") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteNotEqualParser : public TfliteNodeParser {
 public:
  TfliteNotEqualParser() : TfliteNodeParser("NotEqual") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteGreaterParser : public TfliteNodeParser {
 public:
  TfliteGreaterParser() : TfliteNodeParser("Greater") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteGreaterEqualParser : public TfliteNodeParser {
 public:
  TfliteGreaterEqualParser() : TfliteNodeParser("GreaterEqual") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteLessParser : public TfliteNodeParser {
 public:
  TfliteLessParser() : TfliteNodeParser("Less") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

class TfliteLessEqualParser : public TfliteNodeParser {
 public:
  TfliteLessEqualParser() : TfliteNodeParser("LessEqual") {}

  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_ARITHMETIC_PARSER_H
