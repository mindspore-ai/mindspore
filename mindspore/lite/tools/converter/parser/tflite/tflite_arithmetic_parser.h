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

#ifndef PREDICT_TFLITE_MATH_PARSER_H
#define PREDICT_TFLITE_MATH_PARSER_H

#include <memory>
#include <vector>
#include "tools/converter/parser/tflite/tflite_node_parser.h"
#include "tools/converter/parser/tflite/tflite_node_parser_registry.h"

namespace mindspore {
namespace lite {

class TfliteDoubleInputOpParser : public TfliteNodeParser {
 public:
  TfliteDoubleInputOpParser() : TfliteNodeParser("node_name") {}

  STATUS Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
               const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
               const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
               const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet, schema::CNodeT *op,
               TensorCache *tensor_cache, bool quantizedModel) override;
};

class TfliteAddParser : public TfliteDoubleInputOpParser {
 public:
  TfliteAddParser() : TfliteDoubleInputOpParser() {}
};

class TfliteSubParser : public TfliteDoubleInputOpParser {
 public:
  TfliteSubParser() : TfliteDoubleInputOpParser() {}
};

class TfliteMulParser : public TfliteDoubleInputOpParser {
 public:
  TfliteMulParser() : TfliteDoubleInputOpParser() {}
};

class TfliteDivParser : public TfliteDoubleInputOpParser {
 public:
  TfliteDivParser() : TfliteDoubleInputOpParser() {}
};

class TfliteFloorDivParser : public TfliteDoubleInputOpParser {
 public:
  TfliteFloorDivParser() : TfliteDoubleInputOpParser() {}
};

class TfliteFloorModParser : public TfliteDoubleInputOpParser {
 public:
  TfliteFloorModParser() : TfliteDoubleInputOpParser() {}
};

class TfliteSquaredDifferenceParser : public TfliteDoubleInputOpParser {
 public:
  TfliteSquaredDifferenceParser() : TfliteDoubleInputOpParser() {}
};

class TfliteRealDivParser : public TfliteDoubleInputOpParser {
 public:
  TfliteRealDivParser() : TfliteDoubleInputOpParser() {}
};

class TflitePowParser : public TfliteDoubleInputOpParser {
 public:
  TflitePowParser() : TfliteDoubleInputOpParser() {}
};

class TfliteMaximumParser : public TfliteDoubleInputOpParser {
 public:
  TfliteMaximumParser() : TfliteDoubleInputOpParser() {}
};

class TfliteMinimumParser : public TfliteDoubleInputOpParser {
 public:
  TfliteMinimumParser() : TfliteDoubleInputOpParser() {}
};


class TfliteSingleInputOpParser : public TfliteNodeParser {
 public:
  TfliteSingleInputOpParser() : TfliteNodeParser("node_name") {}

  STATUS Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
               const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
               const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
               const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet, schema::CNodeT *op,
               TensorCache *tensor_cache, bool quantizedModel) override;
};

class TfliteAbsParser : public TfliteSingleInputOpParser {
 public:
  TfliteAbsParser() : TfliteSingleInputOpParser() {}
};

class TfliteExpParser : public TfliteSingleInputOpParser {
 public:
  TfliteExpParser() : TfliteSingleInputOpParser() {}
};

class TfliteSqrtParser : public TfliteSingleInputOpParser {
 public:
  TfliteSqrtParser() : TfliteSingleInputOpParser() {}
};

class TfliteSquareParser : public TfliteSingleInputOpParser {
 public:
  TfliteSquareParser() : TfliteSingleInputOpParser() {}
};

class TfliteSinParser : public TfliteSingleInputOpParser {
 public:
  TfliteSinParser() : TfliteSingleInputOpParser() {}
};

class TfliteCosParser : public TfliteSingleInputOpParser {
 public:
  TfliteCosParser() : TfliteSingleInputOpParser() {}
};

class TfliteRsqrtParser : public TfliteSingleInputOpParser {
 public:
  TfliteRsqrtParser() : TfliteSingleInputOpParser() {}
};

class TfliteLogParser : public TfliteSingleInputOpParser {
 public:
  TfliteLogParser() : TfliteSingleInputOpParser() {}
};

class TfliteRoundParser : public TfliteSingleInputOpParser {
 public:
  TfliteRoundParser() : TfliteSingleInputOpParser() {}
};

class TfliteCeilParser : public TfliteSingleInputOpParser {
 public:
  TfliteCeilParser() : TfliteSingleInputOpParser() {}
};

class TfliteFloorParser : public TfliteSingleInputOpParser {
 public:
  TfliteFloorParser() : TfliteSingleInputOpParser() {}
};


class TfliteCompareOpParser : public TfliteNodeParser {
 public:
  TfliteCompareOpParser() : TfliteNodeParser("node_name") {}

  STATUS Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
               const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
               const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
               const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet, schema::CNodeT *op,
               TensorCache *tensor_cache, bool quantizedModel) override;
};

class TfliteEqualParser : public TfliteCompareOpParser {
 public:
  TfliteEqualParser() : TfliteCompareOpParser() {}
};

class TfliteNotEqualParser : public TfliteCompareOpParser {
 public:
  TfliteNotEqualParser() : TfliteCompareOpParser() {}
};

class TfliteGreaterParser : public TfliteCompareOpParser {
 public:
  TfliteGreaterParser() : TfliteCompareOpParser() {}
};

class TfliteGreaterEqualParser : public TfliteCompareOpParser {
 public:
  TfliteGreaterEqualParser() : TfliteCompareOpParser() {}
};

class TfliteLessParser : public TfliteCompareOpParser {
 public:
  TfliteLessParser() : TfliteCompareOpParser() {}
};

class TfliteLessEqualParser : public TfliteCompareOpParser {
 public:
  TfliteLessEqualParser() : TfliteCompareOpParser() {}
};

}  // namespace lite
}  // namespace mindspore

#endif  // PREDICT_TFLITE_MATH_PARSER_H

