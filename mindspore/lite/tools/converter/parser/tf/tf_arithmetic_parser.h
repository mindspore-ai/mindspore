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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_TF_ARITHMETIC_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_TF_ARITHMETIC_PARSER_H_

#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser.h"

namespace mindspore {
namespace lite {
class TFAddParser : public TFNodeParser {
 public:
  TFAddParser() = default;
  ~TFAddParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFSubParser : public TFNodeParser {
 public:
  TFSubParser() = default;
  ~TFSubParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFMulParser : public TFNodeParser {
 public:
  TFMulParser() = default;
  ~TFMulParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFDivParser : public TFNodeParser {
 public:
  TFDivParser() = default;
  ~TFDivParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFMaximumParser : public TFNodeParser {
 public:
  TFMaximumParser() = default;
  ~TFMaximumParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFMinimumParser : public TFNodeParser {
 public:
  TFMinimumParser() = default;
  ~TFMinimumParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFGreaterParser : public TFNodeParser {
 public:
  TFGreaterParser() = default;
  ~TFGreaterParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFGreaterEqualParser : public TFNodeParser {
 public:
  TFGreaterEqualParser() = default;
  ~TFGreaterEqualParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFLessParser : public TFNodeParser {
 public:
  TFLessParser() = default;
  ~TFLessParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFLessEqualParser : public TFNodeParser {
 public:
  TFLessEqualParser() = default;
  ~TFLessEqualParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFEqualParser : public TFNodeParser {
 public:
  TFEqualParser() = default;
  ~TFEqualParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFNotEqualParser : public TFNodeParser {
 public:
  TFNotEqualParser() = default;
  ~TFNotEqualParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFSquaredDifferenceParser : public TFNodeParser {
 public:
  TFSquaredDifferenceParser() = default;
  ~TFSquaredDifferenceParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFRsqrtParser : public TFNodeParser {
 public:
  TFRsqrtParser() = default;
  ~TFRsqrtParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFRoundParser : public TFNodeParser {
 public:
  TFRoundParser() = default;
  ~TFRoundParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFCeilParser : public TFNodeParser {
 public:
  TFCeilParser() = default;
  ~TFCeilParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFExpParser : public TFNodeParser {
 public:
  TFExpParser() = default;
  ~TFExpParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFFloorParser : public TFNodeParser {
 public:
  TFFloorParser() = default;
  ~TFFloorParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFFloorDivParser : public TFNodeParser {
 public:
  TFFloorDivParser() = default;
  ~TFFloorDivParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFFloorModParser : public TFNodeParser {
 public:
  TFFloorModParser() = default;
  ~TFFloorModParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFLogParser : public TFNodeParser {
 public:
  TFLogParser() = default;
  ~TFLogParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFSqrtParser : public TFNodeParser {
 public:
  TFSqrtParser() = default;
  ~TFSqrtParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFCosParser : public TFNodeParser {
 public:
  TFCosParser() = default;
  ~TFCosParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFSinParser : public TFNodeParser {
 public:
  TFSinParser() = default;
  ~TFSinParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFSquareParser : public TFNodeParser {
 public:
  TFSquareParser() = default;
  ~TFSquareParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFPowParser : public TFNodeParser {
 public:
  TFPowParser() = default;
  ~TFPowParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};

class TFAbsParser : public TFNodeParser {
 public:
  TFAbsParser() = default;
  ~TFAbsParser() override = default;

  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_TF_ARITHMETIC_PARSER_H_
