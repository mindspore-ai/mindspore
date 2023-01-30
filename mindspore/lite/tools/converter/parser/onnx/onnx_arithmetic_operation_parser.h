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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_ARITHMETIC_OPERATION_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_ARITHMETIC_OPERATION_PARSER_H_

#include "tools/converter/parser/onnx/onnx_node_parser.h"
#include "tools/converter/parser/onnx/onnx_node_parser_registry.h"

namespace mindspore {
namespace lite {
class OnnxAddParser : public OnnxNodeParser {
 public:
  OnnxAddParser() : OnnxNodeParser("Add") {}
  ~OnnxAddParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxSubParser : public OnnxNodeParser {
 public:
  OnnxSubParser() : OnnxNodeParser("Sub") {}
  ~OnnxSubParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxMulParser : public OnnxNodeParser {
 public:
  OnnxMulParser() : OnnxNodeParser("Mul") {}
  ~OnnxMulParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxDivParser : public OnnxNodeParser {
 public:
  OnnxDivParser() : OnnxNodeParser("Div") {}
  ~OnnxDivParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxPowParser : public OnnxNodeParser {
 public:
  OnnxPowParser() : OnnxNodeParser("Power") {}
  ~OnnxPowParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxEqualParser : public OnnxNodeParser {
 public:
  OnnxEqualParser() : OnnxNodeParser("Equal") {}
  ~OnnxEqualParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxLessParser : public OnnxNodeParser {
 public:
  OnnxLessParser() : OnnxNodeParser("Less") {}
  ~OnnxLessParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxGreaterParser : public OnnxNodeParser {
 public:
  OnnxGreaterParser() : OnnxNodeParser("Greater") {}
  ~OnnxGreaterParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxMinParser : public OnnxNodeParser {
 public:
  OnnxMinParser() : OnnxNodeParser("Min") {}
  ~OnnxMinParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxMaxParser : public OnnxNodeParser {
 public:
  OnnxMaxParser() : OnnxNodeParser("Max") {}
  ~OnnxMaxParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxEltwiseParser : public OnnxNodeParser {
 public:
  OnnxEltwiseParser() : OnnxNodeParser("Eltwise") {}
  ~OnnxEltwiseParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxFloorParser : public OnnxNodeParser {
 public:
  OnnxFloorParser() : OnnxNodeParser("Floor") {}
  ~OnnxFloorParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxAbsParser : public OnnxNodeParser {
 public:
  OnnxAbsParser() : OnnxNodeParser("Abs") {}
  ~OnnxAbsParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxNegParser : public OnnxNodeParser {
 public:
  OnnxNegParser() : OnnxNodeParser("Neg") {}
  ~OnnxNegParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxExpParser : public OnnxNodeParser {
 public:
  OnnxExpParser() : OnnxNodeParser("Exp") {}
  ~OnnxExpParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxCosParser : public OnnxNodeParser {
 public:
  OnnxCosParser() : OnnxNodeParser("Cos") {}
  ~OnnxCosParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxSinParser : public OnnxNodeParser {
 public:
  OnnxSinParser() : OnnxNodeParser("Sin") {}
  ~OnnxSinParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxSqrtParser : public OnnxNodeParser {
 public:
  OnnxSqrtParser() : OnnxNodeParser("Sqrt") {}
  ~OnnxSqrtParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxCeilParser : public OnnxNodeParser {
 public:
  OnnxCeilParser() : OnnxNodeParser("Ceil") {}
  ~OnnxCeilParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxLogParser : public OnnxNodeParser {
 public:
  OnnxLogParser() : OnnxNodeParser("Log") {}
  ~OnnxLogParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxTanParser : public OnnxNodeParser {
 public:
  OnnxTanParser() : OnnxNodeParser("Tan") {}
  ~OnnxTanParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxAtanParser : public OnnxNodeParser {
 public:
  OnnxAtanParser() : OnnxNodeParser("Atan") {}
  ~OnnxAtanParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxAsinParser : public OnnxNodeParser {
 public:
  OnnxAsinParser() : OnnxNodeParser("Asin") {}
  ~OnnxAsinParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxAndParser : public OnnxNodeParser {
 public:
  OnnxAndParser() : OnnxNodeParser("And") {}
  ~OnnxAndParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxOrParser : public OnnxNodeParser {
 public:
  OnnxOrParser() : OnnxNodeParser("Or") {}
  ~OnnxOrParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxNotParser : public OnnxNodeParser {
 public:
  OnnxNotParser() : OnnxNodeParser("Not") {}
  ~OnnxNotParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxRoundParser : public OnnxNodeParser {
 public:
  OnnxRoundParser() : OnnxNodeParser("Round") {}
  ~OnnxRoundParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxReciprocalParser : public OnnxNodeParser {
 public:
  OnnxReciprocalParser() : OnnxNodeParser("Reciprocal") {}
  ~OnnxReciprocalParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxModParser : public OnnxNodeParser {
 public:
  OnnxModParser() : OnnxNodeParser("Mod") {}
  ~OnnxModParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_ARITHMETIC_OPERATION_PARSER_H_
