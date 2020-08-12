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

#ifndef MS_ONNX_ARITHMETIC_OPREATION_PARSER_H
#define MS_ONNX_ARITHMETIC_OPREATION_PARSER_H

#include "tools/converter/parser/onnx/onnx_node_parser.h"
#include "tools/converter/parser/onnx/onnx_node_parser_registry.h"

namespace mindspore {
namespace lite {
class OnnxAddParser : public OnnxNodeParser {
 public:
  OnnxAddParser() : OnnxNodeParser("Add") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxSubParser : public OnnxNodeParser {
 public:
  OnnxSubParser() : OnnxNodeParser("Sub") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxMulParser : public OnnxNodeParser {
 public:
  OnnxMulParser() : OnnxNodeParser("Mul") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxDivParser : public OnnxNodeParser {
 public:
  OnnxDivParser() : OnnxNodeParser("Div") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxMeanParser : public OnnxNodeParser {
 public:
  OnnxMeanParser() : OnnxNodeParser("Mean") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxPowParser : public OnnxNodeParser {
 public:
  OnnxPowParser() : OnnxNodeParser("Power") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxEqualParser : public OnnxNodeParser {
 public:
  OnnxEqualParser() : OnnxNodeParser("Equal") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxLessParser : public OnnxNodeParser {
 public:
  OnnxLessParser() : OnnxNodeParser("Less") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxGreaterParser : public OnnxNodeParser {
 public:
  OnnxGreaterParser() : OnnxNodeParser("Greater") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxMinParser : public OnnxNodeParser {
 public:
  OnnxMinParser() : OnnxNodeParser("Min") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxEltwiseParser : public OnnxNodeParser {
 public:
  OnnxEltwiseParser() : OnnxNodeParser("Eltwise") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxFloorParser : public OnnxNodeParser {
 public:
  OnnxFloorParser() : OnnxNodeParser("Floor") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxAbsParser : public OnnxNodeParser {
 public:
  OnnxAbsParser() : OnnxNodeParser("Abs") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxNegParser : public OnnxNodeParser {
 public:
  OnnxNegParser() : OnnxNodeParser("Neg") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxExpParser : public OnnxNodeParser {
 public:
  OnnxExpParser() : OnnxNodeParser("Exp") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxCosParser : public OnnxNodeParser {
 public:
  OnnxCosParser() : OnnxNodeParser("Cos") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxSinParser : public OnnxNodeParser {
 public:
  OnnxSinParser() : OnnxNodeParser("Sin") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxSqrtParser : public OnnxNodeParser {
 public:
  OnnxSqrtParser() : OnnxNodeParser("Sqrt") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxCeilParser : public OnnxNodeParser {
 public:
  OnnxCeilParser() : OnnxNodeParser("Ceil") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxLogParser : public OnnxNodeParser {
 public:
  OnnxLogParser() : OnnxNodeParser("Log") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxTanParser : public OnnxNodeParser {
 public:
  OnnxTanParser() : OnnxNodeParser("Tan") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxAtanParser : public OnnxNodeParser {
 public:
  OnnxAtanParser() : OnnxNodeParser("Atan") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxAsinParser : public OnnxNodeParser {
 public:
  OnnxAsinParser() : OnnxNodeParser("Asin") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};

class OnnxTanhParser : public OnnxNodeParser {
 public:
  OnnxTanhParser() : OnnxNodeParser("Tanh") {}
  STATUS Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MS_ONNX_ARITHMETIC_OPREATION_PARSER_H

