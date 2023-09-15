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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_CUSTOM_OP_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_CUSTOM_OP_PARSER_H_

#include "tools/converter/parser/onnx/onnx_node_parser.h"
#include "tools/converter/parser/onnx/onnx_node_parser_registry.h"

namespace mindspore {
namespace lite {
class OnnxAffineGridParser : public OnnxNodeParser {
 public:
  OnnxAffineGridParser() : OnnxNodeParser("affine_grid") {}
  ~OnnxAffineGridParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxHistogramParser : public OnnxNodeParser {
 public:
  OnnxHistogramParser() : OnnxNodeParser("Histogram") {}
  ~OnnxHistogramParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxLogicalNotParser : public OnnxNodeParser {
 public:
  OnnxLogicalNotParser() : OnnxNodeParser("logical_not") {}
  ~OnnxLogicalNotParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxRot90Parser : public OnnxNodeParser {
 public:
  OnnxRot90Parser() : OnnxNodeParser("Rot90") {}
  ~OnnxRot90Parser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxXlogyParser : public OnnxNodeParser {
 public:
  OnnxXlogyParser() : OnnxNodeParser("xlogy") {}
  ~OnnxXlogyParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxRandomUniformLikeParser : public OnnxNodeParser {
 public:
  OnnxRandomUniformLikeParser() : OnnxNodeParser("RandomUniformLike") {}
  ~OnnxRandomUniformLikeParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxBlendFaceBgPartOneParser : public OnnxNodeParser {
 public:
  OnnxBlendFaceBgPartOneParser() : OnnxNodeParser("BlendFaceBgPartOne") {}
  ~OnnxBlendFaceBgPartOneParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_ONNX_CUSTOM_OP_PARSER_H_
