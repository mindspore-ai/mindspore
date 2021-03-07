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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_CONV_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_CONV_PARSER_H

#include <memory>
#include <vector>
#include "tools/converter/parser/onnx/onnx_node_parser.h"
#include "tools/converter/parser/onnx/onnx_node_parser_registry.h"
#include "ops/primitive_c.h"

namespace mindspore {
namespace lite {
class OnnxConvParser : public OnnxNodeParser {
 public:
  OnnxConvParser() : OnnxNodeParser("Conv") {}
  ~OnnxConvParser() override = default;

  ops::PrimitiveC *Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

STATUS ParseVecAttr(const onnx::NodeProto &onnx_node, std::vector<int64_t> *kernels, std::vector<int64_t> *strides,
                    std::vector<int64_t> *dilation, std::vector<int64_t> *pads);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_CONV_PARSER_H
