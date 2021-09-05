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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_NODE_REGISTRY_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_NODE_REGISTRY_H

#include <string>
#include <unordered_map>
#include "tools/converter/parser/onnx/onnx_node_parser.h"

namespace mindspore {
namespace lite {
class OnnxNodeParserRegistry {
 public:
  virtual ~OnnxNodeParserRegistry();

  static OnnxNodeParserRegistry &GetInstance();

  OnnxNodeParser *GetNodeParser(const std::string &name) const;

  void RegNodeParser(const std::string &name, OnnxNodeParser *parser);

 private:
  OnnxNodeParserRegistry();

 private:
  std::unordered_map<std::string, OnnxNodeParser *> parsers{};
};

class OnnxNodeRegistrar {
 public:
  OnnxNodeRegistrar(const std::string &name, OnnxNodeParser *parser) {
    OnnxNodeParserRegistry::GetInstance().RegNodeParser(name, parser);
  }
  ~OnnxNodeRegistrar() = default;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_NODE_REGISTRY_H
