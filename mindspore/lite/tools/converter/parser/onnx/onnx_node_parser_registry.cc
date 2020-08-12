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

#include "tools/converter/parser/onnx/onnx_node_parser_registry.h"
#include <string>

namespace mindspore {
namespace lite {
OnnxNodeParserRegistry::OnnxNodeParserRegistry() = default;

OnnxNodeParserRegistry::~OnnxNodeParserRegistry() = default;

OnnxNodeParserRegistry *OnnxNodeParserRegistry::GetInstance() {
  static OnnxNodeParserRegistry instance;
  return &instance;
}

OnnxNodeParser *OnnxNodeParserRegistry::GetNodeParser(const std::string &name) {
  auto it = parsers.find(name);
  if (it != parsers.end()) {
    return it->second;
  }
  /* should not support vague name, otherwise may get wrong parser. ex. PRelu and Relu
  for (auto const &i : parsers) {
    if (name.find(i.first) != std::string::npos) {
      return i.second;
    }
  }
   */
  return nullptr;
}
}  // namespace lite
}  // namespace mindspore
