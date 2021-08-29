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

OnnxNodeParserRegistry::~OnnxNodeParserRegistry() {
  for (auto ite : parsers) {
    if (ite.second != nullptr) {
      delete ite.second;
      ite.second = nullptr;
    }
  }
}

OnnxNodeParserRegistry &OnnxNodeParserRegistry::GetInstance() {
  static OnnxNodeParserRegistry instance;
  return instance;
}

OnnxNodeParser *OnnxNodeParserRegistry::GetNodeParser(const std::string &name) const {
  auto it = parsers.find(name);
  if (it != parsers.end()) {
    return it->second;
  }
  return nullptr;
}

void OnnxNodeParserRegistry::RegNodeParser(const std::string &name, OnnxNodeParser *parser) {
  if (parser == nullptr) {
    MS_LOG(WARNING) << "Input OnnxNodeParser is nullptr";
    return;
  }
  if (this->parsers.find(name) != this->parsers.end()) {
    MS_LOG(WARNING) << "OnnxNodeParser " << name << " is already exist";
    return;
  }
  this->parsers[name] = parser;
}
}  // namespace lite
}  // namespace mindspore
