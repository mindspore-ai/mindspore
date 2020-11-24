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

#include "tools/converter/parser/tflite/tflite_node_parser_registry.h"

namespace mindspore {
namespace lite {
TfliteNodeParserRegistry::TfliteNodeParserRegistry() = default;

TfliteNodeParserRegistry::~TfliteNodeParserRegistry() {
  for (auto ite : parsers) {
    if (ite.second != nullptr) {
      delete ite.second;
      ite.second = nullptr;
    }
  }
}

TfliteNodeParserRegistry *TfliteNodeParserRegistry::GetInstance() {
  static TfliteNodeParserRegistry instance;
  return &instance;
}

TfliteNodeParser *TfliteNodeParserRegistry::GetNodeParser(const tflite::BuiltinOperator &type) {
  auto it = parsers.find(type);
  if (it != parsers.end()) {
    return it->second;
  }
  return nullptr;
}
}  // namespace lite
}  // namespace mindspore
