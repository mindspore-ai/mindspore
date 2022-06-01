/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TEST_UT_TOOLS_CONVERTER_REGISTRY_PARSER_NODE_PARSER_TEST_H_
#define MINDSPORE_LITE_TEST_UT_TOOLS_CONVERTER_REGISTRY_PARSER_NODE_PARSER_TEST_H_

#include <memory>
#include <string>
#include <unordered_map>
#include "ops/base_operator.h"
#include "mindapi/base/shared_ptr.h"
#include "src/common/log_adapter.h"

namespace mindspore {
using BaseOperatorPtr = api::SharedPtr<ops::BaseOperator>;
class NodeParserTest {
 public:
  NodeParserTest() = default;

  virtual ~NodeParserTest() {}

  virtual BaseOperatorPtr Parse() { return nullptr; }
};
using NodeParserTestPtr = std::shared_ptr<NodeParserTest>;

class NodeParserTestRegistry {
 public:
  static NodeParserTestRegistry *GetInstance() {
    static NodeParserTestRegistry instance;
    return &instance;
  }

  NodeParserTestPtr GetNodeParser(const std::string &name) {
    if (parsers_.find(name) == parsers_.end()) {
      MS_LOG(ERROR) << "cannot find node parser.";
      return nullptr;
    }
    return parsers_[name];
  }

  void RegNodeParser(const std::string &name, const NodeParserTestPtr node_parser) { parsers_[name] = node_parser; }

 private:
  NodeParserTestRegistry() = default;
  virtual ~NodeParserTestRegistry() = default;
  std::unordered_map<std::string, NodeParserTestPtr> parsers_;
};

class RegisterNodeParserTest {
 public:
  RegisterNodeParserTest(const std::string &name, NodeParserTestPtr node_parser) {
    NodeParserTestRegistry::GetInstance()->RegNodeParser(name, node_parser);
  }
};

#define REGISTER_NODE_PARSER_TEST(name, node_parser) \
  static RegisterNodeParserTest g_##name##_node_parser(name, node_parser);
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TEST_UT_TOOLS_CONVERTER_REGISTRY_PARSER_NODE_PARSER_TEST_H_
