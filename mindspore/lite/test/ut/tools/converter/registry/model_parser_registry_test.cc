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

#include <functional>
#include "common/common_test.h"
#include "include/registry/model_parser_registry.h"
#include "tools/converter/model_parser.h"
#include "tools/converter/converter_flags.h"

using mindspore::lite::ModelRegistrar;
using mindspore::lite::converter::Flags;
namespace mindspore {
class ModelParserRegistryTest : public mindspore::CommonTest {
 public:
  ModelParserRegistryTest() = default;
};

class ModelParserTest : public lite::ModelParser {
 public:
  ModelParserTest() = default;
};

lite::ModelParser *TestModelParserCreator() {
  auto *parser = new (std::nothrow) ModelParserTest();
  if (parser == nullptr) {
    MS_LOG(ERROR) << "new model parser failed";
    return nullptr;
  }
  return parser;
}
REG_MODEL_PARSER(TEST, TestModelParserCreator);

TEST_F(ModelParserRegistryTest, TestRegistry) {
  auto model_parser = lite::ModelParserRegistry::GetInstance()->GetModelParser("TEST");
  ASSERT_NE(model_parser, nullptr);
  Flags flags;
  auto func_graph = model_parser->Parse(flags);
  ASSERT_EQ(func_graph, nullptr);
}
}  // namespace mindspore
