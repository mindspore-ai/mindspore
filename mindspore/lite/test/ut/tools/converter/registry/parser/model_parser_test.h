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

#ifndef LITE_TEST_UT_TOOLS_CONVERTER_REGISTRY_MODEL_PARSER_TEST_H
#define LITE_TEST_UT_TOOLS_CONVERTER_REGISTRY_MODEL_PARSER_TEST_H

#include <map>
#include <string>
#include <vector>
#include "include/registry/model_parser.h"
#include "include/registry/model_parser_registry.h"
#include "ut/tools/converter/registry/parser/node_parser_test.h"

namespace mindspore {
class ModelParserTest : public converter::ModelParser {
 public:
  ModelParserTest() = default;
  api::FuncGraphPtr Parse(const converter::ConverterParameters &flag) override;

 private:
  int InitOriginModelStructure();
  int BuildGraphInputs();
  int BuildGraphNodes();
  int BuildGraphOutputs();
  std::map<std::string, AnfNodePtr> nodes_;
  std::map<std::string, std::vector<std::string>> model_layers_info_;
  std::vector<std::string> model_structure_;
};

converter::ModelParser *TestModelParserCreator();
}  // namespace mindspore

#endif  // LITE_TEST_UT_TOOLS_CONVERTER_REGISTRY_MODEL_PARSER_TEST_H
