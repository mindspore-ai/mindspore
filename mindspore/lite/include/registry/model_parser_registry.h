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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MODEL_PARSER_REGISTRY_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_MODEL_PARSER_REGISTRY_H
#include <string>
#include <memory>
#include <unordered_map>
#include "include/lite_utils.h"

namespace mindspore::lite {
class MS_API ModelParser;
typedef ModelParser *(*ModelParserCreator)();

class MS_API ModelParserRegistry {
 public:
  ModelParserRegistry() = default;
  ~ModelParserRegistry() = default;

  static ModelParserRegistry *GetInstance();
  ModelParser *GetModelParser(const std::string &fmk);
  void RegParser(const std::string &fmk, ModelParserCreator creator);

  std::unordered_map<std::string, ModelParserCreator> parsers_;
};

class MS_API ModelRegistrar {
 public:
  ModelRegistrar(const std::string &fmk, ModelParserCreator creator) {
    ModelParserRegistry::GetInstance()->RegParser(fmk, creator);
  }
  ~ModelRegistrar() = default;
};

#define REG_MODEL_PARSER(fmk, parserCreator) static ModelRegistrar g_##type##fmk##ModelParserReg(#fmk, parserCreator);
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MODEL_PARSER_REGISTRY_H
