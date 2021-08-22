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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MODEL_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_MODEL_PARSER_H
#include <google/protobuf/message.h>
#include <string>
#include <memory>
#include "schema/inner/model_generated.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "include/registry/model_parser_registry.h"
#include "utils/log_adapter.h"

namespace mindspore::converter {
class ModelParser {
 public:
  ModelParser() = default;

  virtual ~ModelParser() = default;

  virtual FuncGraphPtr Parse(const converter::ConverterParameters &flags) { return this->res_graph_; }

 protected:
  FuncGraphPtr res_graph_ = nullptr;
};

typedef ModelParser *(*ModelParserCreator)();

template <class T>
ModelParser *LiteModelParserCreator() {
  auto *parser = new (std::nothrow) T();
  if (parser == nullptr) {
    MS_LOG(ERROR) << "new model parser failed";
    return nullptr;
  }
  return parser;
}
}  // namespace mindspore::converter

#endif
