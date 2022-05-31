/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_LITE_MODEL_PARSER_CREATOR_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_LITE_MODEL_PARSER_CREATOR_H_

#include "include/registry/model_parser.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "src/common/log_adapter.h"

namespace mindspore::lite {
template <class T>
converter::ModelParser *LiteModelParserCreator() {
  auto *parser = new (std::nothrow) T();
  if (parser == nullptr) {
    MS_LOG(ERROR) << "new model parser failed";
    return nullptr;
  }
  return parser;
}
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_LITE_MODEL_PARSER_CREATOR_H_
