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

#include "ut/tools/converter/parser/tflite/tflite_parsers_test_utils.h"
#include <string>
#include "schema/inner/model_generated.h"
#include "tools/converter/parser/tflite/tflite_model_parser.h"

namespace mindspore {

schema::MetaGraphT *TestTfliteParser::LoadAndConvert(const string &model_path, const string &weight_path) {
  lite::TfliteModelParser parser;
  meta_graph = parser.ParseToFb(model_path, weight_path, schema::QuantType_QUANT_NONE);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Parse to metaGraph return nullptr";
    return nullptr;
  }
  return meta_graph;
}

void TestTfliteParser::TearDown() { free(meta_graph); }

}  // namespace mindspore
