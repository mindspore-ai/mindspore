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

#include "tools/converter/parser/tflite/tflite_skip_gram_parser.h"
#include <vector>
#include <memory>
#include "ops/skip_gram.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteSkipGramParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                             const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::SkipGram>();

  MS_ASSERT(tflite_op != nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsSkipGramOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get SkipGram attr failed";
    return nullptr;
  }
  prim->set_include_all_grams(tflite_attr->include_all_ngrams);
  prim->set_max_skip_size(tflite_attr->max_skip_size);
  prim->set_ngram_size(tflite_attr->ngram_size);

  return prim.release();
}

TfliteNodeRegister g_tfliteSkiGramParser(tflite::BuiltinOperator_SKIP_GRAM, new TfliteSkipGramParser());
}  // namespace lite
}  // namespace mindspore
