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

#include "tools/converter/parser/tflite/tflite_concat_parser.h"
#include <vector>
#include <memory>
#include "ops/concat.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteConcatParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                           const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Concat>();

  MS_ASSERT(tflite_op != nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsConcatenationOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op concat attr failed";
    return nullptr;
  }
  prim->set_axis(tflite_attr->axis);

  return prim.release();
}

TfliteNodeRegister g_tfliteConcatParser(tflite::BuiltinOperator_CONCATENATION, new TfliteConcatParser());
}  // namespace lite
}  // namespace mindspore
