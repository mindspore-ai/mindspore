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
 * distributed under the License is distributed on an AS
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tools/converter/parser/tflite/tflite_while_parser.h"
#include <vector>
#include <memory>
#include "tools/converter/ops/while.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteWhileParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<While>();

  MS_ASSERT(tflite_op != nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsWhileOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get While attr failed";
    return nullptr;
  }
  prim->set_cond_subgraph_index(tflite_attr->cond_subgraph_index);
  prim->set_body_subgraph_index(tflite_attr->body_subgraph_index);

  return prim.release();
}

TfliteNodeRegister g_tfliteWhileParser(tflite::BuiltinOperator_WHILE, new TfliteWhileParser());
}  // namespace lite
}  // namespace mindspore
