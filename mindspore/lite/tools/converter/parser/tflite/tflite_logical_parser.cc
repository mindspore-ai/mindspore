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

#include "tools/converter/parser/tflite/tflite_logical_parser.h"
#include <vector>
#include <memory>
#include "ops/logical_and.h"
#include "ops/logical_not.h"
#include "ops/logical_or.h"

namespace mindspore {
namespace lite {

ops::PrimitiveC *TfliteLogicalAndParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                               const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::LogicalAnd>();
  return prim.release();
}

ops::PrimitiveC *TfliteLogicalNotParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                               const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::LogicalNot>();
  return prim.release();
}

ops::PrimitiveC *TfliteLogicalOrParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                              const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::LogicalOr>();
  return prim.release();
}

TfliteNodeRegister g_tfliteLogicalAndParser(tflite::BuiltinOperator_LOGICAL_AND, new TfliteLogicalAndParser());
TfliteNodeRegister g_tfliteLogicalNotParser(tflite::BuiltinOperator_LOGICAL_NOT, new TfliteLogicalNotParser());
TfliteNodeRegister g_tfliteLogicalOrParser(tflite::BuiltinOperator_LOGICAL_OR, new TfliteLogicalOrParser());
}  // namespace lite
}  // namespace mindspore
