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

#include "tools/converter/parser/tflite/tflite_reduce_parser.h"
#include <vector>
#include <memory>
#include "ops/fusion/reduce_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteReduceParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                           const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::ReduceFusion>();

  MS_ASSERT(tflite_op != nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsReducerOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get reduce attr failed";
    return nullptr;
  }
  prim->set_keep_dims(tflite_attr->keep_dims);

  auto tflite_op_type = (tflite_model->operator_codes[tflite_op->opcode_index])->builtin_code;
  if (tflite_op_type == tflite::BuiltinOperator_REDUCE_MAX) {
    prim->set_mode(mindspore::ReduceMode::Reduce_Max);
  } else if (tflite_op_type == tflite::BuiltinOperator_REDUCE_MIN) {
    prim->set_mode(mindspore::ReduceMode::Reduce_Min);
  } else if (tflite_op_type == tflite::BuiltinOperator_REDUCE_PROD) {
    prim->set_mode(mindspore::ReduceMode::Reduce_Prod);
  } else if (tflite_op_type == tflite::BuiltinOperator_SUM) {
    prim->set_mode(mindspore::ReduceMode::Reduce_Sum);
  } else if (tflite_op_type == tflite::BuiltinOperator_MEAN) {
    prim->set_mode(mindspore::ReduceMode::Reduce_Mean);
  } else {
    MS_LOG(ERROR) << "unsupported reduce mode:" << tflite_op_type;
    return nullptr;
  }

  return prim.release();
}

TfliteNodeRegister g_TfliteSumParser(tflite::BuiltinOperator_SUM, new TfliteReduceParser());
TfliteNodeRegister g_TfliteMeanParser(tflite::BuiltinOperator_MEAN, new TfliteReduceParser());
TfliteNodeRegister g_TfliteReduceMaxParser(tflite::BuiltinOperator_REDUCE_MAX, new TfliteReduceParser());
TfliteNodeRegister g_TfliteReduceMinParser(tflite::BuiltinOperator_REDUCE_MIN, new TfliteReduceParser());
TfliteNodeRegister g_TfliteReduceProdParser(tflite::BuiltinOperator_REDUCE_PROD, new TfliteReduceParser());
}  // namespace lite
}  // namespace mindspore
