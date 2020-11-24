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
#include <string>

namespace mindspore {
namespace lite {
PrimitiveC *TfliteReduceParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                   const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto &tflite_subgraph = tflite_model->subgraphs.front();
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is null";
    return nullptr;
  }

  std::unique_ptr<schema::ReduceT> attr = std::make_unique<schema::ReduceT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  const auto &tflite_attr = tflite_op->builtin_options.AsReducerOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op reduce attr failed";
    return nullptr;
  }
  attr->keepDims = tflite_attr->keep_dims;

  auto tflite_op_type = (tflite_model->operator_codes[tflite_op->opcode_index])->builtin_code;
  if (tflite_op_type == tflite::BuiltinOperator_REDUCE_MAX) {
    MS_LOG(DEBUG) << "parse TfliteReduceMaxParser";
    attr->mode = schema::ReduceMode_ReduceMax;
  } else if (tflite_op_type == tflite::BuiltinOperator_REDUCE_MIN) {
    MS_LOG(DEBUG) << "parse TfliteReduceMinParser";
    attr->mode = schema::ReduceMode_ReduceMin;
  } else if (tflite_op_type == tflite::BuiltinOperator_REDUCE_PROD) {
    MS_LOG(DEBUG) << "parse TfliteReduceProdParser";
    attr->mode = schema::ReduceMode_ReduceProd;
  } else if (tflite_op_type == tflite::BuiltinOperator_SUM) {
    MS_LOG(DEBUG) << "parse TfliteSumParser";
    attr->mode = schema::ReduceMode_ReduceSum;
  } else if (tflite_op_type == tflite::BuiltinOperator_MEAN) {
    MS_LOG(DEBUG) << "parse TfliteMeanParser";
    attr->mode = schema::ReduceMode_ReduceMean;
  } else if (tflite_op_type == tflite::BuiltinOperator_REDUCE_ANY) {
    // attr->mode;
    MS_LOG(ERROR) << "ms-lite haven't supported REDUCE_ANY now";
    return nullptr;
  }

  if (GetTfliteData(tflite_op->inputs[1], tflite_subgraph->tensors, tflite_model->buffers, attr->axes)) {
    MS_LOG(ERROR) << "get reduce -> axes failed";
    return nullptr;
  }

  primitive->value.type = schema::PrimitiveType_Reduce;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_TfliteSumParser(tflite::BuiltinOperator_SUM, new TfliteReduceParser());
TfliteNodeRegister g_TfliteMeanParser(tflite::BuiltinOperator_MEAN, new TfliteReduceParser());
TfliteNodeRegister g_TfliteReduceMaxParser(tflite::BuiltinOperator_REDUCE_MAX, new TfliteReduceParser());
TfliteNodeRegister g_TfliteReduceMinParser(tflite::BuiltinOperator_REDUCE_MIN, new TfliteReduceParser());
TfliteNodeRegister g_TfliteReduceProdParser(tflite::BuiltinOperator_REDUCE_PROD, new TfliteReduceParser());
TfliteNodeRegister g_TfliteReduceAnyParser(tflite::BuiltinOperator_REDUCE_ANY, new TfliteReduceParser());
}  // namespace lite
}  // namespace mindspore
