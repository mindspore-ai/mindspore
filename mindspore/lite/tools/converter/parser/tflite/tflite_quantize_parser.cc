/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/tflite/tflite_quantize_parser.h"
#include <vector>
#include <memory>

namespace mindspore {
namespace lite {
PrimitiveC *TfliteQuantizeParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                     const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto &tflite_subgraph = tflite_model->subgraphs.front();
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is null";
    return nullptr;
  }

  const auto &in_tensor = tflite_subgraph->tensors[tflite_op->inputs[0]];
  if (in_tensor == nullptr) {
    MS_LOG(ERROR) << "input tensor is null";
    return nullptr;
  }
  const auto &out_tensor = tflite_subgraph->tensors[tflite_op->outputs[0]];
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "output tensor is null";
    return nullptr;
  }
  if ((GetTfliteDataType(out_tensor->type) == kNumberTypeInt8 ||
       GetTfliteDataType(out_tensor->type) == kNumberTypeUInt8)) {
    std::unique_ptr<schema::QuantDTypeCastT> attr = std::make_unique<schema::QuantDTypeCastT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    attr->srcT = GetTfliteDataType(in_tensor->type);
    attr->dstT = GetTfliteDataType(out_tensor->type);
    primitive->value.type = schema::PrimitiveType_QuantDTypeCast;
    primitive->value.value = attr.release();
  } else {
    std::unique_ptr<schema::CastT> attr = std::make_unique<schema::CastT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return nullptr;
    }
    attr->srcT = GetTfliteDataType(in_tensor->type);
    attr->dstT = GetTfliteDataType(out_tensor->type);
    primitive->value.type = schema::PrimitiveType_Cast;
    primitive->value.value = attr.release();
  }
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_tfliteQuantizeParser(tflite::BuiltinOperator_QUANTIZE, new TfliteQuantizeParser());
}  // namespace lite
}  // namespace mindspore
