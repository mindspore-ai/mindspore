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
STATUS TfliteQuantizeParser::Parse(TfliteTensorsInfo *tensors_info, const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                   const std::unique_ptr<tflite::ModelT> &tflite_model,
                                   const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "parse TfliteQuantizeNParser";
  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  MS_ASSERT(tflite_subgraph != nullptr);
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  const auto &in_tensor = tflite_subgraph->tensors[tflite_op->inputs[0]];
  if (in_tensor == nullptr) {
    MS_LOG(ERROR) << "input tensor is null";
    return RET_NULL_PTR;
  }
  const auto &out_tensor = tflite_subgraph->tensors[tflite_op->outputs[0]];
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "output tensor is null";
    return RET_NULL_PTR;
  }
  if (GetTfliteDataType(in_tensor->type) != GetTfliteDataType(out_tensor->type) &&
      (GetTfliteDataType(out_tensor->type) == kNumberTypeInt8 ||
       GetTfliteDataType(out_tensor->type) == kNumberTypeUInt8)) {
    std::unique_ptr<schema::QuantDTypeCastT> attr = std::make_unique<schema::QuantDTypeCastT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return RET_NULL_PTR;
    }
    attr->srcT = GetTfliteDataType(in_tensor->type);
    attr->dstT = GetTfliteDataType(out_tensor->type);
    op->primitive->value.type = schema::PrimitiveType_QuantDTypeCast;
    op->primitive->value.value = attr.release();
  } else {
    std::unique_ptr<schema::CastT> attr = std::make_unique<schema::CastT>();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new op failed";
      return RET_NULL_PTR;
    }
    attr->srcT = GetTfliteDataType(in_tensor->type);
    attr->dstT = GetTfliteDataType(out_tensor->type);
    op->primitive->value.type = schema::PrimitiveType_Cast;
    op->primitive->value.value = attr.release();
  }

  AddOpInput(op, tensors_info, tflite_op->inputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  AddOpOutput(op, tensors_info, tflite_op->outputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_tfliteQuantizeParser("QUANTIZE", new TfliteQuantizeParser());
}  // namespace lite
}  // namespace mindspore
