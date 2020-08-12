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

#include "tools/converter/parser/tflite/tflite_reshape_parser.h"
#include <vector>
#include <memory>
#include <map>

namespace mindspore {
namespace lite {
STATUS TfliteReshapeParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                  const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                                  const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                                  schema::CNodeT *op,
                                  std::vector<int32_t> *tensors_id,
                                  std::vector<schema::Format> *tensors_format,
                                  std::map<int, int>  *tensors_id_map) {
  MS_LOG(DEBUG) << "parse TfliteReshapeParser";

  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::ReshapeT> attr(new schema::ReshapeT());

  const auto &tfliteAttr = tflite_op->builtin_options.AsReshapeOptions();
  if (tfliteAttr == nullptr) {
    if (tflite_op->inputs.size() < 2) {
      MS_LOG(ERROR) << "expected two input tensors, but got: " << tflite_op->inputs.size();
      return RET_ERROR;
    }
    auto shape_tensor_index = tflite_op->inputs[1];
    const auto & shape_tensor = tflite_tensors[shape_tensor_index];
    if (shape_tensor == nullptr) {
      MS_LOG(ERROR) << "shape_tensor is null";
      return RET_NULL_PTR;
    }
    if (GetTfliteData(tflite_op->inputs[1], tflite_tensors, tflite_model_buffer, attr->shape)) {
      MS_LOG(ERROR) << "get reshape -> shape failed";
      return RET_ERROR;
    }
  } else {
    attr->format = schema::Format_NHWC;
    attr->shape.resize(tfliteAttr->new_shape.size());
    for (size_t i = 0; i < tfliteAttr->new_shape.size(); ++i) {
      attr->shape[i] = tfliteAttr->new_shape[i];
    }
  }

  op->primitive->value.type = schema::PrimitiveType_Reshape;
  op->primitive->value.value = attr.release();

  for (int i = 0; i < tflite_op->inputs.size(); i++) {
    AddOpInput(op, tensors_id, tensors_format, tensors_id_map,
               tflite_op->inputs[i], tensors_id->size(), tflite_tensors.size(), schema::Format_NHWC);
  }
  AddOpOutput(op, tensors_id, tensors_format, tensors_id_map,
              tflite_op->outputs[0], tensors_id->size(), tflite_tensors.size(), schema::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_tfliteReshapeParser("Reshape", new TfliteReshapeParser());
}  // namespace lite
}  // namespace mindspore
