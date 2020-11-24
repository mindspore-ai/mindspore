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

namespace mindspore::lite {
lite::PrimitiveC *TfliteReshapeParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  const auto &tflite_subgraph = tflite_model->subgraphs.front();
  std::unique_ptr<schema::ReshapeT> attr = std::make_unique<schema::ReshapeT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  const auto &tflite_attr = tflite_op->builtin_options.AsReshapeOptions();
  if (tflite_attr == nullptr) {
    if (tflite_op->inputs.size() < 2) {
      MS_LOG(ERROR) << "expected two input tensors, but got: " << tflite_op->inputs.size();
      return nullptr;
    }
    auto shape_tensor_index = tflite_op->inputs[1];
    const auto &shape_tensor = tflite_subgraph->tensors[shape_tensor_index];
    if (shape_tensor == nullptr) {
      MS_LOG(ERROR) << "shape_tensor is null";
      return nullptr;
    }
    auto &buf_data = tflite_model->buffers[shape_tensor->buffer];
    if (buf_data == nullptr) {
      MS_LOG(ERROR) << "buf_data is null";
      return nullptr;
    }
    if (!buf_data->data.empty()) {
      if (GetTfliteData(tflite_op->inputs[1], tflite_subgraph->tensors, tflite_model->buffers, attr->shape)) {
        MS_LOG(ERROR) << "get reshape -> shape failed";
        return nullptr;
      }
    }
  } else {
    attr->format = schema::Format::Format_NHWC;
    attr->shape.resize(tflite_attr->new_shape.size());
    for (size_t i = 0; i < tflite_attr->new_shape.size(); ++i) {
      attr->shape[i] = tflite_attr->new_shape[i];
    }
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_Reshape;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_tfliteReshapeParser(tflite::BuiltinOperator_RESHAPE, new TfliteReshapeParser());
}  // namespace mindspore::lite
