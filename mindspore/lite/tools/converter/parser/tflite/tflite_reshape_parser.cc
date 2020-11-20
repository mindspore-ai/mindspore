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

namespace mindspore {
namespace lite {
STATUS TfliteReshapeParser::Parse(TfliteTensorsInfo *tensors_info, const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                  const std::unique_ptr<tflite::ModelT> &tflite_model,
                                  const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, schema::CNodeT *op) {
  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  MS_ASSERT(tflite_subgraph != nullptr);
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

  std::unique_ptr<schema::ReshapeT> attr = std::make_unique<schema::ReshapeT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  const auto &tfliteAttr = tflite_op->builtin_options.AsReshapeOptions();
  if (tfliteAttr == nullptr) {
    if (tflite_op->inputs.size() < 2) {
      MS_LOG(ERROR) << "expected two input tensors, but got: " << tflite_op->inputs.size();
      return RET_ERROR;
    }
    auto shape_tensor_index = tflite_op->inputs[1];
    const auto &shape_tensor = tflite_subgraph->tensors[shape_tensor_index];
    if (shape_tensor == nullptr) {
      MS_LOG(ERROR) << "shape_tensor is null";
      return RET_NULL_PTR;
    }
    auto &buf_data = tflite_model->buffers[shape_tensor->buffer];
    if (buf_data == nullptr) {
      MS_LOG(ERROR) << "buf_data is null";
      return RET_NULL_PTR;
    }
    if (!buf_data->data.empty()) {
      if (GetTfliteData(tflite_op->inputs[1], tflite_subgraph->tensors, tflite_model->buffers, attr->shape)) {
        MS_LOG(ERROR) << "get reshape -> shape failed";
        return RET_ERROR;
      }
    }
  } else {
    attr->format = schema::Format::Format_NHWC;
    attr->shape.resize(tfliteAttr->new_shape.size());
    for (size_t i = 0; i < tfliteAttr->new_shape.size(); ++i) {
      attr->shape[i] = tfliteAttr->new_shape[i];
    }
  }

  op->primitive->value.type = schema::PrimitiveType_Reshape;
  op->primitive->value.value = attr.release();

  for (int input : tflite_op->inputs) {
    AddOpInput(op, tensors_info, input, tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  }
  AddOpOutput(op, tensors_info, tflite_op->outputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_tfliteReshapeParser("Reshape", new TfliteReshapeParser());
}  // namespace lite
}  // namespace mindspore
