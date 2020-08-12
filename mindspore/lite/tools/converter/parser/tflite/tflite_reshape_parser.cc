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

#include <vector>
#include <memory>
#include "tools/converter/parser/tflite/tflite_reshape_parser.h"

namespace mindspore {
namespace lite {
STATUS TfliteReshapeParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                                  const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                                  const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                  const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                                  schema::CNodeT *op, TensorCache *tensor_cache, bool quantizedModel) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  MS_LOG(DEBUG) << "parse TfliteReshapeParser";
  std::unique_ptr<schema::ReshapeT> attr(new schema::ReshapeT());

  const auto &tfliteAttr = tfliteOp->builtin_options.AsReshapeOptions();
  if (tfliteAttr == nullptr) {
    if (tfliteOp->inputs.size() < 2) {
      MS_LOG(ERROR) << "expected two input tensors, but got: " << tfliteOp->inputs.size();
      return RET_ERROR;
    }
    auto shape_tensor_index = tfliteOp->inputs[1];
    const auto & shape_tensor = tfliteTensors[shape_tensor_index];
    if (shape_tensor == nullptr) {
      MS_LOG(ERROR) << "shape_tensor is null";
      return RET_NULL_PTR;
    }
    if (GetTfliteData(tfliteOp->inputs[1], tfliteTensors, tfliteModelBuffer, attr->shape)) {
      MS_LOG(ERROR) << "get reshape->shape error";
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
  return RET_OK;
}

TfliteNodeRegister g_tfliteReshapeParser("Reshape", new TfliteReshapeParser());
}  // namespace lite
}  // namespace mindspore
