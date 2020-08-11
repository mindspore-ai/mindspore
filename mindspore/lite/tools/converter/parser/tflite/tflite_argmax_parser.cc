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

#include "tools/converter/parser/tflite/tflite_argmax_parser.h"
#include <memory>
#include <vector>

namespace mindspore {
namespace lite {
STATUS TfliteArgmaxParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                                 const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                                 const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                 const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                                 schema::CNodeT *op,
                                 TensorCache *tensor_cache,
                                 bool quantizedModel) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  MS_LOG(DEBUG) << "parse TfliteArgmaxParser";
  std::unique_ptr<schema::ArgMaxT> attr(new schema::ArgMaxT());

  attr->outMaxValue = false;
  attr->topK = 1;
  attr->keepDims = false;
  attr->axisType = 1;

  auto axis_idx = tfliteOp->inputs[1];
  std::for_each(tfliteTensors[axis_idx]->shape.begin(), tfliteTensors[axis_idx]->shape.end(), [&](int32_t sha){});
  auto &buf_data = tfliteModelBuffer[tfliteTensors[axis_idx]->buffer];
  if (buf_data == nullptr) {
    MS_LOG(ERROR) << "the buf data is null";
    return RET_NULL_PTR;
  }
  auto data_ptr = buf_data->data.data();
  if (data_ptr == nullptr) {
    MS_LOG(ERROR) << "the data is null";
    return RET_NULL_PTR;
  }
  attr->axis = *(static_cast<int32_t *>(static_cast<void *>(data_ptr)));

  op->primitive->value.type = schema::PrimitiveType_ArgMax;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

TfliteNodeRegister g_TfliteArgmaxParser("Argmax", new TfliteArgmaxParser());
}  // namespace lite
}  // namespace mindspore
