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

#include "tools/converter/parser/tflite/tflite_argmin_parser.h"
#include <memory>
#include <vector>
#include <map>

namespace mindspore::lite {
PrimitiveC *TfliteArgminParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                   const std::unique_ptr<tflite::ModelT> &tflite_model) {
  const auto &tflite_subgraph = tflite_model->subgraphs.front();
  std::unique_ptr<schema::ArgMinT> attr = std::make_unique<schema::ArgMinT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  attr->outMaxValue = false;
  attr->topK = 1;
  attr->keepDims = false;
  attr->axisType = 1;

  // get axis attr
  auto axis_idx = tflite_op->inputs[1];
  auto buffer_idx = tflite_subgraph->tensors[axis_idx]->buffer;
  auto &buf_data = tflite_model->buffers[buffer_idx];
  if (buf_data == nullptr) {
    MS_LOG(ERROR) << "the buf data is null";
    return nullptr;
  }
  auto data_ptr = buf_data->data.data();
  if (data_ptr == nullptr) {
    MS_LOG(ERROR) << "the data is null";
    return nullptr;
  }
  attr->axis = *(static_cast<int32_t *>(static_cast<void *>(data_ptr)));
  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_ArgMin;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_tfliteArgminParser(tflite::BuiltinOperator_ARG_MIN, new TfliteArgminParser());
}  // namespace mindspore::lite
