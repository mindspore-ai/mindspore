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

#include "mindspore/lite/tools/converter/parser/tflite/tflite_argmin_parser.h"
#include <memory>
#include <vector>

namespace mindspore {
namespace lite {
STATUS TfliteArgminParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                                 const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                                 const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                 const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                                 schema::CNodeT *op,
                                 TensorCache *tensor_cache,
                                 bool quantizedModel) {
  MS_LOG(DEBUG) << "parse TfliteArgminParser";
  std::unique_ptr<schema::ArgMinT> attr(new schema::ArgMinT());
  const auto &tflite_attr = tfliteOp->builtin_options.AsArgMinOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
  }

  // get axis
  auto axis_idx = tfliteOp->inputs[1];
  std::for_each(tfliteTensors[axis_idx]->shape.begin(), tfliteTensors[axis_idx]->shape.end(), [&](int32_t sha){});
  auto &buf_data = tfliteModelBuffer[tfliteTensors[axis_idx]->buffer];
  auto data_ptr = buf_data->data.data();
  attr->axis = *(static_cast<int32_t *>(static_cast<void *>(data_ptr)));

  // the following use default values
  attr->outMaxValue = false;
  attr->topK = 1;
  attr->keepDims = false;
  attr->axisType = 0;

  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_ArgMin;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

TfliteNodeRegister g_TfliteArgminParser("Argmin", new TfliteArgminParser());
}  // namespace lite
}  // namespace mindspore
