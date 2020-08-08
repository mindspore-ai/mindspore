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

#include "tools/converter/parser/tflite/tflite_range_parser.h"
#include <vector>
#include <memory>

namespace mindspore {
namespace lite {
STATUS TfliteRangeParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                              const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                              const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                              const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                              schema::CNodeT *op,
                              TensorCache *tensor_cache,
                              bool quantizedModel) {
  MS_LOG(DEBUG) << "parse TfliteRangeParser";
  std::unique_ptr<schema::RangeT> attr(new schema::RangeT());
  const auto &tflite_attr = tfliteOp->builtin_options.AsRangeOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
  }

  auto start_idx = tfliteOp->inputs[1];
  std::for_each(tfliteTensors[start_idx]->shape.begin(), tfliteTensors[start_idx]->shape.end(), [&](int32_t sha){});
  auto &start_buf_data = tfliteModelBuffer[tfliteTensors[start_idx]->buffer];
  auto start_data_ptr = start_buf_data->data.data();
  attr->start = *(static_cast<int32_t *>(static_cast<void *>(start_data_ptr)));

  auto limit_idx = tfliteOp->inputs[2];
  std::for_each(tfliteTensors[limit_idx]->shape.begin(), tfliteTensors[limit_idx]->shape.end(), [&](int32_t sha){});
  auto &limit_buf_data = tfliteModelBuffer[tfliteTensors[limit_idx]->buffer];
  auto limit_data_ptr = limit_buf_data->data.data();
  attr->limit = *(static_cast<int32_t *>(static_cast<void *>(limit_data_ptr)));

  if (tfliteOp->inputs.size() > 2) {
    auto delta_idx = tfliteOp->inputs[3];
    std::for_each(tfliteTensors[delta_idx]->shape.begin(), tfliteTensors[delta_idx]->shape.end(), [&](int32_t sha){});
    auto &delta_buf_data = tfliteModelBuffer[tfliteTensors[delta_idx]->buffer];
    auto delta_data_ptr = delta_buf_data->data.data();
    attr->delta = *(static_cast<int32_t *>(static_cast<void *>(delta_data_ptr)));
  } else {
    attr->delta = 0;    // default
  }

  attr->dType = 0;   // default

  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Range;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

TfliteNodeRegister g_tfliteRangeParser("Range", new TfliteRangeParser());
}  // namespace lite
}  // namespace mindspore


