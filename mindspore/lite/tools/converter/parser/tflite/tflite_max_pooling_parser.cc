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
#include "tools/converter/parser/tflite/tflite_max_pooling_parser.h"

namespace mindspore {
namespace lite {
STATUS TfliteMaxPoolingParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                     const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                                     const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                     const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                                     schema::CNodeT *op,
                                     TensorCache *tensor_cache, bool quantizedModel) {
  MS_LOG(DEBUG) << "parse TfliteMaxPoolingParser";
  std::unique_ptr<schema::PoolingT> attr(new schema::PoolingT());
  const auto &tflite_attr = tflite_op->builtin_options.AsPool2DOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
    return RET_NULL_PTR;
  }

  attr->format = schema::Format_NHWC;
  // attr->global
  attr->poolingMode = schema::PoolMode_MAX_POOLING;
  attr->windowW = tflite_attr->filter_width;
  attr->windowH = tflite_attr->filter_height;
  attr->strideW = tflite_attr->stride_w;
  attr->strideH = tflite_attr->stride_h;
  attr->padMode = GetPadMode(tflite_attr->padding);

  // calculate pad params

  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Pooling;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

TfliteNodeRegister g_tfliteMaxPoolingParser("MaxPooling", new TfliteMaxPoolingParser());
}  // namespace lite
}  // namespace mindspore


