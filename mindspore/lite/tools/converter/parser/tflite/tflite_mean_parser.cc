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
#include "tools/converter/parser/tflite/tflite_mean_parser.h"

namespace mindspore {
namespace lite {
STATUS TfliteMeanParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                               const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                               const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                               const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                               schema::CNodeT *op, TensorCache *tensor_cache, bool quantizedModel) {
  MS_LOG(DEBUG) << "parse TfliteMeanParser";
  std::unique_ptr<schema::MeanT> attr(new schema::MeanT());

  const auto &tflite_attr = tfliteOp->builtin_options.AsReducerOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
    return RET_NULL_PTR;
  }
  attr->keepDims = tflite_attr->keep_dims;

  if (GetTfliteData(tfliteOp->inputs[1], tfliteTensors, tfliteModelBuffer, attr->axis)) {
    MS_LOG(ERROR) << "Mean get axis attr failed";
    return RET_ERROR;
  }

  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Mean;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

TfliteNodeRegister g_tfliteMeanParser("Mean", new TfliteMeanParser());
}  // namespace lite
}  // namespace mindspore
