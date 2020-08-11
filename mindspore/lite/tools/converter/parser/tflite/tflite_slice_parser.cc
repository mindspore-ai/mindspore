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

#include "tools/converter/parser/tflite/tflite_slice_parser.h"
#include <vector>
#include <memory>

namespace mindspore {
namespace lite {
STATUS TfliteSliceParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                                  const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                                  const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                  const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                                  schema::CNodeT *op,
                                  TensorCache *tensor_cache, bool quantizedModel) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  MS_LOG(DEBUG) << "parse TfliteSliceParser";
  std::unique_ptr<schema::SliceT> attr(new schema::SliceT());

  if (GetTfliteData(tfliteOp->inputs[1], tfliteTensors, tfliteModelBuffer, attr->begin)) {
    MS_LOG(ERROR) << "get slice -> begin failed";
    return RET_ERROR;
  }
  if (GetTfliteData(tfliteOp->inputs[2], tfliteTensors, tfliteModelBuffer, attr->size)) {
    MS_LOG(ERROR) << "get slice -> size failed";
    return RET_ERROR;
  }

  op->primitive->value.type = schema::PrimitiveType_Slice;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

TfliteNodeRegister g_tfliteSliceParser("Slice", new TfliteSliceParser());
}  // namespace lite
}  // namespace mindspore

