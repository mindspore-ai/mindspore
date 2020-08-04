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
#include "tools/converter/parser/tflite/tflite_strided_slice_parser.h"

namespace mindspore {
namespace lite {
STATUS TfliteStridedSliceParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                                       const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                                       const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                       const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                                       schema::CNodeT *op, TensorCache *tensor_cache, bool quantizedModel) {
  MS_LOG(INFO) << "parse TfliteStridedSliceParser";
  std::unique_ptr<schema::StridedSliceT> attr(new schema::StridedSliceT());
  const auto &tflite_attr = tfliteOp->builtin_options.AsStridedSliceOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name << " attr failed";
    return RET_NULL_PTR;
  }

  if (GetTfliteData(tfliteOp->inputs[1], tfliteTensors, tfliteModelBuffer, attr->begin)) {
    MS_LOG(ERROR) << "STRIDED_SLICE get begin attr failed";
    return RET_ERROR;
  }
  if (GetTfliteData(tfliteOp->inputs[2], tfliteTensors, tfliteModelBuffer, attr->end)) {
    MS_LOG(ERROR) << "STRIDED_SLICE get end attr failed";
    return RET_ERROR;
  }
  if (GetTfliteData(tfliteOp->inputs[3], tfliteTensors, tfliteModelBuffer, attr->stride)) {
    MS_LOG(ERROR) << "STRIDED_SLICE get stride attr failed";
    return RET_ERROR;
  }
  attr->beginMask = tflite_attr->begin_mask;
  attr->endMask = tflite_attr->end_mask;
  attr->ellipsisMask = tflite_attr->ellipsis_mask;
  attr->newAxisMask = tflite_attr->new_axis_mask;
  attr->shrinkAxisMask = tflite_attr->shrink_axis_mask;
  // attr->isScale;  // isScale is actually not used in ms-lite

  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_StridedSlice;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

TfliteNodeRegister g_TfliteStridedSliceParser("StridedSlice", new TfliteStridedSliceParser());
}  // namespace lite
}  // namespace mindspore
