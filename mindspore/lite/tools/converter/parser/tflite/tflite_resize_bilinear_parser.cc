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
#include "tools/converter/parser/tflite/tflite_resize_bilinear_parser.h"

namespace mindspore {
namespace lite {
STATUS TfliteResizeBilinearParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
                                         const std::vector<std::unique_ptr<tflite::TensorT>> &tfliteTensors,
                                         const std::vector<std::unique_ptr<tflite::BufferT>> &tfliteModelBuffer,
                                         const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tfliteOpSet,
                                         schema::CNodeT *op, TensorCache *tensor_cache, bool quantizedModel) {
  MS_LOG(DEBUG) << "parse TfliteResizeBilinearParser";
  std::unique_ptr<schema::ResizeT> attr(new schema::ResizeT());
  const auto &tfliteAttr = tfliteOp->builtin_options.AsResizeBilinearOptions();
  if (tfliteAttr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
    return RET_NULL_PTR;
  }

  attr->format = schema::Format_NHWC;
  attr->method = schema::ResizeMethod_BILINEAR;
  attr->alignCorners = tfliteAttr->align_corners;
  auto tfliteResizeTensorIndex = tfliteOp->inputs[1];
  auto resizeTensorBufferIndex = tfliteTensors.at(tfliteResizeTensorIndex)->buffer;
  auto buffData = reinterpret_cast<int32_t *>(tfliteModelBuffer.at(resizeTensorBufferIndex)->data.data());
  auto height = buffData[0];
  auto width = buffData[1];
  attr->newWidth = width;
  attr->newHeight = height;
  // attr->preserveAspectRatio

  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_Resize;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

TfliteNodeRegister g_tfliteResizeBilinearParser("ResizeBilinear", new TfliteResizeBilinearParser());
}  // namespace lite
}  // namespace mindspore


