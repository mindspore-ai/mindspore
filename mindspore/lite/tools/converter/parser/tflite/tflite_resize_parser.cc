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
#include <string>
#include "tools/converter/parser/tflite/tflite_resize_parser.h"

namespace mindspore {
namespace lite {
STATUS TfliteResizeParser::Parse(const std::unique_ptr<tflite::OperatorT> &tfliteOp,
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

  std::unique_ptr<schema::ResizeT> attr(new schema::ResizeT());

  std::vector<std::string> node_name_str;
  Split(op->name.data(), &node_name_str, "-");
  const char *node_name = node_name_str.data()->c_str();

  if (std::strcmp(node_name, "ResizeBilinear") == 0) {
    MS_LOG(DEBUG) << "parse TfliteResizeBilinearParser";
    const auto &tfliteAttr = tfliteOp->builtin_options.AsResizeBilinearOptions();
    if (tfliteAttr == nullptr) {
      MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
      return RET_NULL_PTR;
    }
    attr->alignCorners = tfliteAttr->align_corners;
    attr->method = schema::ResizeMethod_BILINEAR;
  }  else if (std::strcmp(node_name, "NearestNeighbor") == 0) {
    MS_LOG(DEBUG) << "parse TfliteResizeNearestNeighborParser";
    const auto &tfliteAttr = tfliteOp->builtin_options.AsResizeNearestNeighborOptions();
    if (tfliteAttr == nullptr) {
      MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
      return RET_NULL_PTR;
    }
    attr->alignCorners = tfliteAttr->align_corners;
    attr->method = schema::ResizeMethod_NEAREST_NEIGHBOR;
  } else {
    MS_LOG(ERROR) << "wrong resize type";
    return RET_ERROR;
  }

  attr->format = schema::Format_NHWC;
  attr->preserveAspectRatio = false;

  auto tfliteResizeTensorIndex = tfliteOp->inputs[1];
  const auto & shape_tensor = tfliteTensors[tfliteResizeTensorIndex];
  if (shape_tensor == nullptr) {
    MS_LOG(ERROR) << "shape_tensor is null";
    return RET_NULL_PTR;
  }
  auto resizeTensorBufferIndex = shape_tensor->buffer;
  const auto & buff = tfliteModelBuffer.at(resizeTensorBufferIndex);
  if (buff == nullptr) {
    MS_LOG(ERROR) << "buff_data is null";
    return RET_NULL_PTR;
  }
  auto buffData = reinterpret_cast<int32_t *>(buff->data.data());
  auto height = buffData[0];
  auto width = buffData[1];
  attr->newWidth = width;
  attr->newHeight = height;

  op->primitive->value.type = schema::PrimitiveType_Resize;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

TfliteNodeRegister g_tfliteResizeBilinearParser("ResizeBilinear", new TfliteResizeBilinearParser());
TfliteNodeRegister g_tfliteResizeNearestNeighborParser("NearestNeighbor",
                                                       new TfliteResizeNearestNeighborParser());
}  // namespace lite
}  // namespace mindspore


