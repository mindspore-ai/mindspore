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

#include "tools/converter/parser/tflite/tflite_resize_parser.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mindspore {
namespace lite {
PrimitiveC *TfliteResizeParser::ParseLitePrimitive(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                   const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto &tflite_subgraph = tflite_model->subgraphs.front();
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is null";
    return nullptr;
  }

  std::unique_ptr<schema::ResizeT> attr = std::make_unique<schema::ResizeT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  attr->coordinateTransformMode = schema::CoordinateTransformMode_ASYMMETRIC;
  auto tflite_op_type = (tflite_model->operator_codes[tflite_op->opcode_index])->builtin_code;
  if (tflite_op_type == tflite::BuiltinOperator_RESIZE_BILINEAR) {
    MS_LOG(DEBUG) << "parse TfliteResizeBilinearParser";
    const auto &tfliteAttr = tflite_op->builtin_options.AsResizeBilinearOptions();
    if (tfliteAttr == nullptr) {
      MS_LOG(ERROR) << "get op ResizeBilinear attr failed";
      return nullptr;
    }
    if (tfliteAttr->align_corners) {
      attr->coordinateTransformMode = schema::CoordinateTransformMode_ALIGN_CORNERS;
    }
    if (tfliteAttr->half_pixel_centers) {
      MS_LOG(ERROR) << "Does not support half pixel centers";
      return nullptr;
    }
    attr->method = schema::ResizeMethod_LINEAR;
  } else if (tflite_op_type == tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR) {
    MS_LOG(DEBUG) << "parse TfliteResizeNearestNeighborParser";
    const auto &tfliteAttr = tflite_op->builtin_options.AsResizeNearestNeighborOptions();
    if (tfliteAttr == nullptr) {
      MS_LOG(ERROR) << "get op ResizeNearestNeighbor attr failed";
      return nullptr;
    }
    if (tfliteAttr->align_corners) {
      attr->coordinateTransformMode = schema::CoordinateTransformMode_ALIGN_CORNERS;
    }
    if (tfliteAttr->half_pixel_centers) {
      MS_LOG(ERROR) << "Does not support half pixel centers";
      return nullptr;
    }
    attr->method = schema::ResizeMethod_NEAREST;
    attr->nearestMode = schema::NearestMode_NORMAL;
  } else {
    MS_LOG(ERROR) << "wrong resize type";
    return nullptr;
  }

  attr->format = schema::Format::Format_NHWC;
  attr->preserveAspectRatio = false;

  auto tfliteResizeTensorIndex = tflite_op->inputs[1];
  const auto &shape_tensor = tflite_subgraph->tensors[tfliteResizeTensorIndex];
  if (shape_tensor == nullptr) {
    MS_LOG(ERROR) << "shape_tensor is null";
    return nullptr;
  }
  auto resizeTensorBufferIndex = shape_tensor->buffer;
  const auto &buff = tflite_model->buffers.at(resizeTensorBufferIndex);
  if (buff == nullptr) {
    MS_LOG(ERROR) << "buff_data is null";
    return nullptr;
  }
  auto buffData = reinterpret_cast<int32_t *>(buff->data.data());
  if (buffData != nullptr) {
    auto height = buffData[0];
    auto width = buffData[1];
    attr->newWidth = width;
    attr->newHeight = height;
  }

  primitive->value.type = schema::PrimitiveType_Resize;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

TfliteNodeRegister g_tfliteResizeBilinearParser(tflite::BuiltinOperator_RESIZE_BILINEAR, new TfliteResizeParser());
TfliteNodeRegister g_tfliteResizeNearestNeighborParser(tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                                                       new TfliteResizeParser());
}  // namespace lite
}  // namespace mindspore
