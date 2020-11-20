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
STATUS TfliteResizeParser::Parse(TfliteTensorsInfo *tensors_info, const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                 const std::unique_ptr<tflite::ModelT> &tflite_model,
                                 const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, schema::CNodeT *op) {
  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  MS_ASSERT(tflite_subgraph != nullptr);
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::ResizeT> attr = std::make_unique<schema::ResizeT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }
  attr->coordinateTransformMode = schema::CoordinateTransformMode_COMMON;
  std::vector<std::string> node_name_str;
  Split(op->name, &node_name_str, "-");
  const char *node_name = node_name_str.data()->c_str();
  if (std::strcmp(node_name, "ResizeBilinear") == 0) {
    MS_LOG(DEBUG) << "parse TfliteResizeBilinearParser";
    const auto &tfliteAttr = tflite_op->builtin_options.AsResizeBilinearOptions();
    if (tfliteAttr == nullptr) {
      MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
      return RET_NULL_PTR;
    }
    if (tfliteAttr->align_corners) {
      attr->alignCorners = tfliteAttr->align_corners;
      attr->coordinateTransformMode = schema::CoordinateTransformMode_ALIGN_CORNERS;
    }
    if (tfliteAttr->half_pixel_centers) {
      attr->coordinateTransformMode = (attr->coordinateTransformMode == schema::CoordinateTransformMode_COMMON
                                         ? schema::CoordinateTransformMode_TF_HALF_PIXEL
                                         : schema::CoordinateTransformMode_ALIGN_CORNERS_WITH_HALF_PIEXL);
    }
    attr->method = schema::ResizeMethod_LINEAR;
  } else if (std::strcmp(node_name, "NearestNeighbor") == 0) {
    MS_LOG(DEBUG) << "parse TfliteResizeNearestNeighborParser";
    const auto &tfliteAttr = tflite_op->builtin_options.AsResizeNearestNeighborOptions();
    if (tfliteAttr == nullptr) {
      MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
      return RET_NULL_PTR;
    }
    if (tfliteAttr->align_corners) {
      attr->alignCorners = tfliteAttr->align_corners;
      attr->coordinateTransformMode = schema::CoordinateTransformMode_ALIGN_CORNERS;
    }
    if (tfliteAttr->half_pixel_centers) {
      attr->coordinateTransformMode = (attr->coordinateTransformMode == schema::CoordinateTransformMode_COMMON
                                         ? schema::CoordinateTransformMode_TF_HALF_PIXEL
                                         : schema::CoordinateTransformMode_ALIGN_CORNERS_WITH_HALF_PIEXL);
    }
    attr->method = schema::ResizeMethod_NEAREST;
    attr->nearestMode = schema::NearestMode_NORMAL;
  } else {
    MS_LOG(ERROR) << "wrong resize type";
    return RET_ERROR;
  }

  attr->format = schema::Format::Format_NHWC;
  attr->preserveAspectRatio = false;

  auto tfliteResizeTensorIndex = tflite_op->inputs[1];
  const auto &shape_tensor = tflite_subgraph->tensors[tfliteResizeTensorIndex];
  if (shape_tensor == nullptr) {
    MS_LOG(ERROR) << "shape_tensor is null";
    return RET_NULL_PTR;
  }
  auto resizeTensorBufferIndex = shape_tensor->buffer;
  const auto &buff = tflite_model->buffers.at(resizeTensorBufferIndex);
  if (buff == nullptr) {
    MS_LOG(ERROR) << "buff_data is null";
    return RET_NULL_PTR;
  }
  auto buffData = reinterpret_cast<int32_t *>(buff->data.data());
  if (buffData != nullptr) {
    auto height = buffData[0];
    auto width = buffData[1];
    attr->newWidth = width;
    attr->newHeight = height;
  }

  op->primitive->value.type = schema::PrimitiveType_Resize;
  op->primitive->value.value = attr.release();

  AddOpInput(op, tensors_info, tflite_op->inputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  if (buffData == nullptr) {
    AddOpInput(op, tensors_info, tflite_op->inputs[1], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  }
  AddOpOutput(op, tensors_info, tflite_op->outputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_tfliteResizeBilinearParser("ResizeBilinear", new TfliteResizeParser());
TfliteNodeRegister g_tfliteResizeNearestNeighborParser("NearestNeighbor", new TfliteResizeParser());
}  // namespace lite
}  // namespace mindspore
