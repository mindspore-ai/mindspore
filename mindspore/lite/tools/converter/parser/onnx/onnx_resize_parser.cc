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

#include "tools/converter/parser/onnx/onnx_resize_parser.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mindspore {
namespace lite {
lite::PrimitiveC *OnnxResizeParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                       const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx ResizeParser";
  auto attr = std::make_unique<schema::ResizeT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  attr->format = schema::Format_NCHW;
  attr->nearestMode = schema::NearestMode_ROUND_HALF_DOWN;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "coordinate_transformation_mode") {
      std::map<std::string, schema::CoordinateTransformMode> transform_map = {
        {"half_pixel", schema::CoordinateTransformMode_HALF_PIXEL},
        {"pytorch_half_pixel", schema::CoordinateTransformMode_HALF_PIXEL},
        {"align_corners", schema::CoordinateTransformMode_ALIGN_CORNERS},
        {"asymmetric", schema::CoordinateTransformMode_ASYMMETRIC},
        {"tf_crop_and_resize", schema::CoordinateTransformMode_TF_CROP_AND_RESIZE},
      };
      if (transform_map.find(onnx_node_attr.s()) != transform_map.end()) {
        attr->coordinateTransformMode = transform_map[onnx_node_attr.s()];
      } else {
        MS_LOG(ERROR) << "Unsupport coordinate transform mode: " << attribute_name;
        return nullptr;
      }
    } else if (attribute_name == "cubic_coeff_a") {
      attr->cubicCoeff = onnx_node_attr.f();
    } else if (attribute_name == "exclude_outside") {
      attr->excludeOutside = onnx_node_attr.i();
    } else if (attribute_name == "extrapolation_value") {
      attr->extrapolationValue = onnx_node_attr.f();
    } else if (attribute_name == "mode") {
      attr->method = [&]() {
        std::map<std::string, schema::ResizeMethod> resize_mode = {
          {"nearest", schema::ResizeMethod_NEAREST},
          {"linear", schema::ResizeMethod_LINEAR},
          {"cubic", schema::ResizeMethod_CUBIC},
        };
        return resize_mode[onnx_node_attr.s()];
      }();
    } else if (attribute_name == "nearest_mode") {
      attr->nearestMode = [&]() {
        std::map<std::string, schema::NearestMode> nearest_mode = {
          {"round_prefer_floor", schema::NearestMode_ROUND_HALF_DOWN},
          {"round_prefer_ceil", schema::NearestMode_ROUND_HALF_UP},
          {"floor", schema::NearestMode_FLOOR},
          {"ceil", schema::NearestMode_CEIL},
        };
        return nearest_mode[onnx_node_attr.s()];
      }();
    }
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_Resize;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

OnnxNodeRegistrar g_onnxResizeParser("Resize", new OnnxResizeParser());
}  // namespace lite
}  // namespace mindspore
