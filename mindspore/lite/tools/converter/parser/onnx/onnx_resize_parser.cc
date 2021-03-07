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
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "ops/resize.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxResizeParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Resize>();

  prim->set_format(mindspore::Format::NCHW);
  prim->set_nearest_mode(mindspore::NearestMode::ROUND_HALF_DOWN);

  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "coordinate_transformation_mode") {
      std::map<std::string, mindspore::CoordinateTransformMode> transform_map = {
        {"half_pixel", mindspore::CoordinateTransformMode::HALF_PIXEL},
        {"pytorch_half_pixel", mindspore::CoordinateTransformMode::HALF_PIXEL},
        {"align_corners", mindspore::CoordinateTransformMode::ALIGN_CORNERS},
        {"asymmetric", mindspore::CoordinateTransformMode::ASYMMETRIC}};
      if (transform_map.find(onnx_node_attr.s()) != transform_map.end()) {
        prim->set_coordinate_transform_mode(transform_map[onnx_node_attr.s()]);
      } else {
        MS_LOG(ERROR) << "Unsupported coordinate transform mode: " << attribute_name;
        return nullptr;
      }
    } else if (attribute_name == "cubic_coeff_a") {
      prim->set_cubic_coeff(onnx_node_attr.f());
    } else if (attribute_name == "exclude_outside") {
      prim->set_exclude_outside(onnx_node_attr.i());
    } else if (attribute_name == "extrapolation_value") {
      prim->set_extrapolation_value(onnx_node_attr.f());
    } else if (attribute_name == "mode") {
      std::map<std::string, mindspore::ResizeMethod> resize_mode = {
        {"nearest", mindspore::ResizeMethod::NEAREST},
        {"linear", mindspore::ResizeMethod::LINEAR},
        {"cubic", mindspore::ResizeMethod::CUBIC},
      };
      prim->set_method(resize_mode[onnx_node_attr.s()]);
    } else if (attribute_name == "nearest_mode") {
      std::map<std::string, mindspore::NearestMode> nearest_mode = {
        {"round_prefer_floor", mindspore::NearestMode::ROUND_HALF_DOWN},
        {"round_prefer_ceil", mindspore::NearestMode::ROUND_HALF_UP},
        {"floor", mindspore::NearestMode::FLOOR},
        {"ceil", mindspore::NearestMode::CEIL},
      };
      prim->set_nearest_mode(nearest_mode[onnx_node_attr.s()]);
    }
  }

  return prim.release();
}

OnnxNodeRegistrar g_onnxResizeParser("Resize", new OnnxResizeParser());
}  // namespace lite
}  // namespace mindspore
