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

#include "tools/converter/parser/onnx/onnx_upsample_parser.h"
#include <string>
#include <vector>
#include <memory>
#include "ops/resize.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxUpsampleParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  // use bilinear method
  auto primitive_c = new (std::nothrow) ops::Resize;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Resize failed";
    return nullptr;
  }

  primitive_c->set_method(mindspore::ResizeMethod::NEAREST);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "mode") {
      if (onnx_node_attr.s() != "nearest" && onnx_node_attr.s() != "linear") {
        MS_LOG(ERROR) << "the UpSample mode don't support now.";
        return nullptr;
      }
      primitive_c->set_method(onnx_node_attr.s() == "nearest" ? mindspore::ResizeMethod::NEAREST
                                                              : mindspore::ResizeMethod::LINEAR);
    }
  }
  primitive_c->set_coordinate_transform_mode(mindspore::CoordinateTransformMode::ASYMMETRIC);

  return primitive_c;
}

OnnxNodeRegistrar g_onnxUpsampleParser("Upsample", new OnnxUpsampleParser());
}  // namespace lite
}  // namespace mindspore
