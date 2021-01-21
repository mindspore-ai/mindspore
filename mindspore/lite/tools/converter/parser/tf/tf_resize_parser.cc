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
#include "tools/converter/parser/tf/tf_resize_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"

namespace mindspore {
namespace lite {
STATUS TFResizeParser::Parse(const tensorflow::NodeDef &tf_op,
                             const std::map<string, const tensorflow::NodeDef *> &tf_node_map, PrimitiveC **primitiveC,
                             std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(INFO) << "TF ResizeParser";
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "New PrimitiveT failed";
    return RET_NULL_PTR;
  }
  auto attr = std::make_unique<schema::ResizeT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new attr failed";
    return RET_NULL_PTR;
  }
  tensorflow::AttrValue attr_value;
  attr->format = schema::Format_NHWC;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "align_corners", &attr_value)) {
    MS_LOG(ERROR) << "The align_corners attr should be specified";
    return RET_ERROR;
  }
  if (attr_value.b()) {
    attr->coordinateTransformMode = schema::CoordinateTransformMode_ALIGN_CORNERS;
  } else {
    attr->coordinateTransformMode = schema::CoordinateTransformMode_ASYMMETRIC;
  }
  if (tf_op.op() == "ResizeBilinear") {
    attr->method = schema::ResizeMethod_LINEAR;
  } else if (tf_op.op() == "ResizeNearestNeighbor") {
    attr->method = schema::ResizeMethod_NEAREST;
  } else {
    attr->method = schema::ResizeMethod_UNKNOWN;
  }
  auto size_node = tf_node_map.at(tf_op.input(1));
  if (size_node == nullptr) {
    MS_LOG(ERROR) << "Find size input failed.";
    return RET_ERROR;
  }
  if (!TensorFlowUtils::FindAttrValue(tf_op, "value", &attr_value)) {
    MS_LOG(WARNING) << "The value attr should be specified";
  }
  auto tensor_proto = attr_value.tensor();
  auto size_ptr = reinterpret_cast<const int32_t *>(tensor_proto.tensor_content().data());
  attr->newHeight = size_ptr[0];
  attr->newWidth = size_ptr[1];

  primitive->value.type = schema::PrimitiveType_Resize;
  primitive->value.value = attr.release();
  *primitiveC = PrimitiveC::Create(primitive.release());
  if (*primitiveC == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_ERROR;
  }

  *output_size = 1;
  auto status = AddOpInput(tf_op, 0, inputs);
  if (status != RET_OK) {
    return status;
  }
  status = AddOpInput(tf_op, 1, inputs);
  return status;
}
TFNodeRegistrar g_tfResizeBilinearParser("ResizeBilinear", new TFResizeParser());
TFNodeRegistrar g_tfResizeNearestNeighborParser("ResizeNearestNeighbor", new TFResizeParser());
}  // namespace lite
}  // namespace mindspore
