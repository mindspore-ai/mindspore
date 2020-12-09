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

#include "tools/converter/parser/onnx/onnx_constant_of_shape_parser.h"
#include <memory>
#include "tools/converter/parser/onnx/onnx_model_parser.h"

namespace mindspore {
namespace lite {
lite::PrimitiveC *OnnxConstantOfShapeParser::ParseLitePrimitive(const onnx::GraphProto &onnx_graph,
                                                                const onnx::NodeProto &onnx_node) {
  MS_LOG(DEBUG) << "onnx ConstantOfShapeParser";
  auto attr = std::make_unique<schema::ConstantOfShapeT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "value") {
      switch (onnx_node_attr.type()) {
        case onnx::AttributeProto_AttributeType_FLOAT:
          attr->dataType = OnnxModelParser::GetDataTypeFromOnnx(onnx::TensorProto_DataType_FLOAT);
          attr->value.push_back(onnx_node_attr.f());
          break;
        case onnx::AttributeProto_AttributeType_INT:
          attr->dataType = OnnxModelParser::GetDataTypeFromOnnx(onnx::TensorProto_DataType_INT32);
          attr->value.push_back(static_cast<float>(onnx_node_attr.i()));
          break;
        case onnx::AttributeProto_AttributeType_TENSOR: {
          const auto &tensor = onnx_node_attr.t();
          auto ret = GetTensorDataFromOnnx(tensor, &attr->value, &attr->dataType);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "get data from tensor failed";
            return nullptr;
          }
        } break;
        default:
          MS_LOG(ERROR) << "The data type is not supported.";
          return nullptr;
      }
    }
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  primitive->value.type = schema::PrimitiveType_ConstantOfShape;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

OnnxNodeRegistrar g_onnxConstantOfShapeParser("ConstantOfShape", new OnnxConstantOfShapeParser());
}  // namespace lite
}  // namespace mindspore
