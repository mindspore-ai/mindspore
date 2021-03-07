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
#include <vector>
#include "tools/converter/parser/onnx/onnx_model_parser.h"
#include "ops/constant_of_shape.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *OnnxConstantOfShapeParser::Parse(const onnx::GraphProto &onnx_graph,
                                                  const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::ConstantOfShape>();

  int data_type = 0;
  std::vector<float> values;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "value") {
      switch (onnx_node_attr.type()) {
        case onnx::AttributeProto_AttributeType_FLOAT:
          data_type = OnnxModelParser::GetDataTypeFromOnnx(onnx::TensorProto_DataType_FLOAT);
          values.push_back(onnx_node_attr.f());
          break;
        case onnx::AttributeProto_AttributeType_INT:
          data_type = OnnxModelParser::GetDataTypeFromOnnx(onnx::TensorProto_DataType_INT32);
          values.push_back(static_cast<float>(onnx_node_attr.i()));
          break;
        case onnx::AttributeProto_AttributeType_TENSOR: {
          const auto &tensor = onnx_node_attr.t();
          auto ret = GetTensorDataFromOnnx(tensor, &values, &data_type);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "get data from tensor failed";
            return nullptr;
          }
        } break;
        default:
          MS_LOG(ERROR) << "Datatype : " << onnx_node_attr.type() << " is not supported.";
          return nullptr;
      }
    }
  }
  if (values.empty()) {
    values = {0};
  }
  prim->set_value(values);
  prim->set_data_type((int64_t)data_type);

  return prim.release();
}

OnnxNodeRegistrar g_onnxConstantOfShapeParser("ConstantOfShape", new OnnxConstantOfShapeParser());
}  // namespace lite
}  // namespace mindspore
